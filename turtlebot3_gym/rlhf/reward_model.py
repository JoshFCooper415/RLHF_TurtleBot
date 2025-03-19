#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import os

class RewardModel(nn.Module):
    def __init__(self, input_dim=364):  # Adjust based on your observation space
        super(RewardModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class RewardModelTrainer:
    def __init__(self, model=None, learning_rate=0.001, input_dim=364):
        # Force CPU usage regardless of CUDA availability
        self.device = torch.device("cpu")
        self.model = model if model else RewardModel(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def load_preferences(self, filepath='~/ros2_ws/src/turtlebot3_gym/feedback_log.json'):
        """Load human preferences from feedback log"""
        filepath = os.path.expanduser(filepath)
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, 'r') as f:
            preferences = json.load(f)
        return preferences
    
    def prepare_training_data(self, preferences, trajectory_manager):
        """Convert preferences to training examples"""
        training_data = []
        
        # Handle both the original data structure and newer formats
        if isinstance(preferences, dict) and 'feedback' in preferences:
            preference_list = preferences['feedback']
        elif isinstance(preferences, dict) and 'comparisons' in preferences:
            preference_list = preferences['comparisons'] + preferences.get('feedback', [])
        else:
            preference_list = preferences
        
        for pref in preference_list:
            # Skip entries where human couldn't decide
            preferred_id = pref.get('preferred_trajectory', pref.get('preferred'))
            if preferred_id == 'similar':
                continue
            
            rejected_id = pref.get('rejected_trajectory', pref.get('rejected'))
            
            try:
                preferred_traj = trajectory_manager.load_trajectory(preferred_id)
                rejected_traj = trajectory_manager.load_trajectory(rejected_id)
                
                # Convert observations to tensors
                preferred_obs = [torch.tensor(obs, dtype=torch.float32) 
                            for obs in preferred_traj['observations']]
                rejected_obs = [torch.tensor(obs, dtype=torch.float32) 
                            for obs in rejected_traj['observations']]
                
                # Create pairs of observations for comparison
                # We can sample a few points from each trajectory
                for _ in range(10):  # Sample 10 pairs
                    if len(preferred_obs) > 0 and len(rejected_obs) > 0:
                        p_idx = random.randint(0, len(preferred_obs) - 1)
                        r_idx = random.randint(0, len(rejected_obs) - 1)
                        
                        training_data.append({
                            'preferred': preferred_obs[p_idx],
                            'rejected': rejected_obs[r_idx],
                            'confidence': pref.get('confidence', 1.0)
                        })
            except Exception as e:
                print(f"Error processing preference {preferred_id} vs {rejected_id}: {e}")
        
        return training_data
    
    def train(self, training_data, epochs=100, batch_size=64):
        """Train the reward model on human preference data"""
        if len(training_data) == 0:
            print("No training data available")
            return
            
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            random.shuffle(training_data)
            
            # Process in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                batch_loss = self._train_batch(batch)
                epoch_loss += batch_loss
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(training_data):.4f}")
    
    def _train_batch(self, batch):
        self.optimizer.zero_grad()
        
        # Compute rewards for both preferred and rejected trajectories
        preferred_rewards = torch.stack([self.model(item['preferred'].to(self.device)) 
                                       for item in batch])
        rejected_rewards = torch.stack([self.model(item['rejected'].to(self.device)) 
                                      for item in batch])
        
        # The probability that the preferred trajectory has higher reward
        logits = preferred_rewards - rejected_rewards
        
        # Target is always 1 (preferred should have higher reward)
        targets = torch.ones_like(logits) * torch.tensor([[item['confidence']] for item in batch])
        targets = targets.to(self.device)
        
        # Compute loss and update weights
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item() * len(batch)
    
    def save_model(self, filepath='~/ros2_ws/src/turtlebot3_gym/reward_model.pt'):
        """Save the trained model"""
        filepath = os.path.expanduser(filepath)
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath='~/ros2_ws/src/turtlebot3_gym/reward_model.pt'):
        """Load a trained model"""
        filepath = os.path.expanduser(filepath)
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.eval()
            return True
        return False
    
    def predict_reward(self, observation):
        """Predict reward for a given observation"""
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
            else:
                obs_tensor = observation.to(self.device)
            return self.model(obs_tensor).item()
