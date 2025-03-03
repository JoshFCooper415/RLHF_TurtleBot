#!/usr/bin/env python3

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardModelCallback(BaseCallback):
    """Callback for using a learned reward model with PPO"""
    
    def __init__(self, reward_model, original_weight=0.2, verbose=0):
        super(RewardModelCallback, self).__init__(verbose)
        self.reward_model = reward_model
        self.original_weight = original_weight  # Weight for the original reward
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        
        This is where you can modify rewards using the learned reward model.
        """
        # Calculate the learned reward
        learned_rewards = []
        for obs in self.locals['new_obs']:
            learned_rewards.append(self.reward_model.predict_reward(obs))
        
        # Combine original and learned rewards
        original_rewards = self.locals['rewards']
        combined_rewards = (self.original_weight * original_rewards + 
                           (1 - self.original_weight) * np.array(learned_rewards))
        
        # Replace the rewards
        self.locals['rewards'] = combined_rewards
        
        return True
