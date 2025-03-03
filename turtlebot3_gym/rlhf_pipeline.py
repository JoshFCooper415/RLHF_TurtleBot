#!/usr/bin/env python3

import os
import time
import gymnasium as gym
import turtlebot3_gym
import numpy as np
import torch
import argparse
from stable_baselines3 import PPO
from turtlebot3_gym.rlhf.trajectory_manager import TrajectoryManager
from turtlebot3_gym.rlhf.reward_model import RewardModel, RewardModelTrainer
from turtlebot3_gym.envs.trajectory_collector import record_trajectory
from turtlebot3_gym.rlhf.ppo_reward_callback import RewardModelCallback
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='RLHF Pipeline for TurtleBot3')
    parser.add_argument('--collect_trajectories', action='store_true', 
                        help='Collect new trajectories')
    parser.add_argument('--num_trajectories', type=int, default=10,
                        help='Number of trajectories to collect')
    parser.add_argument('--train_reward', action='store_true',
                        help='Train the reward model')
    parser.add_argument('--start_feedback_server', action='store_true',
                        help='Start the feedback collection server')
    parser.add_argument('--train_policy', action='store_true',
                        help='Train policy with learned reward')
    parser.add_argument('--policy_steps', type=int, default=100000,
                        help='Training steps for policy')
    parser.add_argument('--original_reward_weight', type=float, default=0.2,
                        help='Weight for original reward (0-1)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the trained policy')
    
    return parser.parse_args()

def collect_trajectories(num_episodes=10, random_policy=True):
    """Collect trajectories for human feedback"""
    env = gym.make('TurtleBot3-v0')
    manager = TrajectoryManager()
    
    if random_policy:
        # Use a random policy
        def policy(obs):
            return env.action_space.sample()
    else:
        # Use the trained policy if available
        if os.path.exists(os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/policy_model.zip')):
            policy = PPO.load(os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/policy_model.zip'))
            print("Using trained policy for trajectory collection")
        else:
            def policy(obs):
                return env.action_space.sample()
            print("No trained policy found, using random policy")
    
    print(f"Collecting {num_episodes} trajectories...")
    trajectories = record_trajectory(env, policy, num_episodes=num_episodes)
    
    # Save trajectories
    for traj in trajectories:
        manager.save_trajectory(traj)
    
    print(f"Collected and saved {len(trajectories)} trajectories")
    env.close()

def start_feedback_server():
    """Start the web server for human feedback collection"""
    from turtlebot3_gym.rlhf.feedback_server import app
    print("Starting feedback server at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)

def train_reward_model():
    """Train the reward model from human feedback"""
    manager = TrajectoryManager()
    trainer = RewardModelTrainer()
    
    print("Loading human preferences...")
    preferences = trainer.load_preferences()
    if len(preferences) == 0:
        print("No human preferences found. Please collect feedback first.")
        return
    
    print(f"Found {len(preferences)} preference data points")
    
    # Prepare training data
    print("Preparing training data...")
    training_data = trainer.prepare_training_data(preferences, manager)
    
    if len(training_data) == 0:
        print("No valid training examples could be created. Please check your feedback data.")
        return
    
    print(f"Created {len(training_data)} training examples")
    
    # Train the reward model
    print("Training reward model...")
    trainer.train(training_data, epochs=50)
    
    # Save the trained model
    trainer.save_model()
    print("Reward model trained and saved")

def train_policy_with_learned_reward(policy_steps=100000, original_reward_weight=0.2):
    """Train a policy using the learned reward function"""
    env = gym.make('TurtleBot3-v0')
    
    # Load reward model
    reward_model = RewardModel()
    trainer = RewardModelTrainer(model=reward_model)
    
    if not trainer.load_model():
        print("No trained reward model found. Please train the reward model first.")
        return
    
    print("Successfully loaded reward model")
    
    # Create callback for reward modification
    reward_callback = RewardModelCallback(
        trainer, 
        original_weight=original_reward_weight, 
        verbose=1
    )
    
    # Train policy
    print(f"Training policy with learned reward (original reward weight: {original_reward_weight})...")
    print(f"Training for {policy_steps} steps...")
    
    # Initialize PPO agent
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    
    # Train with callback
    model.learn(total_timesteps=policy_steps, callback=reward_callback)
    
    # Save the policy
    model_path = os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/policy_model.zip')
    model.save(model_path)
    print(f"Policy trained and saved to {model_path}")
    env.close()

def evaluate_policy(num_episodes=10):
    """Evaluate the trained policy"""
    env = gym.make('TurtleBot3-v0')
    model_path = os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/policy_model.zip')
    
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        
        successes = 0
        total_rewards = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                step += 1
                
                if info.get('success', False):
                    successes += 1
            
            total_rewards += episode_reward
            print(f"Episode {episode+1}: Reward={episode_reward:.2f}, " +
                  f"Steps={step}, Success={info.get('success', False)}")
        
        print(f"\nEvaluation over {num_episodes} episodes:")
        print(f"Average reward: {total_rewards/num_episodes:.2f}")
        print(f"Success rate: {successes/num_episodes:.2f} ({successes}/{num_episodes})")
    else:
        print("No trained policy found. Please train a policy first.")
    
    env.close()

def main():
    """Main function that runs the RLHF pipeline based on command-line arguments"""
    args = parse_args()
    
    if args.collect_trajectories:
        collect_trajectories(num_episodes=args.num_trajectories)
    
    if args.start_feedback_server:
        start_feedback_server()
    
    if args.train_reward:
        train_reward_model()
    
    if args.train_policy:
        train_policy_with_learned_reward(
            policy_steps=args.policy_steps, 
            original_reward_weight=args.original_reward_weight
        )
    
    if args.evaluate:
        evaluate_policy()
    
    if not any([args.collect_trajectories, args.start_feedback_server, 
                args.train_reward, args.train_policy, args.evaluate]):
        print("No action specified. Use --help to see available options.")

if __name__ == "__main__":
    main()