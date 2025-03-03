#!/usr/bin/env python3

import gymnasium as gym
import turtlebot3_gym
import os
import numpy as np
from stable_baselines3 import PPO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained TurtleBot3 model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    args = parser.parse_args()
    
    # Create the environment
    env = gym.make('TurtleBot3-v0')
    
    # Load the trained model
    model = PPO.load(args.model)
    
    # Evaluate the model
    total_rewards = []
    success_count = 0
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if info.get('success', False):
                success_count += 1
                print(f"Episode {episode+1}: Success! Reached target in {step_count} steps.")
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1} finished with reward: {total_reward:.2f}")
    
    # Print evaluation summary
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / args.episodes
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{args.episodes})")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
