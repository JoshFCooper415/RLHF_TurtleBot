#!/usr/bin/env python3

import gymnasium as gym
import turtlebot3_gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train TurtleBot3 using PPO')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train for')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create the environment
    env = gym.make('TurtleBot3-v0')
    
    # Wrap the environment in a vectorized environment
    env = DummyVecEnv([lambda: env])
    
    # Create the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto",
    )
    
    # Create a checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=args.save_dir,
        name_prefix="turtlebot3_model"
    )
    
    # Train the agent
    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    
    # Save the final model
    model.save(os.path.join(args.save_dir, "turtlebot3_final"))
    
    # Close the environment
    env.close()
    
    print("Training completed successfully!")
    print(f"Final model saved to {os.path.join(args.save_dir, 'turtlebot3_final')}")

if __name__ == "__main__":
    main()
