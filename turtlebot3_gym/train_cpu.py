#!/usr/bin/env python3

import gymnasium as gym
import turtlebot3_gym
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train TurtleBot3 using PPO (CPU only)')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train for')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    args = parser.parse_args()
    
    print("Creating save directory...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Creating environment...")
    env = gym.make('TurtleBot3-v0')
    
    # Wrap the environment in a vectorized environment
    env = DummyVecEnv([lambda: env])
    
    print("Creating PPO agent (CPU only)...")
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
        device="cpu",  # Force CPU usage
    )
    
    # Train the agent without using checkpoints
    print(f"Training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps)
        
        # Save the final model
        model_path = os.path.join(args.save_dir, "turtlebot3_final")
        model.save(model_path)
        print("Training completed successfully!")
        print(f"Final model saved to {model_path}")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Close the environment
        env.close()

if __name__ == "__main__":
    main()
