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

def get_default_obstacles():
    """Return a set of default obstacles matching the simple_obstacles.world"""
    return [
        # Box
        {
            'type': 'rectangle',
            'x': 2.0,
            'y': 0.0,
            'width': 1.0,
            'height': 1.0
        },
        # Cylinder
        {
            'type': 'circle',
            'x': 0.0,
            'y': 2.0,
            'radius': 0.5
        },
        # Wall (rotated rectangle)
        {
            'type': 'rectangle',
            'x': -2.0,
            'y': -1.0,
            'width': 3.0,
            'height': 0.2,
            'rotation': 0.7
        }
    ]

def collect_trajectories_directly(env, policy, num_episodes=10):
    """Record complete trajectories using the given policy with explicit obstacle handling"""
    trajectories = []
    
    # Unwrap the environment to access the original environment
    original_env = env
    while hasattr(original_env, 'env'):
        original_env = original_env.env
    
    # Check if we reached our TurtleBot3Env
    if not hasattr(original_env, 'target_position'):
        print("Warning: Could not access target_position from environment")
    
    # Get the default obstacles to ensure we always have some
    default_obstacles = get_default_obstacles()
    
    for episode in range(num_episodes):
        # Create trajectory structure with direct inclusion of obstacles
        trajectory = {
            'id': f"episode_{episode}_{time.time()}",
            'metadata': {
                'policy_version': getattr(policy, 'version', 'unknown'),
                'timestamp': time.time(),
                'success': False,
                'steps': 0,
                'target_position': None,
                'final_distance': 0.0,
                'obstacles': default_obstacles,  # Always include default obstacles
                'obstacle_count': len(default_obstacles)
            },
            'visualization_data': {
                'positions': [],
                'orientations': [],
                'target_position': None,
                'obstacles': default_obstacles,  # Always include default obstacles
                'success': False,
                'steps': 0
            }
        }
        
        # Try to get environment-specific obstacles if available
        try:
            if hasattr(original_env, 'get_obstacles'):
                env_obstacles = original_env.get_obstacles()
                if env_obstacles and len(env_obstacles) > 0:
                    print(f"Found {len(env_obstacles)} obstacles in environment")
                    trajectory['metadata']['obstacles'] = env_obstacles
                    trajectory['visualization_data']['obstacles'] = env_obstacles
                    trajectory['metadata']['obstacle_count'] = len(env_obstacles)
        except Exception as e:
            print(f"Error getting obstacles from environment: {e}")
            # We already have default obstacles, so continue
        
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Save the target position if available
        if hasattr(original_env, 'target_position'):
            target_pos = original_env.target_position.tolist()
            trajectory['metadata']['target_position'] = target_pos
            trajectory['visualization_data']['target_position'] = target_pos
        
        # Log what we're including in the trajectory
        print(f"Trajectory includes {len(trajectory['visualization_data']['obstacles'])} obstacles")
        
        while not done and step < getattr(original_env, 'max_steps', 500):
            # Get action from policy (handle both callable functions and stable-baselines models)
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs)
            else:
                action = policy(obs)
            
            # Execute action and get result
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Record position and orientation
            if hasattr(original_env, 'robot_position'):
                position = original_env.robot_position.copy().tolist()
                orientation = original_env.robot_orientation
            else:
                position = [0, 0]
                orientation = 0
                
            # Add to visualization data
            trajectory['visualization_data']['positions'].append(position)
            trajectory['visualization_data']['orientations'].append(orientation)
            
            # Update for next step
            obs = next_obs
            step += 1
        
        # Add final outcome information
        success = info.get('success', False)
        trajectory['metadata']['success'] = success
        trajectory['visualization_data']['success'] = success
        trajectory['metadata']['steps'] = step
        trajectory['visualization_data']['steps'] = step
        
        # Calculate final distance if possible
        if hasattr(original_env, 'target_position') and hasattr(original_env, 'robot_position'):
            trajectory['metadata']['final_distance'] = float(np.linalg.norm(
                original_env.target_position - original_env.robot_position))
        else:
            trajectory['metadata']['final_distance'] = 0.0
        
        # Debug verification of obstacles
        obstacle_count = len(trajectory['visualization_data']['obstacles'])
        print(f"Final trajectory has {obstacle_count} obstacles in visualization_data")
        
        trajectories.append(trajectory)
        print(f"Recorded trajectory {episode+1}/{num_episodes}: " +
              f"Success={trajectory['metadata']['success']}, " +
              f"Steps={trajectory['metadata']['steps']}, " +
              f"Obstacles={trajectory['metadata']['obstacle_count']}")
    
    return trajectories

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
    # Use our direct implementation
    trajectories = collect_trajectories_directly(env, policy, num_episodes=num_episodes)
    
    # Save trajectories
    for traj in trajectories:
        manager.save_trajectory(traj)
    
    print(f"Collected and saved {len(trajectories)} trajectories")
    env.close()

def start_feedback_server():
    """Start the web server for human feedback collection"""
    from turtlebot3_gym.rlhf.feedback_server import app
    
    # Patch the feedback server to ensure obstacles are included in comparison pairs
    try:
        from turtlebot3_gym.rlhf.feedback_server import routes
        
        # Find the function that prepares comparison pairs
        if hasattr(routes, 'get_comparison_pair'):
            original_func = routes.get_comparison_pair
            
            def patched_get_comparison_pair():
                result = original_func()
                
                # Ensure obstacles are included
                if 'trajectory1' in result and 'visualization_data' in result['trajectory1']:
                    if 'obstacles' not in result['trajectory1']['visualization_data'] or \
                       result['trajectory1']['visualization_data']['obstacles'] is None:
                        result['trajectory1']['visualization_data']['obstacles'] = get_default_obstacles()
                
                if 'trajectory2' in result and 'visualization_data' in result['trajectory2']:
                    if 'obstacles' not in result['trajectory2']['visualization_data'] or \
                       result['trajectory2']['visualization_data']['obstacles'] is None:
                        result['trajectory2']['visualization_data']['obstacles'] = get_default_obstacles()
                
                return result
            
            # Replace the function
            routes.get_comparison_pair = patched_get_comparison_pair
            print("Successfully patched feedback server to include obstacles")
    except Exception as e:
        print(f"Could not patch feedback server: {e}")
    
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