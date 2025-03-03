#!/usr/bin/env python3

import time
import numpy as np
import gymnasium as gym

def record_trajectory(env, policy, num_episodes=10):
    """Record complete trajectories using the given policy"""
    trajectories = []
    
    # Unwrap the environment to access the original environment
    original_env = env
    while hasattr(original_env, 'env'):
        original_env = original_env.env
    
    # Check if we reached our TurtleBot3Env
    if not hasattr(original_env, 'target_position'):
        print("Warning: Could not access target_position from environment")
    
    for episode in range(num_episodes):
        trajectory = {
            'states': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'episode_id': f"episode_{episode}_{time.time()}",
            'metadata': {
                'policy_version': getattr(policy, 'version', 'unknown'),
                'timestamp': time.time(),
                'success': False,
                'steps': 0,
                'target_position': None
            }
        }
        
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Save the target position if available
        if hasattr(original_env, 'target_position'):
            trajectory['metadata']['target_position'] = original_env.target_position.tolist()
        
        while not done and step < getattr(original_env, 'max_steps', 500):
            # Get action from policy (handle both callable functions and stable-baselines models)
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs)
            else:
                action = policy(obs)
            
            # Execute action and get result
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Record everything
            if hasattr(original_env, 'robot_position'):
                position = original_env.robot_position.copy()
                orientation = original_env.robot_orientation
            else:
                position = [0, 0]
                orientation = 0
                
            trajectory['states'].append({
                'position': position,
                'orientation': orientation,
                'velocity': info.get('velocity', [0, 0])
            })
            
            # Handle different action types
            if isinstance(action, np.ndarray):
                action_to_save = action.copy()
            else:
                action_to_save = action
                
            trajectory['actions'].append(action_to_save)
            trajectory['observations'].append(obs.copy())
            trajectory['rewards'].append(reward)
            
            # Update for next step
            obs = next_obs
            step += 1
        
        # Add final outcome information
        trajectory['metadata']['success'] = info.get('success', False)
        trajectory['metadata']['steps'] = step
        
        # Calculate final distance if possible
        if hasattr(original_env, 'target_position') and hasattr(original_env, 'robot_position'):
            trajectory['metadata']['final_distance'] = np.linalg.norm(
                original_env.target_position - original_env.robot_position)
        else:
            trajectory['metadata']['final_distance'] = 0.0
        
        trajectories.append(trajectory)
        print(f"Recorded trajectory {episode+1}/{num_episodes}: " +
              f"Success={trajectory['metadata']['success']}, " +
              f"Steps={trajectory['metadata']['steps']}")
    
    return trajectories