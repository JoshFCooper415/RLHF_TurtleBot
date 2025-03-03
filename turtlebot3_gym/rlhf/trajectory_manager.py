#!/usr/bin/env python3

import os
import json
import numpy as np
import pickle
import time

class TrajectoryManager:
    def __init__(self, storage_dir="~/ros2_ws/src/turtlebot3_gym/turtlebot3_gym/rlhf/trajectories"):
        self.storage_dir = os.path.expanduser(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.trajectory_index = self._load_index()
    
    def _load_index(self):
        index_path = os.path.join(self.storage_dir, "index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        with open(os.path.join(self.storage_dir, "index.json"), 'w') as f:
            json.dump(self.trajectory_index, f, indent=2)
    
    def save_trajectory(self, trajectory):
        """Save a trajectory to storage"""
        # Get ID from either 'id' or 'episode_id' field
        trajectory_id = trajectory.get('id', trajectory.get('episode_id'))
        if not trajectory_id:
            trajectory_id = f"trajectory_{time.time()}"
            
        file_path = os.path.join(self.storage_dir, f"{trajectory_id}.pkl")
        
        # Ensure trajectory has required structure
        self._ensure_complete_structure(trajectory)
        
        # Convert numpy arrays to lists for serialization
        serializable_trajectory = self._make_serializable(trajectory)
        
        # Save the trajectory data
        with open(file_path, 'wb') as f:
            pickle.dump(serializable_trajectory, f)
        
        # Update the index
        self.trajectory_index[trajectory_id] = {
            'path': file_path,
            'success': trajectory['metadata']['success'],
            'steps': trajectory['metadata']['steps'],
            'policy_version': trajectory['metadata'].get('policy_version', 'unknown'),
            'timestamp': trajectory['metadata'].get('timestamp', time.time()),
            'obstacle_count': len(trajectory['visualization_data'].get('obstacles', []))
        }
        self._save_index()
        
        return trajectory_id
    
    def _ensure_complete_structure(self, trajectory):
        """Ensure trajectory has complete structure with obstacles"""
        # Create default obstacles if missing
        default_obstacles = [
            {"type": "rectangle", "x": 2.0, "y": 0.0, "width": 1.0, "height": 1.0},
            {"type": "circle", "x": 0.0, "y": 2.0, "radius": 0.5},
            {"type": "rectangle", "x": -2.0, "y": -1.0, "width": 3.0, "height": 0.2, "rotation": 0.7}
        ]
        
        # Initialize key sections if missing
        if 'metadata' not in trajectory:
            trajectory['metadata'] = {}
        
        if 'visualization_data' not in trajectory:
            trajectory['visualization_data'] = {}
        
        # Ensure obstacles exist in both metadata and visualization_data
        if 'obstacles' not in trajectory['metadata'] or not trajectory['metadata']['obstacles']:
            trajectory['metadata']['obstacles'] = default_obstacles
            
        if 'obstacles' not in trajectory['visualization_data'] or not trajectory['visualization_data']['obstacles']:
            trajectory['visualization_data']['obstacles'] = trajectory['metadata']['obstacles']
        
        # Update obstacle count
        trajectory['metadata']['obstacle_count'] = len(trajectory['metadata']['obstacles'])
        
        return trajectory
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_trajectory(self, trajectory_id):
        """Load a trajectory from storage"""
        if trajectory_id not in self.trajectory_index:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        
        file_path = self.trajectory_index[trajectory_id]['path']
        with open(file_path, 'rb') as f:
            trajectory = pickle.load(f)
        
        # Ensure trajectory has complete structure with obstacles
        trajectory = self._ensure_complete_structure(trajectory)
        
        return trajectory
    
    def get_random_pair(self):
        """Get a random pair of trajectories for comparison"""
        if len(self.trajectory_index) < 2:
            raise ValueError("Not enough trajectories for comparison")
        
        trajectory_ids = list(self.trajectory_index.keys())
        # Select two different random trajectories
        idx1, idx2 = np.random.choice(len(trajectory_ids), 2, replace=False)
        
        traj1 = self.load_trajectory(trajectory_ids[idx1])
        traj2 = self.load_trajectory(trajectory_ids[idx2])
        
        # Ensure both trajectories have obstacles
        traj1 = self._ensure_complete_structure(traj1)
        traj2 = self._ensure_complete_structure(traj2)
        
        return (traj1, traj2)
    
    def prepare_comparison_pair(self, traj1, traj2):
        """Prepare two trajectories for comparison in the web interface"""
        # Ensure both trajectories have obstacles
        traj1 = self._ensure_complete_structure(traj1)
        traj2 = self._ensure_complete_structure(traj2)
        
        # Format for web interface
        return {
            'trajectory1': {
                'id': traj1.get('id', traj1.get('episode_id', 'unknown')),
                'visualization_data': {
                    'positions': traj1['visualization_data']['positions'],
                    'target_position': traj1['visualization_data']['target_position'],
                    'obstacles': traj1['visualization_data']['obstacles'],
                    'success': traj1['metadata']['success']
                },
                'metadata': {
                    'success': traj1['metadata']['success'],
                    'steps': traj1['metadata']['steps'],
                    'final_distance': traj1['metadata'].get('final_distance', 0.0),
                    'obstacle_count': len(traj1['visualization_data']['obstacles'])
                }
            },
            'trajectory2': {
                'id': traj2.get('id', traj2.get('episode_id', 'unknown')),
                'visualization_data': {
                    'positions': traj2['visualization_data']['positions'],
                    'target_position': traj2['visualization_data']['target_position'],
                    'obstacles': traj2['visualization_data']['obstacles'],
                    'success': traj2['metadata']['success']
                },
                'metadata': {
                    'success': traj2['metadata']['success'],
                    'steps': traj2['metadata']['steps'],
                    'final_distance': traj2['metadata'].get('final_distance', 0.0),
                    'obstacle_count': len(traj2['visualization_data']['obstacles'])
                }
            }
        }