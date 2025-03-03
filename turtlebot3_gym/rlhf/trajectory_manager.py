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
        trajectory_id = trajectory['episode_id']
        file_path = os.path.join(self.storage_dir, f"{trajectory_id}.pkl")
        
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
            'policy_version': trajectory['metadata']['policy_version'],
            'timestamp': trajectory['metadata']['timestamp']
        }
        self._save_index()
        
        return trajectory_id
    
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
        
        return trajectory
    
    def get_random_pair(self):
        """Get a random pair of trajectories for comparison"""
        if len(self.trajectory_index) < 2:
            raise ValueError("Not enough trajectories for comparison")
        
        trajectory_ids = list(self.trajectory_index.keys())
        # Select two different random trajectories
        idx1, idx2 = np.random.choice(len(trajectory_ids), 2, replace=False)
        
        return (self.load_trajectory(trajectory_ids[idx1]), 
                self.load_trajectory(trajectory_ids[idx2]))