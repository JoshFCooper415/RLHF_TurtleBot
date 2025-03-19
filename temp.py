#!/usr/bin/env python3

import os
import json
import pickle
import argparse
from pathlib import Path


def ensure_complete_structure(trajectory):
    """Ensure trajectory has complete structure with obstacles"""
    # Create default obstacles if missing
    #default_obstacles = #get_fallback_obstacles()
    
    # Initialize key sections if missing
    if 'metadata' not in trajectory:
        trajectory['metadata'] = {}
    
    if 'visualization_data' not in trajectory:
        trajectory['visualization_data'] = {}
    
    # Ensure obstacles exist in both metadata and visualization_data
    if 'obstacles' not in trajectory['metadata'] or not trajectory['metadata']['obstacles']:
        print(f"Adding obstacles to metadata")
        #trajectory['metadata']['obstacles'] = default_obstacles
            
    if 'obstacles' not in trajectory['visualization_data'] or not trajectory['visualization_data']['obstacles']:
        print(f"Adding obstacles to visualization_data")
        trajectory['visualization_data']['obstacles'] = trajectory['metadata']['obstacles']
    
    # Update obstacle count
    trajectory['metadata']['obstacle_count'] = len(trajectory['metadata']['obstacles'])
    
    # Ensure other required fields exist
    if 'positions' not in trajectory['visualization_data']:
        trajectory['visualization_data']['positions'] = [[0, 0]]
    
    if 'target_position' not in trajectory['visualization_data']:
        trajectory['visualization_data']['target_position'] = [1, 1]
    
    if 'success' not in trajectory['metadata']:
        trajectory['metadata']['success'] = False
    
    if 'success' not in trajectory['visualization_data']:
        trajectory['visualization_data']['success'] = trajectory['metadata']['success']
    
    if 'steps' not in trajectory['metadata']:
        trajectory['metadata']['steps'] = len(trajectory['visualization_data']['positions'])
    
    return trajectory

def make_json_serializable(obj):
    """Convert objects like numpy arrays to standard Python types"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # Handle numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='Convert trajectory pkl files to JSON and print structure')
    parser.add_argument('--dir', type=str, default='~/ros2_ws/src/turtlebot3_gym/turtlebot3_gym/rlhf/trajectories',
                        help='Directory containing trajectory pkl files')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save JSON files (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print full JSON content')
    args = parser.parse_args()
    
    trajectory_dir = os.path.expanduser(args.dir)
    if not os.path.exists(trajectory_dir):
        print(f"Error: Directory not found: {trajectory_dir}")
        return
    
    # Create output directory if specified
    output_dir = None
    if args.output:
        output_dir = os.path.expanduser(args.output)
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all pkl files
    pkl_files = list(Path(trajectory_dir).glob('*.pkl'))
    print(f"Found {len(pkl_files)} trajectory files")
    
    # Process each file
    for i, pkl_path in enumerate(pkl_files):
        print(f"\nProcessing {i+1}/{len(pkl_files)}: {pkl_path.name}")
        
        try:
            # Load the pickle file
            with open(pkl_path, 'rb') as f:
                trajectory = pickle.load(f)
            
            # Ensure complete structure
            trajectory = ensure_complete_structure(trajectory)
            
            # Convert to JSON-serializable format
            json_data = make_json_serializable(trajectory)
            
            # Print structure information
            print(f"ID: {trajectory.get('id', 'Unknown')}")
            print(f"Metadata keys: {list(trajectory['metadata'].keys())}")
            print(f"Visualization data keys: {list(trajectory['visualization_data'].keys())}")
            
            # Check obstacles
            if 'obstacles' in trajectory['visualization_data'] and trajectory['visualization_data']['obstacles']:
                obstacle_count = len(trajectory['visualization_data']['obstacles'])
                print(f"Obstacles found: {obstacle_count}")
                
                # Print first obstacle as sample
                if obstacle_count > 0:
                    print(f"First obstacle: {trajectory['visualization_data']['obstacles'][0]}")
            else:
                print("No obstacles found in visualization_data")
            
            # Print full JSON if requested
            if args.verbose:
                json_str = json.dumps(json_data, indent=2)
                print(f"Full JSON content:\n{json_str}")
            
            # Save to file if output directory is specified
            if output_dir:
                output_path = os.path.join(output_dir, f"{pkl_path.stem}.json")
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {pkl_path.name}: {e}")
    
    print("\nProcessing complete")

if __name__ == "__main__":
    main()