#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import numpy as np
import time
from turtlebot3_gym.rlhf.trajectory_manager import TrajectoryManager

app = Flask(__name__, 
            template_folder=os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/templates'),
            static_folder=os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/static'))

manager = TrajectoryManager()
feedback_log_path = os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/feedback_log.json')

# Initialize feedback log
if os.path.exists(feedback_log_path):
    with open(feedback_log_path, 'r') as f:
        feedback_log = json.load(f)
else:
    feedback_log = []

@app.route('/')
def index():
    """Main page for trajectory comparison"""
    return render_template('compare.html')

@app.route('/api/get_comparison_pair')
def get_comparison_pair():
    """Get a pair of trajectories to compare"""
    try:
        traj1, traj2 = manager.get_random_pair()
        return jsonify({
            'trajectory1': {
                'id': traj1['episode_id'],
                'metadata': traj1['metadata'],
                'visualization_data': prepare_visualization_data(traj1)
            },
            'trajectory2': {
                'id': traj2['episode_id'],
                'metadata': traj2['metadata'],
                'visualization_data': prepare_visualization_data(traj2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit_preference', methods=['POST'])
def submit_preference():
    """Record a human preference between two trajectories"""
    data = request.json
    
    preference = {
        'preferred_trajectory': data['preferred'],
        'rejected_trajectory': data['rejected'],
        'reason': data.get('reason', ''),
        'timestamp': time.time(),
        'confidence': data.get('confidence', 1.0)
    }
    
    feedback_log.append(preference)
    
    # Save feedback to disk
    with open(feedback_log_path, 'w') as f:
        json.dump(feedback_log, f, indent=2)
    
    return jsonify({'status': 'success'})

def prepare_visualization_data(trajectory):
    """Extract the necessary data for visualizing a trajectory"""
    # Convert trajectory data to a format suitable for visualization
    return {
        'positions': [state['position'] for state in trajectory['states']],
        'orientations': [state['orientation'] for state in trajectory['states']],
        'target_position': trajectory['metadata'].get('target_position', [0, 0]),
        'steps': trajectory['metadata']['steps'],
        'success': trajectory['metadata']['success']
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
