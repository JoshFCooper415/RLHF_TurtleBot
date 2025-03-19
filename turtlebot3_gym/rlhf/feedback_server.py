#!/usr/bin/env python3

import os
import json
import random
import time
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import logging
import sys
from turtlebot3_gym.gazebo_world_parser import get_obstacles_from_world

# Configure Flask app with correct template paths
# Find the package directory to correctly reference templates and static files
package_dir = os.path.expanduser("~/ros2_ws/src/turtlebot3_gym")
template_dir = os.path.join(package_dir, "templates")
static_dir = os.path.join(package_dir, "static")

# Create app with properly configured template and static folders
app = Flask(__name__, 
            static_folder=static_dir, 
            template_folder=template_dir)
app.config['SECRET_KEY'] = 'rlhf_feedback_server_key'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('feedback_server')

# Global variables
trajectories = []
feedback_data = []
pairwise_comparisons = []
active_ros_node = None

def get_obstacles():
    """Get obstacles from Gazebo world file"""
    try:
        obstacles = get_obstacles_from_world(logger=logger)
        logger.info(f"Loaded {len(obstacles)} obstacles from world file")
        return obstacles
    except Exception as e:
        logger.error(f"Error loading obstacles from world file: {e}")
        raise RuntimeError("Failed to load obstacles from world file")

def ros_spin_thread(node):
    """Thread function to spin the ROS node"""
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except Exception as e:
        node.get_logger().error(f"Error in ROS executor: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()

def load_trajectories():
    """Load trajectories from files in the trajectory directory"""
    global trajectories
    
    trajectory_dir = os.path.expanduser("~/ros2_ws/src/turtlebot3_gym/turtlebot3_gym/rlhf/trajectories")
    if not os.path.exists(trajectory_dir):
        logger.warning(f"Trajectory directory not found: {trajectory_dir}")
        return False
    
    # Load obstacles from world file
    try:
        obstacles = get_obstacles() 
        print("test")
        print(obstacles)
    except Exception as e:
        logger.error(f"Error getting obstacles: {e}")
        return False
    
    # Load index file if it exists
    index_path = os.path.join(trajectory_dir, "index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
                
            loaded_trajectories = []
            
            # Load each trajectory file from the index
            for traj_id, traj_info in index.items():
                traj_path = traj_info.get('path')
                if traj_path and os.path.exists(traj_path):
                    try:
                        with open(traj_path, 'rb') as f:
                            import pickle
                            trajectory = pickle.load(f)
                        
                        # Ensure trajectory has ID
                        if 'id' not in trajectory:
                            trajectory['id'] = traj_id
                            
                        # Ensure complete structure
                        trajectory = ensure_complete_structure(trajectory, obstacles)
                        loaded_trajectories.append(trajectory)
                    except Exception as e:
                        logger.error(f"Error loading trajectory {traj_id}: {e}")
            
            trajectories = loaded_trajectories
            logger.info(f"Loaded {len(trajectories)} trajectories from index")
            
            # Debug log to check obstacle presence
            for i, traj in enumerate(trajectories[:3]):  # Log first 3 for brevity
                has_obstacles = 'obstacles' in traj['visualization_data'] and traj['visualization_data']['obstacles']
                obstacle_count = len(traj['visualization_data'].get('obstacles', []))
                logger.info(f"Trajectory {i}: Has obstacles: {has_obstacles}, Count: {obstacle_count}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading trajectory index: {e}")
    
    # If we couldn't load from index, look for pickle files directly
    try:
        traj_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl')]
        loaded_trajectories = []
        
        for traj_file in traj_files:
            try:
                file_path = os.path.join(trajectory_dir, traj_file)
                with open(file_path, 'rb') as f:
                    import pickle
                    trajectory = pickle.load(f)
                
                # Ensure trajectory has ID
                if 'id' not in trajectory:
                    trajectory['id'] = os.path.splitext(traj_file)[0]
                    
                # Ensure complete structure with obstacles
                trajectory = ensure_complete_structure(trajectory, obstacles)
                loaded_trajectories.append(trajectory)
            except Exception as e:
                logger.error(f"Error loading trajectory {traj_file}: {e}")
        
        trajectories = loaded_trajectories
        logger.info(f"Loaded {len(trajectories)} trajectories directly from files")
        
        # Debug log to check obstacle presence
        for i, traj in enumerate(trajectories[:3]):  # Log first 3 for brevity
            has_obstacles = 'obstacles' in traj['visualization_data'] and traj['visualization_data']['obstacles']
            obstacle_count = len(traj['visualization_data'].get('obstacles', []))
            logger.info(f"Trajectory {i}: Has obstacles: {has_obstacles}, Count: {obstacle_count}")
            
        return True
    except Exception as e:
        logger.error(f"Error loading trajectories from files: {e}")
        return False

def ensure_complete_structure(trajectory, obstacles):
    """Ensure trajectory has complete structure with obstacles"""
    # Initialize key sections if missing
    if 'metadata' not in trajectory:
        trajectory['metadata'] = {}
    
    if 'visualization_data' not in trajectory:
        trajectory['visualization_data'] = {}
    
    # Ensure obstacles exist in both metadata and visualization_data
    if 'obstacles' not in trajectory['metadata'] or not trajectory['metadata']['obstacles']:
        trajectory['metadata']['obstacles'] = obstacles
            
    if 'obstacles' not in trajectory['visualization_data'] or not trajectory['visualization_data']['obstacles']:
        trajectory['visualization_data']['obstacles'] = obstacles
    
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

def load_feedback():
    """Load existing feedback data"""
    global feedback_data, pairwise_comparisons
    
    feedback_file = os.path.expanduser("~/ros2_ws/src/turtlebot3_gym/feedback_log.json")
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                feedback_data = data.get('feedback', [])
                pairwise_comparisons = data.get('comparisons', [])
            elif isinstance(data, list):
                feedback_data = data
                
            logger.info(f"Loaded {len(feedback_data)} feedback entries and {len(pairwise_comparisons)} comparisons")
            return True
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
    
    return False

def save_feedback():
    """Save feedback data to file"""
    try:
        feedback_file = os.path.expanduser("~/ros2_ws/src/turtlebot3_gym/feedback_log.json")
        with open(feedback_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'feedback': feedback_data,
                'comparisons': pairwise_comparisons
            }, f, indent=2)
        logger.info(f"Saved {len(feedback_data)} feedback entries to {feedback_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False

def select_comparison_pair():
    """Select two trajectories to compare"""
    if len(trajectories) < 2:
        logger.error("Not enough trajectories for comparison")
        raise ValueError("Not enough trajectories for comparison")
    
    # Ensure we have obstacles for visualization
    obstacles = get_obstacles()
    
    # Select two different trajectories
    indices = random.sample(range(len(trajectories)), 2)
    
    # Helper function to create states from positions
    def create_states_from_positions(positions):
        return [{"position": pos} for pos in positions]
    
    # Create properly structured comparison data
    pair = {
        'trajectory1': {
            'id': trajectories[indices[0]].get('id', f'traj_{indices[0]}'),
            # Add states at the root level in the format expected by the visualizer
            'states': create_states_from_positions(trajectories[indices[0]]['visualization_data'].get('positions', [[0, 0]])),
            'visualization_data': {
                'positions': trajectories[indices[0]]['visualization_data'].get('positions', [[0, 0]]),
                'orientations': trajectories[indices[0]]['visualization_data'].get('orientations', [0]),
                'target_position': trajectories[indices[0]]['visualization_data'].get('target_position', [1, 1]),
                'obstacles': obstacles,  # Use obstacles from world file
                'success': trajectories[indices[0]]['metadata'].get('success', False),
                # Add states in visualization_data too
                'states': create_states_from_positions(trajectories[indices[0]]['visualization_data'].get('positions', [[0, 0]]))
            },
            'metadata': {
                'success': trajectories[indices[0]]['metadata'].get('success', False),
                'steps': trajectories[indices[0]]['metadata'].get('steps', 0),
                'final_distance': trajectories[indices[0]]['metadata'].get('final_distance', 1.0),
                'obstacle_count': len(obstacles)
            }
        },
        'trajectory2': {
            'id': trajectories[indices[1]].get('id', f'traj_{indices[1]}'),
            # Add states at the root level in the format expected by the visualizer
            'states': create_states_from_positions(trajectories[indices[1]]['visualization_data'].get('positions', [[0, 0]])),
            'visualization_data': {
                'positions': trajectories[indices[1]]['visualization_data'].get('positions', [[0, 0]]),
                'orientations': trajectories[indices[1]]['visualization_data'].get('orientations', [0]),
                'target_position': trajectories[indices[1]]['visualization_data'].get('target_position', [1, 1]),
                'obstacles': obstacles,  # Use obstacles from world file
                'success': trajectories[indices[1]]['metadata'].get('success', False),
                # Add states in visualization_data too
                'states': create_states_from_positions(trajectories[indices[1]]['visualization_data'].get('positions', [[0, 0]]))
            },
            'metadata': {
                'success': trajectories[indices[1]]['metadata'].get('success', False),
                'steps': trajectories[indices[1]]['metadata'].get('steps', 0),
                'final_distance': trajectories[indices[1]]['metadata'].get('final_distance', 1.0),
                'obstacle_count': len(obstacles)
            }
        }
    }
    
    # Log the structure for debugging
    logger.info(f"Generated comparison with states arrays: " + 
               f"T1 states: {len(pair['trajectory1']['states'])}, " +
               f"T2 states: {len(pair['trajectory2']['states'])}")
    
    return pair

def sanitize_trajectory_for_json(trajectory):
    """Ensure trajectory data is JSON serializable"""
    if isinstance(trajectory, dict):
        result = {}
        for key, value in trajectory.items():
            if key == 'obstacles' and value is None:
                # Replace None obstacles with obstacles from world file
                result[key] = get_obstacles()
            elif isinstance(value, (dict, list)):
                result[key] = sanitize_trajectory_for_json(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                result[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                result[key] = int(value)
            else:
                result[key] = value
        return result
    elif isinstance(trajectory, list):
        return [sanitize_trajectory_for_json(item) for item in trajectory]
    else:
        return trajectory

# Flask routes
@app.route('/')
def index():
    """Main page listing all trajectories"""
    return render_template('index.html', 
                          feedback_count=len(feedback_data),
                          rated_count=len(pairwise_comparisons),
                          total_count=len(trajectories),
                          trajectories=trajectories)

@app.route('/compare')
def compare():
    """Page for comparing trajectories"""
    return render_template('compare.html')

@app.route('/trajectory/<trajectory_id>')
def view_trajectory(trajectory_id):
    """View a single trajectory"""
    # Find the trajectory
    trajectory = None
    for traj in trajectories:
        if traj.get('id') == trajectory_id:
            trajectory = traj
            break
    
    if not trajectory:
        return "Trajectory not found", 404
    
    # Get feedback for this trajectory
    traj_feedback = [f for f in feedback_data if f.get('trajectory_id') == trajectory_id]
    
    # Ensure trajectory has obstacles
    obstacles = get_obstacles()
    if 'visualization_data' in trajectory:
        trajectory['visualization_data']['obstacles'] = obstacles
    if 'metadata' in trajectory:
        trajectory['metadata']['obstacles'] = obstacles
        trajectory['metadata']['obstacle_count'] = len(obstacles)
    
    return render_template('trajectory.html', 
                           trajectory=trajectory,
                           feedback=traj_feedback)

@app.route('/refresh', methods=['POST'])
def refresh_trajectories():
    """Refresh the trajectory list"""
    success = load_trajectories()
    return jsonify({
        'success': success,
        'message': "Trajectories refreshed" if success else "Error refreshing trajectories",
        'feedback_count': len(feedback_data),
        'rated_count': len(set([f.get('trajectory_id') for f in feedback_data])),
    })

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a trajectory"""
    try:
        trajectory_id = request.form.get('trajectory_id')
        rating = int(request.form.get('rating', 0))
        comment = request.form.get('comment', '')
        
        if not trajectory_id or rating < 1 or rating > 5:
            return jsonify({
                'success': False,
                'message': "Invalid feedback data"
            })
        
        # Create feedback entry
        feedback_entry = {
            'trajectory_id': trajectory_id,
            'rating': rating,
            'comment': comment,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        feedback_data.append(feedback_entry)
        save_feedback()
        
        return jsonify({
            'success': True,
            'message': "Feedback submitted successfully",
            'feedback_count': len(feedback_data)
        })
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        })

@app.route('/api/get_comparison_pair', methods=['GET'])
def get_comparison_pair():
    """API endpoint to get a pair of trajectories to compare"""
    # Get the trajectories
    pair = select_comparison_pair()
    
    # Sanitize data to ensure it's JSON serializable
    sanitized_pair = sanitize_trajectory_for_json(pair)
    
    # Force obstacles if they're still missing (final check)
    obstacles = get_obstacles()
    if 'visualization_data' in sanitized_pair['trajectory1']:
        if 'obstacles' not in sanitized_pair['trajectory1']['visualization_data'] or not sanitized_pair['trajectory1']['visualization_data']['obstacles']:
            logger.warning("Adding obstacles to T1 before sending")
            sanitized_pair['trajectory1']['visualization_data']['obstacles'] = obstacles
    
    if 'visualization_data' in sanitized_pair['trajectory2']:
        if 'obstacles' not in sanitized_pair['trajectory2']['visualization_data'] or not sanitized_pair['trajectory2']['visualization_data']['obstacles']:
            logger.warning("Adding obstacles to T2 before sending")
            sanitized_pair['trajectory2']['visualization_data']['obstacles'] = obstacles
    
    # Add 'states' property based on positions for compatibility
    for traj_key in ['trajectory1', 'trajectory2']:
        if 'visualization_data' in sanitized_pair[traj_key]:
            positions = sanitized_pair[traj_key]['visualization_data'].get('positions', [])
            sanitized_pair[traj_key]['states'] = positions  # Add at root level
            sanitized_pair[traj_key]['visualization_data']['states'] = positions  # Add in visualization_data too
    
    # Log the complete data structure being sent
    import json
    logger.info(f"Sending trajectory pair data: {json.dumps(sanitized_pair, indent=2)[:500]}...")  # Log first 500 chars
    
    logger.info(f"Final check - T1 has {len(sanitized_pair['trajectory1']['visualization_data'].get('obstacles', []))} obstacles")
    logger.info(f"Final check - T2 has {len(sanitized_pair['trajectory2']['visualization_data'].get('obstacles', []))} obstacles")
    
    return jsonify(sanitized_pair)

@app.route('/api/submit_preference', methods=['POST'])
def submit_preference():
    """API endpoint to submit preference data"""
    try:
        data = request.json
        
        # Validate required fields
        if not all(key in data for key in ['preferred', 'rejected']):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # Record the feedback
        feedback_entry = {
            'timestamp': time.time(),
            'preferred_trajectory': data['preferred'],
            'rejected_trajectory': data['rejected'],
            'reason': data.get('reason', ''),
            'confidence': data.get('confidence', 1.0)
        }
        
        feedback_data.append(feedback_entry)
        
        # Record the pairwise comparison
        comparison = {
            'preferred': data['preferred'],
            'rejected': data['rejected']
        }
        pairwise_comparisons.append(comparison)
        
        # Save feedback to file
        save_feedback()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error submitting preference: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint to get statistics"""
    return jsonify({
        'trajectories': len(trajectories),
        'feedback': len(feedback_data),
        'comparisons': len(pairwise_comparisons)
    })

# Main function to start the server
def start_server():
    global active_ros_node
    
    # Initialize ROS2
    rclpy.init(args=None)
    active_ros_node = Node('feedback_server_node')
    
    # Start ROS spinning thread
    ros_thread = threading.Thread(target=ros_spin_thread, args=(active_ros_node,))
    ros_thread.daemon = True
    ros_thread.start()

        
    # Load trajectories and feedback
    load_trajectories()
    print("load_trajectories")
    load_feedback()
    
    # Check if templates directory exists
    if not os.path.exists(template_dir):
        logger.error(f"Templates directory not found: {template_dir}")
        logger.error("Creating templates directory structure...")
        
        # Create template directory
        os.makedirs(template_dir, exist_ok=True)
        
        # Copy templates from package
        package_templates = os.path.join(package_dir, "templates")
        if os.path.exists(package_templates):
            import shutil
            for template in os.listdir(package_templates):
                src = os.path.join(package_templates, template)
                dst = os.path.join(template_dir, template)
                shutil.copy(src, dst)
                logger.info(f"Copied template: {template}")
    
    # Start the Flask server
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    logger.info(f"Starting feedback server at http://{host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    finally:
        # Clean up ROS resources
        if active_ros_node:
            active_ros_node.destroy_node()
        rclpy.shutdown()
