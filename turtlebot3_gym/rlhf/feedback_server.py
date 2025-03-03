#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from datetime import datetime
import traceback
import pickle
import time
from turtlebot3_gym.rlhf.trajectory_manager import TrajectoryManager

class RLHFFeedbackServer(Node):
    """
    A Flask-based server for collecting human feedback on robot trajectories.
    This server provides a web interface for humans to view, rate, and compare
    robot trajectories for Reinforcement Learning from Human Feedback (RLHF).
    """
    
    def __init__(self):
        super().__init__('rlhf_feedback_server')
        
        # Get package paths
        self.package_path = os.path.expanduser('~/ros2_ws/src/turtlebot3_gym')
        self.trajectories_path = os.path.join(self.package_path, 'turtlebot3_gym', 'rlhf', 'trajectories')
        self.feedback_path = os.path.join(self.package_path, 'feedback_log.json')
        
        # Correct paths for static and template folders
        self.static_path = os.path.join(self.package_path, 'static')
        self.template_path = os.path.join(self.package_path, 'templates')
        
        # Create directory for saving plots if it doesn't exist
        os.makedirs(os.path.join(self.static_path, 'plots'), exist_ok=True)
        
        # Initialize trajectory manager
        self.trajectory_manager = TrajectoryManager()
        
        # Load existing feedback if available
        self.feedback_data = []
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, 'r') as f:
                    self.feedback_data = json.load(f)
                self.get_logger().info(f"Loaded {len(self.feedback_data)} existing feedback entries")
            except json.JSONDecodeError:
                self.get_logger().warning(f"Error decoding feedback file, starting with empty feedback data")
        
        # Load trajectories
        self.trajectories = self._load_trajectories()
        self.get_logger().info(f"Loaded {len(self.trajectories)} trajectories")
        
        self.get_logger().info("Feedback server initialized")
    
    def _load_trajectories(self):
        """Load all trajectory files from the trajectories directory"""
        trajectories = []
        
        if not os.path.exists(self.trajectories_path):
            self.get_logger().warning(f"Trajectories directory not found: {self.trajectories_path}")
            return trajectories
        
        # Get a list of files in the directory
        self.get_logger().info(f"Looking for trajectories in: {self.trajectories_path}")
        files = os.listdir(self.trajectories_path)
        self.get_logger().info(f"Found {len(files)} files in trajectory directory")
        
        # Load each pickle file (except index.json)
        for filename in files:
            if filename.endswith('.pkl'):
                try:
                    file_path = os.path.join(self.trajectories_path, filename)
                    self.get_logger().info(f"Loading trajectory file: {filename}")
                    
                    with open(file_path, 'rb') as f:
                        trajectory_data = pickle.load(f)
                    
                    # Add filename to the trajectory data
                    trajectory_data['filename'] = filename
                    
                    # Ensure trajectory has complete structure with obstacles
                    trajectory_data = self._ensure_trajectory_structure(trajectory_data)
                    
                    # Generate visualization
                    plot_path = self._generate_trajectory_plot(trajectory_data, filename)
                    trajectory_data['plot_path'] = plot_path
                    
                    # Add to trajectories list
                    trajectories.append(trajectory_data)
                    self.get_logger().info(f"Successfully loaded trajectory: {filename}")
                except Exception as e:
                    self.get_logger().error(f"Error loading trajectory {filename}: {e}")
                    traceback.print_exc()
        
        # Sort trajectories by timestamp if available
        trajectories.sort(key=lambda x: x.get('metadata', {}).get('timestamp', 0), reverse=True)
        
        return trajectories
    
    def _ensure_trajectory_structure(self, trajectory):
        """Ensure trajectory has complete structure with obstacles"""
        # Define default obstacles
        default_obstacles = [
            {"type": "rectangle", "x": 2.0, "y": 0.0, "width": 1.0, "height": 1.0},
            {"type": "circle", "x": 0.0, "y": 2.0, "radius": 0.5},
            {"type": "rectangle", "x": -2.0, "y": -1.0, "width": 3.0, "height": 0.2, "rotation": 0.7}
        ]
        
        # Initialize metadata if missing
        if 'metadata' not in trajectory:
            trajectory['metadata'] = {}
        
        # Initialize visualization_data if missing
        if 'visualization_data' not in trajectory:
            trajectory['visualization_data'] = {}
            
            # If no positions in visualization_data, try to get from states
            if 'positions' not in trajectory['visualization_data'] and 'states' in trajectory:
                if all(isinstance(state, dict) and 'position' in state for state in trajectory['states']):
                    trajectory['visualization_data']['positions'] = [state['position'] for state in trajectory['states']]
                    if all('orientation' in state for state in trajectory['states']):
                        trajectory['visualization_data']['orientations'] = [state['orientation'] for state in trajectory['states']]
        
        # Ensure target position exists in both places
        if 'target_position' in trajectory['metadata']:
            trajectory['visualization_data']['target_position'] = trajectory['metadata']['target_position']
        elif 'target_position' in trajectory['visualization_data']:
            trajectory['metadata']['target_position'] = trajectory['visualization_data']['target_position']
        
        # Check for obstacles in various locations
        obstacles = None
        
        # Check visualization_data first
        if 'obstacles' in trajectory['visualization_data'] and trajectory['visualization_data']['obstacles']:
            obstacles = trajectory['visualization_data']['obstacles']
        
        # Then check metadata
        elif 'obstacles' in trajectory['metadata'] and trajectory['metadata']['obstacles']:
            obstacles = trajectory['metadata']['obstacles']
        
        # Then check top level
        elif 'obstacles' in trajectory and trajectory['obstacles']:
            obstacles = trajectory['obstacles']
        
        # Use default obstacles if none found
        if not obstacles:
            obstacles = default_obstacles
            self.get_logger().info(f"No obstacles found in trajectory, using default obstacles")
        
        # Ensure obstacles exist in both metadata and visualization_data
        trajectory['metadata']['obstacles'] = obstacles
        trajectory['visualization_data']['obstacles'] = obstacles
        trajectory['metadata']['obstacle_count'] = len(obstacles)
        
        # Ensure success and steps are consistent
        if 'success' in trajectory['metadata']:
            trajectory['visualization_data']['success'] = trajectory['metadata']['success']
        elif 'success' in trajectory['visualization_data']:
            trajectory['metadata']['success'] = trajectory['visualization_data']['success']
        
        if 'steps' in trajectory['metadata']:
            trajectory['visualization_data']['steps'] = trajectory['metadata']['steps']
        elif 'steps' in trajectory['visualization_data']:
            trajectory['metadata']['steps'] = trajectory['visualization_data']['steps']
            
        return trajectory
    
    def _generate_trajectory_plot(self, trajectory, filename):
        """Generate a visualization of the trajectory with obstacles"""
        try:
            # Get positions from visualization_data or states
            positions = None
            if 'visualization_data' in trajectory and 'positions' in trajectory['visualization_data']:
                positions = trajectory['visualization_data']['positions']
            elif 'states' in trajectory:
                positions = [state['position'] for state in trajectory['states']]
            
            if not positions:
                self.get_logger().warning(f"No position data found for trajectory {filename}")
                return None
                
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            
            # Get target position
            target_pos = None
            if 'visualization_data' in trajectory and 'target_position' in trajectory['visualization_data']:
                target_pos = trajectory['visualization_data']['target_position']
            elif 'metadata' in trajectory and 'target_position' in trajectory['metadata']:
                target_pos = trajectory['metadata']['target_position']
            
            # Create figure
            plt.figure(figsize=(8, 8))
            
            # Plot trajectory
            plt.plot(x_positions, y_positions, 'b-', label='Robot Path')
            plt.scatter(x_positions[0], y_positions[0], color='green', s=100, marker='o', label='Start')
            plt.scatter(x_positions[-1], y_positions[-1], color='red', s=100, marker='x', label='End')
            
            # Plot target if available
            if target_pos:
                plt.scatter(target_pos[0], target_pos[1], color='purple', s=100, marker='*', label='Target')
            
            # Get obstacles from visualization_data or metadata
            obstacles = []
            if 'visualization_data' in trajectory and 'obstacles' in trajectory['visualization_data']:
                obstacles = trajectory['visualization_data']['obstacles']
            elif 'metadata' in trajectory and 'obstacles' in trajectory['metadata']:
                obstacles = trajectory['metadata']['obstacles']
            elif 'obstacles' in trajectory:
                obstacles = trajectory['obstacles']
            
            # Draw obstacles
            for obstacle in obstacles:
                try:
                    # Check what kind of obstacle data we have
                    if isinstance(obstacle, dict):
                        if 'type' in obstacle and obstacle['type'] == 'circle':
                            # Circle with type field
                            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['radius'], color='gray', alpha=0.7)
                            plt.gca().add_patch(circle)
                        elif 'type' in obstacle and obstacle['type'] == 'rectangle':
                            # Rectangle with type field
                            rect = plt.Rectangle(
                                (obstacle['x'] - obstacle['width']/2, obstacle['y'] - obstacle['height']/2),
                                obstacle['width'], obstacle['height'], color='gray', alpha=0.7
                            )
                            plt.gca().add_patch(rect)
                        elif 'position' in obstacle and 'radius' in obstacle:
                            # Circular obstacle with position field
                            position = obstacle['position']
                            radius = obstacle['radius']
                            circle = plt.Circle((position[0], position[1]), radius, color='gray', alpha=0.7)
                            plt.gca().add_patch(circle)
                        elif 'position' in obstacle and ('size' in obstacle or 'dimensions' in obstacle):
                            # Rectangular obstacle with position field
                            position = obstacle['position']
                            size = obstacle.get('size', obstacle.get('dimensions'))
                            rect = plt.Rectangle(
                                (position[0] - size[0]/2, position[1] - size[1]/2),
                                size[0], size[1], color='gray', alpha=0.7
                            )
                            plt.gca().add_patch(rect)
                        elif 'x' in obstacle and 'y' in obstacle and 'width' in obstacle and 'height' in obstacle:
                            # Rectangle with explicit dimensions
                            rect = plt.Rectangle(
                                (obstacle['x'] - obstacle['width']/2, obstacle['y'] - obstacle['height']/2),
                                obstacle['width'], obstacle['height'], color='gray', alpha=0.7
                            )
                            plt.gca().add_patch(rect)
                    elif isinstance(obstacle, list) or isinstance(obstacle, tuple):
                        # List/tuple format
                        if len(obstacle) >= 3:  # Assuming [x, y, radius] format
                            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.7)
                            plt.gca().add_patch(circle)
                        elif len(obstacle) == 2:  # Just a point
                            plt.scatter(obstacle[0], obstacle[1], color='gray', s=100, marker='s')
                except Exception as e:
                    self.get_logger().warning(f"Error drawing obstacle {obstacle}: {e}")
            
            # Add grid and labels
            plt.grid(True)
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Robot Trajectory with Obstacles')
            
            # Add obstacle to legend if we found obstacles
            if obstacles:
                # Create a dummy artist for the legend
                from matplotlib.patches import Patch
                obstacle_patch = Patch(color='gray', alpha=0.7, label='Obstacles')
                handles, labels = plt.gca().get_legend_handles_labels()
                handles.append(obstacle_patch)
                plt.legend(handles=handles)
            else:
                plt.legend()
            
            # Equal aspect ratio for proper distance perception
            plt.gca().set_aspect('equal', 'box')
            
            # Save plot to static directory
            base_filename = os.path.splitext(filename)[0]
            plot_filename = f"{base_filename}.png"
            plot_path = os.path.join('plots', plot_filename)
            full_plot_path = os.path.join(self.static_path, plot_path)
            plt.savefig(full_plot_path)
            plt.close()
            
            self.get_logger().info(f"Generated plot for {filename} at {full_plot_path}")
            return plot_path
        except Exception as e:
            self.get_logger().error(f"Error generating plot for {filename}: {e}")
            traceback.print_exc()
            return None
    
    def prepare_visualization_data(self, trajectory):
        """Extract the necessary data for visualizing a trajectory including obstacles"""
        # First make sure the trajectory has a consistent structure
        trajectory = self._ensure_trajectory_structure(trajectory)
        
        # Default obstacles as fallback
        default_obstacles = [
            {"type": "rectangle", "x": 2.0, "y": 0.0, "width": 1.0, "height": 1.0},
            {"type": "circle", "x": 0.0, "y": 2.0, "radius": 0.5},
            {"type": "rectangle", "x": -2.0, "y": -1.0, "width": 3.0, "height": 0.2, "rotation": 0.7}
        ]
        
        # Get positions from visualization_data or states
        positions = []
        if 'visualization_data' in trajectory and 'positions' in trajectory['visualization_data']:
            positions = trajectory['visualization_data']['positions']
        elif 'states' in trajectory and trajectory['states']:
            positions = [state['position'] for state in trajectory['states']]
        
        # Get orientations if available
        orientations = []
        if 'visualization_data' in trajectory and 'orientations' in trajectory['visualization_data']:
            orientations = trajectory['visualization_data']['orientations']
        elif 'states' in trajectory and trajectory['states'] and 'orientation' in trajectory['states'][0]:
            orientations = [state['orientation'] for state in trajectory['states']]
        
        # Get target position
        target_position = None
        if 'visualization_data' in trajectory and 'target_position' in trajectory['visualization_data']:
            target_position = trajectory['visualization_data']['target_position']
        elif 'metadata' in trajectory and 'target_position' in trajectory['metadata']:
            target_position = trajectory['metadata']['target_position']
        else:
            target_position = [0, 0]  # Default if none found
        
        # Get success and steps info
        success = False
        if 'metadata' in trajectory and 'success' in trajectory['metadata']:
            success = trajectory['metadata']['success']
        elif 'visualization_data' in trajectory and 'success' in trajectory['visualization_data']:
            success = trajectory['visualization_data']['success']
            
        steps = len(positions)
        if 'metadata' in trajectory and 'steps' in trajectory['metadata']:
            steps = trajectory['metadata']['steps']
        elif 'visualization_data' in trajectory and 'steps' in trajectory['visualization_data']:
            steps = trajectory['visualization_data']['steps']
        
        # Create the basic visualization data structure
        visualization_data = {
            'positions': positions,
            'orientations': orientations,
            'target_position': target_position,
            'steps': steps,
            'success': success,
            'obstacles': default_obstacles  # Default fallback
        }
        
        # Get obstacles from various possible locations
        obstacles = None
        if 'visualization_data' in trajectory and 'obstacles' in trajectory['visualization_data']:
            obstacles = trajectory['visualization_data']['obstacles']
        elif 'metadata' in trajectory and 'obstacles' in trajectory['metadata']:
            obstacles = trajectory['metadata']['obstacles']
        elif 'obstacles' in trajectory:
            obstacles = trajectory['obstacles']
        
        # If we found obstacles, convert them to a standard format
        if obstacles:
            standardized_obstacles = []
            for obstacle in obstacles:
                try:
                    if isinstance(obstacle, dict):
                        if 'type' in obstacle:
                            # Already in standard format, add directly
                            if obstacle['type'] in ['rectangle', 'circle', 'point']:
                                standardized_obstacles.append(obstacle)
                        elif 'position' in obstacle and 'radius' in obstacle:
                            # Circular obstacle with position field
                            standardized_obstacles.append({
                                'type': 'circle',
                                'x': obstacle['position'][0],
                                'y': obstacle['position'][1],
                                'radius': obstacle['radius']
                            })
                        elif 'position' in obstacle and ('size' in obstacle or 'dimensions' in obstacle):
                            # Rectangular obstacle with position field
                            size = obstacle.get('size', obstacle.get('dimensions'))
                            standardized_obstacles.append({
                                'type': 'rectangle',
                                'x': obstacle['position'][0],
                                'y': obstacle['position'][1],
                                'width': size[0],
                                'height': size[1]
                            })
                    elif isinstance(obstacle, list) or isinstance(obstacle, tuple):
                        if len(obstacle) >= 3:  # [x, y, radius] format
                            standardized_obstacles.append({
                                'type': 'circle',
                                'x': obstacle[0],
                                'y': obstacle[1],
                                'radius': obstacle[2]
                            })
                        elif len(obstacle) == 2:  # Just a point
                            standardized_obstacles.append({
                                'type': 'point',
                                'x': obstacle[0],
                                'y': obstacle[1]
                            })
                except Exception as e:
                    self.get_logger().warning(f"Error converting obstacle {obstacle}: {e}")
            
            # Only replace if we successfully converted obstacles
            if standardized_obstacles:
                visualization_data['obstacles'] = standardized_obstacles
                self.get_logger().info(f"Using {len(standardized_obstacles)} standardized obstacles")
            else:
                self.get_logger().warning(f"No valid obstacles found, using defaults")
        
        # Log what we're returning
        self.get_logger().info(f"Prepared visualization data with {len(visualization_data['obstacles'])} obstacles")
        
        return visualization_data    
    
    def create_flask_app(self):
        """Create and configure the Flask application"""
        # Log template directory and verify it exists
        template_dir = self.template_path
        self.get_logger().info(f"Template directory: {template_dir}")
        self.get_logger().info(f"Template directory exists: {os.path.exists(template_dir)}")
        
        if os.path.exists(template_dir):
            files = os.listdir(template_dir)
            self.get_logger().info(f"Files in template directory: {files}")
        
        app = Flask(__name__, 
                    template_folder=template_dir,
                    static_folder=self.static_path)
        
        # Store reference to the server instance for route handlers
        server = self
        
        @app.route('/')
        def index():
            try:
                # Count how many trajectories have feedback
                rated_trajectories = set()
                for feedback in server.feedback_data:
                    if 'trajectory_id' in feedback:
                        rated_trajectories.add(feedback.get('trajectory_id', ''))
                    elif 'preferred_trajectory' in feedback:
                        rated_trajectories.add(feedback.get('preferred_trajectory', ''))
                        rated_trajectories.add(feedback.get('rejected_trajectory', ''))
                
                server.get_logger().info(f"Rendering index page with {len(server.trajectories)} trajectories")
                return render_template('index.html', 
                                      trajectories=server.trajectories,
                                      feedback_count=len(server.feedback_data),
                                      rated_count=len(rated_trajectories),
                                      total_count=len(server.trajectories))
            except Exception as e:
                server.get_logger().error(f"Error rendering index page: {e}")
                traceback.print_exc()
                return f"Error loading page: {str(e)}", 500
        
        @app.route('/compare')
        def compare():
            """Page for trajectory comparison"""
            try:
                return render_template('compare.html')
            except Exception as e:
                server.get_logger().error(f"Error rendering compare page: {e}")
                traceback.print_exc()
                return f"Error loading comparison page: {str(e)}", 500
        
        @app.route('/api/get_comparison_pair')
        def get_comparison_pair():
            """Get a pair of trajectories to compare"""
            try:
                # Use the trajectory manager to get a random pair if available
                # Otherwise pick random trajectories from our loaded list
                try:
                    traj1, traj2 = server.trajectory_manager.get_random_pair()
                except (AttributeError, Exception) as e:
                    server.get_logger().warning(f"Could not use trajectory manager: {e}. Using local trajectories.")
                    if len(server.trajectories) < 2:
                        return jsonify({'error': 'Not enough trajectories available'}), 400
                    
                    import random
                    traj1, traj2 = random.sample(server.trajectories, 2)
                
                # Prepare visualization data ensuring obstacles are included
                viz_data1 = server.prepare_visualization_data(traj1)
                viz_data2 = server.prepare_visualization_data(traj2)
                
                # Log obstacle counts for debugging
                server.get_logger().info(f"Sending traj1 with {len(viz_data1['obstacles'])} obstacles")
                server.get_logger().info(f"Sending traj2 with {len(viz_data2['obstacles'])} obstacles")
                
                # Build response
                comparison_data = {
                    'trajectory1': {
                        'id': traj1.get('episode_id', traj1.get('filename', 'unknown')),
                        'metadata': {
                            'success': traj1.get('metadata', {}).get('success', False),
                            'steps': traj1.get('metadata', {}).get('steps', 0),
                            'final_distance': traj1.get('metadata', {}).get('final_distance', 0.0),
                            'obstacle_count': len(viz_data1['obstacles'])
                        },
                        'visualization_data': viz_data1,
                        'plot_path': traj1.get('plot_path', '')
                    },
                    'trajectory2': {
                        'id': traj2.get('episode_id', traj2.get('filename', 'unknown')),
                        'metadata': {
                            'success': traj2.get('metadata', {}).get('success', False),
                            'steps': traj2.get('metadata', {}).get('steps', 0),
                            'final_distance': traj2.get('metadata', {}).get('final_distance', 0.0),
                            'obstacle_count': len(viz_data2['obstacles'])
                        },
                        'visualization_data': viz_data2,
                        'plot_path': traj2.get('plot_path', '')
                    }
                }
                
                return jsonify(comparison_data)
            except Exception as e:
                server.get_logger().error(f"Error getting comparison pair: {e}")
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/debug_trajectory')
        def debug_trajectory():
            """Debug endpoint to help troubleshoot trajectory visualization issues"""
            try:
                # Get a random trajectory
                if not server.trajectories:
                    return jsonify({'error': 'No trajectories available'}), 404
                
                import random
                trajectory = random.choice(server.trajectories)
                
                # Prepare visualization data
                viz_data = server.prepare_visualization_data(trajectory)
                
                # Get a summary of the trajectory structure
                structure_summary = {
                    'has_visualization_data': 'visualization_data' in trajectory,
                    'has_metadata': 'metadata' in trajectory,
                    'has_states': 'states' in trajectory,
                    'has_positions': 'positions' in viz_data,
                    'has_obstacles': 'obstacles' in viz_data,
                    'obstacle_count': len(viz_data.get('obstacles', [])),
                    'first_obstacle': viz_data.get('obstacles', [])[0] if viz_data.get('obstacles', []) else None
                }
                
                return jsonify({
                    'trajectory_id': trajectory.get('episode_id', trajectory.get('filename', 'unknown')),
                    'structure': structure_summary,
                    'visualization_data': viz_data
                })
            except Exception as e:
                server.get_logger().error(f"Error in debug_trajectory: {e}")
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @app.route('/trajectory/<trajectory_id>')
        def view_trajectory(trajectory_id):
            try:
                # Find the trajectory
                trajectory = None
                for t in server.trajectories:
                    if t['filename'] == trajectory_id or t.get('episode_id') == trajectory_id:
                        trajectory = t
                        break
                
                if not trajectory:
                    return "Trajectory not found", 404
                
                # Check if this trajectory has feedback
                existing_feedback = []
                for feedback in server.feedback_data:
                    if feedback.get('trajectory_id') == trajectory_id:
                        existing_feedback.append(feedback)
                
                return render_template('trajectory.html', 
                                      trajectory=trajectory,
                                      feedback=existing_feedback)
            except Exception as e:
                server.get_logger().error(f"Error rendering trajectory page: {e}")
                traceback.print_exc()
                return f"Error loading trajectory: {str(e)}", 500
        
        @app.route('/submit_feedback', methods=['POST'])
        def submit_feedback():
            try:
                # Extract data from form
                trajectory_id = request.form.get('trajectory_id')
                rating = float(request.form.get('rating'))
                comment = request.form.get('comment', '')
                
                # Create feedback entry
                feedback_entry = {
                    'trajectory_id': trajectory_id,
                    'rating': rating,
                    'comment': comment,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add to feedback data
                server.feedback_data.append(feedback_entry)
                
                # Save to file
                with open(server.feedback_path, 'w') as f:
                    json.dump(server.feedback_data, f, indent=2)
                
                return jsonify({'success': True, 
                                'message': 'Feedback submitted successfully',
                                'feedback_count': len(server.feedback_data)})
            except Exception as e:
                server.get_logger().error(f"Error submitting feedback: {e}")
                traceback.print_exc()
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @app.route('/api/submit_preference', methods=['POST'])
        def submit_preference():
            """Record a human preference between two trajectories"""
            try:
                data = request.json
                
                preference = {
                    'preferred_trajectory': data['preferred'],
                    'rejected_trajectory': data['rejected'],
                    'reason': data.get('reason', ''),
                    'timestamp': datetime.now().isoformat(),
                    'confidence': data.get('confidence', 1.0)
                }
                
                server.feedback_data.append(preference)
                
                # Save feedback to disk
                with open(server.feedback_path, 'w') as f:
                    json.dump(server.feedback_data, f, indent=2)
                
                return jsonify({'status': 'success'})
            except Exception as e:
                server.get_logger().error(f"Error submitting preference: {e}")
                traceback.print_exc()
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @app.route('/refresh', methods=['POST'])
        def refresh_trajectories():
            try:
                # Reload trajectories
                server.trajectories = server._load_trajectories()
                
                # Count how many trajectories have feedback
                rated_trajectories = set()
                for feedback in server.feedback_data:
                    if 'trajectory_id' in feedback:
                        rated_trajectories.add(feedback.get('trajectory_id', ''))
                    elif 'preferred_trajectory' in feedback:
                        rated_trajectories.add(feedback.get('preferred_trajectory', ''))
                        rated_trajectories.add(feedback.get('rejected_trajectory', ''))
                
                return jsonify({
                    'success': True,
                    'trajectory_count': len(server.trajectories),
                    'feedback_count': len(server.feedback_data),
                    'rated_count': len(rated_trajectories)
                })
            except Exception as e:
                server.get_logger().error(f"Error refreshing trajectories: {e}")
                traceback.print_exc()
                return jsonify({'success': False, 'message': str(e)}), 500
        
        return app
    
    def destroy_node(self):
        """Clean up resources when shutting down"""
        super().destroy_node()

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the ROS2 node
    server = RLHFFeedbackServer()
    
    # Create the Flask app
    app = server.create_flask_app()
    
    # Start a separate thread for ROS2 spinning
    def spin_ros():
        try:
            rclpy.spin(server)
        except KeyboardInterrupt:
            pass
        finally:
            server.destroy_node()
            rclpy.shutdown()
    
    ros_thread = threading.Thread(target=spin_ros)
    ros_thread.daemon = True
    ros_thread.start()
    
    # Run the Flask app in the main thread
    try:
        server.get_logger().info("Starting Flask server at http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        server.get_logger().info("Shutting down server")


if __name__ == '__main__':
    main()