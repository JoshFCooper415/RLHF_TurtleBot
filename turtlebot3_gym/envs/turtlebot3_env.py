#!/usr/bin/env python3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import time
from rclpy.executors import SingleThreadedExecutor
import threading
import os
import xml.etree.ElementTree as ET

class GazeboWorldParser:
    """
    Parse Gazebo world files to extract obstacle information
    for visualization and collision checking.
    """
    
    def __init__(self, node=None):
        """
        Initialize the parser with an optional ROS node for logging.
        
        Args:
            node: ROS2 node for logging, if available
        """
        self.node = node
        self.cached_obstacles = None
    
    def log(self, level, message):
        """Log a message using the ROS node if available, otherwise print"""
        if self.node is not None and hasattr(self.node, 'get_logger'):
            if level == 'info':
                self.node.get_logger().info(message)
            elif level == 'warn':
                self.node.get_logger().warn(message)
            elif level == 'error':
                self.node.get_logger().error(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def find_world_file(self):
        """
        Find the Gazebo world file path using various methods:
        1. Check GAZEBO_WORLD_FILE environment variable
        2. Look in standard ROS2 directory structure
        3. Try to get from ROS parameter if a node is available
        """
        # First, try to get from environment variable
        world_file = os.environ.get('GAZEBO_WORLD_FILE')
        if world_file and os.path.exists(world_file):
            self.log('info', f"Found world file from environment: {world_file}")
            return world_file
        
        # Check for common world file names in standard locations
        world_names = ['simple_obstacles.world', 'turtlebot3_world.world', 'empty_world.world']
        search_paths = [
            os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/worlds'),
            os.path.expanduser('~/ros2_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds'),
            os.path.expanduser('~/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/worlds'),
            '/opt/ros/galactic/share/turtlebot3_gazebo/worlds',  # Adjust ROS distro as needed
            '/opt/ros/humble/share/turtlebot3_gazebo/worlds',
            '/opt/ros/foxy/share/turtlebot3_gazebo/worlds',
            '/opt/ros/rolling/share/turtlebot3_gazebo/worlds',
        ]
        
        for path in search_paths:
            for name in world_names:
                full_path = os.path.join(path, name)
                if os.path.exists(full_path):
                    self.log('info', f"Found world file: {full_path}")
                    return full_path
        
        # If we have a ROS node, try to get from parameter
        if self.node is not None:
            try:
                if hasattr(self.node, 'get_parameter'):
                    world_param = self.node.get_parameter('world_file').get_parameter_value().string_value
                    if world_param and os.path.exists(world_param):
                        self.log('info', f"Found world file from parameter: {world_param}")
                        return world_param
            except Exception as e:
                self.log('warn', f"Failed to get world file from parameter: {e}")
        
        self.log('warn', "Could not find world file, will use fallback obstacles")
        return None
    
    def parse_world_file(self, world_file=None):
        """
        Parse a Gazebo world file to extract obstacle information.
        
        Args:
            world_file: Path to the Gazebo world file. If None, will try to find it.
            
        Returns:
            List of obstacle dictionaries in a format suitable for visualization.
        """
        # If obstacles were already parsed, return cached result
        if self.cached_obstacles is not None:
            return self.cached_obstacles
            
        # Find world file if not provided
        if world_file is None:
            world_file = self.find_world_file()
        
        obstacles = []
        
        # If we can't find a world file, use fallback obstacles
        if world_file is None or not os.path.exists(world_file):
            self.log('warn', f"World file not found: {world_file}")
            # Return fallback obstacles that match simple_obstacles.world
            obstacles = self._get_fallback_obstacles()
            self.cached_obstacles = obstacles
            return obstacles
        
        try:
            # Parse the XML file
            self.log('info', f"Parsing world file: {world_file}")
            tree = ET.parse(world_file)
            root = tree.getroot()
            
            # Find the world element (handle different namespace formats)
            world = root
            if root.tag != 'world':
                for child in root:
                    if child.tag.endswith('world'):
                        world = child
                        break
            
            # Find all model elements (potential obstacles)
            models = world.findall('.//model')
            if len(models) == 0:
                # Try with specific namespace
                ns = {'sdf': 'http://sdformat.org/schemas/root.xsd'}
                models = world.findall('.//sdf:model', ns)
            
            self.log('info', f"Found {len(models)} models in world file")
            
            # Process each model
            for model in models:
                model_name = model.get('name', 'unknown')
                
                # Skip standard models like ground_plane and sun
                if model_name in ['ground_plane', 'sun']:
                    continue
                
                # Get the pose
                pose_elem = model.find('./pose')
                if pose_elem is None:
                    # Try with namespace
                    pose_elem = model.find('.//pose')
                
                if pose_elem is None or not pose_elem.text:
                    self.log('warn', f"Model {model_name} has no pose, skipping")
                    continue
                    
                pose_values = pose_elem.text.strip().split()
                if len(pose_values) < 6:
                    self.log('warn', f"Model {model_name} has invalid pose format: {pose_elem.text}")
                    continue
                    
                try:
                    x, y, z, roll, pitch, yaw = map(float, pose_values)
                except ValueError:
                    self.log('warn', f"Could not parse pose values: {pose_values}")
                    continue
                
                # Only consider obstacles on the ground (z close to 0)
                # Adjust this threshold based on your world
                if abs(z) > 1.0 and z < 0:
                    continue
                
                # Process box obstacles
                # Look for link/collision/geometry/box
                box_size = None
                for link in model.findall('.//link'):
                    for collision in link.findall('.//collision'):
                        for geometry in collision.findall('.//geometry'):
                            box_elem = geometry.find('.//box/size')
                            if box_elem is not None and box_elem.text:
                                box_size = box_elem.text.strip().split()
                                break
                
                if box_size and len(box_size) >= 3:
                    try:
                        width, depth, height = map(float, box_size)
                        obstacles.append({
                            'type': 'rectangle',
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': depth,
                            'rotation': yaw
                        })
                        self.log('info', f"Added box obstacle '{model_name}' at ({x}, {y})")
                        continue
                    except ValueError:
                        self.log('warn', f"Could not parse box size: {box_size}")
                
                # Process cylinder obstacles
                cylinder_dims = None
                for link in model.findall('.//link'):
                    for collision in link.findall('.//collision'):
                        for geometry in collision.findall('.//geometry'):
                            cylinder_elem = geometry.find('.//cylinder')
                            if cylinder_elem is not None:
                                radius_elem = cylinder_elem.find('./radius')
                                length_elem = cylinder_elem.find('./length')
                                
                                if radius_elem is None:
                                    radius_elem = cylinder_elem.find('.//radius')
                                if length_elem is None:
                                    length_elem = cylinder_elem.find('.//length')
                                
                                if radius_elem is not None and radius_elem.text and \
                                   length_elem is not None and length_elem.text:
                                    try:
                                        radius = float(radius_elem.text)
                                        length = float(length_elem.text)
                                        cylinder_dims = (radius, length)
                                        break
                                    except ValueError:
                                        pass
                
                if cylinder_dims:
                    radius, _ = cylinder_dims
                    obstacles.append({
                        'type': 'circle',
                        'x': x,
                        'y': y,
                        'radius': radius
                    })
                    self.log('info', f"Added cylinder obstacle '{model_name}' at ({x}, {y})")
                    continue
            
            # If we didn't find any obstacles but the file was parsed successfully,
            # it might be using a different format or naming convention
            if len(obstacles) == 0:
                self.log('warn', "No obstacles found in world file, using fallback obstacles")
                obstacles = self._get_fallback_obstacles()
            
        except Exception as e:
            self.log('error', f"Error parsing world file: {e}")
            # Return fallback obstacles
            obstacles = self._get_fallback_obstacles()
        
        self.cached_obstacles = obstacles
        self.log('info', f"Total obstacles found: {len(obstacles)}")
        return obstacles
    
    def _get_fallback_obstacles(self):
        """Provide fallback obstacles if parsing fails"""
        # These match the simple_obstacles.world layout
        obstacles = [
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
        self.log('info', f"Using {len(obstacles)} fallback obstacles")
        return obstacles

def initialize_obstacles_from_world(node=None):
    """Get obstacles from Gazebo world file"""
    parser = GazeboWorldParser(node)
    return parser.parse_world_file()

class TurtleBot3Env(gym.Env):
    """
    TurtleBot3 Gymnasium environment for navigation tasks.
    This environment simulates a navigation task where the TurtleBot3
    must navigate to a target position while avoiding obstacles.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super(TurtleBot3Env, self).__init__()
        
        # Initialize observation data - do this first to avoid attribute errors
        self.lidar_data = np.ones(360)  # Default to max range
        self.robot_position = np.zeros(2)
        self.robot_orientation = 0.0
        self.target_position = np.array([2.0, 0.0])  # Default target
        self.min_obstacle_distance = 1.0
        
        # Status flags
        self.scan_received = False
        self.odom_received = False
        
        # Initialize ROS2 node
        rclpy.init(args=None)
        self.node = Node('turtlebot3_gym_node')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self._run_executor)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        # Set up QoS profiles for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-0.22, -2.5]), 
            high=np.array([0.22, 2.5]),
            dtype=np.float32
        )
        
        # Observation space: [lidar_scans + robot_position + target_position]
        # 360 lidar scans (1 degree resolution) + 2 for robot position + 2 for target position
        self.observation_space = spaces.Box(
            low=np.zeros(364),
            high=np.ones(364),
            dtype=np.float32
        )
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.node.create_publisher(
            Twist, 'cmd_vel', 10)
        
        self.odom_sub = self.node.create_subscription(
            Odometry, 'odom', self._odom_callback, qos_profile)
        
        self.scan_sub = self.node.create_subscription(
            LaserScan, 'scan', self._scan_callback, qos_profile)
        
        # Initialize episode parameters
        self.collision_threshold = 0.2  # Distance considered a collision
        self.success_threshold = 0.3    # Distance considered reaching the target
        self.max_steps = 500
        self.current_step = 0
        
        # Initialize obstacles from the Gazebo world file
        self.obstacles = initialize_obstacles_from_world(self.node)
        self.node.get_logger().info(f"Initialized {len(self.obstacles)} obstacles from world file")
        
        # Wait for data from Gazebo
        self._wait_for_data()
    
    def get_obstacles(self):
        """Return the list of obstacles for visualization"""
        if not hasattr(self, 'obstacles') or self.obstacles is None:
            from turtlebot3_gym.gazebo_world_parser import GazeboWorldParser
            parser = GazeboWorldParser(self.node)
            self.obstacles = parser.parse_world_file()
        return self.obstacles
        
    def _run_executor(self):
        while rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                self.node.get_logger().error(f"Error in executor: {e}")
    
    def _wait_for_data(self):
        """Wait for initial data from Gazebo"""
        self.node.get_logger().info('Waiting for sensor data...')
        timeout = 10.0  # seconds
        start_time = time.time()
        
        while not (self.scan_received and self.odom_received):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.node.get_logger().warning('Timeout waiting for sensor data!')
                break
            time.sleep(0.1)
        
        if self.scan_received and self.odom_received:
            self.node.get_logger().info('Received initial sensor data!')
    
    def _odom_callback(self, msg):
        """Process odometry data"""
        try:
            self.robot_position[0] = msg.pose.pose.position.x
            self.robot_position[1] = msg.pose.pose.position.y
            
            # Extract yaw from quaternion
            q = msg.pose.pose.orientation
            self.robot_orientation = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            self.odom_received = True
        except Exception as e:
            self.node.get_logger().error(f"Error in odom callback: {e}")
    
    def _scan_callback(self, msg):
        """Process laser scan data"""
        try:
            # Process ranges (clipping to max range and normalizing)
            ranges = np.array(msg.ranges)
            
            # Replace inf with max_range
            max_range = msg.range_max
            ranges = np.where(np.isinf(ranges), max_range, ranges)
            
            # Clip ranges between min and max range
            ranges = np.clip(ranges, msg.range_min, max_range)
            
            # Normalize to [0, 1]
            self.lidar_data = ranges / max_range
            
            # Calculate minimum distance to obstacles
            self.min_obstacle_distance = np.min(ranges)
            
            self.scan_received = True
        except Exception as e:
            self.node.get_logger().error(f"Error in scan callback: {e}")
    
    def _get_observation(self):
        """Construct the observation vector"""
        # Normalized robot position (assuming a 10x10 world)
        norm_robot_pos = self.robot_position / 10.0 + 0.5  # Normalize to [0, 1]
        
        # Normalized target position relative to robot
        target_rel = self.target_position - self.robot_position
        distance_to_target = np.linalg.norm(target_rel)
        
        # Convert to robot frame
        angle_to_target = math.atan2(target_rel[1], target_rel[0]) - self.robot_orientation
        
        # Normalize to [0, 1]
        norm_distance = min(distance_to_target / 10.0, 1.0)
        norm_angle = (angle_to_target + math.pi) / (2 * math.pi)
        
        # Combine all observations
        obs = np.concatenate([
            self.lidar_data,
            [norm_distance],
            [norm_angle],
            norm_robot_pos
        ])
        
        return obs
    
    def _calculate_reward(self):
        """Calculate the reward based on the current state"""
        # Calculate distance to target
        distance_to_target = np.linalg.norm(self.target_position - self.robot_position)
        
        # Base reward is negative distance to goal (encourage getting closer)
        reward = -distance_to_target
        
        # Check for collision
        if self.min_obstacle_distance < self.collision_threshold:
            reward -= 100  # Big penalty for collision
            return reward, True  # Done
        
        # Check for success
        if distance_to_target < self.success_threshold:
            reward += 200  # Big reward for reaching the goal
            return reward, True  # Done
        
        # Small penalty for each step (to encourage faster solutions)
        reward -= 0.1
        
        # Encourage facing the target
        target_rel = self.target_position - self.robot_position
        angle_to_target = math.atan2(target_rel[1], target_rel[0])
        angle_diff = abs(angle_to_target - self.robot_orientation) % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # Add a small reward for facing the target
        reward += (1 - angle_diff / math.pi) * 0.5
        
        return reward, False
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Generate a new random target position
        if seed is not None:
            np.random.seed(seed)
        
        # Generate target in a random position within a radius
        angle = np.random.uniform(0, 2 * math.pi)
        distance = np.random.uniform(1.0, 3.0)
        self.target_position = np.array([
            distance * math.cos(angle),
            distance * math.sin(angle)
        ])
        
        # Reset robot position by stopping it
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        # Wait a bit for the simulation to stabilize
        time.sleep(0.5)
        
        # Wait for fresh sensor data
        self._wait_for_data()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        # Publish action
        vel_msg = Twist()
        vel_msg.linear.x = float(action[0])
        vel_msg.angular.z = float(action[1])
        self.cmd_vel_pub.publish(vel_msg)
        
        # Wait for simulation to apply the action and update sensors
        time.sleep(0.1)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward and check if done
        reward, done_by_goal = self._calculate_reward()
        
        # Increment step counter
        self.current_step += 1
        
        # Check if we've exceeded max steps
        done_by_timeout = self.current_step >= self.max_steps
        
        # Check if done
        done = done_by_goal or done_by_timeout
        
        # Create info dictionary
        info = {
            'target_position': self.target_position,
            'robot_position': self.robot_position,
            'min_obstacle_distance': self.min_obstacle_distance,
            'steps': self.current_step,
            'timeout': done_by_timeout,
            'success': done_by_goal and reward > 0,
            'obstacles': self.get_obstacles()  # Include obstacles in info
        }
        
        return observation, reward, done, False, info
    
    def record_trajectory(self, policy, num_episodes=10):
        """Record complete trajectories using the given policy"""
        trajectories = []
        
        for episode in range(num_episodes):
            # Get obstacles first - make sure they exist!
            obstacles_list = self.get_obstacles()
            
            # Print debug info about obstacles
            self.node.get_logger().info(f"Including {len(obstacles_list)} obstacles in trajectory")
            
            # Create trajectory with explicit obstacles in visualization_data
            trajectory = {
                'id': f"episode_{episode}_{time.time()}",
                'metadata': {
                    'policy_version': getattr(policy, 'version', 'unknown'),
                    'timestamp': time.time(),
                    'success': False,
                    'steps': 0,
                    'target_position': None,
                    'obstacles': obstacles_list,  # Include obstacles here
                    'obstacle_count': len(obstacles_list)
                },
                'visualization_data': {
                    'positions': [],
                    'orientations': [],
                    'target_position': None,
                    'obstacles': obstacles_list,  # IMPORTANT: Include obstacles here too
                    'success': False,
                    'steps': 0
                }
            }
            
            # Remove the old fields we don't need for visualization
            # (states, actions, observations, rewards)
            
            obs, _ = self.reset()
            done = False
            step = 0
            
            # Save target position
            target_pos = self.target_position.tolist()
            trajectory['metadata']['target_position'] = target_pos
            trajectory['visualization_data']['target_position'] = target_pos
            
            while not done and step < self.max_steps:
                # Get action from policy
                if hasattr(policy, 'predict'):
                    action, _ = policy.predict(obs)
                else:
                    action = policy(obs)
                
                # Execute action and get result
                next_obs, reward, done, _, info = self.step(action)
                
                # Add position and orientation to visualization data
                trajectory['visualization_data']['positions'].append(self.robot_position.tolist())
                trajectory['visualization_data']['orientations'].append(self.robot_orientation)
                
                # Update for next step
                obs = next_obs
                step += 1
            
            # Add final outcome information
            success = info.get('success', False)
            trajectory['metadata']['success'] = success
            trajectory['visualization_data']['success'] = success
            trajectory['metadata']['steps'] = step
            trajectory['visualization_data']['steps'] = step
            trajectory['metadata']['final_distance'] = float(np.linalg.norm(
                self.target_position - self.robot_position))
            
            trajectories.append(trajectory)
            
            self.node.get_logger().info(
                f"Recorded trajectory {episode+1}/{num_episodes}: " +
                f"Success={trajectory['metadata']['success']}, " +
                f"Steps={trajectory['metadata']['steps']}, " +
                f"Obstacles={len(trajectory['visualization_data']['obstacles'])}"
            )
        
        return trajectories

    def close(self):
        """Clean up resources"""
        # Stop the robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        # Destroy executor thread
        self.executor.shutdown()
        if self.executor_thread.is_alive():
            self.executor_thread.join()
        
        # Clean up ROS2 resources
        self.node.destroy_node()
        
        # Shutdown ROS2 if no other nodes are using it
        if rclpy.ok():
            rclpy.shutdown()