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
        
        # Wait for data from Gazebo
        self._wait_for_data()
        
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
            'success': done_by_goal and reward > 0
        }
        
        return observation, reward, done, False, info
    
    def record_trajectory(self, policy, num_episodes=10):
        """Record complete trajectories using the given policy"""
        trajectories = []
        
        for episode in range(num_episodes):
            trajectory = {
                'states': [],
                'actions': [],
                'observations': [],
                'rewards': [],
                'episode_id': f"episode_{episode}_{time.time()}",
                'metadata': {
                    'policy_version': getattr(policy, 'version', 'unknown'),
                    'timestamp': time.time()
                }
            }
            
            obs, _ = self.reset()
            done = False
            step = 0
            
            while not done and step < self.max_steps:
                # Get action from policy
                action, _ = policy.predict(obs)
                
                # Execute action and get result
                next_obs, reward, done, _, info = self.step(action)
                
                # Record everything
                trajectory['states'].append({
                    'position': self.robot_position.copy(),
                    'orientation': self.robot_orientation,
                    'velocity': info.get('velocity', [0, 0])
                })
                trajectory['actions'].append(action.copy())
                trajectory['observations'].append(obs.copy())
                trajectory['rewards'].append(reward)
                
                # Update for next step
                obs = next_obs
                step += 1
            
            # Add final outcome information
            trajectory['metadata']['success'] = info.get('success', False)
            trajectory['metadata']['steps'] = step
            trajectory['metadata']['final_distance'] = np.linalg.norm(
                self.target_position - self.robot_position)
            
            trajectories.append(trajectory)
        
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
