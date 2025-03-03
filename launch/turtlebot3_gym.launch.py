import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Get the turtlebot3 gazebo launch file
    turtlebot3_gazebo_launch_dir = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'launch')
    
    # Set the model type
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    
    # Set up the world
    world = 'empty_world.launch.py'
    
    return LaunchDescription([
        # Launch the turtlebot3 in Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_gazebo_launch_dir, world)
            ),
        ),
    ])
