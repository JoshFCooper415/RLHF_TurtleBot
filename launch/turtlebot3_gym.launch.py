import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get package directories
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    turtlebot3_gym_pkg = get_package_share_directory('turtlebot3_gym')
    
    # Set the model type
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    
    # Launch arguments
    use_custom_world = DeclareLaunchArgument(
        'use_custom_world',
        default_value='true',
        description='Use custom world from turtlebot3_gym package'
    )
    
    world_name = DeclareLaunchArgument(
        'world_name',
        default_value='rlhf_arena.world',
        description='Name of the world file to load'
    )
    
    # World file paths
    custom_world_path = os.path.join(turtlebot3_gym_pkg, 'worlds', LaunchConfiguration('world_name'))
    default_world_path = os.path.join(turtlebot3_gazebo_pkg, 'worlds', 'empty_world.sdf')
    
    # Check if custom world exists
    def get_world_path(context):
        use_custom = context.launch_configurations['use_custom_world'].lower() == 'true'
        world_file = context.launch_configurations['world_name']
        custom_path = os.path.join(turtlebot3_gym_pkg, 'worlds', world_file)
        
        if use_custom and os.path.exists(custom_path):
            print(f"Using custom world: {custom_path}")
            return custom_path
        else:
            if use_custom:
                print(f"Custom world not found: {custom_path}")
                print(f"Falling back to default world: {default_world_path}")
            return default_world_path
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': get_world_path
        }.items()
    )
    
    # TurtleBot3 spawn
    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_pkg, 'launch', 'spawn_turtlebot3.launch.py')
        )
    )
    
    # Return the launch description
    return LaunchDescription([
        use_custom_world,
        world_name,
        gazebo,
        spawn_turtlebot
    ])