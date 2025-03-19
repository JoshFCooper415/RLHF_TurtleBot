import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get package directories
    try:
        turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
        print(f"Found turtlebot3_gazebo at: {turtlebot3_gazebo_pkg}")
    except:
        print("Error: turtlebot3_gazebo package not found. Make sure it's installed.")
        turtlebot3_gazebo_pkg = ""  # Set to empty to avoid further errors
    
    turtlebot3_gym_pkg = get_package_share_directory('turtlebot3_gym')
    
    # Set the model type
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    
    # Check for custom world
    world_file = os.path.join(turtlebot3_gym_pkg, 'worlds', 'simple_obstacles.world')
    if os.path.exists(world_file):
        print(f"Using custom world: {world_file}")
    else:
        print(f"Custom world not found: {world_file}")
        # Use empty world from TurtleBot3 gazebo package if available
        if turtlebot3_gazebo_pkg:
            world_file = os.path.join(turtlebot3_gazebo_pkg, 'worlds', 'empty_world.world')
            if os.path.exists(world_file):
                print(f"Using TurtleBot3 empty world: {world_file}")
            else:
                print(f"TurtleBot3 empty world not found at: {world_file}")
                # Fallback to a basic empty world
                world_file = ""
    
    # Declare headless argument
    headless = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Run Gazebo without GUI'
    )
    
    # Create launch description
    ld = LaunchDescription([headless])
    
    # If the TurtleBot3 Gazebo package was found
    if turtlebot3_gazebo_pkg and os.path.exists(os.path.join(turtlebot3_gazebo_pkg, 'launch')):
        # Check which launch files are available
        available_launches = []
        for file in os.listdir(os.path.join(turtlebot3_gazebo_pkg, 'launch')):
            if file.endswith('.launch.py'):
                available_launches.append(file)
        
        print(f"Available turtlebot3_gazebo launch files: {available_launches}")
        
        # Use empty_world.launch.py if available, otherwise try others
        if 'empty_world.launch.py' in available_launches:
            # Use the gazebo_ros package directly with headless mode
            gazebo_ros_pkg = get_package_share_directory('gazebo_ros')
            gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
                ),
                launch_arguments={
                    'world': world_file,
                    'gui': 'false'  # This sets headless mode
                }.items()
            )
            ld.add_action(gazebo)
            
            # Try to find and include the turtlebot3 spawn launch
            if 'spawn_turtlebot3.launch.py' in available_launches:
                spawn = IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(turtlebot3_gazebo_pkg, 'launch', 'spawn_turtlebot3.launch.py')
                    ),
                )
                ld.add_action(spawn)
    else:
        # Fallback to basic gazebo with empty world
        print("Using fallback Gazebo launch")
        gazebo_ros_pkg = get_package_share_directory('gazebo_ros')
        gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={
                'gui': 'false'  # This sets headless mode
            }.items()
        )
        ld.add_action(gazebo)
    
    return ld