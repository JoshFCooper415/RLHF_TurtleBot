from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'turtlebot3_gym'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        # Add world files if they're not included elsewhere
        (os.path.join('share', package_name, 'worlds'),
         glob(os.path.join('worlds', '*.world'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='TurtleBot3 Gymnasium environment for reinforcement learning',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = turtlebot3_gym.train_turtlebot3:main',
            'train_simple = turtlebot3_gym.train_simple:main',
            'train_cpu = turtlebot3_gym.train_cpu:main',
            'evaluate = turtlebot3_gym.evaluate_turtlebot3:main',
            'rlhf = turtlebot3_gym.rlhf_pipeline:main',
            'parse_world = turtlebot3_gym.gazebo_world_parser:main',  # Add CLI for world parser
        ],
    },
)