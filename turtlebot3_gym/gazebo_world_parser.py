import os
import xml.etree.ElementTree as ET
import logging

class GazeboWorldParser:
    """
    Parse Gazebo world files to extract obstacle information for visualization.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the parser with an optional logger.
        
        Args:
            logger: Logger object for logging messages
        """
        self.logger = logger or logging.getLogger('gazebo_world_parser')
        self.cached_obstacles = None
    
    def log(self, level, message):
        """Log a message using the provided logger or print if no logger"""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warn':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def find_world_file(self):
        """
        Find the Gazebo world file path using various methods.
        """
        # First, try to get from environment variable
        world_file = os.environ.get('GAZEBO_WORLD_FILE')
        if world_file and os.path.exists(world_file):
            self.log('info', f"Found world file from environment: {world_file}")
            return world_file
        
        # Look for common world files in standard locations
        world_names = ['simple_obstacles.world', 'turtlebot3_world.world', 'empty_world.world']
        search_paths = [
            os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/worlds'),
            os.path.expanduser('~/ros2_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds'),
            os.path.expanduser('~/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/worlds'),
            '/opt/ros/humble/share/turtlebot3_gazebo/worlds',
            '/opt/ros/foxy/share/turtlebot3_gazebo/worlds',
        ]
        
        for path in search_paths:
            for name in world_names:
                full_path = os.path.join(path, name)
                if os.path.exists(full_path):
                    self.log('info', f"Found world file: {full_path}")
                    return full_path
        
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
            obstacles = self._get_fallback_obstacles()
            self.cached_obstacles = obstacles
            return obstacles
        
        try:
            # Parse the XML file
            self.log('info', f"Parsing world file: {world_file}")
            tree = ET.parse(world_file)
            root = tree.getroot()
            
            # Find the world element
            world = root
            if root.tag != 'world':
                for child in root:
                    if child.tag.endswith('world'):
                        world = child
                        break
            
            # Find all model elements (potential obstacles)
            models = world.findall('.//model')
            if len(models) == 0:
                # Try with specific namespace if needed
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
                
                # Process box obstacles
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
            
            # If we didn't find any obstacles, use fallback obstacles
            if len(obstacles) == 0:
                self.log('warn', "No obstacles found in world file, using fallback obstacles")
                obstacles = self._get_fallback_obstacles()
            
        except Exception as e:
            self.log('error', f"Error parsing world file: {e}")
            obstacles = self._get_fallback_obstacles()
        
        self.cached_obstacles = obstacles
        self.log('info', f"Total obstacles found: {len(obstacles)}")
        return obstacles
    
    def _get_fallback_obstacles(self):
        """Provide fallback obstacles if parsing fails"""
        # These match the simple_obstacles.world layout provided
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

def get_obstacles_from_world(world_file=None, logger=None):
    """Helper function to get obstacles from a world file"""
    parser = GazeboWorldParser(logger)
    return parser.parse_world_file(world_file)

# For testing
if __name__ == "__main__":
    import sys
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('test_parser')
    
    # Get world file path from command line if provided
    world_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Parse world file
    parser = GazeboWorldParser(logger)
    obstacles = parser.parse_world_file(world_file)
    
    # Print results
    print(f"Found {len(obstacles)} obstacles:")
    print(json.dumps(obstacles, indent=2))