"""
Semantic SLAM Agent: DeepLabV3+ based Semantic Mapping
=======================================================
Part of SemanticVLN-MCP Framework

Implements semantic mapping by combining:
- DeepLabV3+ for semantic segmentation
- Occupancy grid for navigation
- Webots GPS/IMU for pose estimation (simplified SLAM)

Author: SemanticVLN-MCP Team
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/torchvision not installed. Run: pip install torch torchvision")


@dataclass
class SemanticCell:
    """Single cell in the semantic grid map."""
    x: int  # Grid x coordinate
    y: int  # Grid y coordinate
    occupancy: float  # 0=free, 1=occupied, 0.5=unknown
    semantic_class: int  # Semantic class ID
    semantic_name: str  # Semantic class name
    confidence: float  # Confidence of semantic label
    last_updated: float  # Timestamp


@dataclass
class SemanticMap:
    """Complete semantic map of the environment."""
    grid: np.ndarray  # Occupancy grid (H, W)
    semantic_grid: np.ndarray  # Semantic labels (H, W)
    confidence_grid: np.ndarray  # Confidence values (H, W)
    resolution: float  # Meters per cell
    origin: np.ndarray  # World position of grid (0,0)
    width: int  # Grid width
    height: int  # Grid height
    
    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_pos = (world_pos[:2] - self.origin[:2]) / self.resolution
        return int(grid_pos[0]), int(grid_pos[1])
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        world_pos = np.array([grid_x, grid_y]) * self.resolution + self.origin[:2]
        return world_pos
    
    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_free(self, x: int, y: int) -> bool:
        """Check if cell is free for navigation."""
        if not self.is_valid_cell(x, y):
            return False
        return self.grid[y, x] < 0.5


# DeepLabV3+ class names (COCO)
DEEPLABV3_CLASSES = {
    0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'potted plant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'
}

# Navigation-relevant classes
WALKABLE_CLASSES = {0}  # background (floor)
OBSTACLE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}


class SemanticSLAMAgent:
    """
    Semantic SLAM Agent for building environment maps.
    
    Features:
    - DeepLabV3+ semantic segmentation
    - Occupancy grid mapping from depth camera
    - Semantic annotation of grid cells
    - Simplified pose tracking using Webots sensors
    """
    
    def __init__(self, 
                 resolution: float = 0.05,
                 map_size: Tuple[int, int] = (200, 200),
                 device: str = "cuda"):
        """
        Initialize Semantic SLAM agent.
        
        Args:
            resolution: Grid resolution in meters per cell
            map_size: Grid size (width, height) in cells
            device: 'cuda' or 'cpu'
        """
        self.resolution = resolution
        self.map_width, self.map_height = map_size
        self.device = device
        
        # Initialize DeepLabV3+ model
        self.segmentation_model = None
        self.transform = None
        if TORCH_AVAILABLE:
            print("Loading DeepLabV3+ model...")
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.segmentation_model = deeplabv3_resnet50(weights=weights)
            self.segmentation_model.eval()
            self.segmentation_model.to(device)
            self.transform = weights.transforms()
            print(f"DeepLabV3+ loaded on {device}")
        else:
            print("DeepLabV3+ not available - running in mock mode")
        
        # Initialize map
        self.semantic_map = self._create_empty_map()
        
        # Robot pose (from Webots GPS/IMU)
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        
        # Semantic regions detected
        self.semantic_regions: Dict[str, np.ndarray] = {}
    
    def _create_empty_map(self) -> SemanticMap:
        """Create an empty semantic map with pre-loaded arena walls."""
        # Origin at center of map
        origin = np.array([
            -self.map_width * self.resolution / 2,
            -self.map_height * self.resolution / 2,
            0.0
        ])
        
        # Start with free space (0.0) instead of unknown (0.5)
        grid = np.zeros((self.map_height, self.map_width))
        
        # Pre-load arena walls (10x10m arena centered at 0,0)
        # Convert world coords to grid coords
        # Arena: -5m to +5m in both x and y
        # Grid: 200x200 with resolution 0.05m
        # So wall at x=-5 -> grid_x = 0, wall at x=+5 -> grid_x = 199
        
        # Add boundary walls (mark as occupied = 1.0)
        wall_thickness = 3  # cells
        grid[0:wall_thickness, :] = 1.0           # Bottom wall (y=-5)
        grid[-wall_thickness:, :] = 1.0           # Top wall (y=+5)
        grid[:, 0:wall_thickness] = 1.0           # Left wall (x=-5)
        grid[:, -wall_thickness:] = 1.0           # Right wall (x=+5)
        
        # Pre-load furniture obstacles (world coords -> grid coords)
        # Grid formula: grid_pos = (world_pos + 5) / 0.05 = (world_pos + 5) * 20
        def world_to_grid(x, y):
            gx = int((x + 5) * 20)
            gy = int((y + 5) * 20)
            return max(0, min(199, gx)), max(0, min(199, gy))
        
        # ONLY arena boundary walls - room walls detected by depth sensor
        # This ensures A* always finds a valid path
        # Depth-based obstacle avoidance (40cm) prevents actual collisions
        
        # No pre-loaded furniture - let depth sensor handle it
        # This fixes "Start/Goal is invalid" errors
        
        return SemanticMap(
            grid=grid,
            semantic_grid=np.zeros((self.map_height, self.map_width), dtype=np.int32),
            confidence_grid=np.zeros((self.map_height, self.map_width)),
            resolution=self.resolution,
            origin=origin,
            width=self.map_width,
            height=self.map_height
        )
    
    def update(self, 
               rgb_frame: np.ndarray,
               depth_frame: np.ndarray,
               robot_pose: np.ndarray,
               lidar_scan: Optional[np.ndarray] = None) -> SemanticMap:
        """
        Update semantic map with new sensor data.
        
        Args:
            rgb_frame: RGB image (H, W, 3)
            depth_frame: Depth image (H, W) in meters
            robot_pose: Robot pose [x, y, theta] from Webots
            
        Returns:
            Updated SemanticMap
        """
        self.robot_pose = robot_pose
        
        # Step 1: Semantic segmentation
        semantic_mask = self._segment_image(rgb_frame)
        
        # Step 2: Project depth to point cloud
        points_3d = self._depth_to_points(depth_frame, robot_pose)
        
        # Step 3: Update occupancy grid from depth
        self._update_occupancy(points_3d, semantic_mask, depth_frame)
        
        # Step 4: Update occupancy from Lidar (360 degrees)
        if lidar_scan is not None:
            self._update_from_lidar(lidar_scan, robot_pose)
        
        # Step 5: Extract semantic regions
        self._extract_semantic_regions()
        
        # Log map stats periodically
        if not hasattr(self, '_map_log_counter'):
            self._map_log_counter = 0
        self._map_log_counter += 1
        
        # MAP DECAY: Reduce old obstacle confidence to prevent clutter
        if self._map_log_counter % 60 == 0:  # Every ~2 seconds
            # Decay obstacles by 2% (cells > 0.5 slowly return toward 0.5)
            high_cells = self.semantic_map.grid > 0.55
            self.semantic_map.grid[high_cells] *= 0.98  # Decay by 2%
            
            # Decay free space too (cells < 0.45 slowly return toward 0.5)
            low_cells = self.semantic_map.grid < 0.45
            self.semantic_map.grid[low_cells] = self.semantic_map.grid[low_cells] * 0.98 + 0.5 * 0.02
            
            occupied = np.sum(self.semantic_map.grid > 0.5)
            total = self.semantic_map.grid.size
            print(f"[SLAM] Map: {occupied}/{total} cells occupied ({100*occupied/total:.1f}%)")
        
        return self.semantic_map
    
    def _segment_image(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Run DeepLabV3+ semantic segmentation."""
        if self.segmentation_model is None:
            # Mock segmentation
            return np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.int32)
        
        # Prepare input
        input_tensor = self.transform(
            torch.from_numpy(rgb_frame).permute(2, 0, 1)
        ).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out'][0]
            semantic_mask = output.argmax(0).cpu().numpy()
        
        # Log detected classes (periodically to avoid spam)
        unique_classes = np.unique(semantic_mask)
        detected = [DEEPLABV3_CLASSES.get(c, f"class_{c}") for c in unique_classes if c != 0]
        if detected and hasattr(self, '_seg_log_counter'):
            self._seg_log_counter += 1
            if self._seg_log_counter % 30 == 0:  # Log every ~1 second
                print(f"[DEEPLABV3] Segmented: {', '.join(detected)}")
        else:
            self._seg_log_counter = 0
        
        return semantic_mask
    
    def _depth_to_points(self, 
                         depth_frame: np.ndarray, 
                         robot_pose: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D points in world frame.
        
        Uses pinhole camera model with known intrinsics.
        """
        # Camera intrinsics (calculated for 60deg FOV @ 640px)
        fx, fy = 585.0, 585.0
        cx, cy = 320.0, 240.0
        
        h, w = depth_frame.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Filter noise: 0.2m is min range for TB3 Burger camera proto
        z_cam = np.where(np.isfinite(depth_frame) & (depth_frame > 0.2), depth_frame, 0.0)
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        
        # Transform to Robot frame: X_rob=Forward, Y_rob=Left, Z_rob=Up
        # Z_cam is forward, X_cam is right, Y_cam is down
        x_rob = z_cam
        y_rob = -x_cam
        z_rob = -y_cam + 0.1 # Camera is 0.1m high
        
        points_robot = np.stack([x_rob, y_rob, z_rob], axis=-1)
        
        # Transform to World frame using robot pose (rotate X,Y by theta)
        theta = robot_pose[2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Rotation only affects X and Y
        x_world = robot_pose[0] + (x_rob * cos_t - y_rob * sin_t)
        y_world = robot_pose[1] + (x_rob * sin_t + y_rob * cos_t)
        z_world = z_rob
        
        points_world = np.stack([x_world, y_world, z_world], axis=-1)
        
        return points_world
    
    def _update_occupancy(self, 
                          points_3d: np.ndarray,
                          semantic_mask: np.ndarray,
                          depth_frame: np.ndarray):
        """Update occupancy grid with new observations."""
        h, w = depth_frame.shape
        
        for v in range(0, h, 4):  # Subsample for speed
            for u in range(0, w, 4):
                depth = depth_frame[v, u]
                if depth <= 0.1 or depth > 8.0:  # Invalid or too far
                    continue
                
                # Get world position and height
                world_pos_3d = points_3d[v, u]
                z_world = world_pos_3d[2]
                
                # Height filter: only obstacles within robot height range are solid
                if z_world < 0.1 or z_world > 1.2:
                    continue
                
                # Get world position for grid
                world_pos = world_pos_3d[:2]
                
                # Convert to grid coordinates
                gx, gy = self.semantic_map.world_to_grid(world_pos)
                
                if not self.semantic_map.is_valid_cell(gx, gy):
                    continue
                
                # Get semantic class
                semantic_class = semantic_mask[v, u]
                
                # Update occupancy using BOTH semantics and depth
                # DeepLabV3+ class 0 = background (usually floor)
                # Close obstacles (< 1m) are likely walls/furniture
                
                if semantic_class == 0 and depth > 1.0:
                    # Floor/background far away - definitely free
                    self.semantic_map.grid[gy, gx] = max(
                        0.0, 
                        self.semantic_map.grid[gy, gx] - 0.15
                    )
                elif depth < 0.5:
                    # Very close obstacle - definitely occupied (wall/furniture)
                    self.semantic_map.grid[gy, gx] = min(
                        1.0,
                        self.semantic_map.grid[gy, gx] + 0.4
                    )
                elif semantic_class in OBSTACLE_CLASSES:
                    # DeepLabV3+ detected obstacle
                    self.semantic_map.grid[gy, gx] = min(
                        1.0,
                        self.semantic_map.grid[gy, gx] + 0.25
                    )
                else:
                    # Unknown - slight tendency toward free
                    self.semantic_map.grid[gy, gx] = max(
                        0.0,
                        self.semantic_map.grid[gy, gx] - 0.05
                    )
                
                # Update semantic label
                self.semantic_map.semantic_grid[gy, gx] = semantic_class
                self.semantic_map.confidence_grid[gy, gx] = min(
                    1.0,
                    self.semantic_map.confidence_grid[gy, gx] + 0.1
                )
    
    def _update_from_lidar(self, scan: np.ndarray, robot_pose: np.ndarray):
        """Update occupancy grid from 360-degree lidar scan."""
        num_points = len(scan)
        theta_robot = robot_pose[2]
        
        # Assuming 360 points = 1 degree resolution
        # Webots Lidar: index 0 is front, goes counter-clockwise?
        # Let's assume point i is at (i * 2*pi / num_points) relative to robot front
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        for i in range(num_points):
            dist = scan[i]
            if dist <= 0.1 or dist > 10.0:  # Invalid or too far
                continue
                
            # Angle in world frame
            angle_world = theta_robot + angles[i]
            
            # World position
            wx = robot_pose[0] + dist * np.cos(angle_world)
            wy = robot_pose[1] + dist * np.sin(angle_world)
            
            # Convert to grid
            gx, gy = self.semantic_map.world_to_grid(np.array([wx, wy]))
            
            if self.semantic_map.is_valid_cell(gx, gy):
                # Update occupancy (Lidar is very reliable for walls)
                self.semantic_map.grid[gy, gx] = min(
                    1.0,
                    self.semantic_map.grid[gy, gx] + 0.5
                )
                
                # Clear space along the lidar beam (raycasting)
                # For efficiency, we just clear a few points
                for step in np.linspace(0.2, dist - 0.1, 5):
                    sx = robot_pose[0] + step * np.cos(angle_world)
                    sy = robot_pose[1] + step * np.sin(angle_world)
                    sgx, sgy = self.semantic_map.world_to_grid(np.array([sx, sy]))
                    if self.semantic_map.is_valid_cell(sgx, sgy):
                        self.semantic_map.grid[sgy, sgx] = max(
                            0.0,
                            self.semantic_map.grid[sgy, sgx] - 0.2
                        )

    def _extract_semantic_regions(self):
        """Extract semantic regions (e.g., kitchen, living room) from map."""
        # Find clusters of semantic classes
        for class_id, class_name in DEEPLABV3_CLASSES.items():
            if class_id == 0:  # Skip background
                continue
            
            # Find cells with this class
            mask = self.semantic_map.semantic_grid == class_id
            if not np.any(mask):
                continue
            
            # Get centroid
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                world_pos = self.semantic_map.grid_to_world(
                    int(centroid_x), int(centroid_y)
                )
                self.semantic_regions[class_name] = world_pos
    
    def get_semantic_location(self, object_name: str) -> Optional[np.ndarray]:
        """Get world location of a semantic object/region."""
        object_lower = object_name.lower()
        
        # Direct match
        if object_lower in self.semantic_regions:
            return self.semantic_regions[object_lower]
        
        # Partial match
        for name, location in self.semantic_regions.items():
            if object_lower in name.lower() or name.lower() in object_lower:
                return location
        
        return None
    
    def get_occupancy_grid(self) -> np.ndarray:
        """Get the occupancy grid for path planning."""
        return self.semantic_map.grid
    
    def is_navigable(self, world_pos: np.ndarray) -> bool:
        """Check if a world position is navigable."""
        gx, gy = self.semantic_map.world_to_grid(world_pos)
        return self.semantic_map.is_free(gx, gy)
    
    def get_robot_pose(self) -> np.ndarray:
        """Get current robot pose."""
        return self.robot_pose
    
    # MCP Tool Interface
    def mcp_tool_definition(self) -> dict:
        """Export MCP tool definitions."""
        return {
            "name": "slam_agent",
            "description": "Semantic SLAM for building environment maps",
            "tools": [
                {
                    "name": "update_map",
                    "description": "Update map with current sensor data",
                    "input_schema": {"type": "object", "properties": {}}
                },
                {
                    "name": "get_semantic_location",
                    "description": "Get world location of a semantic object",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "object_name": {"type": "string"}
                        },
                        "required": ["object_name"]
                    }
                },
                {
                    "name": "is_navigable",
                    "description": "Check if a position is navigable",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["x", "y"]
                    }
                }
            ]
        }


# Standalone test
if __name__ == "__main__":
    print("Testing Semantic SLAM Agent...")
    
    agent = SemanticSLAMAgent(device="cuda")
    
    # Create test images
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
    pose = np.array([0.0, 0.0, 0.0])
    
    # Update map
    semantic_map = agent.update(rgb, depth, pose)
    
    print(f"Map size: {semantic_map.width}x{semantic_map.height}")
    print(f"Resolution: {semantic_map.resolution}m/cell")
    print(f"Semantic regions found: {list(agent.semantic_regions.keys())}")
    
    print("\nSemantic SLAM Agent test complete!")
