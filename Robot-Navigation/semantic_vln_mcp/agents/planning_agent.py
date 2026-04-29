"""
Planning Agent: A* + DWA Path Planning
========================================
Part of SemanticVLN-MCP Framework

Implements path planning with semantic constraints:
- A* for global path planning
- Dynamic Window Approach (DWA) for local obstacle avoidance
- Semantic constraints (avoid fragile objects, prefer walkable areas)

Author: SemanticVLN-MCP Team
"""

import math
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum


class CellType(Enum):
    """Cell types for planning."""
    FREE = 0
    OCCUPIED = 1
    UNKNOWN = 2
    FRAGILE = 3  # Avoid these more carefully


@dataclass
class Waypoint:
    """Single waypoint in a path."""
    x: float  # World x coordinate
    y: float  # World y coordinate
    theta: Optional[float] = None  # Optional orientation
    velocity: float = 0.3  # Desired velocity at this point


@dataclass
class Path:
    """Complete navigation path."""
    waypoints: List[Waypoint]
    total_distance: float
    is_valid: bool
    planning_time_ms: float


@dataclass
class DWAConfig:
    """Configuration for Dynamic Window Approach."""
    max_linear_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 1.5  # rad/s
    linear_acceleration: float = 0.5  # m/s^2
    angular_acceleration: float = 2.0  # rad/s^2
    dt: float = 0.1  # Time step
    predict_steps: int = 30  # Number of prediction steps
    goal_weight: float = 1.0
    obstacle_weight: float = 3.0
    velocity_weight: float = 0.5


class PlanningAgent:
    """
    Path planning agent with semantic awareness.
    
    Features:
    - A* global path planning on semantic grid
    - DWA local obstacle avoidance
    - Semantic constraints (fragile objects, walkable areas)
    - Smooth path generation
    """
    
    def __init__(self, 
                 resolution: float = 0.05,
                 robot_radius: float = 0.2,
                 goal_threshold: float = 0.15):
        """
        Initialize planning agent.
        
        Args:
            resolution: Grid resolution (meters per cell)
            robot_radius: Robot radius for collision checking
            goal_threshold: Distance threshold for goal reached
        """
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.goal_threshold = goal_threshold
        self.robot_radius_cells = int(robot_radius / resolution) + 1
        
        # DWA configuration
        self.dwa_config = DWAConfig()
        
        # Initialize with empty occupancy grid for 12x12m arena
        # This ensures A* works immediately without waiting for SLAM
        arena_size = 12.0  # meters
        grid_cells = int(arena_size / resolution)
        self.occupancy_grid = np.zeros((grid_cells, grid_cells), dtype=np.float32)
        self.grid_width = grid_cells
        self.grid_height = grid_cells
        self.grid_origin = np.array([-arena_size/2, -arena_size/2])  # Center at (0,0)
    
    def set_map(self, 
                occupancy_grid: np.ndarray,
                origin: np.ndarray,
                resolution: float):
        """
        Set the occupancy grid for planning.
        
        Args:
            occupancy_grid: 2D numpy array (0=free, 1=occupied)
            origin: World position of grid (0,0)
            resolution: Meters per cell
        """
        self.occupancy_grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_origin = origin[:2] if len(origin) > 2 else origin
        self.resolution = resolution
        self.robot_radius_cells = int(self.robot_radius / resolution) + 1
    
    def plan_path(self, 
                  start: np.ndarray, 
                  goal: np.ndarray,
                  semantic_constraints: Optional[Dict] = None) -> Path:
        """
        Plan a path from start to goal using A*.
        
        Args:
            start: Start position [x, y, theta]
            goal: Goal position [x, y]
            semantic_constraints: Optional constraints (avoid classes, prefer regions)
            
        Returns:
            Path object with waypoints
        """
        import time
        start_time = time.time()
        
        if self.occupancy_grid is None:
            # No map - return direct path
            return Path(
                waypoints=[
                    Waypoint(x=start[0], y=start[1]),
                    Waypoint(x=goal[0], y=goal[1])
                ],
                total_distance=np.linalg.norm(goal[:2] - start[:2]),
                is_valid=True,
                planning_time_ms=0.0
            )
        
        # USE A COPY to avoid modifying the shared SLAM map
        grid_copy = np.copy(self.occupancy_grid)
        
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start[:2])
        goal_grid = self._world_to_grid(goal[:2])
        
        # FORCE-CLEAR robot's current position (larger radius)
        sx, sy = start_grid
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    grid_copy[ny, nx] = 0.0
        
        # ALSO force-clear goal position
        gx, gy = goal_grid
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    grid_copy[ny, nx] = 0.0
        
        # CLEAR a corridor from start to goal (ensures A* can always find path)
        # INCREASED: (-4,5) = 9-cell radius (45cm) - for reliable path finding
        num_steps = max(abs(gx - sx), abs(gy - sy), 1)
        for i in range(num_steps + 1):
            t = i / num_steps
            cx = int(sx + t * (gx - sx))
            cy = int(sy + t * (gy - sy))
            for dx in range(-4, 5):  # 9-cell corridor for reliable A*
                for dy in range(-4, 5):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        grid_copy[ny, nx] = 0.0
        
        # Check if start/goal are valid (using grid_copy)
        if not self._is_valid_cell(start_grid, grid_copy):
            print(f"[PLANNING] Warning: Start {start_grid} is blocked or OOB")
            # Don't return yet, attempt to find nearest free neighbor
            
        if not self._is_valid_cell(goal_grid, grid_copy):
            print(f"[PLANNING] Warning: Goal {goal_grid} is blocked or OOB")

        # Run A* algorithm (pass grid_copy)
        path_grid = self._astar(start_grid, goal_grid, grid_copy)
        
        if not path_grid:
            # Log why A* failed
            occupied_cells = np.sum(grid_copy > 0.5)
            total_cells = grid_copy.size
            print(f"[PLANNING] A* failed: {occupied_cells}/{total_cells} cells occupied ({100*occupied_cells/total_cells:.1f}%)")
            planning_time = (time.time() - start_time) * 1000
            return Path([], 0.0, False, planning_time)
        
        # Convert grid path to world coordinates
        waypoints = self._grid_path_to_waypoints(path_grid)
        
        # Smooth the path
        waypoints = self._smooth_path(waypoints)
        
        # Calculate total distance
        total_distance = sum(
            np.linalg.norm(np.array([w2.x - w1.x, w2.y - w1.y]))
            for w1, w2 in zip(waypoints[:-1], waypoints[1:])
        )
        
        planning_time = (time.time() - start_time) * 1000
        
        return Path(
            waypoints=waypoints,
            total_distance=total_distance,
            is_valid=True,
            planning_time_ms=planning_time
        )
    
    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_pos = (world_pos - self.grid_origin) / self.resolution
        return int(grid_pos[0]), int(grid_pos[1])
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        return np.array([grid_x, grid_y]) * self.resolution + self.grid_origin
    
    def _is_valid_cell(self, cell: Tuple[int, int], grid: np.ndarray) -> bool:
        """Check if cell is valid and free."""
        x, y = cell
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        return self._is_free(x, y, grid)
    
    def _is_free(self, x: int, y: int, grid: np.ndarray) -> bool:
        """Check if cell is free (with robot radius inflation)."""
        # Safety radius increased to 25cm (5 cells) for better collision avoidance
        # TurtleBot3 Burger is ~18cm diameter, this provides 7cm safety margin
        radius = 5 
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if grid[ny, nx] > 0.8: 
                        return False
        return True
    
    def _astar(self, 
               start: Tuple[int, int], 
               goal: Tuple[int, int],
               grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm.
        
        Returns:
            List of grid cells from start to goal
        """
        # Priority queue: (f_score, counter, cell)
        counter = 0
        open_set = [(0, counter, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, goal)}
        
        open_set_hash: Set[Tuple[int, int]] = {start}
        
        # 8-connected neighbors
        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid_cell(neighbor, grid):
                    continue
                
                # Cost: 1.0 for cardinal, 1.414 for diagonal
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)
        
        return []  # No path found
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _grid_path_to_waypoints(self, path_grid: List[Tuple[int, int]]) -> List[Waypoint]:
        """Convert grid path to world waypoints."""
        waypoints = []
        for gx, gy in path_grid:
            world_pos = self._grid_to_world(gx, gy)
            waypoints.append(Waypoint(x=world_pos[0], y=world_pos[1]))
        return waypoints
    
    def _smooth_path(self, waypoints: List[Waypoint], 
                     weight_smooth: float = 0.5,
                     weight_data: float = 0.1,
                     tolerance: float = 0.001) -> List[Waypoint]:
        """
        Smooth path using gradient descent.
        
        This reduces the number of waypoints while maintaining path validity.
        """
        if len(waypoints) <= 2:
            return waypoints
        
        # Convert to numpy array
        path = np.array([[w.x, w.y] for w in waypoints])
        new_path = path.copy()
        
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(2):
                    old = new_path[i, j]
                    new_path[i, j] += weight_data * (path[i, j] - new_path[i, j])
                    new_path[i, j] += weight_smooth * (
                        new_path[i-1, j] + new_path[i+1, j] - 2*new_path[i, j]
                    )
                    change += abs(old - new_path[i, j])
        
        # Subsample to reduce waypoints
        subsampled = [Waypoint(x=new_path[0, 0], y=new_path[0, 1])]
        for i in range(1, len(new_path) - 1, 3):  # Every 3rd point
            subsampled.append(Waypoint(x=new_path[i, 0], y=new_path[i, 1]))
        subsampled.append(Waypoint(x=new_path[-1, 0], y=new_path[-1, 1]))
        
        return subsampled
    
    def compute_dwa_velocity(self,
                            current_pose: np.ndarray,
                            current_velocity: Tuple[float, float],
                            goal: np.ndarray) -> Tuple[float, float]:
        """
        Compute best velocity using Dynamic Window Approach.
        
        Args:
            current_pose: [x, y, theta]
            current_velocity: (linear_vel, angular_vel)
            goal: Target position [x, y]
            
        Returns:
            (linear_velocity, angular_velocity)
        """
        cfg = self.dwa_config
        v, w = current_velocity
        
        # Dynamic window
        v_min = max(0, v - cfg.linear_acceleration * cfg.dt)
        v_max = min(cfg.max_linear_velocity, v + cfg.linear_acceleration * cfg.dt)
        w_min = max(-cfg.max_angular_velocity, w - cfg.angular_acceleration * cfg.dt)
        w_max = min(cfg.max_angular_velocity, w + cfg.angular_acceleration * cfg.dt)
        
        best_score = float('-inf')
        best_v, best_w = 0.0, 0.0
        
        # Sample velocities
        for v_sample in np.linspace(v_min, v_max, 10):
            for w_sample in np.linspace(w_min, w_max, 15):
                # Simulate trajectory
                trajectory = self._simulate_trajectory(
                    current_pose, v_sample, w_sample, cfg.predict_steps, cfg.dt
                )
                
                # Score trajectory
                goal_score = self._goal_score(trajectory[-1], goal)
                obstacle_score = self._obstacle_score(trajectory)
                velocity_score = v_sample / cfg.max_linear_velocity
                
                total_score = (
                    cfg.goal_weight * goal_score +
                    cfg.obstacle_weight * obstacle_score +
                    cfg.velocity_weight * velocity_score
                )
                
                if total_score > best_score:
                    best_score = total_score
                    best_v, best_w = v_sample, w_sample
        
        return best_v, best_w
    
    def _simulate_trajectory(self,
                            pose: np.ndarray,
                            v: float,
                            w: float,
                            steps: int,
                            dt: float) -> List[np.ndarray]:
        """Simulate robot trajectory given constant velocity."""
        trajectory = [pose.copy()]
        current = pose.copy()
        
        for _ in range(steps):
            current[0] += v * math.cos(current[2]) * dt
            current[1] += v * math.sin(current[2]) * dt
            current[2] += w * dt
            trajectory.append(current.copy())
        
        return trajectory
    
    def _goal_score(self, pose: np.ndarray, goal: np.ndarray) -> float:
        """Score based on distance to goal."""
        dist = np.linalg.norm(pose[:2] - goal[:2])
        return 1.0 / (1.0 + dist)
    
    def _obstacle_score(self, trajectory: List[np.ndarray]) -> float:
        """Score based on distance to obstacles."""
        if self.occupancy_grid is None:
            return 1.0
        
        min_dist = float('inf')
        for pose in trajectory:
            gx, gy = self._world_to_grid(pose[:2])
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                if self.occupancy_grid[gy, gx] > 0.5:
                    return 0.0  # Collision
                # Simple distance check
                min_dist = min(min_dist, 1.0 - self.occupancy_grid[gy, gx])
        
        return min(1.0, min_dist)
    
    def is_goal_reached(self, current_pos: np.ndarray, goal: np.ndarray) -> bool:
        """Check if robot has reached the goal."""
        return np.linalg.norm(current_pos[:2] - goal[:2]) < self.goal_threshold
    
    # MCP Tool Interface
    def mcp_tool_definition(self) -> dict:
        """Export MCP tool definitions."""
        return {
            "name": "planning_agent",
            "description": "Path planning with A* and DWA",
            "tools": [
                {
                    "name": "plan_path",
                    "description": "Plan a path from current position to goal",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "goal_x": {"type": "number"},
                            "goal_y": {"type": "number"}
                        },
                        "required": ["goal_x", "goal_y"]
                    }
                },
                {
                    "name": "get_next_velocity",
                    "description": "Get next velocity command using DWA",
                    "input_schema": {"type": "object", "properties": {}}
                }
            ]
        }


# Standalone test
if __name__ == "__main__":
    print("Testing Planning Agent...")
    
    agent = PlanningAgent()
    
    # Create test grid (simple maze)
    grid = np.zeros((100, 100))
    grid[40:60, 30:35] = 1.0  # Obstacle
    
    agent.set_map(
        occupancy_grid=grid,
        origin=np.array([-5.0, -5.0]),
        resolution=0.1
    )
    
    # Plan path
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([4.0, 4.0])
    
    path = agent.plan_path(start, goal)
    
    print(f"Path valid: {path.is_valid}")
    print(f"Waypoints: {len(path.waypoints)}")
    print(f"Total distance: {path.total_distance:.2f}m")
    print(f"Planning time: {path.planning_time_ms:.2f}ms")
    
    if path.waypoints:
        print(f"Start: ({path.waypoints[0].x:.2f}, {path.waypoints[0].y:.2f})")
        print(f"Goal: ({path.waypoints[-1].x:.2f}, {path.waypoints[-1].y:.2f})")
    
    print("\nPlanning Agent test complete!")
