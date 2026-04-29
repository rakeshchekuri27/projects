"""
MCP Orchestrator: Multi-Agent Coordination
============================================
Part of SemanticVLN-MCP Framework

NOVEL CONTRIBUTION: First application of MCP to robotics

Coordinates all agents:
- Perception Agent (YOLOv8)
- Semantic SLAM Agent (DeepLabV3+)
- Planning Agent (A* + DWA)
- Reasoning Agent (Ollama)
- Gesture Agent (MediaPipe)
- TSG Algorithm (Temporal Semantic Grounding)

Author: SemanticVLN-MCP Team
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

# Import all agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.perception_agent import PerceptionAgent, PerceptionResult
from agents.slam_agent import SemanticSLAMAgent, SemanticMap
from agents.planning_agent import PlanningAgent, Path, Waypoint
from agents.reasoning_agent import ReasoningAgent, ReasoningResult
from agents.gesture_agent import GestureAgent, GestureResult, GestureType
from algorithms.tsg import TemporalSemanticGrounding

# Evaluation logging
try:
    from evaluation_logger import get_logger as get_eval_logger
    EVAL_LOGGING_ENABLED = True
except ImportError:
    EVAL_LOGGING_ENABLED = False
    def get_eval_logger():
        return None


class RobotState(Enum):
    """Robot state machine states."""
    IDLE = "idle"
    PROCESSING_INSTRUCTION = "processing"
    NAVIGATING = "navigating"
    SEARCHING = "searching"
    FOLLOWING = "following"
    EXPLORING = "exploring"  # NEW: Visit all rooms
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class NavigationContext:
    """Shared context for the MCP loop."""
    # Sensor data
    latest_rgb_frame: Optional[np.ndarray] = None
    latest_depth_frame: Optional[np.ndarray] = None
    latest_lidar_scan: Optional[np.ndarray] = None
    robot_pose: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    # Agent outputs
    detections: List[Dict] = field(default_factory=list)
    semantic_map: Optional[SemanticMap] = None
    current_path: Optional[Path] = None
    current_instruction: Optional[str] = None
    parsed_instruction: Optional[Dict] = None
    
    # TSG beliefs
    object_beliefs: Dict[str, Any] = field(default_factory=dict)
    
    # Goals
    goal_position: Optional[np.ndarray] = None
    target_location: Optional[str] = None
    target_object: Optional[str] = None
    current_waypoint_index: int = 0
    
    # Control
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    
    # Latency tracking (benchmarking)
    latencies: Dict[str, float] = field(default_factory=lambda: {
        "perception": 0.0,
        "slam": 0.0,
        "planning": 0.0,
        "tsg": 0.0,
        "reasoning": 0.0,
        "gesture": 0.0,
        "mcp_total": 0.0
    })
    
    # State
    state: RobotState = RobotState.IDLE
    timestamp: float = 0.0
    
    # Stuck detection
    last_pose: Optional[np.ndarray] = None
    stuck_start_time: float = 0.0
    recovery_mode: bool = False
    recovery_end_time: float = 0.0
    
    # Backup loop prevention
    backup_start_time: float = 0.0
    is_backing_up: bool = False
    
    # Detection logging throttle
    detection_log_counter: int = 0
    last_logged_detections: str = ""
    
    # Dynamic goal update cooldown
    last_goal_update_time: float = 0.0
    
    # Exploration mode
    exploration_rooms: List[str] = field(default_factory=list)
    exploration_index: int = 0
    exploring_rotation: float = 0.0  # Track rotation at each room


class MCPOrchestrator:
    """
    NOVEL: MCP-based Multi-Agent Orchestrator for Robotics
    
    First application of Anthropic's Model Context Protocol to embodied AI.
    
    Features:
    - Asynchronous agent coordination
    - Parallel execution of perception and SLAM
    - Shared context management
    - State machine for robot control
    - MCP tool interface for external access
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the MCP orchestrator.
        
        Args:
            device: 'cuda' or 'cpu' for neural network inference
        """
        print("Initializing MCP Orchestrator...")
        
        # Initialize agents with lower confidence for better detection
        self.perception_agent = PerceptionAgent(device=device, confidence_threshold=0.3)
        self.slam_agent = SemanticSLAMAgent(device=device)
        self.planning_agent = PlanningAgent()
        self.reasoning_agent = ReasoningAgent()
        self.gesture_agent = GestureAgent()
        
        # Initialize TSG algorithm
        self.tsg = TemporalSemanticGrounding()
        
        # Shared context
        self.context = NavigationContext()
        
        # Control loop timing
        self.last_perception_time = 0.0
        self.last_slam_time = 0.0
        self.last_planning_time = 0.0
        self.perception_interval = 0.033  # 30 Hz
        self.slam_interval = 0.1  # 10 Hz
        self.planning_interval = 0.2  # 5 Hz
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_goal_reached: Optional[Callable] = None
        
        print("MCP Orchestrator initialized!")
    
    async def process_instruction(self, instruction: str):
        """
        Process a natural language navigation instruction.
        
        This is the main entry point for NL commands.
        
        Args:
            instruction: Natural language command
        """
        print(f"[ORCHESTRATOR] Processing: \"{instruction}\"")
        self.context.current_instruction = instruction
        self.context.state = RobotState.PROCESSING_INSTRUCTION
        
        # Step 1: Parse instruction using reasoning agent
        reasoning_result = await self.reasoning_agent.parse_instruction(
            instruction,
            context={
                "detected_objects": [d.get('class_name') for d in self.context.detections],
                "current_room": self._get_current_room()
            }
        )
        
        self.context.parsed_instruction = {
            "goal_type": reasoning_result.parsed_instruction.goal_type,
            "target_object": reasoning_result.parsed_instruction.target_object,
            "target_location": reasoning_result.parsed_instruction.target_location,
            "sub_tasks": reasoning_result.parsed_instruction.sub_tasks
        }
        
        # Store targets for recovery
        self.context.target_location = reasoning_result.parsed_instruction.target_location
        self.context.target_object = reasoning_result.parsed_instruction.target_object
        
        print(f"[REASONING] Parsed: {self.context.parsed_instruction}")
        
        # Check for STOP command first
        if reasoning_result.parsed_instruction.goal_type == "stop":
            print("[ORCHESTRATOR] Stopping robot")
            self.context.state = RobotState.IDLE
            self.context.goal_position = None
            self.context.current_path = None
            return
        
        # Check for EXPLORE command - visit all rooms
        if reasoning_result.parsed_instruction.goal_type == "explore":
            print("[ORCHESTRATOR] Starting exploration mode - will visit all rooms")
            self.context.exploration_rooms = ["kitchen", "living_room", "bedroom", "bathroom"]
            self.context.exploration_index = 0
            self.context.exploring_rotation = 0.0
            self.context.state = RobotState.EXPLORING
            # Navigate to first room
            await self._navigate_to_next_exploration_room()
            return
        
        # Step 2: Determine goal position
        goal = await self._resolve_goal(reasoning_result)
        
        if goal is not None:
            self.context.goal_position = goal
            
            # Step 3: Plan path to goal
            path = self.planning_agent.plan_path(
                start=self.context.robot_pose,
                goal=goal
            )
            
            if path.is_valid:
                self.context.current_path = path
                self.context.current_waypoint_index = 0
                self.context.state = RobotState.NAVIGATING
                print(f"[PLANNING] Path found with {len(path.waypoints)} waypoints")
            else:
                # FALLBACK: Create direct path through center when A* fails
                print("[PLANNING] A* failed, using direct navigation")
                from semantic_vln_mcp.agents.planning_agent import Waypoint, Path
                
                # Create simple path: current → center (0,0) → goal
                current = self.context.robot_pose[:2]
                center = np.array([0.0, 0.0])
                
                # Only go through center if goal is in different quadrant
                waypoints = []
                if (goal[0] > 1 and current[0] < -1) or (goal[0] < -1 and current[0] > 1) or \
                   (goal[1] > 1 and current[1] < -1) or (goal[1] < -1 and current[1] > 1):
                    waypoints.append(Waypoint(x=center[0], y=center[1]))
                waypoints.append(Waypoint(x=goal[0], y=goal[1]))
                
                fallback_path = Path(waypoints=waypoints, is_valid=True, total_distance=0.0, planning_time_ms=0.0)
                self.context.current_path = fallback_path
                self.context.current_waypoint_index = 0
                self.context.state = RobotState.NAVIGATING
                print(f"[PLANNING] Fallback path with {len(waypoints)} waypoints")
        else:
            # Cannot determine goal - start searching
            self.context.state = RobotState.SEARCHING
            print("[ORCHESTRATOR] Goal unclear, entering search mode")
    
    async def _resolve_goal(self, reasoning_result: ReasoningResult) -> Optional[np.ndarray]:
        """Resolve goal position from reasoning result."""
        parsed = reasoning_result.parsed_instruction
        
        # 12x12 arena - goals in OPEN areas, away from corners
        ROOM_LOCATIONS = {
            # Kitchen: Goal in OPEN area, not in corner
            "kitchen": np.array([2.5, 2.5]),  # Away from fridge corner
            "fridge": np.array([3.5, 3.5]),

            # Living Room: Bottom-Left - OPEN area
            "living room": np.array([-2.5, -2.5]),
            "living": np.array([-2.5, -2.5]),
            "lounge": np.array([-2.5, -2.5]),
            "sofa": np.array([-3.5, -3.5]),
            # Bedroom: Top-Left - OPEN area
            "bedroom": np.array([-2.5, 2.5]),
            "bed": np.array([-3.5, 3.5]),
            # Bathroom: Bottom-Right - OPEN area (moved inward from wall)
            "bathroom": np.array([2.0, -2.0]),
            "toilet": np.array([3.5, -3.5]),
            "restroom": np.array([2.5, -2.5]),
            # Furniture
            "table": np.array([2.5, 3.5]),
            "chair": np.array([2.0, 3.5]),
            # Person walks in open area
            "person": np.array([-1.0, 1.0]),
            "pedestrian": np.array([-1.0, 1.0]),
            "human": np.array([-1.0, 1.0]),
        }
        
        # Helper to find matching location
        def find_location(text: str) -> Optional[np.ndarray]:
            if not text:
                return None
            text_lower = text.lower()
            for name, loc in ROOM_LOCATIONS.items():
                if name in text_lower:
                    print(f"[GOAL] Using predefined location for '{name}': {loc}")
                    return loc
            return None
        
        # Try TSG first for object locations (NOVEL: Uses tracked locations from YOLO)
        if parsed.target_object:
            try:
                location, confidence = self.tsg.query_location(parsed.target_object)
                print(f"[TSG] Query for '{parsed.target_object}': loc={location}, conf={confidence:.2f}")
                if location is not None and confidence > 0.2:  # Lower threshold for more responsive tracking
                    # Offset goal 0.5m towards robot (navigate NEAR object, not ON it)
                    robot_pos = self.context.robot_pose[:2]
                    direction = robot_pos - location
                    dist = np.linalg.norm(direction)
                    if dist > 0.5:
                        direction = direction / dist  # Normalize
                        location = location + direction * 0.5  # 0.5m offset
                    print(f"[TSG] *** FOUND {parsed.target_object} via TSG at {location} (conf={confidence:.2f}) ***")
                    return location
                else:
                    print(f"[TSG] {parsed.target_object} not confident enough (conf={confidence:.2f})")
            except Exception as e:
                print(f"[TSG] Query error: {e}")
            # Try predefined location for target_object as fallback
            loc = find_location(parsed.target_object)
            if loc is not None:
                print(f"[TSG] Using fallback predefined location for {parsed.target_object}: {loc}")
                return loc
        
        # Try predefined locations for target_location
        # BUT only if target_object wasn't found via TSG (prevent LLM false associations)
        if parsed.target_location and not parsed.target_object:
            loc = find_location(parsed.target_location)
            if loc is not None:
                return loc
        
        # Try semantic map for locations
        if parsed.target_location:
            location = self.slam_agent.get_semantic_location(parsed.target_location)
            if location is not None:
                print(f"[GOAL] Found {parsed.target_location} via SLAM at {location}")
                return location
        
        # Try original text for any keywords
        if parsed.original_text:
            loc = find_location(parsed.original_text)
            if loc is not None:
                return loc
        
        # Try to resolve implicit goals
        implicit_goal = self.reasoning_agent.resolve_implicit_goal(
            parsed.original_text,
            [d.get('class_name', '') for d in self.context.detections]
        )
        if implicit_goal:
            # Check predefined locations for implicit goal
            loc = find_location(implicit_goal)
            if loc is not None:
                return loc
            
            location = self.slam_agent.get_semantic_location(implicit_goal)
            if location is not None:
                return location
        
        print("[GOAL] Could not resolve goal position")
        return None
    
    async def _navigate_to_next_exploration_room(self):
        """Navigate to the next room in exploration sequence."""
        if self.context.exploration_index >= len(self.context.exploration_rooms):
            print("[EXPLORE] Exploration complete! Visited all rooms.")
            self.context.state = RobotState.IDLE
            return
        
        room = self.context.exploration_rooms[self.context.exploration_index]
        print(f"[EXPLORE] Navigating to room {self.context.exploration_index + 1}/{len(self.context.exploration_rooms)}: {room}")
        
        # Room locations
        ROOM_LOCS = {
            "kitchen": np.array([2.5, 2.5]),
            "living_room": np.array([-2.5, -2.5]),
            "bedroom": np.array([-2.5, 2.5]),
            "bathroom": np.array([2.0, -2.0]),
        }
        
        goal = ROOM_LOCS.get(room)
        if goal is not None:
            self.context.goal_position = goal
            path = self.planning_agent.plan_path(
                start=self.context.robot_pose,
                goal=goal
            )
            if path.is_valid:
                self.context.current_path = path
                self.context.current_waypoint_index = 0
                print(f"[EXPLORE] Path to {room} found")
            else:
                # Direct navigation fallback
                from semantic_vln_mcp.agents.planning_agent import Waypoint, Path
                waypoints = [Waypoint(x=goal[0], y=goal[1])]
                self.context.current_path = Path(waypoints=waypoints, is_valid=True, total_cost=0)
                self.context.current_waypoint_index = 0
    
    def advance_exploration(self):
        """Called when robot reaches exploration waypoint - rotate and move to next room."""
        self.context.exploration_index += 1
        self.context.exploring_rotation = 0.0
        print(f"[EXPLORE] Room explored, moving to next")
    
    def _get_current_room(self) -> str:
        """Determine current room based on robot position."""
        # Simple heuristic based on position
        x, y = self.context.robot_pose[:2]
        
        # Room boundaries (would be from SLAM in practice)
        if x > 2.0:
            return "kitchen"
        elif x < -2.0 and y > 0:
            return "bedroom"
        elif x < -2.0 and y < 0:
            return "bathroom"
        else:
            return "living_room"
    
    async def update(self, 
                     rgb_frame: np.ndarray,
                     depth_frame: np.ndarray,
                     robot_pose: np.ndarray,
                     lidar_scan: Optional[np.ndarray] = None) -> tuple:
        """
        Main update loop - process sensor data and compute control.
        
        Args:
            rgb_frame: RGB camera image
            depth_frame: Depth camera image
            robot_pose: Current robot pose [x, y, theta]
            lidar_scan: 360-degree lidar scan (optional)
            
        Returns:
            (linear_velocity, angular_velocity)
        """
        current_time = time.time()
        self.context.latest_rgb_frame = rgb_frame
        self.context.latest_depth_frame = depth_frame
        self.context.latest_lidar_scan = lidar_scan
        self.context.robot_pose = robot_pose
        self.context.timestamp = current_time
        
        # Run perception and SLAM in parallel (async)
        perception_task = None
        slam_task = None
        
        if current_time - self.last_perception_time > self.perception_interval:
            perception_task = asyncio.create_task(
                self._run_perception(rgb_frame, depth_frame)
            )
            self.last_perception_time = current_time
        
        if current_time - self.last_slam_time > self.slam_interval:
            slam_task = asyncio.create_task(
                self._run_slam(rgb_frame, depth_frame, robot_pose, lidar_scan)
            )
            self.last_slam_time = current_time
        
        # Wait for tasks to complete
        if perception_task:
            await perception_task
        if slam_task:
            await slam_task
        
        # --- STUCK DETECTION LOGIC ---
        if self.context.state == RobotState.NAVIGATING:
            if self.context.last_pose is None:
                self.context.last_pose = np.copy(robot_pose)
                self.context.stuck_start_time = current_time
            else:
                dist_moved = np.linalg.norm(robot_pose[:2] - self.context.last_pose[:2])
                # If moved more than 5cm, reset timer
                if dist_moved > 0.05:
                    self.context.last_pose = np.copy(robot_pose)
                    self.context.stuck_start_time = current_time
                elif current_time - self.context.stuck_start_time > 5.0:
                    # Trigger recovery!
                    if not self.context.recovery_mode:
                        print(f"\n[STUCK] Robot hasn't moved in 5s. Entering recovery mode!")
                        self.context.recovery_mode = True
                        self.context.recovery_end_time = current_time + 3.0 # Recover for 3s
        else:
            self.context.last_pose = None
            self.context.recovery_mode = False

        # Update TSG with latest detections
        start_tsg = time.time()
        self.tsg.update(
            detections=self.context.detections,
            robot_pose=robot_pose
        )
        self.context.object_beliefs = self.tsg.get_all_beliefs()
        self.context.latencies["tsg"] = time.time() - start_tsg
        
        # Check for gestures
        start_gest = time.time()
        gesture_result = self.gesture_agent.recognize_gesture(rgb_frame)
        if gesture_result.gesture != GestureType.NONE:
            await self._handle_gesture(gesture_result)
        self.context.latencies["gesture"] = time.time() - start_gest
        
        # Compute control based on state
        start_plan = time.time()
        result = self._compute_control()
        self.context.latencies["planning"] = time.time() - start_plan
        
        # Total MCP Loop Time
        self.context.latencies["mcp_total"] = time.time() - current_time
        return result
    
    async def _run_perception(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """Run perception agent."""
        start_time = time.time()
        result = self.perception_agent.detect_objects(rgb_frame, depth_frame)
        self.context.latencies["perception"] = time.time() - start_time
        
        # Define relevant indoor classes (filter out false positives)
        relevant_classes = {
            'person', 'chair', 'couch', 'table', 'bed', 'tv', 'bottle', 
            'cup', 'refrigerator', 'sink', 'potted plant', 'dining table',
            'laptop', 'book', 'vase', 'clock'
        }
        
        # Transform robot-frame coords to world coords
        robot_x = self.context.robot_pose[0]
        robot_y = self.context.robot_pose[1]
        robot_theta = self.context.robot_pose[2]
        
        self.context.detections = []
        for d in result.detections:
            if d.class_name.lower() not in relevant_classes:
                continue
            
            # Filter low-confidence person detections (reduce false positives)
            if d.class_name.lower() == 'person' and d.confidence < 0.50:
                continue  # Skip low-confidence person detections
            
            world_pos = None
            if d.world_position is not None:
                # Transform from robot frame to world frame
                # Robot frame: x=forward, y=left
                # World frame: x=east, y=north
                rx, ry = d.world_position[0], d.world_position[1]
                world_x = robot_x + rx * np.cos(robot_theta) - ry * np.sin(robot_theta)
                world_y = robot_y + rx * np.sin(robot_theta) + ry * np.cos(robot_theta)
                
                # BOUNDS CHECK: Keep within arena (-5 to +5)
                world_x = np.clip(world_x, -4.5, 4.5)
                world_y = np.clip(world_y, -4.5, 4.5)
                world_pos = (world_x, world_y)
            
            self.context.detections.append({
                'object_id': d.object_id,
                'class_name': d.class_name,
                'class_id': d.class_id,
                'bbox': d.bbox,
                'confidence': d.confidence,
                'depth': d.depth,
                'world_position': world_pos
            })
        
        # LOG DETECTIONS (only when detections CHANGE, throttled)
        if len(self.context.detections) > 0:
            det_summary = ", ".join([f"{d['class_name']}({d['confidence']:.2f})" for d in self.context.detections[:3]])
            # Only log if detections changed OR every 30 frames
            self.context.detection_log_counter += 1
            if det_summary != self.context.last_logged_detections or self.context.detection_log_counter >= 30:
                print(f"[YOLO] Detected: {det_summary}")
                self.context.last_logged_detections = det_summary
                self.context.detection_log_counter = 0
            
            # DYNAMIC GOAL UPDATE: Update goal ONCE when target is first spotted
            # After first update, robot keeps navigating to that location
            if self.context.target_object and self.context.state == RobotState.NAVIGATING:
                # Only update if we haven't updated yet (last_goal_update_time == 0)
                if self.context.last_goal_update_time == 0:
                    for det in self.context.detections:
                        if det['class_name'].lower() == self.context.target_object.lower() and det['world_position']:
                            # Update goal to FIRST detected position
                            new_x, new_y = det['world_position']
                            print(f"[LIVE] Target {self.context.target_object} spotted at ({new_x:.1f}, {new_y:.1f}) - navigating there!")
                            self.context.goal_position = np.array([new_x, new_y])
                            self.context.current_path = None  # Force replan
                            self.context.last_goal_update_time = time.time()  # Mark as updated
                            break
    
    async def _run_slam(self, rgb_frame: np.ndarray, 
                        depth_frame: np.ndarray, 
                        robot_pose: np.ndarray,
                        lidar_scan: Optional[np.ndarray] = None):
        """Run SLAM agent."""
        start_time = time.time()
        semantic_map = self.slam_agent.update(rgb_frame, depth_frame, robot_pose, lidar_scan)
        self.context.latencies["slam"] = time.time() - start_time
        self.context.semantic_map = semantic_map
        
        # Update planning agent with new map
        self.planning_agent.set_map(
            occupancy_grid=semantic_map.grid,
            origin=semantic_map.origin,
            resolution=semantic_map.resolution
        )
        
        # Update TSG with semantic regions
        for name, location in self.slam_agent.semantic_regions.items():
            self.tsg.set_semantic_region(name, location)
    
    async def _handle_gesture(self, gesture: GestureResult):
        """Handle detected gesture."""
        command = self.gesture_agent.gesture_to_command(gesture)
        
        # High confidence required to avoid accidental stops during demo
        if command is None or gesture.confidence < 0.85:
            return
        
        action = command.get("action")
        
        if action == "stop":
            self.context.state = RobotState.STOPPED
            self.context.linear_velocity = 0.0
            self.context.angular_velocity = 0.0
            print("[GESTURE] Stop command received")
        
        elif action == "follow_me":
            self.context.state = RobotState.FOLLOWING
            print("[GESTURE] Follow command received")
        
        elif action == "navigate_direction" and gesture.pointing_direction:
            # Navigate in pointed direction
            dx, dy = gesture.pointing_direction
            target = self.context.robot_pose[:2] + np.array([dx, dy]) * 2.0
            self.context.goal_position = target
            await self.process_instruction(f"Navigate to pointed direction")
            print(f"[GESTURE] Pointing detected, navigating to {target}")
        
        elif action == "confirm":
            print("[GESTURE] Action confirmed")
        
        elif action == "cancel":
            self.context.state = RobotState.IDLE
            self.context.current_path = None
            print("[GESTURE] Action cancelled")
    
    def _compute_control(self) -> tuple:
        """Compute velocity commands based on current state."""
        current_time = time.time()
        
        # 1. Handle Recovery Mode First
        if self.context.recovery_mode:
            if current_time < self.context.recovery_end_time:
                # Decisive recovery: Backup and Pivot
                # Pivot 90 degrees away from what was likely the obstacle side
                return -0.1, 1.5 
            else:
                print("[RECOVERY] Done. Attempting to resume navigation...")
                self.context.recovery_mode = False
                self.context.last_pose = None # Reset stuck detection
                # Re-trigger planning to find a fresh way out
                # Prioritize target_object over target_location
                if self.context.target_object or self.context.target_location:
                    recovery_target = self.context.target_object or self.context.target_location
                    asyncio.create_task(self.process_instruction(
                        f"Navigate to {recovery_target}"
                    ))
                return 0.0, 0.0

        if self.context.state == RobotState.STOPPED:
            return 0.0, 0.0
        
        if self.context.state == RobotState.IDLE:
            return 0.0, 0.0
        
        if self.context.state == RobotState.NAVIGATING:
            return self._navigate_to_waypoint()
        
        if self.context.state == RobotState.FOLLOWING:
            return self._follow_target()
        
        if self.context.state == RobotState.SEARCHING:
            return self._search_behavior()
        
        return 0.0, 0.0
    
    def _navigate_to_waypoint(self) -> tuple:
        """Navigate toward current waypoint using DWA."""
        if self.context.current_path is None:
            return 0.0, 0.0
        
        waypoints = self.context.current_path.waypoints
        if self.context.current_waypoint_index >= len(waypoints):
            # Path complete
            self.context.state = RobotState.IDLE
            if self.on_goal_reached:
                self.on_goal_reached()
            return 0.0, 0.0
        
        # Get current target waypoint
        waypoint = waypoints[self.context.current_waypoint_index]
        target = np.array([waypoint.x, waypoint.y])
        
        # Check if waypoint reached
        dist = np.linalg.norm(self.context.robot_pose[:2] - target)
        if dist < 0.3:  # 30cm threshold - more forgiving
            self.context.current_waypoint_index += 1
            return self._navigate_to_waypoint()
        
        # SIMPLE PROPORTIONAL CONTROL (more reliable than DWA)
        # Calculate angle to target
        dx = target[0] - self.context.robot_pose[0]
        dy = target[1] - self.context.robot_pose[1]
        target_angle = np.arctan2(dy, dx)
        
        # Calculate angle error
        angle_error = target_angle - self.context.robot_pose[2]
        # Normalize to [-pi, pi]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # ===== LIDAR AVOIDANCE FIRST (before any motion) =====
        if self.context.latest_lidar_scan is not None:
            scan = self.context.latest_lidar_scan
            num_points = len(scan)
            
            # Front sector (-45 to +45 degrees for wider coverage)
            front_width = num_points // 8  # 45 degrees each side
            front_indices = list(range(num_points - front_width, num_points)) + list(range(0, front_width + 1))
            left_indices = list(range(front_width + 1, num_points // 4))
            right_indices = list(range(num_points - num_points // 4, num_points - front_width))
            
            front_points = scan[front_indices]
            left_points = scan[left_indices]
            right_points = scan[right_indices]
            
            # Get min distances (ignore zeros/invalid)
            min_front = np.min(front_points[front_points > 0.1]) if np.any(front_points > 0.1) else 9.0
            min_left = np.min(left_points[left_points > 0.1]) if np.any(left_points > 0.1) else 9.0
            min_right = np.min(right_points[right_points > 0.1]) if np.any(right_points > 0.1) else 9.0
            
            # CRITICAL: Stop and backup if too close (VERY AGGRESSIVE)
            if min_front < 1.0:  # Increased from 0.6m for safer detection
                # BACKUP TIMEOUT: Prevent infinite backup loops
                if not self.context.is_backing_up:
                    self.context.backup_start_time = time.time()
                    self.context.is_backing_up = True
                
                # If backing up for >2 seconds, stop and replan
                elif time.time() - self.context.backup_start_time > 2.0:
                    print(f"[BACKUP] Timeout after 2s - triggering replan")
                    self.context.is_backing_up = False
                    self.context.current_path = None  # Clear old path
                    
                    # TRIGGER REPLANNING to continue navigation
                    if self.context.goal_position is not None:
                        target = self.context.target_object or self.context.target_location
                        if target:
                            asyncio.create_task(self.process_instruction(f"Navigate to {target}"))
                        else:
                            # Direct replan to goal position
                            goal = self.context.goal_position
                            path = self.planning_agent.plan_path(
                                start=self.context.robot_pose,
                                goal=goal
                            )
                            if path.is_valid:
                                self.context.current_path = path
                                self.context.current_waypoint_index = 0
                                self.context.state = RobotState.NAVIGATING
                                print(f"[REPLAN] New path with {len(path.waypoints)} waypoints")
                            else:
                                # Fallback: direct path
                                from semantic_vln_mcp.agents.planning_agent import Waypoint, Path
                                wp = Waypoint(x=goal[0], y=goal[1])
                                self.context.current_path = Path(waypoints=[wp], is_valid=True, total_cost=0)
                                self.context.current_waypoint_index = 0
                                self.context.state = RobotState.NAVIGATING
                                print(f"[REPLAN] Direct path to goal")
                    return 0.0, 0.0  # Brief stop before resuming
                
                print(f"[LIDAR] DANGER! Front obstacle at {min_front:.2f}m - BACKUP!")
                v = -0.25  # Faster backup
                w = 2.5 if min_left > min_right else -2.5  # Faster turn
                return v, w
            else:
                # Clear backup state when obstacles are far
                self.context.is_backing_up = False
            
            # CAUTION: Slow down and steer away (earlier detection)
            if min_front < 1.5:  # Increased from 1.2m for earlier warning
                print(f"[LIDAR] Obstacle at {min_front:.2f}m - steering away")
                v = 0.08  # Slower
                w = 2.0 if min_left > min_right else -2.0  # Faster turn
                return v, w
        
        # ===== WAYPOINT NAVIGATION (only if path is clear) =====
        Kp_angular = 2.0
        Kp_linear = 0.5
        
        w = Kp_angular * angle_error
        w = np.clip(w, -1.5, 1.5)
        
        if abs(angle_error) < 0.5:
            v = Kp_linear * dist
            v = np.clip(v, 0.1, 0.4)
        else:
            v = 0.05
        
        # DEPTH CAMERA AVOIDANCE (works alongside Lidar for better coverage)
        if self.context.latest_depth_frame is not None:
            depth = self.context.latest_depth_frame
            h, w_depth = depth.shape[:2] if len(depth.shape) >= 2 else (1, 1)
            
            # Check center region for obstacles
            center_region = depth[h//3:2*h//3, w_depth//3:2*w_depth//3]
            left_region = depth[h//3:2*h//3, 0:w_depth//3]
            right_region = depth[h//3:2*h//3, 2*w_depth//3:]
            
            if center_region.size > 0:
                min_center = np.min(center_region[center_region > 0]) if np.any(center_region > 0) else 999
                min_left = np.min(left_region[left_region > 0]) if np.any(left_region > 0) else 999
                min_right = np.min(right_region[right_region > 0]) if np.any(right_region > 0) else 999
                
                if min_center < 0.35:  # Slightly larger threshold for safety
                    print(f"[RECOVERY] Obstacle too close ({min_center:.2f}m) - Backing up and turning!")
                    v = -0.15  # Faster backup
                    # Turn away from the side that is more blocked
                    w = 1.5 if min_left > min_right else -1.5
                    return v, w
                
                # Moderate avoidance (turning while moving slowly)
                if min_center < 0.6:
                    v = 0.08
                    w = 0.8 if min_left > min_right else -0.8
                    print(f"[AVOID] Wall at {min_center:.2f}m - BACKUP & turn!")
                elif min_center < 0.4:  # Close obstacle - stop and turn
                    v = 0.0  # STOP completely
                    if min_left > min_right:
                        w = 1.0  # Turn left
                    else:
                        w = -1.0  # Turn right
                    print(f"[AVOID] Obstacle at {min_center:.2f}m - turning!")
                elif min_center < 0.6:  # Approaching - slow turn
                    v = 0.1  # Move forward slowly
                    if min_left > min_right:
                        w = 0.5
                    else:
                        w = -0.5
        
        self.context.linear_velocity = v
        self.context.angular_velocity = w
        
        return v, w
    
    def _follow_target(self) -> tuple:
        """Follow a detected person."""
        # Find person in TSG beliefs
        location, confidence = self.tsg.query_location("person")
        
        if location is None or confidence < 0.3:
            # Lost the person - rotate to search
            return 0.0, 0.5
        
        # Compute direction to person
        dx = location[0] - self.context.robot_pose[0]
        dy = location[1] - self.context.robot_pose[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Desired following distance
        follow_dist = 1.5
        
        if dist < follow_dist:
            # Too close - stop
            return 0.0, 0.0
        
        # Navigate toward person
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - self.context.robot_pose[2]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Simple proportional control
        v = min(0.3, 0.3 * (dist - follow_dist))
        w = 1.5 * angle_error
        
        return v, w
    
    def _search_behavior(self) -> tuple:
        """Search behavior - rotate to look around."""
        return 0.0, 0.3  # Slow rotation
    
    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        return {
            "state": self.context.state.value,
            "robot_pose": self.context.robot_pose.tolist(),
            "goal_position": self.context.goal_position.tolist() if self.context.goal_position is not None else None,
            "current_instruction": self.context.current_instruction,
            "detections_count": len(self.context.detections),
            "beliefs_count": len(self.context.object_beliefs),
            "path_waypoints": len(self.context.current_path.waypoints) if self.context.current_path else 0,
            "current_waypoint": self.context.current_waypoint_index,
            "velocity": {
                "linear": self.context.linear_velocity,
                "angular": self.context.angular_velocity
            }
        }
    
    # MCP Tool Interface
    def mcp_tool_definition(self) -> dict:
        """Export MCP tool definitions for external access."""
        return {
            "name": "semantic_vln_mcp",
            "description": "SemanticVLN-MCP Robot Navigation System",
            "tools": [
                {
                    "name": "navigate",
                    "description": "Navigate using natural language instruction",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "Natural language navigation command"
                            }
                        },
                        "required": ["instruction"]
                    }
                },
                {
                    "name": "get_status",
                    "description": "Get current robot status",
                    "input_schema": {"type": "object", "properties": {}}
                },
                {
                    "name": "stop",
                    "description": "Stop all movement",
                    "input_schema": {"type": "object", "properties": {}}
                },
                {
                    "name": "get_detections",
                    "description": "Get current object detections",
                    "input_schema": {"type": "object", "properties": {}}
                },
                {
                    "name": "get_beliefs",
                    "description": "Get TSG object beliefs (including occluded)",
                    "input_schema": {"type": "object", "properties": {}}
                }
            ]
        }


# Standalone test
if __name__ == "__main__":
    print("Testing MCP Orchestrator...")
    
    async def test():
        orchestrator = MCPOrchestrator(device="cuda")
        
        # Create test data
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
        pose = np.array([0.0, 0.0, 0.0])
        
        # Test update
        v, w = await orchestrator.update(rgb, depth, pose)
        print(f"Velocities: v={v:.2f}, w={w:.2f}")
        
        # Test instruction
        await orchestrator.process_instruction("Navigate to the kitchen")
        
        # Get status
        status = orchestrator.get_status()
        print(f"Status: {status}")
    
    asyncio.run(test())
    print("\nMCP Orchestrator test complete!")
