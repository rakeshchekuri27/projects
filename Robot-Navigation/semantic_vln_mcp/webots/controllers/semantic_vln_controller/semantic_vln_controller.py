"""
Webots Robot Controller for SemanticVLN-MCP
============================================
Part of SemanticVLN-MCP Framework

This controller runs inside Webots and:
- Captures sensor data (camera, depth, GPS, IMU)
- Sends data to MCP orchestrator
- Executes velocity commands on robot motors
- Handles keyboard input for manual control

Compatible with: e-Puck, Pioneer 3-DX, TurtleBot3

Author: SemanticVLN-MCP Team
"""

import sys
import os
import asyncio
import numpy as np
from pathlib import Path

# Add project root to path - use absolute path for Webots compatibility
# This ensures the module can be found regardless of where Webots runs from
PROJECT_ROOT = r"c:\Users\Lenovo\Documents\new_pro"
sys.path.insert(0, PROJECT_ROOT)

# Also try relative path as fallback
try:
    rel_root = Path(__file__).parent.parent.parent.parent
    if str(rel_root) not in sys.path:
        sys.path.insert(0, str(rel_root))
except:
    pass

# Evaluation logging for paper graphs
try:
    from evaluation_logger import EvaluationLogger
    EVAL_LOGGER = EvaluationLogger()
    print("[EVAL] Real-time data logging ENABLED")
except ImportError:
    EVAL_LOGGER = None


# Webots imports
try:
    from controller import Robot, Motor, Camera, RangeFinder, GPS, InertialUnit, Keyboard
    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False
    print("Warning: Webots controller module not available. Running in mock mode.")


class SemanticVLNController:
    """
    Webots controller for SemanticVLN-MCP.
    
    Features:
    - Multi-sensor integration (RGB, depth, pose)
    - MCP orchestrator integration
    - Manual keyboard override
    - Real-time velocity control
    """
    
    def __init__(self):
        """Initialize the controller."""
        print("Initializing SemanticVLN Controller...")
        
        if not WEBOTS_AVAILABLE:
            print("Running in mock mode (no Webots)")
            self.robot = None
            self.timestep = 32
            return
        
        # Initialize robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Get robot name to determine type
        self.robot_name = self.robot.getName()
        print(f"Robot: {self.robot_name}")
        
        # Initialize devices based on robot type
        self._init_motors()
        self._init_sensors()
        self._init_keyboard()
        
        # Initialize MCP orchestrator (lazy import to avoid circular deps)
        self.orchestrator = None
        self._init_orchestrator()
        
        # Initialize visualization (OpenCV window for detections)
        self.visualizer = None
        self._init_visualization()
        
        # Control state
        self.manual_mode = False
        self.running = True
        
        print(f"Controller initialized! Timestep: {self.timestep}ms")
    
    def _init_motors(self):
        """Initialize wheel motors."""
        # Different robot types have different motor names
        motor_configs = {
            "e-puck": ("left wheel motor", "right wheel motor"),
            "pioneer": ("left wheel", "right wheel"),
            "turtlebot": ("left wheel motor", "right wheel motor"),
            "turtlebot3": ("left wheel motor", "right wheel motor"),
            "default": ("left wheel motor", "right wheel motor")
        }
        
        # Detect robot type
        robot_type = "default"
        for rtype in motor_configs:
            if rtype in self.robot_name.lower():
                robot_type = rtype
                break
        
        left_name, right_name = motor_configs[robot_type]
        
        try:
            self.left_motor = self.robot.getDevice(left_name)
            self.right_motor = self.robot.getDevice(right_name)
            
            # Set to velocity control mode
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
            print(f"Motors initialized: {left_name}, {right_name}")
        except Exception as e:
            print(f"Warning: Could not initialize motors: {e}")
            self.left_motor = None
            self.right_motor = None
        
        # Robot physical parameters (vary by robot)
        self.wheel_radius = 0.025  # meters
        self.wheel_distance = 0.10  # meters between wheels
        self.max_wheel_velocity = 6.28  # rad/s
    
    def _init_sensors(self):
        """Initialize sensors."""
        # RGB Camera
        try:
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(self.timestep)
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            print(f"Camera: {self.camera_width}x{self.camera_height}")
        except Exception as e:
            print(f"Warning: No camera: {e}")
            self.camera = None
            self.camera_width = 640
            self.camera_height = 480
        
        # Depth Camera / Range Finder
        try:
            self.depth_camera = self.robot.getDevice("range-finder")
            if self.depth_camera is None:
                self.depth_camera = self.robot.getDevice("kinect range")
            self.depth_camera.enable(self.timestep)
            print("Depth camera initialized")
        except Exception as e:
            print(f"Warning: No depth camera: {e}")
            self.depth_camera = None
        
        # Display for showing YOLO detections
        try:
            from controller import Display
            self.display = self.robot.getDevice("detection_display")
            if self.display:
                print("Detection display initialized (640x480)")
            else:
                self.display = None
        except Exception as e:
            print(f"Warning: No display: {e}")
            self.display = None
        
        # Display for showing SLAM map
        try:
            self.slam_display = self.robot.getDevice("slam_display")
            if self.slam_display:
                print("SLAM map display initialized (200x200)")
            else:
                self.slam_display = None
        except Exception as e:
            self.slam_display = None
        
        # GPS for position
        try:
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.timestep)
            print("GPS initialized")
        except Exception as e:
            print(f"Warning: No GPS: {e}")
            self.gps = None
        
        # IMU for orientation
        try:
            self.imu = self.robot.getDevice("inertial unit")
            self.imu.enable(self.timestep)
            print("IMU initialized")
        except Exception as e:
            print(f"Warning: No IMU: {e}")
            self.imu = None
            
        # Lidar for 360 obstacle avoidance
        try:
            self.lidar = self.robot.getDevice("lidar")
            self.lidar.enable(self.timestep)
            print("Lidar initialized")
        except Exception as e:
            print(f"Warning: No Lidar: {e}")
            self.lidar = None
    
    def _init_keyboard(self):
        """Initialize keyboard for manual control."""
        try:
            self.keyboard = self.robot.getKeyboard()
            self.keyboard.enable(self.timestep)
            print("Keyboard enabled (press 'M' for manual mode)")
        except Exception as e:
            print(f"Warning: No keyboard: {e}")
            self.keyboard = None
    
    def _init_orchestrator(self):
        """Initialize MCP orchestrator."""
        try:
            from semantic_vln_mcp.mcp.orchestrator import MCPOrchestrator
            self.orchestrator = MCPOrchestrator(device="cuda")
            print("MCP Orchestrator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize orchestrator: {e}")
            self.orchestrator = None
    
    def _init_visualization(self):
        """Initialize OpenCV visualization for detections."""
        # Using Webots Display device for detection visualization
        self.visualizer = None
        if self.display:
            print("Detection display enabled - bounding boxes will show in Webots!")
        else:
            print("Detection info will be shown in console output")
    
    def visualize_detections(self, rgb_frame: np.ndarray, detections: list):
        """Draw YOLO detections on the Webots display HUD."""
        if self.display is None or rgb_frame is None:
            return
        
        import cv2
        from controller import Display
        
        hud_w, hud_h = self.camera_width, self.camera_height
        image = rgb_frame.copy()
        
        for det in detections:
            bbox = det.get('bbox', {})
            # Handle both xywh (center+size) and x1y1x2y2 formats
            if 'x' in bbox and 'w' in bbox:
                # xywh format (center, width, height)
                cx, cy = bbox.get('x', 0), bbox.get('y', 0)
                w, h = bbox.get('w', 50), bbox.get('h', 50)
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
            else:
                # x1y1x2y2 format
                x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                x2, y2 = int(bbox.get('x2', 100)), int(bbox.get('y2', 100))
            
            # Clamp to image bounds
            x1 = max(0, min(x1, hud_w-1))
            y1 = max(0, min(y1, hud_h-1))
            x2 = max(0, min(x2, hud_w-1))
            y2 = max(0, min(y2, hud_h-1))
            
            # Draw green bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label with confidence
            label = f"{det.get('class_name', 'obj')} {det.get('confidence', 0):.0%}"
            cv2.putText(image, label, (x1, max(15, y1-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        image_bgra = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        try:
            img_ref = self.display.imageNew(image_bgra.tobytes(), Display.BGRA, hud_w, hud_h)
            self.display.imagePaste(img_ref, 0, 0, False)
            self.display.imageDelete(img_ref)
        except: pass
    
    # HINT: If you don't see bounding boxes, right-click the robot and select 
    # Overlays -> detection_display
    
    def visualize_slam_map(self, robot_pose):
        """Draw a smaller SLAM map with robot orientation."""
        if self.slam_display is None or self.orchestrator is None:
            return
        
        import cv2
        from controller import Display
        
        semantic_map = self.orchestrator.context.semantic_map
        if semantic_map is None or not hasattr(semantic_map, 'grid'):
            return
            
        grid = semantic_map.grid
        icon_size = 150
        grid_small = cv2.resize(grid.astype(np.float32), (icon_size, icon_size))
        
        map_img = np.zeros((icon_size, icon_size, 4), dtype=np.uint8)
        map_img[:, :, 0] = (grid_small * 255).astype(np.uint8) # B
        map_img[:, :, 1] = (grid_small * 200).astype(np.uint8) # G
        map_img[:, :, 2] = (grid_small * 100).astype(np.uint8) # R
        map_img[:, :, 3] = 255 # Alpha
        
        # Draw robot
        rx = int((robot_pose[0] + 5) * (icon_size/10))
        ry = int((robot_pose[1] + 5) * (icon_size/10))
        if 0 < rx < icon_size and 0 < ry < icon_size:
            cv2.circle(map_img, (rx, icon_size - ry), 3, (0,0,255,255), -1)
        
        try:
            img_ref = self.slam_display.imageNew(map_img.tobytes(), Display.BGRA, icon_size, icon_size)
            # Only clear the actual map window size
            self.slam_display.setColor(0x000000)
            self.slam_display.fillRectangle(0, 0, icon_size, icon_size)
            self.slam_display.imagePaste(img_ref, 0, 0, False)
            self.slam_display.imageDelete(img_ref)
        except: pass
    
    def get_lidar_frame(self) -> np.ndarray:
        """Capture 360 lidar scan."""
        if self.lidar is None:
            return None
        
        # Get range image from lidar
        ranges = self.lidar.getRangeImage()
        if ranges is None:
            return None
            
        return np.array(ranges, dtype=np.float32)

    def get_rgb_frame(self) -> np.ndarray:
        """Capture RGB frame from camera."""
        if self.camera is None:
            # Return mock frame
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        # Get image data (BGRA format)
        image_data = self.camera.getImage()
        if image_data is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        # Convert to numpy array
        image = np.frombuffer(image_data, dtype=np.uint8)
        image = image.reshape((self.camera_height, self.camera_width, 4))
        
        # Convert BGRA to RGB
        rgb = image[:, :, [2, 1, 0]]
        return rgb
    
    def get_depth_frame(self) -> np.ndarray:
        """Capture depth frame from depth camera."""
        if self.depth_camera is None:
            # Return mock depth (uniform 3m)
            return np.ones((self.camera_height, self.camera_width), dtype=np.float32) * 3.0
        
        # Get range image
        range_image = self.depth_camera.getRangeImage()
        if range_image is None:
            return np.ones((self.camera_height, self.camera_width), dtype=np.float32) * 3.0
        
        # Convert to numpy array
        depth = np.array(range_image, dtype=np.float32)
        depth = depth.reshape((self.depth_camera.getHeight(), self.depth_camera.getWidth()))
        
        return depth
    
    def get_pose(self) -> np.ndarray:
        """Get robot pose [x, y, theta]."""
        x, y, theta = 0.0, 0.0, 0.0
        
        if self.gps is not None:
            pos = self.gps.getValues()
            x, y = pos[0], pos[1]
        
        if self.imu is not None:
            rpy = self.imu.getRollPitchYaw()
            theta = rpy[2]  # Yaw angle
        
        return np.array([x, y, theta])
    
    def set_velocity(self, linear: float, angular: float):
        """
        Set robot velocity.
        
        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        if self.left_motor is None or self.right_motor is None:
            return
        
        # Convert to differential drive wheel velocities
        v_left = (linear - angular * self.wheel_distance / 2) / self.wheel_radius
        v_right = (linear + angular * self.wheel_distance / 2) / self.wheel_radius
        
        # Saturate
        v_left = np.clip(v_left, -self.max_wheel_velocity, self.max_wheel_velocity)
        v_right = np.clip(v_right, -self.max_wheel_velocity, self.max_wheel_velocity)
        
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)
    
    def handle_keyboard(self) -> tuple:
        """
        Handle keyboard input for manual control.
        
        Returns:
            (linear, angular) velocities, or None if in autonomous mode
        """
        if self.keyboard is None:
            return None
        
        key = self.keyboard.getKey()
        
        # Toggle manual/autonomous mode
        if key == ord('M') or key == ord('m'):
            self.manual_mode = not self.manual_mode
            mode_str = "MANUAL" if self.manual_mode else "AUTONOMOUS"
            print(f"Mode: {mode_str}")
            return (0.0, 0.0) if self.manual_mode else None
        
        if not self.manual_mode:
            return None
        
        # Manual control keys
        linear, angular = 0.0, 0.0
        
        if key == ord('W') or key == ord('w'):  # Forward
            linear = 0.3
        elif key == ord('S') or key == ord('s'):  # Backward
            linear = -0.3
        elif key == ord('A') or key == ord('a'):  # Turn left
            angular = 1.0
        elif key == ord('D') or key == ord('d'):  # Turn right
            angular = -1.0
        elif key == ord(' '):  # Stop
            linear, angular = 0.0, 0.0
        elif key == ord('Q') or key == ord('q'):  # Quit
            self.running = False
        # Navigation commands (number keys)
        elif key == ord('1'):  # Go to kitchen
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.process_instruction("Navigate to the kitchen"))
                print("[CMD] Navigate to kitchen")
        elif key == ord('2'):  # Go to living room
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.process_instruction("Navigate to the living room"))
                print("[CMD] Navigate to living room")
        elif key == ord('3'):  # Go to bedroom
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.process_instruction("Navigate to the bedroom"))
                print("[CMD] Navigate to bedroom")
        elif key == ord('4'):  # Find person
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.process_instruction("Find the person"))
                print("[CMD] Find person")
        elif key == ord('5'):  # Stop navigation
            if self.orchestrator:
                from semantic_vln_mcp.mcp.orchestrator import RobotState
                self.orchestrator.context.state = RobotState.IDLE
                print("[CMD] Stopped navigation")
        
        return (linear, angular)
    
    async def run_async(self):
        """Async main control loop."""
        print("\n=== SemanticVLN-MCP Controller Started ===")
        print("Commands:")
        print("  M - Toggle manual/autonomous mode")
        print("  W/S - Forward/Backward (manual mode)")
        print("  A/D - Turn left/right (manual mode)")
        print("  Space - Stop")
        print("  1 - Navigate to KITCHEN")
        print("  2 - Navigate to LIVING ROOM") 
        print("  3 - Navigate to BEDROOM")
        print("  4 - Find PERSON")
        print("  5 - STOP navigation")
        print("  Q - Quit")
        print("==========================================\n")
        
        iteration = 0
        
        while self.running and (self.robot is None or self.robot.step(self.timestep) != -1):
            iteration += 1
            
            # Handle keyboard
            manual_cmd = self.handle_keyboard()
            
            if manual_cmd is not None:
                # Manual control
                self.set_velocity(manual_cmd[0], manual_cmd[1])
                continue
            
            # Check for external commands from command file
            await self._check_command_file()
            
            # Autonomous mode - use orchestrator
            if self.orchestrator is not None:
                # Get sensor data
                rgb = self.get_rgb_frame()
                depth = self.get_depth_frame()
                pose = self.get_pose()
                lidar_scan = self.get_lidar_frame()
                
                # Run orchestrator update
                try:
                    v, w = await self.orchestrator.update(rgb, depth, pose, lidar_scan)
                    self.set_velocity(v, w)
                    
                    # Update visualization (show bounding boxes on Webots Display)
                    if self.display is not None and iteration % 5 == 0:  # Every 5th frame
                        detections = self.orchestrator.context.detections
                        self.visualize_detections(rgb, detections)
                    
                    # Update SLAM map display
                    if self.slam_display is not None and iteration % 10 == 0:  # Every 10th frame
                        self.visualize_slam_map(pose)
                        
                except Exception as e:
                    print(f"Orchestrator error: {e}")
                    self.set_velocity(0.0, 0.0)
            else:
                # No orchestrator - just stop
                self.set_velocity(0.0, 0.0)
            
            # Status output every 100 iterations (with more details)
            if iteration % 100 == 0:
                pose = self.get_pose()
                if self.orchestrator:
                    status = self.orchestrator.get_status()
                    det_count = status.get('detections_count', 0)
                    vel = status.get('velocity', {})
                    v = vel.get('linear', 0)
                    w = vel.get('angular', 0)
                    state = status.get('state', 'N/A')
                    goal = status.get('goal_position', None)
                    
                    goal_str = f"({goal[0]:.1f}, {goal[1]:.1f})" if goal else "None"
                    print(f"[{iteration}] Pose:({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}) "
                          f"State:{state} Goal:{goal_str} V:{v:.2f} W:{w:.2f} Det:{det_count}")
            
            # LOG DATA every 30 iterations for better graphs
            if iteration % 30 == 0 and EVAL_LOGGER is not None and self.orchestrator:
                pose = self.get_pose()
                slam_map = self.orchestrator.context.semantic_map
                if slam_map is not None and hasattr(slam_map, 'grid'):
                    grid = slam_map.grid
                    occupied = int(np.sum(grid > 0.5))
                    free = int(np.sum(grid < 0.5))
                    EVAL_LOGGER.log_slam(occupied, free, pose)
                
                # Log TSG beliefs
                for belief in self.orchestrator.context.object_beliefs:
                    EVAL_LOGGER.log_tsg(
                        belief.get('class_name', 'unknown'),
                        np.array(belief.get('location', [0,0])),
                        belief.get('confidence', 0),
                        belief.get('is_visible', False),
                        belief.get('occlusion_time', 0)
                    )
                
                # Log Latencies (Benchmarking)
                lats = self.orchestrator.context.latencies
                EVAL_LOGGER.log_latency(
                    perception_ms=lats.get("perception", 0) * 1000,
                    slam_ms=lats.get("slam", 0) * 1000,
                    planning_ms=lats.get("planning", 0) * 1000,
                    tsg_ms=lats.get("tsg", 0) * 1000,
                    gesture_ms=lats.get("gesture", 0) * 1000,
                    total_ms=lats.get("mcp_total", 0) * 1000
                )

                # Show detection details every 200 iterations
                if iteration % 200 == 0 and det_count > 0:
                    print("=== DETECTIONS (YOLOv8) ===")
                    for det in self.orchestrator.context.detections[:5]:  # Show max 5
                        name = det.get('class_name', '?')
                        conf = det.get('confidence', 0) * 100
                        bbox = det.get('bbox', {})
                        world_pos = det.get('world_position', None)
                        pos_str = f"({world_pos[0]:.1f},{world_pos[1]:.1f})" if world_pos else "N/A"
                        print(f"  [{name}] conf:{conf:.0f}% pos:{pos_str}")
                    print("=== TSG BELIEFS ===")
                    for belief in self.orchestrator.context.object_beliefs[:5]:
                        name = belief.get('class_name', '?')
                        loc = belief.get('location', [0,0])
                        conf = belief.get('confidence', 0) * 100
                        visible = "visible" if belief.get('is_visible', False) else "hidden"
                        print(f"  [{name}] loc:({loc[0]:.1f},{loc[1]:.1f}) conf:{conf:.0f}% {visible}")
                    
                    # Show SLAM map info
                    if self.orchestrator.context.semantic_map is not None:
                        slam_map = self.orchestrator.context.semantic_map
                        grid = slam_map.grid if hasattr(slam_map, 'grid') else None
                        if grid is not None:
                            print("=== SLAM MAP ===")
                            occupied = np.sum(grid > 0.5) if hasattr(np, 'sum') else 0
                            free = np.sum(grid < 0.5) if hasattr(np, 'sum') else 0
                            print(f"  Grid: {grid.shape if hasattr(grid, 'shape') else 'N/A'}")
                            print(f"  Occupied cells: {occupied}")
                            print(f"  Free cells: {free}")
                            
                            # LOG REAL DATA for paper graphs
                            if EVAL_LOGGER is not None:
                                # Log SLAM data
                                EVAL_LOGGER.log_slam(int(occupied), int(free), pose)
                                
                                # Log TSG beliefs
                                for belief in self.orchestrator.context.object_beliefs:
                                    EVAL_LOGGER.log_tsg(
                                        belief.get('class_name', 'unknown'),
                                        np.array(belief.get('location', [0,0])),
                                        belief.get('confidence', 0),
                                        belief.get('is_visible', False),
                                        belief.get('occlusion_time', 0)
                                    )
                                
                                # Log detections
                                for det in self.orchestrator.context.detections:
                                    if det.get('world_position'):
                                        EVAL_LOGGER.log_detection(
                                            det.get('class_name', 'unknown'),
                                            det.get('confidence', 0),
                                            np.array(det.get('world_position', [0,0]))
                                        )
                    print("===========================")
                else:
                    print(f"[{iteration}] Pose: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) State: N/A")
    
    async def _check_command_file(self):
        """Check for commands from external command interface."""
        command_file = Path(r"c:\Users\Lenovo\Documents\new_pro\robot_command.txt")
        try:
            if command_file.exists():
                command = command_file.read_text().strip()
                if command and self.orchestrator:
                    print(f"\n[RECEIVED COMMAND] {command}")
                    await self.orchestrator.process_instruction(command)
                # Clear the file after reading
                command_file.write_text("")
        except Exception as e:
            pass  # Ignore file errors
    
    def run(self):
        """Synchronous run wrapper."""
        asyncio.run(self.run_async())
    
    def process_command(self, command: str):
        """Process a natural language command."""
        if self.orchestrator is None:
            print("Orchestrator not available")
            return
        
        asyncio.run(self.orchestrator.process_instruction(command))


def save_and_generate_graphs(orchestrator=None):
    """Save collected data and generate graphs."""
    global EVAL_LOGGER
    if EVAL_LOGGER is not None and len(EVAL_LOGGER.tsg_logs) > 0:
        print("\n" + "="*50)
        print("[EVAL] Saving real-time data...")
        data_path = EVAL_LOGGER.save()
        print(f"[EVAL] Data saved to: {data_path}")
        
        # Save final SLAM map image for paper
        if orchestrator and orchestrator.context.semantic_map is not None:
            EVAL_LOGGER.save_map_image(orchestrator.context.semantic_map.grid)
        
        # Generate graphs
        try:
            from evaluation_logger import generate_graphs
            print("[EVAL] Generating paper graphs from real data...")
            generate_graphs(data_path)
            print("[EVAL] Done! Check evaluation_data folder for graphs.")
        except Exception as e:
            print(f"[EVAL] Could not generate graphs: {e}")
        print("="*50 + "\n")


# Entry point for Webots
if __name__ == "__main__":
    controller = SemanticVLNController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping controller...")
    finally:
        save_and_generate_graphs(controller.orchestrator)
