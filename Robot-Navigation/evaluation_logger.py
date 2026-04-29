"""
Evaluation Logger for SemanticVLN-MCP
=====================================
Logs real-time data from simulation for paper graphs and tables.

Run this alongside Webots to collect:
- TSG confidence decay over time
- Navigation success/failure
- System latency (YOLO, DeepLabV3+, A*, TSG)
- SLAM occupancy progress
- Detection accuracy

After collection, generates graphs using matplotlib.
"""

import json
import time
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import numpy as np

# Data file paths
DATA_DIR = "evaluation_data"
os.makedirs(DATA_DIR, exist_ok=True)

@dataclass
class TSGLogEntry:
    """Log TSG belief state."""
    timestamp: float
    object_class: str
    location: List[float]
    confidence: float
    is_visible: bool
    occlusion_time: float

@dataclass
class NavigationLogEntry:
    """Log navigation command result."""
    timestamp: float
    command: str
    goal_type: str
    target: str
    goal_position: List[float]
    start_position: List[float]
    end_position: List[float]
    path_waypoints: int
    success: bool
    time_taken: float
    distance_traveled: float

@dataclass
class LatencyLogEntry:
    """Log system component latencies."""
    timestamp: float
    perception_ms: float
    slam_ms: float
    planning_ms: float
    tsg_ms: float
    gesture_ms: float
    total_ms: float

@dataclass
class SLAMLogEntry:
    """Log SLAM state."""
    timestamp: float
    occupied_cells: int
    free_cells: int
    robot_position: List[float]

@dataclass
class DetectionLogEntry:
    """Log YOLO detections."""
    timestamp: float
    class_name: str
    confidence: float
    detected_position: List[float]
    ground_truth_position: Optional[List[float]]

class EvaluationLogger:
    """Collects and saves evaluation data during simulation."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tsg_logs: List[TSGLogEntry] = []
        self.nav_logs: List[NavigationLogEntry] = []
        self.latency_logs: List[LatencyLogEntry] = []
        self.slam_logs: List[SLAMLogEntry] = []
        self.detection_logs: List[DetectionLogEntry] = []
        self.start_time = time.time()
        
    def log_tsg(self, object_class: str, location: np.ndarray, 
                confidence: float, is_visible: bool, occlusion_time: float):
        """Log TSG belief state."""
        entry = TSGLogEntry(
            timestamp=time.time() - self.start_time,
            object_class=object_class,
            location=location.tolist() if isinstance(location, np.ndarray) else location,
            confidence=confidence,
            is_visible=is_visible,
            occlusion_time=occlusion_time
        )
        self.tsg_logs.append(entry)
        
    def log_navigation(self, command: str, goal_type: str, target: str,
                       goal_pos: np.ndarray, start_pos: np.ndarray, 
                       end_pos: np.ndarray, waypoints: int, success: bool,
                       time_taken: float, distance: float):
        """Log navigation result."""
        entry = NavigationLogEntry(
            timestamp=time.time() - self.start_time,
            command=command,
            goal_type=goal_type,
            target=target,
            goal_position=goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else goal_pos,
            start_position=start_pos.tolist() if isinstance(start_pos, np.ndarray) else start_pos,
            end_position=end_pos.tolist() if isinstance(end_pos, np.ndarray) else end_pos,
            path_waypoints=waypoints,
            success=success,
            time_taken=time_taken,
            distance_traveled=distance
        )
        self.nav_logs.append(entry)
        
    def log_latency(self, perception_ms: float, slam_ms: float, 
                    planning_ms: float, tsg_ms: float, gesture_ms: float, total_ms: float):
        """Log system latencies (input in milliseconds)."""
        entry = LatencyLogEntry(
            timestamp=time.time() - self.start_time,
            perception_ms=perception_ms,
            slam_ms=slam_ms,
            planning_ms=planning_ms,
            tsg_ms=tsg_ms,
            gesture_ms=gesture_ms,
            total_ms=total_ms
        )
        self.latency_logs.append(entry)
        
    def log_slam(self, occupied: int, free: int, robot_pos: np.ndarray):
        """Log SLAM state."""
        entry = SLAMLogEntry(
            timestamp=time.time() - self.start_time,
            occupied_cells=occupied,
            free_cells=free,
            robot_position=robot_pos.tolist() if isinstance(robot_pos, np.ndarray) else robot_pos
        )
        self.slam_logs.append(entry)
        
    def log_detection(self, class_name: str, confidence: float,
                      detected_pos: np.ndarray, ground_truth: Optional[np.ndarray] = None):
        """Log YOLO detection."""
        entry = DetectionLogEntry(
            timestamp=time.time() - self.start_time,
            class_name=class_name,
            confidence=confidence,
            detected_position=detected_pos.tolist() if isinstance(detected_pos, np.ndarray) else detected_pos,
            ground_truth_position=ground_truth.tolist() if ground_truth is not None else None
        )
        self.detection_logs.append(entry)

    def save_map_image(self, grid: np.ndarray, filename: str = "slam_map.png"):
        """Save occupancy grid as a high-quality image for the paper."""
        try:
            import cv2
            base_path = os.path.join(DATA_DIR, self.session_id)
            os.makedirs(base_path, exist_ok=True)
            
            # Convert grid (0-1) to image (0-255)
            # 1.0 (occupied) -> 0 (black), 0.0 (free) -> 255 (white)
            img = ((1.0 - grid) * 255).astype(np.uint8)
            
            # Flip vertically because grid [0,0] is often bottom-left in world but top-left in img
            img = cv2.flip(img, 0)
            
            file_path = os.path.join(base_path, filename)
            cv2.imwrite(file_path, img)
            print(f"[EVAL] Saved SLAM map image to {file_path}")
            return file_path
        except Exception as e:
            print(f"[WARN] Could not save map image: {e}")
            return None
        
    def save(self):
        """Save all logs to JSON files."""
        base_path = os.path.join(DATA_DIR, self.session_id)
        os.makedirs(base_path, exist_ok=True)
        
        with open(os.path.join(base_path, "tsg_logs.json"), 'w') as f:
            json.dump([asdict(e) for e in self.tsg_logs], f, indent=2)
            
        with open(os.path.join(base_path, "nav_logs.json"), 'w') as f:
            json.dump([asdict(e) for e in self.nav_logs], f, indent=2)
            
        with open(os.path.join(base_path, "latency_logs.json"), 'w') as f:
            json.dump([asdict(e) for e in self.latency_logs], f, indent=2)
            
        with open(os.path.join(base_path, "slam_logs.json"), 'w') as f:
            json.dump([asdict(e) for e in self.slam_logs], f, indent=2)
            
        with open(os.path.join(base_path, "detection_logs.json"), 'w') as f:
            json.dump([asdict(e) for e in self.detection_logs], f, indent=2)
            
        print(f"[EVAL] Saved data to {base_path}/")
        return base_path


def generate_graphs(data_path: str):
    """Generate graphs from collected data."""
    import matplotlib.pyplot as plt
    
    output_dir = os.path.join(data_path, "graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================
    # Graph 1: TSG Confidence Decay
    # =====================
    try:
        with open(os.path.join(data_path, "tsg_logs.json"), 'r') as f:
            tsg_data = json.load(f)
        
        if tsg_data:
            # Get person beliefs
            person_data = [d for d in tsg_data if d['object_class'] == 'person']
            
            if person_data:
                times = [d['timestamp'] for d in person_data]
                confidences = [d['confidence'] * 100 for d in person_data]
                visible = [d['is_visible'] for d in person_data]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Color by visibility
                colors = ['green' if v else 'red' for v in visible]
                ax.scatter(times, confidences, c=colors, alpha=0.6, s=30)
                ax.plot(times, confidences, 'b-', alpha=0.3)
                
                ax.set_xlabel('Time (seconds)', fontsize=12)
                ax.set_ylabel('TSG Confidence (%)', fontsize=12)
                ax.set_title('TSG Confidence Decay During Occlusion', fontsize=14)
                ax.set_ylim(0, 105)
                ax.grid(True, alpha=0.3)
                
                # Legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Visible', markersize=10),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Hidden (decaying)', markersize=10)
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "tsg_confidence_decay.png"), dpi=150)
                plt.close()
                print("[GRAPH] Created: tsg_confidence_decay.png")
    except Exception as e:
        print(f"[WARN] Could not create TSG graph: {e}")
    
    # =====================
    # Graph 2: System Latency
    # =====================
    try:
        with open(os.path.join(data_path, "latency_logs.json"), 'r') as f:
            latency_data = json.load(f)
        
        if latency_data:
            components = ['Perception', 'SLAM', 'Planning', 'TSG', 'Gesture']
            avg_latencies = [
                np.mean([d['perception_ms'] for d in latency_data]),
                np.mean([d['slam_ms'] for d in latency_data]),
                np.mean([d['planning_ms'] for d in latency_data]),
                np.mean([d['tsg_ms'] for d in latency_data]),
                np.mean([d['gesture_ms'] for d in latency_data])
            ]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(components, avg_latencies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD'])
            
            ax.set_xlabel('MCP Component', fontsize=12)
            ax.set_ylabel('Average Latency (ms)', fontsize=12)
            ax.set_title('MCP Orchestrator Component Latencies (Benchmarking)', fontsize=14)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, avg_latencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "system_latency.png"), dpi=150)
            plt.close()
            print("[GRAPH] Created: system_latency.png")
    except Exception as e:
        print(f"[WARN] Could not create latency graph: {e}")
    
    # =====================
    # Graph 3: SLAM Occupancy Over Time
    # =====================
    try:
        with open(os.path.join(data_path, "slam_logs.json"), 'r') as f:
            slam_data = json.load(f)
        
        if slam_data:
            times = [d['timestamp'] for d in slam_data]
            occupied = [d['occupied_cells'] for d in slam_data]
            free = [d['free_cells'] for d in slam_data]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(times, 0, occupied, alpha=0.7, label='Occupied Cells', color='#E74C3C')
            ax.fill_between(times, occupied, [o+f for o,f in zip(occupied, free)], 
                           alpha=0.7, label='Free Cells', color='#2ECC71')
            
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Grid Cells', fontsize=12)
            ax.set_title('SLAM Map Building Progress', fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "slam_occupancy.png"), dpi=150)
            plt.close()
            print("[GRAPH] Created: slam_occupancy.png")
    except Exception as e:
        print(f"[WARN] Could not create SLAM graph: {e}")
    
    # =====================
    # Graph 4: Robot Trajectory
    # =====================
    try:
        with open(os.path.join(data_path, "slam_logs.json"), 'r') as f:
            slam_data = json.load(f)
        
        if slam_data:
            x_coords = [d['robot_position'][0] for d in slam_data]
            y_coords = [d['robot_position'][1] for d in slam_data]
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot trajectory
            ax.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=1)
            ax.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='viridis', s=10)
            
            # Mark start and end
            ax.scatter([x_coords[0]], [y_coords[0]], c='green', s=200, marker='^', 
                      label='Start', zorder=5)
            ax.scatter([x_coords[-1]], [y_coords[-1]], c='red', s=200, marker='s', 
                      label='End', zorder=5)
            
            # Room labels - CORRECTED positions to match actual world coordinates
            ax.text(2.5, 2.5, 'Kitchen', fontsize=10, ha='center', fontweight='bold')
            ax.text(-2.5, 2.5, 'Bedroom', fontsize=10, ha='center', fontweight='bold')
            ax.text(-2.5, -2.5, 'Living Room', fontsize=10, ha='center', fontweight='bold')
            ax.text(2.0, -2.0, 'Bathroom', fontsize=10, ha='center', fontweight='bold')
            
            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.set_title('Robot Trajectory', fontsize=14)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "robot_trajectory.png"), dpi=150)
            plt.close()
            print("[GRAPH] Created: robot_trajectory.png")
    except Exception as e:
        print(f"[WARN] Could not create trajectory graph: {e}")
    
    # =====================
    # Table 1: Navigation Summary
    # =====================
    try:
        with open(os.path.join(data_path, "nav_logs.json"), 'r') as f:
            nav_data = json.load(f)
        
        if nav_data:
            success_count = sum(1 for d in nav_data if d['success'])
            total_count = len(nav_data)
            success_rate = success_count / total_count * 100 if total_count > 0 else 0
            avg_time = np.mean([d['time_taken'] for d in nav_data])
            avg_distance = np.mean([d['distance_traveled'] for d in nav_data])
            avg_waypoints = np.mean([d['path_waypoints'] for d in nav_data])
            
            summary = {
                'Total Commands': total_count,
                'Successful': success_count,
                'Failed': total_count - success_count,
                'Success Rate': f"{success_rate:.1f}%",
                'Avg Time (s)': f"{avg_time:.2f}",
                'Avg Distance (m)': f"{avg_distance:.2f}",
                'Avg Waypoints': f"{avg_waypoints:.1f}"
            }
            
            with open(os.path.join(output_dir, "navigation_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            print("[TABLE] Created: navigation_summary.json")
    except Exception as e:
        print(f"[WARN] Could not create navigation summary: {e}")
    
    print(f"\n[EVAL] All graphs saved to: {output_dir}/")
    return output_dir


# Global logger instance
_logger = None

def get_logger() -> EvaluationLogger:
    """Get or create the global evaluation logger."""
    global _logger
    if _logger is None:
        _logger = EvaluationLogger()
    return _logger


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Generate graphs from existing data
        data_path = sys.argv[1]
        if os.path.exists(data_path):
            generate_graphs(data_path)
        else:
            print(f"Data path not found: {data_path}")
    else:
        print("Usage:")
        print("  python evaluation_logger.py <data_path>  # Generate graphs from data")
        print("")
        print("During simulation, the logger collects data automatically.")
        print("Data is saved to: evaluation_data/<session_id>/")
