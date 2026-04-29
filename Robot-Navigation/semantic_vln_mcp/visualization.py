"""
Visualization Module for SemanticVLN-MCP
=========================================
Shows YOLO detections and segmentation results in real-time.

Features:
- Bounding boxes with labels
- Confidence scores
- Semantic segmentation overlay
- Robot status display

Author: SemanticVLN-MCP Team
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


# Colors for different classes (BGR format)
COLORS = {
    'person': (0, 255, 0),      # Green
    'chair': (255, 0, 0),       # Blue
    'table': (0, 165, 255),     # Orange
    'couch': (255, 0, 255),     # Magenta
    'bottle': (0, 255, 255),    # Yellow
    'cup': (255, 255, 0),       # Cyan
    'tv': (128, 0, 128),        # Purple
    'bed': (255, 192, 203),     # Pink
    'potted plant': (0, 128, 0), # Dark Green
    'default': (200, 200, 200), # Gray
}


class DetectionVisualizer:
    """
    Real-time visualization of YOLO detections and segmentation.
    
    Usage:
        visualizer = DetectionVisualizer()
        
        # In main loop:
        visualizer.update(rgb_frame, detections, segmentation, robot_state)
        
        # Will show OpenCV window with bounding boxes
    """
    
    def __init__(self, 
                 window_name: str = "SemanticVLN-MCP Detection",
                 show_segmentation: bool = True,
                 window_size: Tuple[int, int] = (800, 600)):
        """
        Initialize visualizer.
        
        Args:
            window_name: Name of OpenCV window
            show_segmentation: Whether to show segmentation overlay
            window_size: Size of display window
        """
        self.window_name = window_name
        self.show_segmentation = show_segmentation
        self.window_size = window_size
        self.is_initialized = False
        
        # Segmentation colormap (21 classes for DeepLabV3+)
        self.seg_colormap = self._create_colormap()
        
    def _create_colormap(self) -> np.ndarray:
        """Create colormap for segmentation classes."""
        # Pascal VOC colormap
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]       # Background
        colormap[1] = [128, 0, 0]     # Aeroplane
        colormap[2] = [0, 128, 0]     # Bicycle
        colormap[3] = [128, 128, 0]   # Bird
        colormap[4] = [0, 0, 128]     # Boat
        colormap[5] = [128, 0, 128]   # Bottle
        colormap[6] = [0, 128, 128]   # Bus
        colormap[7] = [128, 128, 128] # Car
        colormap[8] = [64, 0, 0]      # Cat
        colormap[9] = [192, 0, 0]     # Chair
        colormap[10] = [64, 128, 0]   # Cow
        colormap[11] = [192, 128, 0]  # Dining table
        colormap[12] = [64, 0, 128]   # Dog
        colormap[13] = [192, 0, 128]  # Horse
        colormap[14] = [64, 128, 128] # Motorbike
        colormap[15] = [192, 128, 128]# Person
        colormap[16] = [0, 64, 0]     # Potted plant
        colormap[17] = [128, 64, 0]   # Sheep
        colormap[18] = [0, 192, 0]    # Sofa
        colormap[19] = [128, 192, 0]  # Train
        colormap[20] = [0, 64, 128]   # TV monitor
        return colormap
    
    def _initialize_window(self):
        """Initialize OpenCV window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        self.is_initialized = True
    
    def update(self,
               rgb_frame: np.ndarray,
               detections: List[Dict],
               segmentation: Optional[np.ndarray] = None,
               robot_state: Optional[Dict] = None) -> bool:
        """
        Update visualization with new frame and detections.
        
        Args:
            rgb_frame: RGB camera image
            detections: List of detections from YOLO
            segmentation: Optional segmentation mask from DeepLabV3+
            robot_state: Optional robot state info
            
        Returns:
            True if window is still open, False if closed
        """
        if not self.is_initialized:
            self._initialize_window()
        
        # Make a copy to draw on
        display = rgb_frame.copy()
        
        # Convert RGB to BGR for OpenCV
        if len(display.shape) == 3 and display.shape[2] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        
        # Draw segmentation overlay if available
        if segmentation is not None and self.show_segmentation:
            display = self._draw_segmentation(display, segmentation)
        
        # Draw bounding boxes
        display = self._draw_detections(display, detections)
        
        # Draw robot state
        if robot_state:
            display = self._draw_status(display, robot_state)
        
        # Draw title bar
        display = self._draw_title(display, len(detections))
        
        # Show
        cv2.imshow(self.window_name, display)
        
        # Check for key press (1ms wait)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            cv2.destroyAllWindows()
            return False
        
        return True
    
    def _draw_detections(self, 
                         image: np.ndarray, 
                         detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        for det in detections:
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0)
            bbox = det.get('bbox', {})
            
            # Get bounding box coordinates
            if isinstance(bbox, dict):
                x1 = int(bbox.get('x1', bbox.get('x', 0)))
                y1 = int(bbox.get('y1', bbox.get('y', 0)))
                x2 = int(bbox.get('x2', x1 + bbox.get('w', 50)))
                y2 = int(bbox.get('y2', y1 + bbox.get('h', 50)))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            else:
                continue
            
            # Get color for this class
            color = COLORS.get(class_name.lower(), COLORS['default'])
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence*100:.0f}%"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(image, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _draw_segmentation(self, 
                           image: np.ndarray, 
                           segmentation: np.ndarray,
                           alpha: float = 0.4) -> np.ndarray:
        """Overlay segmentation mask on image."""
        # Create colored mask
        seg_colored = self.seg_colormap[segmentation.astype(np.uint8)]
        
        # Resize to match image if needed
        if seg_colored.shape[:2] != image.shape[:2]:
            seg_colored = cv2.resize(seg_colored, (image.shape[1], image.shape[0]))
        
        # Blend
        return cv2.addWeighted(image, 1 - alpha, seg_colored, alpha, 0)
    
    def _draw_status(self, image: np.ndarray, state: Dict) -> np.ndarray:
        """Draw robot status overlay."""
        h, w = image.shape[:2]
        
        # Status box
        cv2.rectangle(image, (w - 200, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(image, (w - 200, 10), (w - 10, 100), (255, 255, 255), 1)
        
        # Status text
        robot_state = state.get('state', 'unknown')
        goal = state.get('goal_position', None)
        velocity = state.get('velocity', {})
        
        y = 30
        cv2.putText(image, f"State: {robot_state}", (w - 190, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y += 20
        if goal:
            cv2.putText(image, f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})", (w - 190, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y += 20
        v = velocity.get('linear', 0)
        w_ang = velocity.get('angular', 0)
        cv2.putText(image, f"V:{v:.2f} W:{w_ang:.2f}", (w - 190, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image
    
    def _draw_title(self, image: np.ndarray, det_count: int) -> np.ndarray:
        """Draw title bar."""
        cv2.rectangle(image, (0, 0), (300, 30), (50, 50, 50), -1)
        cv2.putText(image, f"SemanticVLN-MCP | Detections: {det_count}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image
    
    def close(self):
        """Close the visualization window."""
        cv2.destroyAllWindows()
        self.is_initialized = False


def draw_detections_on_frame(frame: np.ndarray, 
                             detections: List[Dict]) -> np.ndarray:
    """
    Simple function to draw detections on a frame.
    
    Args:
        frame: RGB image
        detections: List of detection dicts
        
    Returns:
        Frame with bounding boxes drawn
    """
    visualizer = DetectionVisualizer()
    return visualizer._draw_detections(frame.copy(), detections)


# Test
if __name__ == "__main__":
    print("Testing Detection Visualizer...")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Gray background
    
    # Test detections
    test_detections = [
        {'class_name': 'person', 'confidence': 0.92, 
         'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}},
        {'class_name': 'chair', 'confidence': 0.85,
         'bbox': {'x1': 300, 'y1': 200, 'x2': 400, 'y2': 350}},
        {'class_name': 'bottle', 'confidence': 0.78,
         'bbox': {'x1': 500, 'y1': 150, 'x2': 550, 'y2': 250}},
    ]
    
    visualizer = DetectionVisualizer()
    
    print("Showing test visualization. Press 'q' to exit...")
    
    for i in range(100):  # Show for a few seconds
        # Simulate slight movement
        for det in test_detections:
            det['bbox']['x1'] += np.random.randint(-2, 3)
            det['bbox']['x2'] += np.random.randint(-2, 3)
        
        robot_state = {
            'state': 'navigating',
            'goal_position': [3.0, 0.0],
            'velocity': {'linear': 0.3, 'angular': 0.1}
        }
        
        if not visualizer.update(test_image, test_detections, None, robot_state):
            break
    
    visualizer.close()
    print("Visualization test complete!")
