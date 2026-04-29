"""
Gesture Recognition Agent: MediaPipe Integration
=================================================
Part of SemanticVLN-MCP Framework

INNOVATIVE ADDITION: Gesture-based robot control

Recognizes hand gestures from robot camera:
- POINTING → Navigate to pointed direction
- WAVE → Follow me / Come here
- STOP (palm) → Stop movement
- THUMBS_UP → Confirm action

Author: SemanticVLN-MCP Team
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Run: pip install mediapipe")


class GestureType(Enum):
    """Recognized gesture types."""
    NONE = "none"
    POINTING = "pointing"
    WAVE = "wave"
    STOP = "stop"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    FIST = "fist"
    OPEN_PALM = "open_palm"


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture: GestureType
    confidence: float
    hand_position: Optional[Tuple[float, float]] = None  # Normalized (0-1)
    pointing_direction: Optional[Tuple[float, float]] = None  # For pointing gesture
    timestamp: float = 0.0


@dataclass
class HandLandmarks:
    """Hand landmark positions."""
    wrist: Tuple[float, float, float]
    thumb_tip: Tuple[float, float, float]
    index_tip: Tuple[float, float, float]
    middle_tip: Tuple[float, float, float]
    ring_tip: Tuple[float, float, float]
    pinky_tip: Tuple[float, float, float]
    
    # MCP joints (knuckles)
    index_mcp: Tuple[float, float, float]
    middle_mcp: Tuple[float, float, float]
    ring_mcp: Tuple[float, float, float]
    pinky_mcp: Tuple[float, float, float]


class GestureAgent:
    """
    Gesture recognition agent using MediaPipe.
    
    Features:
    - Real-time hand detection and tracking
    - Gesture classification (pointing, wave, stop, thumbs up)
    - Pointing direction estimation
    - Integration with robot camera
    """
    
    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize gesture recognition agent.
        
        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe
        self.hands = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            print("MediaPipe Hands initialized")
        else:
            print("MediaPipe not available - running in mock mode")
        
        # Gesture history for temporal filtering
        self.gesture_history: List[GestureType] = []
        self.history_size = 15 # Increased from 5 to reduce false positives as requested
        
        # Wave detection state
        self.wave_positions: List[Tuple[float, float]] = []
        self.last_wave_time = 0.0
    
    def recognize_gesture(self, frame: np.ndarray) -> GestureResult:
        """
        Recognize gestures in a camera frame.
        
        Args:
            frame: RGB image from robot camera (H, W, 3)
            
        Returns:
            GestureResult with detected gesture
        """
        timestamp = time.time()
        
        if self.hands is None:
            return GestureResult(
                gesture=GestureType.NONE,
                confidence=0.0,
                timestamp=timestamp
            )
        
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            rgb_frame = frame
        else:
            rgb_frame = frame[:, :, :3]
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return GestureResult(
                gesture=GestureType.NONE,
                confidence=0.0,
                timestamp=timestamp
            )
        
        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = self._extract_landmarks(hand_landmarks)
        
        # Calculate hand center
        hand_center = (
            (landmarks.wrist[0] + landmarks.middle_mcp[0]) / 2,
            (landmarks.wrist[1] + landmarks.middle_mcp[1]) / 2
        )
        
        # Classify gesture
        gesture, confidence = self._classify_gesture(landmarks)
        
        # Get pointing direction if applicable
        pointing_direction = None
        if gesture == GestureType.POINTING:
            pointing_direction = self._get_pointing_direction(landmarks)
        
        # Update gesture history for temporal filtering
        self._update_history(gesture)
        smoothed_gesture = self._get_smoothed_gesture()
        
        return GestureResult(
            gesture=smoothed_gesture,
            confidence=confidence,
            hand_position=hand_center,
            pointing_direction=pointing_direction,
            timestamp=timestamp
        )
    
    def _extract_landmarks(self, hand_landmarks) -> HandLandmarks:
        """Extract key landmarks from MediaPipe result."""
        def get_point(idx):
            lm = hand_landmarks.landmark[idx]
            return (lm.x, lm.y, lm.z)
        
        return HandLandmarks(
            wrist=get_point(0),
            thumb_tip=get_point(4),
            index_tip=get_point(8),
            middle_tip=get_point(12),
            ring_tip=get_point(16),
            pinky_tip=get_point(20),
            index_mcp=get_point(5),
            middle_mcp=get_point(9),
            ring_mcp=get_point(13),
            pinky_mcp=get_point(17)
        )
    
    def _classify_gesture(self, landmarks: HandLandmarks) -> Tuple[GestureType, float]:
        """
        Classify gesture based on hand landmarks.
        
        Uses finger extension and relative positions.
        """
        # Calculate which fingers are extended
        fingers_extended = self._get_extended_fingers(landmarks)
        
        # Count extended fingers
        extended_count = sum(fingers_extended.values())
        
        # POINTING: Only index finger extended
        if (fingers_extended['index'] and 
            not fingers_extended['middle'] and 
            not fingers_extended['ring'] and 
            not fingers_extended['pinky']):
            return GestureType.POINTING, 0.9
        
        # THUMBS UP: Only thumb extended, hand oriented up
        if (fingers_extended['thumb'] and 
            not fingers_extended['index'] and
            not fingers_extended['middle'] and
            not fingers_extended['ring'] and
            not fingers_extended['pinky']):
            # Check thumb is pointing up
            if landmarks.thumb_tip[1] < landmarks.wrist[1]:
                return GestureType.THUMBS_UP, 0.85
            else:
                return GestureType.THUMBS_DOWN, 0.85
        
        # OPEN PALM / STOP: All fingers extended
        if extended_count >= 4:
            return GestureType.STOP, 0.9
        
        # FIST: No fingers extended
        if extended_count == 0:
            return GestureType.FIST, 0.8
        
        # WAVE: Detect oscillating motion (checked externally)
        if self._detect_wave(landmarks):
            return GestureType.WAVE, 0.85
        
        return GestureType.NONE, 0.5
    
    def _get_extended_fingers(self, landmarks: HandLandmarks) -> Dict[str, bool]:
        """Determine which fingers are extended."""
        
        def is_extended(tip, mcp, wrist):
            # Finger is extended if tip is further from wrist than MCP
            tip_dist = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
            mcp_dist = np.sqrt((mcp[0] - wrist[0])**2 + (mcp[1] - wrist[1])**2)
            return tip_dist > mcp_dist * 1.2
        
        wrist = landmarks.wrist
        
        return {
            'thumb': landmarks.thumb_tip[0] < landmarks.index_mcp[0] - 0.05,  # Thumb points outward
            'index': is_extended(landmarks.index_tip, landmarks.index_mcp, wrist),
            'middle': is_extended(landmarks.middle_tip, landmarks.middle_mcp, wrist),
            'ring': is_extended(landmarks.ring_tip, landmarks.ring_mcp, wrist),
            'pinky': is_extended(landmarks.pinky_tip, landmarks.pinky_mcp, wrist),
        }
    
    def _detect_wave(self, landmarks: HandLandmarks) -> bool:
        """Detect waving motion from position history."""
        current_pos = (landmarks.wrist[0], landmarks.wrist[1])
        current_time = time.time()
        
        self.wave_positions.append(current_pos)
        if len(self.wave_positions) > 10:
            self.wave_positions.pop(0)
        
        if len(self.wave_positions) < 5:
            return False
        
        # Check for oscillating X movement
        x_positions = [p[0] for p in self.wave_positions]
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(x_positions) - 1):
            prev_dir = x_positions[i] - x_positions[i-1]
            next_dir = x_positions[i+1] - x_positions[i]
            if prev_dir * next_dir < 0:  # Sign change
                direction_changes += 1
        
        # Wave detected if multiple direction changes in short time
        if direction_changes >= 2 and (current_time - self.last_wave_time) > 0.5:
            self.last_wave_time = current_time
            return True
        
        return False
    
    def _get_pointing_direction(self, landmarks: HandLandmarks) -> Tuple[float, float]:
        """Calculate pointing direction vector."""
        # Direction from wrist through index finger
        dx = landmarks.index_tip[0] - landmarks.wrist[0]
        dy = landmarks.index_tip[1] - landmarks.wrist[1]
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        return (dx, dy)
    
    def _update_history(self, gesture: GestureType):
        """Update gesture history for temporal smoothing."""
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
    
    def _get_smoothed_gesture(self) -> GestureType:
        """Get most common gesture from recent history."""
        if not self.gesture_history:
            return GestureType.NONE
        
        # Count occurrences
        counts = {}
        for g in self.gesture_history:
            counts[g] = counts.get(g, 0) + 1
        
        # Return most common (excluding NONE if others exist)
        max_count = 0
        best_gesture = GestureType.NONE
        for g, count in counts.items():
            if count > max_count and (g != GestureType.NONE or max_count == 0):
                max_count = count
                best_gesture = g
        
        return best_gesture
    
    def gesture_to_command(self, gesture: GestureResult) -> Optional[Dict]:
        """
        Convert gesture to robot command.
        
        Returns:
            Command dictionary or None if no action
        """
        if gesture.gesture == GestureType.NONE:
            return None
        
        commands = {
            GestureType.POINTING: {
                "action": "navigate_direction",
                "direction": gesture.pointing_direction,
                "description": "Navigate in pointed direction"
            },
            GestureType.WAVE: {
                "action": "follow_me",
                "target_position": gesture.hand_position,
                "description": "Follow the person"
            },
            GestureType.STOP: {
                "action": "stop",
                "description": "Stop all movement"
            },
            GestureType.THUMBS_UP: {
                "action": "confirm",
                "description": "Confirm current action"
            },
            GestureType.THUMBS_DOWN: {
                "action": "cancel",
                "description": "Cancel current action"
            },
            GestureType.FIST: {
                "action": "pause",
                "description": "Pause current task"
            }
        }
        
        return commands.get(gesture.gesture)
    
    # MCP Tool Interface
    def mcp_tool_definition(self) -> dict:
        """Export MCP tool definitions."""
        return {
            "name": "gesture_agent",
            "description": "Gesture recognition for human-robot interaction",
            "tools": [
                {
                    "name": "recognize_gesture",
                    "description": "Recognize gestures from camera frame",
                    "input_schema": {"type": "object", "properties": {}}
                },
                {
                    "name": "get_gesture_command",
                    "description": "Convert recognized gesture to robot command",
                    "input_schema": {"type": "object", "properties": {}}
                }
            ]
        }
    
    def close(self):
        """Release resources."""
        if self.hands:
            self.hands.close()


# Standalone test
if __name__ == "__main__":
    print("Testing Gesture Agent...")
    
    agent = GestureAgent()
    
    # Create test image
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test gesture recognition
    result = agent.recognize_gesture(test_frame)
    
    print(f"Detected gesture: {result.gesture.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Hand position: {result.hand_position}")
    print(f"Pointing direction: {result.pointing_direction}")
    
    # Test gesture to command
    command = agent.gesture_to_command(result)
    if command:
        print(f"Command: {command}")
    
    agent.close()
    print("\nGesture Agent test complete!")
