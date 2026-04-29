# SemanticVLN-MCP: Configuration File

import os
from pathlib import Path

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
WEBOTS_WORLD = PROJECT_ROOT / "webots" / "worlds" / "indoor_environment.wbt"
MODELS_DIR = PROJECT_ROOT / "models"

# ==================== YOLO CONFIG ====================
YOLO_CONFIG = {
    "model": "yolov8n.pt",  # Nano model - fast on CPU/GPU
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "device": "cuda" if os.environ.get("USE_GPU", "true").lower() == "true" else "cpu",
    "frame_size": (640, 480),
}

# ==================== DEEPLABV3+ CONFIG ====================
SEGMENTATION_CONFIG = {
    "model": "deeplabv3_resnet50",
    "num_classes": 21,  # COCO classes
    "device": YOLO_CONFIG["device"],
}

# ==================== TSG ALGORITHM CONFIG ====================
TSG_CONFIG = {
    "occlusion_decay_lambda": 0.5,  # Decay rate for occluded objects
    "observation_sigma": 0.1,       # Meters - observation noise
    "semantic_prior_sigma": 0.5,    # Meters - spatial context spread
    "max_occlusion_time_sec": 5.0,  # Max time to track occluded objects
    "motion_sigma": 0.3,            # Meters - motion prediction spread
}

# ==================== PLANNING CONFIG ====================
PLANNING_CONFIG = {
    "grid_resolution": 0.05,       # 5cm per cell
    "robot_radius": 0.2,           # meters
    "max_linear_velocity": 0.3,    # m/s - Reduced from 0.5 for safer navigation
    "max_angular_velocity": 1.5,   # rad/s
    "goal_threshold": 0.15,        # meters - close enough to goal
}

# ==================== OLLAMA CONFIG ====================
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "llama3.1:8b",
    "temperature": 0.7,
    "max_tokens": 512,
}

# ==================== WEBOTS CONFIG ====================
WEBOTS_CONFIG = {
    "timestep": 32,  # ms
    "camera_width": 640,
    "camera_height": 480,
    "camera_fov": 1.047,  # 60 degrees in radians
}

# ==================== GESTURE CONFIG ====================
GESTURE_CONFIG = {
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "gestures": {
        "POINTING": "navigate_direction",
        "WAVE": "follow_me",
        "STOP": "stop_movement",
        "THUMBS_UP": "confirm_action",
    }
}

# ==================== SEMANTIC CLASSES ====================
NAVIGATION_CLASSES = {
    0: "person",
    1: "chair",
    2: "couch",
    3: "dining table",
    4: "tv",
    5: "laptop",
    6: "bottle",
    7: "cup",
    8: "refrigerator",
    9: "microwave",
    10: "oven",
    11: "sink",
    12: "bed",
    13: "toilet",
    14: "door",
    15: "window",
}

# Rooms and their approximate positions (will be refined by SLAM)
ROOM_PRIORS = {
    "kitchen": {"center": (3.0, 2.0), "radius": 2.0},
    "living_room": {"center": (0.0, 0.0), "radius": 3.0},
    "bedroom": {"center": (-3.0, 2.0), "radius": 2.0},
    "bathroom": {"center": (-3.0, -2.0), "radius": 1.5},
}

# Spatial relationships for language grounding
SPATIAL_RELATIONS = {
    "near": 1.0,      # within 1 meter
    "by": 0.8,        # within 0.8 meters
    "next_to": 0.5,   # within 0.5 meters
    "behind": 1.5,    # 1.5m behind
    "in_front_of": 1.0,
    "inside": 0.3,    # within bounds
}
