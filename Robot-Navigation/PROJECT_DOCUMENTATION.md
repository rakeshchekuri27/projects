# SemanticVLN-MCP: Semantic Vision-Language Navigation with Model Context Protocol

## Executive Summary

A novel robotic navigation system that combines **Temporal Semantic Grounding (TSG)** with a **Model Context Protocol (MCP)** architecture to enable natural language-based robot navigation in indoor environments.

---

## 1. Problem Statement

### Existing System Limitations

| Problem | Traditional VLN Systems |
|---------|------------------------|
| **Object Tracking** | Lose track when objects are occluded or out of view |
| **Command Understanding** | Require exact, structured commands; fail with natural language |
| **Multi-Component Coordination** | Monolithic architectures; hard to extend or modify |
| **Navigation Robustness** | Fail when encountering unexpected obstacles |

### Our Solution

| Problem | Our Approach |
|---------|-------------|
| Object Tracking | **TSG Algorithm** - Tracks objects through occlusion with temporal decay |
| Command Understanding | **LLM + Rule-Based Hybrid** - Llama 3.1 8B for natural language parsing |
| Multi-Component Coordination | **MCP Architecture** - Modular agent orchestration |
| Navigation Robustness | **Proportional Control + Reactive Control** - Obstacle avoidance with depth sensing |

---

## 2. Novel Contributions

### 2.1 Temporal Semantic Grounding (TSG)

**File:** `semantic_vln_mcp/algorithms/tsg.py`

**Innovation:** Maintains probabilistic beliefs about object locations even when they're not visible.

**Mathematical Model:**
```
P(location | t) = decay_factor × P_previous + (1 - decay_factor) × P_motion

Where:
  - decay_factor = exp(-λ × occlusion_time)
  - λ = 0.5 (tunable decay rate)
  - P_motion = Gaussian around predicted location based on velocity
```

**Key Features:**
- Tracks objects through temporary occlusion
- Velocity-based motion prediction
- Confidence decay over time
- Spatial relationship reasoning

### 2.2 Model Context Protocol (MCP) Architecture

**File:** `semantic_vln_mcp/mcp/orchestrator.py`

**Innovation:** Unified protocol for coordinating multiple AI agents.

```
┌─────────────────────────────────────────────┐
│            MCP Orchestrator                 │
│  (Centralized Context & Coordination)       │
└────────────────────┬────────────────────────┘
         ┌───────┬───┴───┬───────┬───────┐
         │       │       │       │       │
    ┌────┴──┐ ┌──┴───┐ ┌─┴────┐ ┌┴────┐ ┌┴─────┐
    │Percep.│ │Reason│ │ Plan │ │SLAM │ │Gesture│
    │Agent  │ │Agent │ │Agent │ │Agent│ │Agent  │
    └───────┘ └──────┘ └──────┘ └─────┘ └───────┘
```

**Note:** This is a custom MCP implementation, NOT FastMCP. It's designed specifically for multi-agent robot control with shared context.

---

## 3. Technologies & Approaches

### 3.1 Robot Simulation: Webots R2023b

- **Robot Model:** TurtleBot3 Burger
- **Sensors:** RGB Camera (640×480), Depth Camera, GPS, IMU
- **Environment:** Indoor with rooms (kitchen, bedroom, living room, bathroom)

### 3.2 Object Detection: YOLOv8

- **Model:** yolov8n.pt (Nano variant for speed)
- **Device:** CUDA (GPU accelerated)
- **Output:** Bounding boxes, class names, confidence scores, world coordinates

### 3.3 Scene Understanding: DeepLabV3+

- **Model:** ResNet50 backbone
- **Purpose:** Semantic segmentation (floor, walls, furniture)
- **Device:** CUDA

### 3.4 Language Understanding: Ollama + Llama 3.1 8B

- **Purpose:** Natural language command parsing
- **Approach:** Hybrid LLM + rule-based fallback
- **Handles:** Typos, synonyms, implicit intent ("I'm hungry" → kitchen)

### 3.5 SLAM (Simultaneous Localization and Mapping)

**File:** `semantic_vln_mcp/agents/slam_agent.py`

- **Type:** Semantic SLAM with DeepLabV3+
- **Grid:** 200×200 occupancy grid (0.05m resolution)
- **Pre-loaded Obstacles:** Arena walls + furniture positions
- **Updates:** Continuously from depth sensor
- **Visualization:** SLAM display in Webots (with robot position)

### 3.6 Path Planning: A* + Proportional Control

**File:** `semantic_vln_mcp/agents/planning_agent.py`

- **Global Planning:** A* algorithm on pre-loaded occupancy grid
- **Local Control:** Proportional Controller (turn toward goal, then move)
- **Obstacle Avoidance:** Reactive control at 40cm threshold
- **Fallback:** Direct navigation through center when A* fails

### 3.7 Gesture Recognition: MediaPipe Hands

**File:** `semantic_vln_mcp/agents/gesture_agent.py`

- **Gestures:** STOP, POINT, WAVE, COME_HERE
- **Framework:** MediaPipe Hands
- **Use:** Alternative robot control method

---

## 4. System Architecture

```
User Command ("go to kitchen")
        │
        ▼
┌───────────────────────────────────────────────┐
│         MCP Orchestrator                       │
│  ┌─────────────────────────────────────────┐  │
│  │              Context                    │  │
│  │  - Robot Pose                           │  │
│  │  - Detections (from YOLO)               │  │
│  │  - TSG Beliefs                          │  │
│  │  - Semantic Map (from SLAM)             │  │
│  │  - Current Goal                         │  │
│  └─────────────────────────────────────────┘  │
│                      │                         │
│    ┌─────────┬───────┼───────┬─────────┐      │
│    ▼         ▼       ▼       ▼         ▼      │
│ Perception Reasoning Planning  SLAM  Gesture  │
│  (YOLO)     (LLM)    (A*)   (Depth)  (Hands)  │
└───────────────────────────────────────────────┘
        │
        ▼
   Robot Control (velocity commands)
```

---

## 5. Demo Commands

### Starting the System

**Terminal 1: Start Ollama (LLM Server)**
```powershell
ollama serve
```

**Terminal 2: Start Command Interface**
```powershell
cd c:\Users\Lenovo\Documents\new_pro
.\venv\Scripts\activate
python command_interface.py
```

**Webots: Open World**
1. Open `semantic_vln_mcp/webots/worlds/indoor_environment.wbt`
2. Press **Ctrl+Shift+R** to reload

### Navigation Commands

| Command | Action | Coordinates |
|---------|--------|-------------|
| `go to kitchen` | Navigate to kitchen area | (2, 1) |
| `go to bedroom` | Navigate to bedroom area | (0, 3) |
| `go to living room` | Navigate to living room | (-2, 0.5) |
| `i am hungry` | LLM interprets → kitchen | (2, 1) |
| `find the person` | Use TSG to locate person | Dynamic |
| `stop` | Stop robot immediately | - |

### Verified Features (Working)

| Feature | Status | Console Output |
|---------|--------|----------------|
| Path Planning | ✅ | `Path found with X waypoints` |
| Obstacle Avoidance | ✅ | `[AVOID] Obstacle at 0.16m - stopping!` |
| TSG Tracking | ✅ | `[person] loc:(1.9,1.1) conf:98% visible` |
| SLAM Map | ✅ | `Occupied cells: 4305` |
| LLM Parsing | ✅ | `[LLM] Parsed successfully` |

### TSG Demo Script

**Step 1:** Robot sees person
```
Command> find the person
Console: [person] loc:(1.5,1.2) conf:98% visible
```

**Step 2:** Robot turns away (press M for manual, A to turn)
```
Console: [person] loc:(1.5,1.2) conf:85% hidden  ← TSG still tracking!
```

**Step 3:** Command to find again
```
Command> find the person
Result: Robot uses TSG location → navigates to last known position
```

---

## 6. Key Files

| File | Purpose |
|------|---------|
| `semantic_vln_mcp/mcp/orchestrator.py` | MCP core - coordinates all agents |
| `semantic_vln_mcp/algorithms/tsg.py` | TSG algorithm implementation |
| `semantic_vln_mcp/agents/perception_agent.py` | YOLOv8 detection + world coordinates |
| `semantic_vln_mcp/agents/reasoning_agent.py` | LLM command parsing |
| `semantic_vln_mcp/agents/planning_agent.py` | A* + DWA navigation |
| `semantic_vln_mcp/agents/slam_agent.py` | Semantic SLAM mapping |
| `semantic_vln_mcp/agents/gesture_agent.py` | MediaPipe gesture recognition |
| `semantic_vln_mcp/webots/controllers/semantic_vln_controller/` | Webots robot controller |
| `command_interface.py` | Terminal command interface |

---

## 7. Evaluation (To Be Collected)

Metrics should be collected from real Webots simulation runs:

| Metric | How to Collect |
|--------|----------------|
| Navigation Success Rate | Run 10 navigation commands, count successes |
| TSG Tracking Accuracy | Detect object → turn away → track → verify position |
| Command Understanding | Test various natural language commands |
| Path Completion Time | Measure time from command to goal reached |

**Run `python evaluate.py` after testing with Webots for real metrics.**

---

## 8. Dependencies

```
# Core
python >= 3.8
webots == R2023b
torch >= 2.0
ultralytics >= 8.0  # YOLO
torchvision >= 0.15  # DeepLabV3+
mediapipe >= 0.10  # Gesture recognition
ollama  # LLM serving
httpx  # HTTP client for Ollama
numpy
opencv-python
```

---

## 9. Final Status - All Features Working ✅

### Verified Working Features:

| Feature | Status | Console Proof |
|---------|--------|---------------|
| **Navigation** | ✅ | `Path found with 19 waypoints` |
| **TSG Tracking** | ✅ | `[person] loc:(-4.5,-3.0) conf:98% visible` |
| **TSG Decay** | ✅ | `conf:69% hidden` (remembered after occluded) |
| **YOLOv8** | ✅ | `[person] conf:86%` on CUDA |
| **DeepLabV3+** | ✅ | `Occupied cells: 2461` |
| **LLM Parsing** | ✅ | `iam hungry` → kitchen |
| **Obstacle Avoidance** | ✅ | `turning left!` / `turning right!` |
| **A* Path Planning** | ✅ | Multiple waypoints with force-clear |

### Room Navigation:
- Kitchen: (2.5, 0) ✅
- Living Room: (-2, -1) ✅
- Bedroom: (0, 3.5) ✅
- Bathroom: (0, -3) ✅

### TSG Innovation Demonstrated:
```
1. Robot sees person → conf:98% visible
2. Robot turns away → conf:69% hidden (REMEMBERED!)
3. "Find the person" → Returns hidden location
4. Robot navigates to remembered position!
```

### Key Files:
| File | Purpose |
|------|---------|
| `tsg.py` | Temporal Semantic Grounding algorithm |
| `orchestrator.py` | MCP coordination + obstacle avoidance |
| `slam_agent.py` | DeepLabV3+ semantic mapping |
| `planning_agent.py` | A* path planning |
| `perception_agent.py` | YOLOv8 detection |

---

## 10. How to Demo

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Command Interface
cd c:\Users\Lenovo\Documents\new_pro
.\venv\Scripts\activate
python command_interface.py

# Webots: Open and reset
# File → Open → semantic_vln_mcp\webots\worlds\indoor_environment.wbt
# Ctrl+Shift+R to reset

# Commands:
go to kitchen          # Navigate to kitchen
find the person        # TSG locates person
iam hungry             # LLM interprets → kitchen
go to bedroom          # Navigate through doorway
```

---

## 11. Conclusion

This project demonstrates a **novel approach** to robot navigation that:

1. **Solves occlusion problem** through Temporal Semantic Grounding (TSG) with `exp(-λt)` decay
2. **Enables natural language control** via hybrid LLM + rule-based parsing
3. **Provides modular architecture** through custom MCP implementation
4. **Ensures robust navigation** with depth-based obstacle avoidance + A* planning
5. **Uses semantic perception** combining YOLOv8 detection + DeepLabV3+ mapping

**Project Status: COMPLETE AND WORKING** ✅
