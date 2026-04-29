"""
Test Suite for SemanticVLN-MCP
===============================

Run with: pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTSGAlgorithm:
    """Tests for Temporal Semantic Grounding algorithm."""
    
    def test_tsg_initialization(self):
        """Test TSG initialization."""
        from semantic_vln_mcp.algorithms.tsg import TemporalSemanticGrounding
        
        tsg = TemporalSemanticGrounding()
        assert tsg.lambda_decay == 0.5
        assert tsg.obs_sigma == 0.1
        assert len(tsg.beliefs) == 0
    
    def test_tsg_observation_update(self):
        """Test TSG updates beliefs on observation."""
        from semantic_vln_mcp.algorithms.tsg import TemporalSemanticGrounding
        
        tsg = TemporalSemanticGrounding()
        
        # Add detection
        detections = [{
            'object_id': 'person_1',
            'class_name': 'person',
            'world_position': (2.0, 1.5, 0.0)
        }]
        
        tsg.update(detections, robot_pose=np.array([0, 0, 0]))
        
        # Check belief was created
        assert 'person_1' in tsg.beliefs
        belief = tsg.beliefs['person_1']
        assert belief.is_visible == True
        assert np.allclose(belief.location, [2.0, 1.5])
    
    def test_tsg_occlusion_decay(self):
        """Test TSG decreases confidence during occlusion."""
        from semantic_vln_mcp.algorithms.tsg import TemporalSemanticGrounding
        
        tsg = TemporalSemanticGrounding()
        
        # Initial detection
        detections = [{
            'object_id': 'person_1',
            'class_name': 'person',
            'world_position': (2.0, 1.5, 0.0)
        }]
        tsg.update(detections, robot_pose=np.array([0, 0, 0]))
        initial_confidence = tsg.beliefs['person_1'].confidence
        
        # Occlusion for 2 seconds
        for _ in range(20):
            tsg.update([], robot_pose=np.array([0, 0, 0]), dt=0.1)
        
        # Confidence should decrease
        assert tsg.beliefs['person_1'].confidence < initial_confidence
        assert tsg.beliefs['person_1'].is_visible == False
    
    def test_tsg_query_location(self):
        """Test querying object location."""
        from semantic_vln_mcp.algorithms.tsg import TemporalSemanticGrounding
        
        tsg = TemporalSemanticGrounding()
        
        # Add detection
        detections = [{
            'object_id': 'chair_1',
            'class_name': 'chair',
            'world_position': (1.0, 2.0, 0.0)
        }]
        tsg.update(detections, robot_pose=np.array([0, 0, 0]))
        
        # Query
        location, confidence = tsg.query_location('chair')
        assert location is not None
        assert confidence > 0.5
        assert np.allclose(location, [1.0, 2.0], atol=0.1)
    
    def test_tsg_spatial_relationship_parsing(self):
        """Test parsing spatial relationships from text."""
        from semantic_vln_mcp.algorithms.tsg import TemporalSemanticGrounding
        
        tsg = TemporalSemanticGrounding()
        
        relationships = tsg._parse_spatial_relationships(
            "Find the person near the window"
        )
        
        assert len(relationships) == 1
        assert relationships[0].subject == "person"
        assert relationships[0].relation_type == "near"
        assert relationships[0].anchor == "window"


class TestPerceptionAgent:
    """Tests for Perception Agent."""
    
    def test_perception_initialization(self):
        """Test perception agent initialization."""
        from semantic_vln_mcp.agents.perception_agent import PerceptionAgent
        
        agent = PerceptionAgent(device="cpu")
        assert agent.confidence_threshold == 0.5
    
    def test_perception_mock_detection(self):
        """Test perception returns detections."""
        from semantic_vln_mcp.agents.perception_agent import PerceptionAgent
        
        agent = PerceptionAgent(device="cpu")
        
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = agent.detect_objects(image)
        
        assert result is not None
        assert hasattr(result, 'detections')
        assert hasattr(result, 'processing_time_ms')


class TestPlanningAgent:
    """Tests for Planning Agent."""
    
    def test_planning_initialization(self):
        """Test planning agent initialization."""
        from semantic_vln_mcp.agents.planning_agent import PlanningAgent
        
        agent = PlanningAgent()
        assert agent.robot_radius == 0.2
        assert agent.goal_threshold == 0.15
    
    def test_planning_simple_path(self):
        """Test A* finds path in simple environment."""
        from semantic_vln_mcp.agents.planning_agent import PlanningAgent
        
        agent = PlanningAgent()
        
        # Create simple grid (no obstacles)
        grid = np.zeros((100, 100))
        agent.set_map(
            occupancy_grid=grid,
            origin=np.array([-5.0, -5.0]),
            resolution=0.1
        )
        
        # Plan path
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 2.0])
        
        path = agent.plan_path(start, goal)
        
        assert path.is_valid == True
        assert len(path.waypoints) > 0
        assert path.total_distance > 0
    
    def test_planning_with_obstacle(self):
        """Test A* avoids obstacles."""
        from semantic_vln_mcp.agents.planning_agent import PlanningAgent
        
        agent = PlanningAgent()
        
        # Create grid with obstacle
        grid = np.zeros((100, 100))
        grid[40:60, 45:55] = 1.0  # Obstacle in middle
        
        agent.set_map(
            occupancy_grid=grid,
            origin=np.array([-5.0, -5.0]),
            resolution=0.1
        )
        
        # Plan path that must go around
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([4.0, 0.0])
        
        path = agent.plan_path(start, goal)
        
        assert path.is_valid == True
        # Path should be longer than straight line due to obstacle
        assert path.total_distance > 4.0


class TestReasoningAgent:
    """Tests for Reasoning Agent."""
    
    def test_reasoning_initialization(self):
        """Test reasoning agent initialization."""
        from semantic_vln_mcp.agents.reasoning_agent import ReasoningAgent
        
        agent = ReasoningAgent()
        assert agent.model == "llama3.1:8b"
    
    def test_reasoning_rule_based_parsing(self):
        """Test rule-based instruction parsing."""
        from semantic_vln_mcp.agents.reasoning_agent import ReasoningAgent
        
        agent = ReasoningAgent()
        
        result = agent.parse_instruction_sync("Navigate to the kitchen")
        
        assert result.parsed_instruction.goal_type in ["navigate", "approach"]
        assert result.parsed_instruction.target_location == "kitchen"
    
    def test_reasoning_object_detection(self):
        """Test parsing object from instruction."""
        from semantic_vln_mcp.agents.reasoning_agent import ReasoningAgent
        
        agent = ReasoningAgent()
        
        result = agent.parse_instruction_sync("Find the person near the window")
        
        assert result.parsed_instruction.goal_type == "find"
        assert result.parsed_instruction.target_object == "person"
    
    def test_reasoning_implicit_goal(self):
        """Test resolving implicit goals."""
        from semantic_vln_mcp.agents.reasoning_agent import ReasoningAgent
        
        agent = ReasoningAgent()
        
        goal = agent.resolve_implicit_goal("I'm hungry", [])
        assert goal == "kitchen"
        
        goal = agent.resolve_implicit_goal("I'm tired", [])
        assert goal == "bedroom"


class TestGestureAgent:
    """Tests for Gesture Agent."""
    
    def test_gesture_initialization(self):
        """Test gesture agent initialization."""
        from semantic_vln_mcp.agents.gesture_agent import GestureAgent
        
        agent = GestureAgent()
        assert agent.min_detection_confidence == 0.7
    
    def test_gesture_command_mapping(self):
        """Test gesture to command mapping."""
        from semantic_vln_mcp.agents.gesture_agent import GestureAgent, GestureResult, GestureType
        
        agent = GestureAgent()
        
        # Test stop gesture
        result = GestureResult(
            gesture=GestureType.STOP,
            confidence=0.9,
            timestamp=0.0
        )
        
        command = agent.gesture_to_command(result)
        assert command['action'] == 'stop'
        
        # Test follow gesture
        result.gesture = GestureType.WAVE
        command = agent.gesture_to_command(result)
        assert command['action'] == 'follow_me'


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
