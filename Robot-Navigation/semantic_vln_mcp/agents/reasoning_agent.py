"""
Reasoning Agent: Ollama/Llama 3.1 Integration
==============================================
Part of SemanticVLN-MCP Framework

Uses local LLM (Ollama) for:
- Natural language instruction parsing
- Goal decomposition into sub-tasks
- Spatial relationship extraction
- Ambiguity resolution

Author: SemanticVLN-MCP Team
"""

import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import httpx


@dataclass
class ParsedInstruction:
    """Result of parsing a natural language instruction."""
    original_text: str
    goal_type: str  # "navigate", "find", "approach", "follow"
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    spatial_relationships: List[Dict] = field(default_factory=list)
    sub_tasks: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ReasoningResult:
    """Complete reasoning output."""
    parsed_instruction: ParsedInstruction
    goal_location: Optional[tuple] = None
    reasoning_steps: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class ReasoningAgent:
    """
    Reasoning agent using local Ollama LLM.
    
    Features:
    - Natural language instruction parsing
    - Goal decomposition
    - Spatial relationship extraction
    - Integration with Llama 3.1 8B
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.1:8b",
                 temperature: float = 0.7,
                 max_tokens: int = 512):
        """
        Initialize reasoning agent.
        
        Args:
            base_url: Ollama API base URL
            model: Model to use (e.g., "llama3.1:8b")
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Semantic knowledge for grounding - expanded with more synonyms
        self.room_keywords = {
            "kitchen": ["kitchen", "cook", "cooking", "food", "refrigerator", "fridge", "microwave", 
                       "hungry", "eat", "eating", "meal", "dinner", "lunch", "breakfast", "kichen", "kitchn"],
            "living_room": ["living room", "livingroom", "living", "lounge", "sofa", "couch", "tv", 
                           "television", "relax", "relaxing", "sitting room", "family room"],
            "bedroom": ["bedroom", "bed", "sleep", "sleeping", "rest", "resting", "tired", "nap", 
                       "bedrom", "bedrm"],
            "bathroom": ["bathroom", "toilet", "shower", "bath", "wash", "restroom", "washroom", 
                        "wc", "bathrm", "bathrom"],
        }
        
        self.object_keywords = {
            "person": ["person", "human", "someone", "man", "woman", "people", "guy", "girl", 
                      "pedestrian", "persn", "prson", "eprson", "peron", "humna"],
            "chair": ["chair", "seat", "sit", "sitting", "seating", "chiar", "cair"],
            "table": ["table", "desk", "surface", "tabl", "tabel"],
            "couch": ["couch", "sofa", "settee", "coch", "soffa"],
            "bottle": ["bottle", "drink", "water", "beverage", "botle", "bottel"],
            "cup": ["cup", "mug", "coffee", "tea", "glass", "cip", "cupp"],
            "window": ["window", "windw", "windo"],
            "door": ["door", "entrance", "exit", "doorway", "dor", "dorr"],
        }
        
        # Action verbs for better command understanding
        self.action_keywords = {
            "navigate": ["navigate", "go", "move", "walk", "travel", "head", "proceed", 
                        "naviage", "naviagte", "navigat", "goo", "goto"],
            "find": ["find", "locate", "search", "look", "seek", "discover", "fnd", "fined", 
                    "serach", "seach"],
            "follow": ["follow", "track", "chase", "pursue", "folw", "folow"],
            "approach": ["approach", "come", "get close", "reach", "aproach", "approch"],
            "stop": ["stop", "halt", "pause", "freeze", "stp", "stopp"],
            "explore": ["explore", "scan", "survey", "check", "look around", "exlpore", "explor"],
        }
        
        # Room layout for LLM understanding
        self.ROOM_LAYOUT = """
Room Layout (12m x 12m arena):
- Kitchen: Top-Right quadrant (2.5, 2.5) - contains refrigerator, table, food items
- Living Room: Bottom-Left quadrant (-2.5, -2.5) - contains sofa, TV, relaxation area  
- Bedroom: Top-Left quadrant (-2.5, 2.5) - contains bed, sleeping area
- Bathroom: Bottom-Right quadrant (2.5, -2.5) - contains toilet, shower, soap
- Center: Open navigation area (0, 0)
"""
    
    def _fuzzy_match(self, word: str, target: str, threshold: float = 0.7) -> bool:
        """Simple fuzzy matching for handling typos."""
        if not word or not target:
            return False
        word = word.lower()
        target = target.lower()
        if word == target:
            return True
        if word in target or target in word:
            return True
        # Simple edit distance ratio
        if len(word) < 3 or len(target) < 3:
            return False
        common = sum(1 for a, b in zip(word, target) if a == b)
        ratio = common / max(len(word), len(target))
        return ratio >= threshold
    
    async def parse_instruction(self, 
                                instruction: str,
                                context: Optional[Dict] = None) -> ReasoningResult:
        """
        Parse a natural language instruction.
        
        Uses Ollama LLM as primary parser for semantic understanding,
        with rule-based fallback for speed/reliability.
        
        Args:
            instruction: Natural language command
            context: Optional context (detected objects, current location)
            
        Returns:
            ReasoningResult with parsed instruction and goal
        """
        start_time = time.time()
        
        # Try LLM parsing first (better semantic understanding)
        llm_result = {}
        try:
            llm_result = await self._llm_parsing(instruction, context)
            print(f"[LLM] Parsed successfully")
        except Exception as e:
            print(f"[LLM] Ollama not available: {e}, using rule-based parsing")
        
        # Rule-based parsing (fast fallback)
        parsed = self._rule_based_parsing(instruction)
        
        # Merge LLM results if available (LLM takes priority)
        if llm_result:
            parsed = self._merge_results(parsed, llm_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ReasoningResult(
            parsed_instruction=parsed,
            goal_location=None,  # Will be filled by TSG
            reasoning_steps=parsed.sub_tasks,
            processing_time_ms=processing_time
        )
    
    def _rule_based_parsing(self, instruction: str) -> ParsedInstruction:
        """
        Fast rule-based parsing for common instructions.
        Enhanced with fuzzy matching and expanded keywords.
        """
        instruction_lower = instruction.lower()
        words = instruction_lower.split()
        
        # Detect goal type using action keywords
        goal_type = "navigate"  # default
        for action, keywords in self.action_keywords.items():
            if any(kw in instruction_lower for kw in keywords):
                goal_type = action
                break
        
        # Override for specific patterns
        if any(w in instruction_lower for w in ["where is", "where's", "find me"]):
            goal_type = "find"
        
        # Detect target object
        target_object = None
        for obj, keywords in self.object_keywords.items():
            if any(kw in instruction_lower for kw in keywords):
                target_object = obj
                break
        
        # Detect target location/room
        target_location = None
        for room, keywords in self.room_keywords.items():
            if any(kw in instruction_lower for kw in keywords):
                target_location = room
                break
        
        # Extract spatial relationships
        spatial_relationships = []
        spatial_patterns = [
            (r'(\w+)\s+near\s+(?:the\s+)?(\w+)', 'near'),
            (r'(\w+)\s+by\s+(?:the\s+)?(\w+)', 'by'),
            (r'(\w+)\s+behind\s+(?:the\s+)?(\w+)', 'behind'),
            (r'(\w+)\s+next\s+to\s+(?:the\s+)?(\w+)', 'next_to'),
            (r'(\w+)\s+in\s+front\s+of\s+(?:the\s+)?(\w+)', 'in_front_of'),
        ]
        
        for pattern, relation in spatial_patterns:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                spatial_relationships.append({
                    "subject": match[0],
                    "relation": relation,
                    "anchor": match[1]
                })
        
        # Generate sub-tasks
        sub_tasks = []
        if target_location:
            sub_tasks.append(f"Navigate to {target_location}")
        if target_object:
            sub_tasks.append(f"Locate {target_object}")
            sub_tasks.append(f"Approach {target_object}")
        if goal_type == "follow" and target_object:
            sub_tasks.append(f"Track {target_object} position")
            sub_tasks.append(f"Maintain following distance")
        
        # Calculate confidence
        confidence = 0.5
        if target_object or target_location:
            confidence += 0.3
        if len(sub_tasks) > 0:
            confidence += 0.2
        
        return ParsedInstruction(
            original_text=instruction,
            goal_type=goal_type,
            target_object=target_object,
            target_location=target_location,
            spatial_relationships=spatial_relationships,
            sub_tasks=sub_tasks,
            confidence=min(1.0, confidence)
        )
    
    async def _llm_parsing(self, 
                           instruction: str,
                           context: Optional[Dict] = None) -> Dict:
        """
        Use Ollama LLM for complex reasoning.
        """
        # Build prompt with room layout
        system_prompt = f"""You are a robot navigation assistant in a smart home. Parse user instructions into structured format.

{self.ROOM_LAYOUT}

SEMANTIC UNDERSTANDING - Map abstract requests to locations:
- Food/drinks (apple, water, hungry, cook) → kitchen
- Rest/sleep (tired, nap, rest, exhausted) → bedroom  
- Entertainment (TV, movie, relax, sit) → living_room
- Hygiene (shower, wash, soap, toilet) → bathroom
- "explore" → visit all rooms sequentially

HANDLE TYPOS: persoon→person, kichen→kitchen, bedrom→bedroom

Output JSON:
{{
    "goal_type": "navigate|find|approach|follow|stop|explore",
    "target_object": "object name or null",
    "target_location": "room name (kitchen|living_room|bedroom|bathroom) or null",
    "spatial_relations": [{{"subject": "...", "relation": "near|behind|...", "anchor": "..."}}],
    "sub_tasks": ["task1", "task2", ...],
    "constraints": ["avoid X", "prefer Y", ...]
}}"""

        user_prompt = f"Parse this navigation instruction: \"{instruction}\""
        
        if context:
            user_prompt += f"\n\nContext: Detected objects: {context.get('detected_objects', [])}"
            user_prompt += f"\nCurrent room: {context.get('current_room', 'unknown')}"
        
        # Call Ollama API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Extract JSON from response
                try:
                    # Find JSON in response
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
        return {}
    
    def _merge_results(self, 
                       rule_based: ParsedInstruction, 
                       llm_result: Dict) -> ParsedInstruction:
        """Merge rule-based and LLM parsing results."""
        # Update with LLM results if available
        if llm_result.get("target_object") and not rule_based.target_object:
            rule_based.target_object = llm_result["target_object"]
        
        if llm_result.get("target_location") and not rule_based.target_location:
            rule_based.target_location = llm_result["target_location"]
        
        if llm_result.get("sub_tasks"):
            # Merge sub_tasks, avoiding duplicates
            existing = set(rule_based.sub_tasks)
            for task in llm_result["sub_tasks"]:
                if task not in existing:
                    rule_based.sub_tasks.append(task)
        
        if llm_result.get("spatial_relations"):
            rule_based.spatial_relationships.extend(llm_result["spatial_relations"])
        
        if llm_result.get("constraints"):
            rule_based.constraints.extend(llm_result["constraints"])
        
        # Boost confidence if LLM provided useful info
        if llm_result:
            rule_based.confidence = min(1.0, rule_based.confidence + 0.2)
        
        return rule_based
    
    def parse_instruction_sync(self, instruction: str, context: Optional[Dict] = None) -> ReasoningResult:
        """
        Synchronous version of parse_instruction.
        Falls back to rule-based only if async not available.
        """
        start_time = time.time()
        parsed = self._rule_based_parsing(instruction)
        processing_time = (time.time() - start_time) * 1000
        
        return ReasoningResult(
            parsed_instruction=parsed,
            goal_location=None,
            reasoning_steps=parsed.sub_tasks,
            processing_time_ms=processing_time
        )
    
    def resolve_implicit_goal(self, 
                              instruction: str,
                              detected_objects: List[str]) -> Optional[str]:
        """
        Resolve implicit goals based on context.
        
        Example: "I'm hungry" → kitchen
                 "I'm tired" → bedroom
        """
        instruction_lower = instruction.lower()
        
        # Implicit mappings - semantic understanding of user intent
        implicit_goals = {
            # Kitchen-related (food, drinks, appliances)
            "hungry": "kitchen",
            "thirsty": "kitchen",
            "eat": "kitchen",
            "drink": "kitchen",
            "food": "kitchen",
            "apple": "kitchen",
            "fruit": "kitchen",
            "snack": "kitchen",
            "water": "kitchen",
            "coffee": "kitchen",
            "cook": "kitchen",
            "fridge": "kitchen",
            "refrigerator": "kitchen",
            # Bedroom-related (rest, sleep)
            "tired": "bedroom",
            "sleep": "bedroom",
            "rest": "bedroom",
            "nap": "bedroom",
            "lie down": "bedroom",
            "sleepy": "bedroom",
            "exhausted": "bedroom",
            # Living room-related (entertainment, relaxation)
            "watch tv": "living_room",
            "relax": "living_room",
            "television": "living_room",
            "movie": "living_room",
            "sit": "living_room",
            "couch": "living_room",
            "sofa": "living_room",
            # Bathroom-related (hygiene)
            "bathroom": "bathroom",
            "toilet": "bathroom",
            "shower": "bathroom",
            "wash": "bathroom",
            "soap": "bathroom",
            "brush teeth": "bathroom",
        }
        
        for trigger, goal in implicit_goals.items():
            if trigger in instruction_lower:
                return goal
        
        return None
    
    # MCP Tool Interface
    def mcp_tool_definition(self) -> dict:
        """Export MCP tool definitions."""
        return {
            "name": "reasoning_agent",
            "description": "Natural language understanding for robot navigation",
            "tools": [
                {
                    "name": "parse_instruction",
                    "description": "Parse a natural language navigation instruction",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "Natural language instruction"
                            }
                        },
                        "required": ["instruction"]
                    }
                },
                {
                    "name": "resolve_ambiguity",
                    "description": "Resolve ambiguous references using context",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "reference": {"type": "string"},
                            "candidates": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["reference", "candidates"]
                    }
                }
            ]
        }


# Standalone test
if __name__ == "__main__":
    import asyncio
    
    print("Testing Reasoning Agent...")
    
    agent = ReasoningAgent()
    
    # Test instructions
    test_instructions = [
        "Navigate to the kitchen and find the coffee machine",
        "Find the person near the window",
        "Go to the living room",
        "I'm hungry",
        "Follow that person",
    ]
    
    for instruction in test_instructions:
        print(f"\nInstruction: \"{instruction}\"")
        result = agent.parse_instruction_sync(instruction)
        
        parsed = result.parsed_instruction
        print(f"  Goal type: {parsed.goal_type}")
        print(f"  Target object: {parsed.target_object}")
        print(f"  Target location: {parsed.target_location}")
        print(f"  Spatial relations: {parsed.spatial_relationships}")
        print(f"  Sub-tasks: {parsed.sub_tasks}")
        print(f"  Confidence: {parsed.confidence:.2f}")
    
    print("\nReasoning Agent test complete!")
