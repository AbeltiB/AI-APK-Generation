"""
app/llm/heuristic_provider.py
Rule-based heuristic fallback provider - SCHEMA ALIGNED
"""
import logging
import json
from typing import List, Optional, Dict, Any

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider


logger = logging.getLogger(__name__)


class HeuristicProvider(BaseLLMProvider):
    """
    Schema-aligned rule-based heuristic fallback provider.
    
    Generates responses that match enhanced_schemas.py structure:
    - ArchitectureDesign
    - EnhancedLayoutDefinition
    - EnhancedBlocklyDefinition
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = LLMProvider.HEURISTIC
        
        logger.info("Heuristic provider initialized with schema alignment")
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using rule-based heuristics"""
        
        # Extract user request from messages
        user_message = next(
            (msg.content for msg in messages if msg.role == "user"),
            ""
        ).lower()
        
        # Extract system message to determine what to generate
        system_message = next(
            (msg.content for msg in messages if msg.role == "system"),
            ""
        ).lower()
        
        logger.info(f"Heuristic fallback triggered for: {user_message[:100]}")
        
        # Determine what type of generation is needed
        if "architecture" in system_message or "app design" in system_message:
            content = self._generate_architecture(user_message)
        elif "layout" in system_message or "component position" in system_message:
            content = self._generate_layout(user_message)
        elif "blockly" in system_message or "block" in system_message:
            content = self._generate_blockly(user_message)
        else:
            # Default to architecture
            content = self._generate_architecture(user_message)
        
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            tokens_used=None,
            finish_reason="heuristic",
            model="rule-based",
            metadata={"template_used": "schema_aligned"}
        )
    
    def _generate_architecture(self, message: str) -> str:
        """
        Generate schema-compliant architecture.
        
        Matches schemas.py ArchitectureDesign structure.
        """
        
        # Detect app type from keywords
        app_type = self._detect_app_type(message)
        
        # Generate based on detected type
        if "counter" in message or "increment" in message:
            return self._counter_architecture()
        elif "todo" in message or "task" in message:
            return self._todo_architecture()
        elif "calculator" in message or "calc" in message:
            return self._calculator_architecture()
        elif "notes" in message or "memo" in message:
            return self._notes_architecture()
        else:
            return self._generic_architecture(message)
    
    def _detect_app_type(self, message: str) -> str:
        """Detect app type from message"""
        multi_screen_keywords = ["navigation", "multiple screens", "tabs", "pages"]
        
        if any(kw in message for kw in multi_screen_keywords):
            return "multi-page"
        else:
            return "single-page"
    
    def _counter_architecture(self) -> str:
        """Counter app architecture - Schema compliant"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "main_screen",
                    "name": "Counter",
                    "purpose": "Display counter value with increment and decrement buttons",
                    "components": ["Text", "Button", "Button"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": []
            },
            "state_management": [
                {
                    "name": "count",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": 0
                }
            ],
            "data_flow": {
                "user_interactions": ["increment", "decrement", "reset"],
                "api_calls": [],
                "local_storage": []
            }
        }, indent=2)
    
    def _todo_architecture(self) -> str:
        """Todo app architecture - Schema compliant"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "todo_screen",
                    "name": "Todo List",
                    "purpose": "Manage todo items with add, complete, and delete functionality",
                    "components": ["InputText", "Button", "Text", "Checkbox", "Button"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": []
            },
            "state_management": [
                {
                    "name": "todos",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": []
                },
                {
                    "name": "newTodoText",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": ""
                }
            ],
            "data_flow": {
                "user_interactions": ["add_todo", "toggle_complete", "delete_todo"],
                "api_calls": [],
                "local_storage": ["todos"]
            }
        }, indent=2)
    
    def _calculator_architecture(self) -> str:
        """Calculator app architecture - Schema compliant"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "calc_screen",
                    "name": "Calculator",
                    "purpose": "Perform basic arithmetic operations",
                    "components": ["Text", "Button"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": []
            },
            "state_management": [
                {
                    "name": "display",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": "0"
                },
                {
                    "name": "currentOperation",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": None
                },
                {
                    "name": "previousValue",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": 0
                }
            ],
            "data_flow": {
                "user_interactions": ["number_input", "operation_select", "calculate"],
                "api_calls": [],
                "local_storage": []
            }
        }, indent=2)
    
    def _notes_architecture(self) -> str:
        """Notes app architecture - Schema compliant"""
        return json.dumps({
            "app_type": "multi-page",
            "screens": [
                {
                    "id": "notes_list",
                    "name": "Notes List",
                    "purpose": "Display list of notes",
                    "components": ["Button", "Text"],
                    "navigation": ["note_detail"]
                },
                {
                    "id": "note_detail",
                    "name": "Note Detail",
                    "purpose": "View and edit note content",
                    "components": ["TextArea", "Button"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": [
                    {"from": "notes_list", "to": "note_detail", "label": "View Note"}
                ]
            },
            "state_management": [
                {
                    "name": "notes",
                    "type": "local-state",
                    "scope": "global",
                    "initial_value": []
                },
                {
                    "name": "currentNote",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": None
                }
            ],
            "data_flow": {
                "user_interactions": ["create_note", "edit_note", "delete_note"],
                "api_calls": [],
                "local_storage": ["notes"]
            }
        }, indent=2)
    
    def _generic_architecture(self, message: str) -> str:
        """Generic app architecture - Schema compliant"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "main_screen",
                    "name": "Main Screen",
                    "purpose": f"Main screen for: {message[:100]}",
                    "components": ["Text", "Button"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": []
            },
            "state_management": [
                {
                    "name": "data",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": None
                }
            ],
            "data_flow": {
                "user_interactions": ["interact"],
                "api_calls": [],
                "local_storage": []
            }
        }, indent=2)
    
    def _generate_layout(self, message: str) -> str:
        """
        Generate schema-compliant layout.
        
        Matches enhanced_schemas.py EnhancedLayoutDefinition structure.
        """
        return json.dumps({
            "screenId": "main_screen",
            "layoutType": "flex",
            "backgroundColor": "#FFFFFF",
            "components": [
                {
                    "id": "text_1",
                    "type": "Text",
                    "properties": {
                        "value": "App Content",
                        "fontSize": 18,
                        "fontWeight": "bold"
                    },
                    "position": {"x": 97, "y": 100},
                    "constraints": {
                        "width": "auto",
                        "height": 40,
                        "marginTop": 20,
                        "marginBottom": 16,
                        "marginLeft": 0,
                        "marginRight": 0,
                        "padding": 0,
                        "alignment": "center"
                    },
                    "children": []
                },
                {
                    "id": "button_1",
                    "type": "Button",
                    "properties": {
                        "label": "Action",
                        "size": "medium",
                        "variant": "primary"
                    },
                    "position": {"x": 127, "y": 160},
                    "constraints": {
                        "width": "auto",
                        "height": 44,
                        "marginTop": 16,
                        "marginBottom": 0,
                        "marginLeft": 0,
                        "marginRight": 0,
                        "padding": 12,
                        "alignment": "center"
                    },
                    "children": []
                }
            ]
        }, indent=2)
    
    def _generate_blockly(self, message: str) -> str:
        """
        Generate schema-compliant Blockly blocks.
        
        Matches enhanced_schemas.py EnhancedBlocklyDefinition structure.
        """
        return json.dumps({
            "blocks": {
                "languageVersion": 0,
                "blocks": [
                    {
                        "type": "component_event",
                        "id": "event_1",
                        "fields": {
                            "COMPONENT": "button_1",
                            "EVENT": "onPress"
                        },
                        "next": None
                    }
                ]
            },
            "variables": [
                {
                    "name": "count",
                    "id": "var_1",
                    "type": ""
                }
            ],
            "custom_blocks": []
        }, indent=2)
    
    async def health_check(self) -> bool:
        """Heuristic provider is always available"""
        return True
    
    def get_provider_type(self) -> LLMProvider:
        """Return provider type"""
        return self.provider_name