"""
app/llm/heuristic_provider.py
Rule-based heuristic fallback provider - Enhanced with JSON validation
"""
import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider


logger = logging.getLogger(__name__)


class HeuristicProvider(BaseLLMProvider):
    """
    Enhanced rule-based heuristic fallback provider with:
    - Schema-aligned responses
    - JSON validation
    - Enhanced templates for common app types
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = LLMProvider.HEURISTIC
        self.template_cache: Dict[str, str] = {}
        
        # Load templates
        self._load_templates()
        
        logger.info("Enhanced heuristic provider initialized")
    
    def _load_templates(self):
        """Load JSON templates for different app types"""
        # Architecture templates
        self.template_cache["architecture_counter"] = json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "main_screen",
                    "name": "Counter",
                    "purpose": "Display counter value with increment and decrement buttons",
                    "components": ["Text", "Button", "Button", "Button"],
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
                "local_storage": ["count"]
            },
            "theme": {
                "primary_color": "#007AFF",
                "secondary_color": "#5856D6",
                "background_color": "#FFFFFF"
            }
        }, indent=2)
        
        # Layout templates
        self.template_cache["layout_counter"] = json.dumps({
            "screenId": "main_screen",
            "layoutType": "flex",
            "backgroundColor": "#FFFFFF",
            "components": [
                {
                    "id": "title_text",
                    "type": "Text",
                    "properties": {
                        "value": "Counter App",
                        "fontSize": 24,
                        "fontWeight": "bold",
                        "color": "#000000"
                    },
                    "position": {"x": 0, "y": 80},
                    "constraints": {
                        "width": "100%",
                        "height": 40,
                        "marginTop": 80,
                        "marginBottom": 40,
                        "alignment": "center"
                    }
                },
                {
                    "id": "counter_text",
                    "type": "Text",
                    "properties": {
                        "value": "0",
                        "fontSize": 48,
                        "fontWeight": "bold",
                        "color": "#007AFF"
                    },
                    "position": {"x": 0, "y": 160},
                    "constraints": {
                        "width": "100%",
                        "height": 60,
                        "marginTop": 20,
                        "marginBottom": 40,
                        "alignment": "center"
                    }
                },
                {
                    "id": "increment_button",
                    "type": "Button",
                    "properties": {
                        "label": "Increment +",
                        "size": "large",
                        "variant": "primary"
                    },
                    "position": {"x": 60, "y": 280},
                    "constraints": {
                        "width": 120,
                        "height": 50,
                        "marginTop": 20,
                        "alignment": "center"
                    }
                },
                {
                    "id": "decrement_button",
                    "type": "Button",
                    "properties": {
                        "label": "Decrement -",
                        "size": "large",
                        "variant": "secondary"
                    },
                    "position": {"x": 200, "y": 280},
                    "constraints": {
                        "width": 120,
                        "height": 50,
                        "marginTop": 20,
                        "alignment": "center"
                    }
                },
                {
                    "id": "reset_button",
                    "type": "Button",
                    "properties": {
                        "label": "Reset",
                        "size": "medium",
                        "variant": "outline"
                    },
                    "position": {"x": 140, "y": 360},
                    "constraints": {
                        "width": 100,
                        "height": 40,
                        "marginTop": 20,
                        "alignment": "center"
                    }
                }
            ]
        }, indent=2)
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using enhanced rule-based heuristics"""
        
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
            content, template_type = self._generate_architecture(user_message)
        elif "layout" in system_message or "component position" in system_message:
            content, template_type = self._generate_layout(user_message)
        elif "blockly" in system_message or "block" in system_message:
            content, template_type = self._generate_blockly(user_message)
        else:
            # Default to architecture
            content, template_type = self._generate_architecture(user_message)
        
        # Ensure content is valid JSON
        try:
            json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Heuristic generated invalid JSON, using fallback")
            content = self._get_fallback_json(template_type)
        
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            tokens_used=len(content.split()),
            finish_reason="heuristic_complete",
            model="heuristic-v2",
            metadata={
                "template_used": template_type,
                "generated_at": datetime.now().isoformat(),
                "input_length": len(user_message)
            }
        )
    
    def _generate_architecture(self, message: str) -> tuple[str, str]:
        """Generate schema-compliant architecture with template matching"""
        
        # Detect app type from keywords
        app_type = self._detect_app_type(message)
        
        # Generate based on detected type
        if any(word in message for word in ["counter", "count", "increment", "decrement"]):
            content = self.template_cache.get("architecture_counter", self._generic_architecture(message))
            template_type = "counter_architecture"
        elif any(word in message for word in ["todo", "task", "checklist"]):
            content = self._todo_architecture()
            template_type = "todo_architecture"
        elif any(word in message for word in ["calculator", "calc", "math", "calculate"]):
            content = self._calculator_architecture()
            template_type = "calculator_architecture"
        elif any(word in message for word in ["note", "memo", "text", "write"]):
            content = self._notes_architecture()
            template_type = "notes_architecture"
        elif any(word in message for word in ["weather", "forecast", "temperature"]):
            content = self._weather_architecture()
            template_type = "weather_architecture"
        else:
            content = self._generic_architecture(message)
            template_type = "generic_architecture"
        
        return content, template_type
    
    def _detect_app_type(self, message: str) -> str:
        """Detect app type from message with enhanced keyword matching"""
        multi_screen_keywords = ["navigation", "multiple screens", "tabs", "pages", "multi"]
        
        if any(kw in message for kw in multi_screen_keywords):
            return "multi-page"
        else:
            return "single-page"
    
    def _generic_architecture(self, message: str) -> str:
        """Generic app architecture - Enhanced"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "main_screen",
                    "name": "Main Screen",
                    "purpose": f"Main screen for: {message[:100]}",
                    "components": ["Text", "Button", "InputText"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": []
            },
            "state_management": [
                {
                    "name": "app_data",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": None
                }
            ],
            "data_flow": {
                "user_interactions": ["input_change", "button_press"],
                "api_calls": [],
                "local_storage": []
            },
            "theme": {
                "primary_color": "#007AFF",
                "secondary_color": "#5856D6",
                "background_color": "#F2F2F7"
            }
        }, indent=2)
    
    def _todo_architecture(self) -> str:
        """Enhanced Todo app architecture"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "todo_screen",
                    "name": "Todo List",
                    "purpose": "Manage todo items with add, complete, edit, and delete functionality",
                    "components": ["InputText", "Button", "Text", "Checkbox", "Button", "List"],
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
                    "scope": "global",
                    "initial_value": []
                },
                {
                    "name": "newTodoText",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": ""
                },
                {
                    "name": "filter",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": "all"
                }
            ],
            "data_flow": {
                "user_interactions": ["add_todo", "toggle_complete", "edit_todo", "delete_todo", "filter_todos"],
                "api_calls": [],
                "local_storage": ["todos"]
            },
            "theme": {
                "primary_color": "#34C759",
                "secondary_color": "#FF9500",
                "background_color": "#FFFFFF"
            }
        }, indent=2)
    
    def _weather_architecture(self) -> str:
        """Weather app architecture"""
        return json.dumps({
            "app_type": "single-page",
            "screens": [
                {
                    "id": "weather_screen",
                    "name": "Weather",
                    "purpose": "Display current weather and forecast information",
                    "components": ["Text", "Image", "List", "Button", "InputText"],
                    "navigation": []
                }
            ],
            "navigation": {
                "type": "stack",
                "routes": []
            },
            "state_management": [
                {
                    "name": "current_weather",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": None
                },
                {
                    "name": "forecast",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": []
                },
                {
                    "name": "location",
                    "type": "local-state",
                    "scope": "screen",
                    "initial_value": "New York"
                }
            ],
            "data_flow": {
                "user_interactions": ["refresh", "change_location", "view_details"],
                "api_calls": ["weather_api", "forecast_api"],
                "local_storage": ["location", "last_update"]
            },
            "theme": {
                "primary_color": "#5AC8FA",
                "secondary_color": "#007AFF",
                "background_color": "#F2F2F7"
            }
        }, indent=2)
    
    def _generate_layout(self, message: str) -> tuple[str, str]:
        """Generate schema-compliant layout"""
        if "counter" in message:
            content = self.template_cache.get("layout_counter", self._generic_layout(message))
            template_type = "counter_layout"
        else:
            content = self._generic_layout(message)
            template_type = "generic_layout"
        
        return content, template_type
    
    def _generic_layout(self, message: str) -> str:
        """Generic layout"""
        return json.dumps({
            "screenId": "main_screen",
            "layoutType": "flex",
            "backgroundColor": "#FFFFFF",
            "components": [
                {
                    "id": "title",
                    "type": "Text",
                    "properties": {
                        "value": "App Title",
                        "fontSize": 24,
                        "fontWeight": "bold",
                        "color": "#000000"
                    },
                    "position": {"x": 0, "y": 100},
                    "constraints": {
                        "width": "100%",
                        "height": 40,
                        "marginTop": 100,
                        "alignment": "center"
                    }
                },
                {
                    "id": "content",
                    "type": "Text",
                    "properties": {
                        "value": "App content goes here",
                        "fontSize": 16,
                        "color": "#666666"
                    },
                    "position": {"x": 0, "y": 180},
                    "constraints": {
                        "width": "80%",
                        "height": 100,
                        "marginTop": 20,
                        "marginLeft": "10%",
                        "alignment": "center"
                    }
                },
                {
                    "id": "action_button",
                    "type": "Button",
                    "properties": {
                        "label": "Action",
                        "size": "medium",
                        "variant": "primary"
                    },
                    "position": {"x": 0, "y": 320},
                    "constraints": {
                        "width": 200,
                        "height": 44,
                        "marginTop": 40,
                        "alignment": "center"
                    }
                }
            ]
        }, indent=2)
    
    def _generate_blockly(self, message: str) -> tuple[str, str]:
        """Generate schema-compliant Blockly blocks"""
        content = json.dumps({
            "blocks": {
                "languageVersion": 0,
                "blocks": [
                    {
                        "type": "component_event",
                        "id": "button_click",
                        "fields": {
                            "COMPONENT": "action_button",
                            "EVENT": "onPress"
                        },
                        "inputs": {
                            "DO": {
                                "block": {
                                    "type": "state_update",
                                    "id": "update_state",
                                    "fields": {
                                        "STATE_VAR": "app_data",
                                        "OPERATION": "set",
                                        "VALUE": "updated"
                                    }
                                }
                            }
                        }
                    }
                ]
            },
            "variables": [
                {
                    "name": "app_data",
                    "id": "var_1",
                    "type": "string"
                }
            ],
            "custom_blocks": []
        }, indent=2)
        
        return content, "basic_blockly"
    
    def _get_fallback_json(self, template_type: str) -> str:
        """Get fallback JSON when generation fails"""
        return json.dumps({
            "error": "Failed to generate valid JSON",
            "template_type": template_type,
            "fallback": True,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    
    async def health_check(self) -> bool:
        """Heuristic provider is always available"""
        return True
    
    def get_provider_type(self) -> LLMProvider:
        """Return provider type"""
        return self.provider_name