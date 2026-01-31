"""
Architecture design models for app structure.
"""
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class NavigationStructure(BaseModel):
    """App navigation configuration"""
    type: Literal["stack", "tab", "drawer"] = "stack"
    routes: List[Dict[str, str]] = Field(default_factory=list)


class StateDefinition(BaseModel):
    """State management definition"""
    name: str
    type: Literal["local-state", "global-state", "async-state"]
    scope: Literal["component", "screen", "global"]
    initial_value: Any


class DataFlowDiagram(BaseModel):
    """Data flow representation"""
    user_interactions: List[str] = Field(default_factory=list)
    api_calls: List[str] = Field(default_factory=list)
    local_storage: List[str] = Field(default_factory=list)


class ScreenDefinition(BaseModel):
    """Single screen definition"""
    id: str
    name: str
    purpose: str
    components: List[str] = Field(default_factory=list)
    navigation: List[str] = Field(default_factory=list)


class ArchitectureDesign(BaseModel):
    """Complete app architecture design"""
    app_type: Literal["single-page", "multi-page", "navigation-based"]
    screens: List[ScreenDefinition]
    navigation: NavigationStructure
    state_management: List[StateDefinition]
    data_flow: DataFlowDiagram
    
    class Config:
        json_schema_extra = {
            "example": {
                "app_type": "single-page",
                "screens": [
                    {
                        "id": "screen_1",
                        "name": "Todo List",
                        "purpose": "Main screen for managing todos",
                        "components": ["InputText", "Button", "Text"],
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
                    }
                ],
                "data_flow": {
                    "user_interactions": ["add_todo", "delete_todo"],
                    "api_calls": [],
                    "local_storage": ["todos_list"]
                }
            }
        }