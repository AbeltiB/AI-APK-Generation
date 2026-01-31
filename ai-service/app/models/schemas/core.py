"""
Core type definitions and constants.
"""
from typing import List, Dict, Any, Optional, Literal, Union, Type
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import re


class PropertyValue(BaseModel):
    """Base property value structure"""
    type: Literal["literal", "variable", "expression", "event"]
    value: Any


class ComponentStyle(BaseModel):
    """Component positioning and styling"""
    left: int = Field(..., ge=0, description="X position in pixels")
    top: int = Field(..., ge=0, description="Y position in pixels")
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")
    
    @field_validator('height')
    @classmethod
    def validate_minimum_touch_target(cls, v: int) -> int:
        """Ensure minimum touch target size for accessibility"""
        if v < 44:
            raise ValueError("Height must be at least 44px for touch targets")
        return v


class BaseComponentProperties(BaseModel):
    """Common properties for all components"""
    text: Optional[PropertyValue] = None
    size: Optional[PropertyValue] = Field(
        default=None,
        description="Component size: small, medium, large"
    )
    color: Optional[PropertyValue] = Field(
        default=None,
        description="Text/foreground color"
    )
    backgroundColor: Optional[PropertyValue] = Field(
        default=None,
        description="Background color"
    )
    style: PropertyValue = Field(..., description="Position and dimensions")
    disabled: Optional[PropertyValue] = Field(
        default=None,
        description="Whether component is disabled"
    )
    
    @field_validator('color', 'backgroundColor')
    @classmethod
    def validate_color(cls, v: Optional[PropertyValue]) -> Optional[PropertyValue]:
        """Validate hex color format"""
        if v and v.type == "literal" and isinstance(v.value, str):
            if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', v.value):
                raise ValueError(f"Invalid hex color: {v.value}")
        return v


# Component property type mapping
COMPONENT_PROPERTY_SCHEMAS = {}


# Helper function to register component schemas
def register_component_schema(component_type: str, schema_class: Type[BaseModel]):
    """Register a component property schema"""
    COMPONENT_PROPERTY_SCHEMAS[component_type] = schema_class