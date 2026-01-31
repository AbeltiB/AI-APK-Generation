"""
Component property schemas and enhanced component definitions.
"""
from typing import List, Dict, Any, Optional, Literal, Type
from pydantic import BaseModel, Field, field_validator, model_validator
import re

from .core import (
    BaseComponentProperties,
    PropertyValue,
    ComponentStyle,
    register_component_schema,
)


class ButtonProperties(BaseComponentProperties):
    """Button-specific properties"""
    onPress: Optional[PropertyValue] = Field(
        default=None,
        description="Event handler block ID"
    )
    variant: Optional[PropertyValue] = Field(
        default=None,
        description="Button variant: primary, secondary, outline, ghost"
    )


class InputTextProperties(BaseComponentProperties):
    """InputText-specific properties"""
    placeholder: Optional[PropertyValue] = None
    value: Optional[PropertyValue] = None
    maxLength: Optional[PropertyValue] = None
    keyboardType: Optional[PropertyValue] = Field(
        default=None,
        description="Keyboard type: default, numeric, email, phone"
    )
    onChange: Optional[PropertyValue] = Field(
        default=None,
        description="Change event handler block ID"
    )
    secureTextEntry: Optional[PropertyValue] = Field(
        default=None,
        description="Hide text for passwords"
    )


class TextProperties(BaseComponentProperties):
    """Text display properties"""
    value: PropertyValue = Field(..., description="Text content")
    fontSize: Optional[PropertyValue] = Field(
        default=None,
        description="Font size in pixels (12-32)"
    )
    fontWeight: Optional[PropertyValue] = Field(
        default=None,
        description="Font weight: normal, bold, 100-900"
    )
    textAlign: Optional[PropertyValue] = Field(
        default=None,
        description="Text alignment: left, center, right, justify"
    )
    numberOfLines: Optional[PropertyValue] = Field(
        default=None,
        description="Maximum number of lines (truncate with ...)"
    )


class SwitchProperties(BaseComponentProperties):
    """Switch/Toggle-specific properties"""
    value: PropertyValue = Field(..., description="Boolean value")
    onToggle: Optional[PropertyValue] = Field(
        default=None,
        description="Toggle event handler block ID"
    )
    thumbColor: Optional[PropertyValue] = None
    trackColor: Optional[PropertyValue] = None


class CheckboxProperties(BaseComponentProperties):
    """Checkbox-specific properties"""
    checked: PropertyValue = Field(..., description="Boolean checked state")
    label: Optional[PropertyValue] = None
    onChange: Optional[PropertyValue] = Field(
        default=None,
        description="Change event handler block ID"
    )


class SliderProperties(BaseComponentProperties):
    """Slider-specific properties"""
    min: PropertyValue = Field(..., description="Minimum value")
    max: PropertyValue = Field(..., description="Maximum value")
    value: PropertyValue = Field(..., description="Current value")
    step: Optional[PropertyValue] = Field(
        default=None,
        description="Step increment"
    )
    onChange: Optional[PropertyValue] = Field(
        default=None,
        description="Change event handler block ID"
    )


# Register component schemas
register_component_schema("Button", ButtonProperties)
register_component_schema("InputText", InputTextProperties)
register_component_schema("Text", TextProperties)
register_component_schema("Switch", SwitchProperties)
register_component_schema("Checkbox", CheckboxProperties)
register_component_schema("Slider", SliderProperties)


class EnhancedComponentDefinition(BaseModel):
    """Enhanced component definition with strict validation"""
    component_id: str = Field(..., description="Unique component identifier")
    component_type: Literal[
        "Button", "InputText", "Switch", "Checkbox", "TextArea",
        "Slider", "Spinner", "Text", "Joystick", "ProgressBar",
        "DatePicker", "TimePicker", "ColorPicker", "Map", "Chart"
    ]
    properties: Dict[str, PropertyValue]
    z_index: int = Field(default=0, ge=0, description="Layer order (higher = on top)")
    parent_id: Optional[str] = Field(default=None, description="Parent component ID")
    children_ids: List[str] = Field(default_factory=list)
    substitution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Substitution metadata if component was replaced"
    )
    
    @field_validator('component_id')
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        """Ensure valid component ID format"""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(
                f"Invalid component ID: {v}. Must start with letter/underscore"
            )
        return v
    
    @model_validator(mode='after')
    def validate_component_properties(self) -> 'EnhancedComponentDefinition':
        """Validate properties against component-specific schema"""
        from .core import COMPONENT_PROPERTY_SCHEMAS
        
        schema_class = COMPONENT_PROPERTY_SCHEMAS.get(self.component_type)
        
        if schema_class:
            try:
                # Convert properties dict to schema format
                props_dict = {
                    key: value for key, value in self.properties.items()
                }
                # Validate using component-specific schema
                schema_class(**props_dict)
            except Exception as e:
                raise ValueError(f"Invalid properties for {self.component_type}: {e}")
        
        return self