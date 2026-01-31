"""
Validation helper functions.
"""
from typing import Dict, Any, Optional, Tuple, Type
import re

from .core import COMPONENT_PROPERTY_SCHEMAS
from .components import EnhancedComponentDefinition
from .layout import EnhancedLayoutDefinition


def validate_color(color: str) -> bool:
    """Validate hex color format"""
    return bool(re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color))


def validate_bounds(
    component: EnhancedComponentDefinition,
    canvas_width: int = 375,
    canvas_height: int = 667
) -> Tuple[bool, Optional[str]]:
    """Validate component is within canvas bounds"""
    if 'style' not in component.properties:
        return False, "Component missing style property"
    
    style_val = component.properties['style']
    if not (style_val.type == "literal" and isinstance(style_val.value, dict)):
        return False, "Invalid style property type"
    
    style = style_val.value
    left = style.get('left', 0)
    top = style.get('top', 0)
    width = style.get('width', 0)
    height = style.get('height', 0)
    
    if left < 0 or top < 0:
        return False, f"Negative position: ({left}, {top})"
    
    if left + width > canvas_width:
        return False, f"Exceeds canvas width: {left + width} > {canvas_width}"
    
    if top + height > canvas_height:
        return False, f"Exceeds canvas height: {top + height} > {canvas_height}"
    
    return True, None


def check_collisions(components: list[EnhancedComponentDefinition]) -> Tuple[bool, Optional[str]]:
    """Check for component collisions"""
    bounds_list = []
    
    for comp in components:
        if 'style' not in comp.properties:
            continue
        
        style_val = comp.properties['style']
        if style_val.type == "literal" and isinstance(style_val.value, dict):
            style = style_val.value
            left = style.get('left', 0)
            top = style.get('top', 0)
            width = style.get('width', 0)
            height = style.get('height', 0)
            bounds_list.append((comp.component_id, left, top, left + width, top + height))
    
    # Check all pairs for collision
    for i, (id1, l1_x, l1_y, r1_x, r1_y) in enumerate(bounds_list):
        for j, (id2, l2_x, l2_y, r2_x, r2_y) in enumerate(bounds_list[i+1:], i+1):
            # Check for overlap
            if not (r1_x <= l2_x or r2_x <= l1_x or r1_y <= l2_y or r2_y <= l1_y):
                return False, f"Collision: {id1} overlaps with {id2}"
    
    return True, None


def validate_component(component: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a component dictionary"""
    try:
        comp = EnhancedComponentDefinition(**component)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_layout(layout: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a layout dictionary"""
    try:
        layout_obj = EnhancedLayoutDefinition(**layout)
        return True, None
    except Exception as e:
        return False, str(e)


def get_component_schema(component_type: str) -> Optional[Type]:
    """Get the property schema for a component type"""
    return COMPONENT_PROPERTY_SCHEMAS.get(component_type)


def create_component(
    component_id: str,
    component_type: str,
    properties: Dict[str, Any],
    z_index: int = 0,
    parent_id: Optional[str] = None
) -> EnhancedComponentDefinition:
    """Create a validated component"""
    # Convert properties to PropertyValue if needed
    formatted_props = {}
    for key, value in properties.items():
        if isinstance(value, dict) and 'type' in value and 'value' in value:
            # Already a PropertyValue
            formatted_props[key] = value
        else:
            # Convert to literal PropertyValue
            formatted_props[key] = {"type": "literal", "value": value}
    
    return EnhancedComponentDefinition(
        component_id=component_id,
        component_type=component_type,
        properties=formatted_props,
        z_index=z_index,
        parent_id=parent_id,
        children_ids=[]
    )