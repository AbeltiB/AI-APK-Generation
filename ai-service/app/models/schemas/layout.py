"""
Layout models with validation.
"""
from typing import List, Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel, Field, model_validator

from .components import EnhancedComponentDefinition


class Position(BaseModel):
    """Component position on canvas"""
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class LayoutConstraints(BaseModel):
    """Component layout constraints"""
    width: str = "auto"
    height: int = Field(44, ge=0)
    margin_top: int = Field(0, ge=0)
    margin_bottom: int = Field(0, ge=0)
    margin_left: int = Field(0, ge=0)
    margin_right: int = Field(0, ge=0)
    padding: int = Field(0, ge=0)
    alignment: Literal["left", "center", "right", "stretch"] = "center"


class EnhancedLayoutDefinition(BaseModel):
    """Enhanced layout with collision detection"""
    screen_id: str
    canvas: Dict[str, Any] = Field(
        default_factory=lambda: {
            "width": 375,
            "height": 667,
            "background_color": "#FFFFFF",
            "safe_area_insets": {
                "top": 44,
                "bottom": 34,
                "left": 0,
                "right": 0
            }
        }
    )
    components: List[EnhancedComponentDefinition]
    layout_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_no_collisions(self) -> 'EnhancedLayoutDefinition':
        """Check for component collisions"""
        def get_bounds(comp: EnhancedComponentDefinition) -> Optional[Tuple[int, int, int, int]]:
            """Extract component bounds"""
            if 'style' not in comp.properties:
                return None
            
            style_val = comp.properties['style']
            if style_val.type == "literal" and isinstance(style_val.value, dict):
                style = style_val.value
                left = style.get('left', 0)
                top = style.get('top', 0)
                width = style.get('width', 0)
                height = style.get('height', 0)
                return (left, top, left + width, top + height)
            return None
        
        def rectangles_overlap(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> bool:
            """Check if two rectangles overlap"""
            l1_x, l1_y, r1_x, r1_y = r1
            l2_x, l2_y, r2_x, r2_y = r2
            
            # No overlap if one is to the left/right/above/below the other
            if r1_x <= l2_x or r2_x <= l1_x:
                return False
            if r1_y <= l2_y or r2_y <= l1_y:
                return False
            
            return True
        
        # Check all pairs for collision
        for i, comp1 in enumerate(self.components):
            bounds1 = get_bounds(comp1)
            if not bounds1:
                continue
            
            for comp2 in self.components[i+1:]:
                bounds2 = get_bounds(comp2)
                if not bounds2:
                    continue
                
                if rectangles_overlap(bounds1, bounds2):
                    raise ValueError(
                        f"Component collision detected: {comp1.component_id} "
                        f"overlaps with {comp2.component_id}"
                    )
        
        return self
    
    @model_validator(mode='after')
    def validate_unique_ids(self) -> 'EnhancedLayoutDefinition':
        """Ensure all component IDs are unique"""
        ids = [comp.component_id for comp in self.components]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate component IDs found: {set(duplicates)}")
        return self
    
    @model_validator(mode='after')
    def validate_bounds(self) -> 'EnhancedLayoutDefinition':
        """Ensure components are within canvas bounds"""
        canvas_width = self.canvas.get("width", 375)
        canvas_height = self.canvas.get("height", 667)
        
        for comp in self.components:
            if 'style' not in comp.properties:
                continue
            
            style_val = comp.properties['style']
            if style_val.type == "literal" and isinstance(style_val.value, dict):
                style = style_val.value
                left = style.get('left', 0)
                top = style.get('top', 0)
                width = style.get('width', 0)
                height = style.get('height', 0)
                
                if left < 0 or top < 0:
                    raise ValueError(f"Component {comp.component_id} has negative position")
                
                if left + width > canvas_width:
                    raise ValueError(
                        f"Component {comp.component_id} exceeds canvas width: "
                        f"{left + width} > {canvas_width}"
                    )
                
                if top + height > canvas_height:
                    raise ValueError(
                        f"Component {comp.component_id} exceeds canvas height: "
                        f"{top + height} > {canvas_height}"
                    )
        
        return self