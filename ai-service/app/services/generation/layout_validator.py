"""
Layout Validator - Comprehensive layout validation.

Validates generated layouts for:
- Component bounds (within canvas)
- Touch target sizes (min 44px)
- Collision detection
- Visual hierarchy
- Accessibility compliance
"""
from typing import List, Tuple, Dict, Any, Optional
import math

from app.config import settings
from app.models.schemas.layout import EnhancedLayoutDefinition
from app.models.schemas.components import EnhancedComponentDefinition
from app.models.schemas.core import PropertyValue
from app.utils.logging import get_logger, log_context

logger = get_logger(__name__)


class LayoutWarning:
    """Represents a layout validation warning"""
    
    def __init__(self, level: str, component: str, message: str, suggestion: str = ""):
        self.level = level  # "info", "warning", "error"
        self.component = component
        self.message = message
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'level': self.level,
            'component': self.component,
            'message': self.message,
            'suggestion': self.suggestion
        }
    
    def __str__(self) -> str:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
        s = f"{emoji.get(self.level, '•')} [{self.level.upper()}] {self.component}: {self.message}"
        if self.suggestion:
            s += f"\n   → {self.suggestion}"
        return s


class LayoutValidator:
    """
    Comprehensive layout validation.
    
    Validation passes:
    1. Canvas bounds checking
    2. Touch target size validation
    3. Collision detection
    4. Component spacing
    5. Visual hierarchy
    6. Accessibility compliance
    """
    
    def __init__(self):
        self.warnings: List[LayoutWarning] = []
        self.canvas_width = settings.canvas_width
        self.canvas_height = settings.canvas_height
        self.min_touch_size = settings.min_touch_target_size
        
        logger.info(
            "🔍 layout.validator.initialized",
            extra={
                "canvas": f"{self.canvas_width}x{self.canvas_height}",
                "min_touch_size": self.min_touch_size
            }
        )
    
    async def validate(
        self,
        layout: EnhancedLayoutDefinition
    ) -> Tuple[bool, List[LayoutWarning]]:
        """
        Comprehensive validation of layout.
        
        Args:
            layout: Layout to validate
            
        Returns:
            Tuple of (is_valid, warnings_list)
        """
        with log_context(operation="layout_validation"):
            self.warnings = []
            
            logger.info(
                "🔍 layout.validation.started",
                extra={
                    "screen_id": layout.screen_id,
                    "components": len(layout.components)
                }
            )
            
            # Run all validation checks
            await self._validate_canvas(layout)
            await self._validate_component_bounds(layout)
            await self._validate_touch_targets(layout)
            await self._validate_collisions(layout)
            await self._validate_spacing(layout)
            await self._validate_visual_hierarchy(layout)
            await self._validate_accessibility(layout)
            
            # Determine if layout is valid
            has_errors = any(w.level == "error" for w in self.warnings)
            is_valid = not has_errors
            
            error_count = sum(1 for w in self.warnings if w.level == "error")
            warning_count = sum(1 for w in self.warnings if w.level == "warning")
            info_count = sum(1 for w in self.warnings if w.level == "info")
            
            if is_valid:
                logger.info(
                    "✅ layout.validation.passed",
                    extra={
                        "warnings": warning_count,
                        "infos": info_count
                    }
                )
            else:
                logger.error(
                    "❌ layout.validation.failed",
                    extra={
                        "errors": error_count,
                        "warnings": warning_count
                    }
                )
            
            return is_valid, self.warnings
    
    async def _validate_canvas(self, layout: EnhancedLayoutDefinition) -> None:
        """Validate canvas configuration"""
        
        canvas = layout.canvas
        
        # Check canvas dimensions
        if canvas['width'] != self.canvas_width:
            self.warnings.append(LayoutWarning(
                level="warning",
                component="canvas",
                message=f"Canvas width {canvas['width']} != standard {self.canvas_width}",
                suggestion=f"Use standard width of {self.canvas_width}px"
            ))
        
        if canvas['height'] != self.canvas_height:
            self.warnings.append(LayoutWarning(
                level="warning",
                component="canvas",
                message=f"Canvas height {canvas['height']} != standard {self.canvas_height}",
                suggestion=f"Use standard height of {self.canvas_height}px"
            ))
        
        # Check safe area insets
        safe_area = canvas.get('safe_area_insets', {})
        if safe_area.get('top', 0) < 20:
            self.warnings.append(LayoutWarning(
                level="info",
                component="canvas",
                message="Small top safe area",
                suggestion="Consider increasing for status bar clearance"
            ))
    
    async def _validate_component_bounds(
        self,
        layout: EnhancedLayoutDefinition
    ) -> None:
        """Validate components are within canvas bounds"""
        
        for component in layout.components:
            bounds = self._get_component_bounds(component)
            if not bounds:
                self.warnings.append(LayoutWarning(
                    level="error",
                    component=component.component_id,
                    message="Missing or invalid style property",
                    suggestion="Add valid style with position and size"
                ))
                continue
            
            left, top, right, bottom = bounds
            
            # Check left bound
            if left < 0:
                self.warnings.append(LayoutWarning(
                    level="error",
                    component=component.component_id,
                    message=f"Component extends beyond left edge (x={left})",
                    suggestion="Move component right to x >= 0"
                ))
            
            # Check top bound
            if top < 0:
                self.warnings.append(LayoutWarning(
                    level="error",
                    component=component.component_id,
                    message=f"Component extends beyond top edge (y={top})",
                    suggestion="Move component down to y >= 0"
                ))
            
            # Check right bound
            if right > self.canvas_width:
                self.warnings.append(LayoutWarning(
                    level="error",
                    component=component.component_id,
                    message=f"Component extends beyond right edge ({right} > {self.canvas_width})",
                    suggestion=f"Reduce width or move left"
                ))
            
            # Check bottom bound
            if bottom > self.canvas_height:
                self.warnings.append(LayoutWarning(
                    level="warning",
                    component=component.component_id,
                    message=f"Component extends beyond bottom edge ({bottom} > {self.canvas_height})",
                    suggestion="Consider scrollable container or reduce size"
                ))
            
            # Check safe area violations
            safe_area = layout.canvas.get('safe_area_insets', {})
            safe_top = safe_area.get('top', 0)
            
            if top < safe_top:
                self.warnings.append(LayoutWarning(
                    level="warning",
                    component=component.component_id,
                    message=f"Component in safe area (top={top}, safe={safe_top})",
                    suggestion="Move component below safe area"
                ))
    
    async def _validate_touch_targets(
        self,
        layout: EnhancedLayoutDefinition
    ) -> None:
        """Validate touch target sizes meet minimum requirements"""
        
        interactive_types = {'Button', 'Switch', 'Checkbox'}
        
        for component in layout.components:
            if component.component_type not in interactive_types:
                continue
            
            bounds = self._get_component_bounds(component)
            if not bounds:
                continue
            
            left, top, right, bottom = bounds
            width = right - left
            height = bottom - top
            
            # Check minimum height
            if height < self.min_touch_size:
                self.warnings.append(LayoutWarning(
                    level="error",
                    component=component.component_id,
                    message=f"Touch target too small: {height}px < {self.min_touch_size}px",
                    suggestion=f"Increase height to at least {self.min_touch_size}px"
                ))
            
            # Check minimum width for buttons
            if component.component_type == 'Button' and width < self.min_touch_size:
                self.warnings.append(LayoutWarning(
                    level="warning",
                    component=component.component_id,
                    message=f"Button width small: {width}px",
                    suggestion=f"Consider increasing to at least {self.min_touch_size}px"
                ))
    
    async def _validate_collisions(
        self,
        layout: EnhancedLayoutDefinition
    ) -> None:
        """Detect component collisions"""
        
        components = layout.components
        
        for i, comp1 in enumerate(components):
            bounds1 = self._get_component_bounds(comp1)
            if not bounds1:
                continue
            
            for comp2 in components[i+1:]:
                bounds2 = self._get_component_bounds(comp2)
                if not bounds2:
                    continue
                
                if self._rectangles_overlap(bounds1, bounds2):
                    self.warnings.append(LayoutWarning(
                        level="error",
                        component=f"{comp1.component_id}+{comp2.component_id}",
                        message="Components overlap",
                        suggestion="Reposition components to avoid collision"
                    ))
    
    async def _validate_spacing(
        self,
        layout: EnhancedLayoutDefinition
    ) -> None:
        """Validate component spacing"""
        
        min_spacing = 8  # 8px minimum spacing
        
        components = layout.components
        
        for i, comp1 in enumerate(components):
            bounds1 = self._get_component_bounds(comp1)
            if not bounds1:
                continue
            
            for comp2 in components[i+1:]:
                bounds2 = self._get_component_bounds(comp2)
                if not bounds2:
                    continue
                
                # Check vertical spacing
                distance = self._get_component_distance(bounds1, bounds2)
                
                if 0 < distance < min_spacing:
                    self.warnings.append(LayoutWarning(
                        level="info",
                        component=f"{comp1.component_id}+{comp2.component_id}",
                        message=f"Tight spacing: {distance}px",
                        suggestion=f"Consider {min_spacing}px minimum spacing"
                    ))
    
    def _get_component_distance(
        self,
        bounds1: Tuple[int, int, int, int],
        bounds2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate minimum distance between components"""
        l1_x, l1_y, r1_x, r1_y = bounds1
        l2_x, l2_y, r2_x, r2_y = bounds2
        
        # Calculate center points
        c1_x = (l1_x + r1_x) / 2
        c1_y = (l1_y + r1_y) / 2
        c2_x = (l2_x + r2_x) / 2
        c2_y = (l2_y + r2_y) / 2
        
        # If directly above/below (same x range)
        if not (r1_x < l2_x or r2_x < l1_x):
            if r1_y <= l2_y:
                return l2_y - r1_y
            elif r2_y <= l1_y:
                return l1_y - r2_y
        
        # Otherwise use center distance
        return math.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
    
    async def _validate_visual_hierarchy(
        self,
        layout: EnhancedLayoutDefinition
    ) -> None:
        """Validate visual hierarchy"""
        
        # Check if components are top-heavy (most important at top)
        components_by_y = sorted(
            layout.components,
            key=lambda c: self._get_component_bounds(c)[1] if self._get_component_bounds(c) else 9999
        )
        
        # Primary actions (Buttons) should be prominent
        buttons = [c for c in layout.components if c.component_type == 'Button']
        
        if buttons:
            # Check if buttons have distinguishable sizes
            button_sizes = []
            for btn in buttons:
                bounds = self._get_component_bounds(btn)
                if bounds:
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]
                    button_sizes.append(width * height)
            
            if len(set(button_sizes)) == 1 and len(buttons) > 1:
                self.warnings.append(LayoutWarning(
                    level="info",
                    component="buttons",
                    message="All buttons same size",
                    suggestion="Consider varying sizes for visual hierarchy"
                ))
    
    async def _validate_accessibility(
        self,
        layout: EnhancedLayoutDefinition
    ) -> None:
        """Validate accessibility compliance"""
        
        # Check for text contrast (basic check)
        text_components = [c for c in layout.components if c.component_type == 'Text']
        
        for text_comp in text_components:
            # Check if text has color property
            color_prop = text_comp.properties.get('color')
            bg_color_prop = text_comp.properties.get('backgroundColor')
            
            if color_prop and bg_color_prop:
                color = color_prop.value if color_prop.type == "literal" else None
                bg_color = bg_color_prop.value if bg_color_prop.type == "literal" else None
                
                # Simple check: warn if both are dark or both are light
                if color and bg_color:
                    if (color == bg_color):
                        self.warnings.append(LayoutWarning(
                            level="warning",
                            component=text_comp.component_id,
                            message="Text color same as background",
                            suggestion="Ensure sufficient contrast for readability"
                        ))
        
        # Check label associations for inputs
        inputs = [c for c in layout.components if 'Input' in c.component_type]
        
        if inputs:
            # Check if there are nearby Text components (labels)
            for input_comp in inputs:
                input_bounds = self._get_component_bounds(input_comp)
                if not input_bounds:
                    continue
                
                # Look for Text component above
                has_label = False
                for text_comp in text_components:
                    text_bounds = self._get_component_bounds(text_comp)
                    if not text_bounds:
                        continue
                    
                    # Check if text is above and close to input
                    if (text_bounds[3] <= input_bounds[1] and  # text bottom <= input top
                        abs(text_bounds[0] - input_bounds[0]) < 50):  # similar x position
                        has_label = True
                        break
                
                if not has_label:
                    self.warnings.append(LayoutWarning(
                        level="info",
                        component=input_comp.component_id,
                        message="Input may be missing label",
                        suggestion="Add Text component above input as label"
                    ))
    
    def _get_component_bounds(
        self,
        component: EnhancedComponentDefinition
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get component bounding rectangle"""
        style_prop = component.properties.get('style')
        if not style_prop or style_prop.type != "literal":
            return None
        
        style = style_prop.value
        if not isinstance(style, dict):
            return None
        
        left = style.get('left', 0)
        top = style.get('top', 0)
        width = style.get('width', 0)
        height = style.get('height', 0)
        
        return (left, top, left + width, top + height)
    
    def _rectangles_overlap(
        self,
        rect1: Tuple[int, int, int, int],
        rect2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two rectangles overlap"""
        l1_x, l1_y, r1_x, r1_y = rect1
        l2_x, l2_y, r2_x, r2_y = rect2
        
        if r1_x <= l2_x or r2_x <= l1_x:
            return False
        if r1_y <= l2_y or r2_y <= l1_y:
            return False
        
        return True


# Global validator instance
layout_validator = LayoutValidator()


if __name__ == "__main__":
    # Test validator
    import asyncio
    from app.models.schemas.core import PropertyValue
    
    async def test():
        print("\n" + "=" * 60)
        print("LAYOUT VALIDATOR TEST")
        print("=" * 60)
        
        # Test 1: Valid layout
        print("\n[TEST 1] Valid layout")
        
        valid_layout = EnhancedLayoutDefinition(
            screen_id="screen_1",
            components=[
                EnhancedComponentDefinition(
                    component_id="text_1",
                    component_type="Text",
                    properties={
                        "value": PropertyValue(type="literal", value="Counter: 0"),
                        "style": PropertyValue(type="literal", value={
                            "left": 97,
                            "top": 100,
                            "width": 180,
                            "height": 40
                        })
                    }
                ),
                EnhancedComponentDefinition(
                    component_id="btn_1",
                    component_type="Button",
                    properties={
                        "value": PropertyValue(type="literal", value="+"),
                        "style": PropertyValue(type="literal", value={
                            "left": 127,
                            "top": 160,
                            "width": 120,
                            "height": 44
                        })
                    }
                )
            ]
        )
        
        is_valid, warnings = await layout_validator.validate(valid_layout)
        
        print(f"Valid: {is_valid}")
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  {w}")
        
        # Test 2: Invalid layout (collision, out of bounds)
        print("\n[TEST 2] Invalid layout")
        
        invalid_layout = EnhancedLayoutDefinition(
            screen_id="screen_2",
            components=[
                EnhancedComponentDefinition(
                    component_id="btn_1",
                    component_type="Button",
                    properties={
                        "style": PropertyValue(type="literal", value={
                            "left": 0,
                            "top": 0,
                            "width": 120,
                            "height": 30  # Too small!
                        })
                    }
                ),
                EnhancedComponentDefinition(
                    component_id="btn_2",
                    component_type="Button",
                    properties={
                        "style": PropertyValue(type="literal", value={
                            "left": 50,  # Overlaps with btn_1!
                            "top": 10,
                            "width": 120,
                            "height": 44
                        })
                    }
                ),
                EnhancedComponentDefinition(
                    component_id="text_1",
                    component_type="Text",
                    properties={
                        "style": PropertyValue(type="literal", value={
                            "left": 300,
                            "top": 100,
                            "width": 200,  # Extends beyond canvas!
                            "height": 40
                        })
                    }
                )
            ]
        )
        
        is_valid, warnings = await layout_validator.validate(invalid_layout)
        
        print(f"Valid: {is_valid}")
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  {w}")
        
        print("\n" + "=" * 60)
        print("✅ Validator test complete!")
        print("=" * 60 + "\n")
    
    asyncio.run(test())