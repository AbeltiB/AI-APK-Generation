"""
Layout Generator - Phase 3
Uses LLM Orchestrator (Llama3 → Heuristic fallback)
"""
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas.architecture import ArchitectureDesign, ScreenDefinition
from app.models.schemas.components import EnhancedComponentDefinition
from app.models.schemas.layout import EnhancedLayoutDefinition
from app.models.schemas.core import PropertyValue
from app.models.prompts import prompts
from app.services.generation.layout_validator import layout_validator
from app.llm.orchestrator import LLMOrchestrator
from app.llm.base import LLMMessage
from app.utils.logging import get_logger, log_context, trace_async

logger = get_logger(__name__)


class LayoutGenerationError(Exception):
    """Base exception for layout generation errors"""
    pass

class CollisionError(Exception):
    """Raised when UI elements collide during layout generation."""
    pass


class LayoutGenerator:
    """
    Phase 3 Layout Generator using LLM Orchestrator.
    
    Generation Flow:
    1. 🎯 Try Llama3 via orchestrator
    2. 🔄 Retry with corrections if needed
    3. 🛡️ Fall back to heuristic if all retries fail
    4. ✅ Validate and resolve collisions
    
    Features:
    - Llama3 as primary LLM
    - Automatic heuristic template fallback
    - Collision detection and resolution
    - Touch target validation
    """
    
    def __init__(self, orchestrator: Optional[LLMOrchestrator] = None):
        # Initialize LLM orchestrator
        if orchestrator:
            self.orchestrator = orchestrator
        else:
            config = {
                "failure_threshold": 3,
                "failure_window_minutes": 5,
                "llama3_api_url": settings.llama3_api_url,
                "llama3_api_key": settings.llama3_api_key
            }
            self.orchestrator = LLMOrchestrator(config)
        
        # Canvas constraints
        self.canvas_width = settings.canvas_width
        self.canvas_height = settings.canvas_height
        self.safe_area_top = settings.canvas_safe_area_top
        self.safe_area_bottom = settings.canvas_safe_area_bottom
        
        # Component sizing defaults (width, height)
        self.component_defaults = {
            'Button': (120, 44),
            'InputText': (280, 44),
            'Switch': (51, 31),
            'Checkbox': (24, 24),
            'TextArea': (280, 100),
            'Slider': (280, 44),
            'Spinner': (40, 40),
            'Text': (280, 40),
            'Joystick': (100, 100),
            'ProgressBar': (280, 8),
            'DatePicker': (280, 44),
            'TimePicker': (280, 44),
            'ColorPicker': (280, 44),
            'Map': (340, 200),
            'Chart': (340, 200)
        }
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'collisions_resolved': 0,
            'heuristic_fallbacks': 0,
            'llama3_successes': 0
        }
        
        logger.info(
            "layout.generator.initialized",
            extra={
                "llm_provider": "llama3",
                "canvas": f"{self.canvas_width}x{self.canvas_height}",
                "heuristic_fallback_enabled": True
            }
        )
    
    @trace_async("layout.generation")
    async def generate(
        self,
        architecture: ArchitectureDesign,
        screen_id: str
    ) -> Tuple[EnhancedLayoutDefinition, Dict[str, Any]]:
        """
        Generate layout for a specific screen.
        
        Args:
            architecture: Complete architecture design
            screen_id: Screen to generate layout for
            
        Returns:
            Tuple of (EnhancedLayoutDefinition, metadata)
            
        Raises:
            LayoutGenerationError: If generation fails
        """
        self.stats['total_requests'] += 1
        
        # Find the screen
        screen = None
        for s in architecture.screens:
            if s.id == screen_id:
                screen = s
                break
        
        if not screen:
            raise LayoutGenerationError(f"Screen '{screen_id}' not found in architecture")
        
        with log_context(operation="layout_generation", screen_id=screen_id):
            logger.info(
                f"📐 layout.generation.started",
                extra={
                    "screen_name": screen.name,
                    "components": len(screen.components)
                }
            )
            
            # Try LLM first
            layout = None
            metadata = {}
            used_heuristic = False
            
            try:
                layout_data, metadata = await self._generate_with_llm(
                    screen=screen,
                    architecture=architecture
                )
                
                # Convert to enhanced components
                components = await self._convert_to_enhanced_components(
                    layout_data['components'],
                    screen_id
                )
                
                self.stats['llama3_successes'] += 1
                logger.info(
                    "✅ layout.llm.success",
                    extra={
                        "components": len(components),
                        "provider": metadata.get('provider', 'llama3')
                    }
                )
                
            except Exception as llm_error:
                logger.warning(
                    "⚠️ layout.llm.failed",
                    extra={"error": str(llm_error)},
                    exc_info=llm_error
                )
                
                # Fall back to heuristic
                logger.info("🛡️ layout.fallback.initiating")
                
                try:
                    components = await self._generate_heuristic_layout(screen)
                    metadata = {
                        'generation_method': 'heuristic',
                        'fallback_reason': str(llm_error),
                        'provider': 'heuristic',
                        'tokens_used': 0,
                        'api_duration_ms': 0
                    }
                    
                    used_heuristic = True
                    self.stats['heuristic_fallbacks'] += 1
                    
                    logger.info(
                        "✅ layout.heuristic.success",
                        extra={"components": len(components)}
                    )
                    
                except Exception as heuristic_error:
                    logger.error(
                        "❌ layout.heuristic.failed",
                        extra={"error": str(heuristic_error)},
                        exc_info=heuristic_error
                    )
                    
                    self.stats['failed'] += 1
                    raise LayoutGenerationError(
                        f"Both LLM and heuristic generation failed. "
                        f"LLM: {llm_error}, Heuristic: {heuristic_error}"
                    )
            
            # Resolve collisions
            components = await self._resolve_collisions(components)
            
            # Create layout definition
            layout = EnhancedLayoutDefinition(
                screen_id=screen_id,
                canvas=self._get_default_canvas(),
                components=components,
                layout_metadata=metadata
            )
            
            # Validate layout
            logger.info("🔍 layout.validation.starting")
            
            try:
                is_valid, warnings = await layout_validator.validate(layout)
                
                error_count = sum(1 for w in warnings if w.level == "error")
                warning_count = sum(1 for w in warnings if w.level == "warning")
                
                if not is_valid:
                    logger.error(
                        "❌ layout.validation.failed",
                        extra={
                            "errors": error_count,
                            "warnings": warning_count
                        }
                    )
                    # Don't fail - validation warnings are informational
                
                logger.info(
                    "✅ layout.validation.completed",
                    extra={
                        "warnings": warning_count,
                        "used_heuristic": used_heuristic
                    }
                )
                
            except Exception as validation_error:
                logger.warning(
                    "⚠️ layout.validation.error",
                    extra={"error": str(validation_error)}
                )
            
            # Update metadata
            metadata.update({
                'used_heuristic': used_heuristic,
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z"
            })
            
            self.stats['successful'] += 1
            
            logger.info(
                "🎉 layout.generation.completed",
                extra={
                    "screen": screen.name,
                    "components": len(components),
                    "used_heuristic": used_heuristic
                }
            )
            
            return layout, metadata
    
    async def _generate_with_llm(
        self,
        screen: ScreenDefinition,
        architecture: ArchitectureDesign
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate layout using LLM orchestrator with retries"""
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"🔄 layout.llm.attempt",
                    extra={
                        "attempt": attempt,
                        "screen": screen.name
                    }
                )
                
                # Determine primary action
                primary_action = "view content"
                if any('Button' in comp for comp in screen.components):
                    primary_action = "button interaction"
                elif any('Input' in comp for comp in screen.components):
                    primary_action = "text input"
                
                # Format prompt
                system_prompt, user_prompt = prompts.LAYOUT_GENERATE.format(
                    components=", ".join(settings.available_components),
                    prompt=f"Layout for {screen.name}",
                    screen_architecture=json.dumps({
                        'id': screen.id,
                        'name': screen.name,
                        'purpose': screen.purpose
                    }, indent=2),
                    required_components=", ".join(screen.components),
                    primary_action=primary_action
                )
                
                # Create messages
                messages = [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt)
                ]
                
                # Call LLM via orchestrator
                start_time = asyncio.get_event_loop().time()
                
                response = await self.orchestrator.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096
                )
                
                api_duration = int((asyncio.get_event_loop().time() - start_time) * 1000)
                
                logger.debug(
                    "layout.llm.response_received",
                    extra={
                        "response_length": len(response.content),
                        "api_duration_ms": api_duration,
                        "provider": response.provider.value
                    }
                )
                
                # Parse response
                layout_data = await self._parse_layout_json(response.content)
                
                # Build metadata
                metadata = {
                    'generation_method': 'llm',
                    'provider': response.provider.value,
                    'tokens_used': response.tokens_used,
                    'api_duration_ms': api_duration,
                    'screen_id': screen.id,
                    'screen_name': screen.name
                }
                
                return layout_data, metadata
                
            except Exception as e:
                last_error = e
                
                logger.warning(
                    f"⚠️ layout.llm.retry",
                    extra={
                        "attempt": attempt,
                        "error": str(e)[:200],
                        "will_retry": attempt < self.max_retries
                    }
                )
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "❌ layout.llm.exhausted",
                        extra={
                            "total_attempts": attempt,
                            "final_error": str(last_error)
                        }
                    )
                    raise last_error
        
        raise last_error or LayoutGenerationError("All retries failed")
    
    async def _parse_layout_json(self, response_text: str) -> Dict[str, Any]:
        """Parse layout JSON from LLM response"""
        
        # Remove markdown code blocks
        if response_text.startswith("```"):
            parts = response_text.split("```")
            if len(parts) >= 3:
                response_text = parts[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
        
        # Parse JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise LayoutGenerationError(f"Could not parse layout JSON: {e}")
    
    async def _convert_to_enhanced_components(
        self,
        components_data: List[Dict[str, Any]],
        screen_id: str
    ) -> List[EnhancedComponentDefinition]:
        """Convert LLM's component data to enhanced definitions"""
        
        enhanced_components = []
        
        for idx, comp_data in enumerate(components_data):
            try:
                comp_id = comp_data.get('id', f"comp_{screen_id}_{idx}")
                comp_type = comp_data.get('type')
                
                if comp_type not in settings.available_components:
                    logger.warning(f"Unsupported component type: {comp_type}, skipping")
                    continue
                
                # Convert properties to PropertyValue format
                properties = {}
                raw_properties = comp_data.get('properties', {})
                
                for key, value in raw_properties.items():
                    if isinstance(value, dict) and 'type' in value:
                        properties[key] = PropertyValue(**value)
                    else:
                        properties[key] = PropertyValue(type="literal", value=value)
                
                # Extract or generate position/size
                position = comp_data.get('position', {'x': 0, 'y': 0})
                constraints = comp_data.get('constraints', {})
                
                # Ensure style property
                if 'style' not in properties:
                    width, height = self.component_defaults.get(comp_type, (280, 44))
                    
                    width_str = constraints.get('width', 'auto')
                    if width_str != 'auto':
                        try:
                            if '%' in str(width_str):
                                percentage = float(str(width_str).rstrip('%'))
                                width = int(self.canvas_width * percentage / 100)
                            else:
                                width = int(str(width_str).rstrip('px'))
                        except:
                            pass
                    
                    height = constraints.get('height', height)
                    
                    properties['style'] = PropertyValue(
                        type="literal",
                        value={
                            'left': position.get('x', 0),
                            'top': position.get('y', 0),
                            'width': width,
                            'height': height
                        }
                    )
                
                enhanced = EnhancedComponentDefinition(
                    component_id=comp_id,
                    component_type=comp_type,
                    properties=properties,
                    z_index=idx,
                    parent_id=None,
                    children_ids=[]
                )
                
                enhanced_components.append(enhanced)
                
            except Exception as e:
                logger.warning(f"Failed to convert component {idx}: {e}")
                continue
        
        return enhanced_components
    
    async def _generate_heuristic_layout(
        self,
        screen: ScreenDefinition
    ) -> List[EnhancedComponentDefinition]:
        """Generate layout using heuristic templates"""
        
        logger.info(
            "🛡️ layout.heuristic.generating",
            extra={"screen": screen.name}
        )
        
        components = []
        current_y = self.safe_area_top + 20
        
        for idx, comp_type in enumerate(screen.components):
            if comp_type not in settings.available_components:
                continue
            
            width, height = self.component_defaults.get(comp_type, (280, 44))
            x = (self.canvas_width - width) // 2
            
            comp_id = f"{comp_type.lower()}_{idx}"
            
            properties = {
                'style': PropertyValue(
                    type="literal",
                    value={
                        'left': x,
                        'top': current_y,
                        'width': width,
                        'height': height
                    }
                )
            }
            
            # Add component-specific properties
            if comp_type == 'Button':
                properties['value'] = PropertyValue(type="literal", value="Click Me")
            elif comp_type == 'Text':
                properties['value'] = PropertyValue(type="literal", value="Text")
            elif comp_type == 'InputText':
                properties['placeholder'] = PropertyValue(type="literal", value="Enter text")
            
            component = EnhancedComponentDefinition(
                component_id=comp_id,
                component_type=comp_type,
                properties=properties,
                z_index=idx
            )
            
            components.append(component)
            current_y += height + 16
        
        logger.info(
            "layout.heuristic.generated",
            extra={"components": len(components)}
        )
        
        return components
    
    async def _resolve_collisions(
        self,
        components: List[EnhancedComponentDefinition]
    ) -> List[EnhancedComponentDefinition]:
        """Detect and resolve component collisions"""
        
        if len(components) <= 1:
            return components
        
        logger.debug("Checking for collisions...")
        
        # Check for collisions
        has_collisions = False
        for i, comp1 in enumerate(components):
            bounds1 = self._get_component_bounds(comp1)
            if not bounds1:
                continue
            
            for comp2 in components[i+1:]:
                bounds2 = self._get_component_bounds(comp2)
                if not bounds2:
                    continue
                
                if self._rectangles_overlap(bounds1, bounds2):
                    has_collisions = True
                    break
            
            if has_collisions:
                break
        
        if not has_collisions:
            logger.debug("No collisions detected")
            return components
        
        logger.info(f"⚠️ Collisions detected, resolving...")
        self.stats['collisions_resolved'] += 1
        
        # Simple vertical stack layout
        current_y = self.safe_area_top + 20
        
        for component in components:
            style_prop = component.properties.get('style')
            if not style_prop or style_prop.type != "literal":
                continue
            
            style = style_prop.value
            width = style.get('width', 280)
            height = style.get('height', 44)
            
            # Center horizontally
            x = (self.canvas_width - width) // 2
            
            # Update position
            style['left'] = x
            style['top'] = current_y
            
            # Move to next position
            current_y += height + 16
        
        logger.info(f"✅ Collisions resolved: stacked vertically")
        
        return components
    
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
    
    def _get_default_canvas(self) -> Dict[str, Any]:
        """Get default canvas configuration"""
        return {
            "width": self.canvas_width,
            "height": self.canvas_height,
            "background_color": "#FFFFFF",
            "safe_area_insets": {
                "top": self.safe_area_top,
                "bottom": self.safe_area_bottom,
                "left": 0,
                "right": 0
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        total = self.stats['total_requests']
        
        return {
            **self.stats,
            'success_rate': (self.stats['successful'] / total * 100) if total > 0 else 0,
            'heuristic_fallback_rate': (self.stats['heuristic_fallbacks'] / total * 100) if total > 0 else 0,
            'llama3_success_rate': (self.stats['llama3_successes'] / total * 100) if total > 0 else 0,
            'collisions_resolved': self.stats['collisions_resolved']
        }


# Global layout generator instance
layout_generator = LayoutGenerator()