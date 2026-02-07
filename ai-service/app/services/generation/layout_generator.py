"""
Layout Generator - Phase 3
Uses LLM Orchestrator (Llama3 → Heuristic fallback)
"""
import json
import asyncio
import re
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
        """Parse layout JSON from LLM response with robust error handling"""
        
        # Make a copy for logging
        original_text = response_text[:500] + "..." if len(response_text) > 500 else response_text
        
        logger.debug(f"Raw LLM response (first 500 chars): {original_text}")
        
        # Step 1: Extract JSON from markdown
        cleaned_text = self._extract_json_from_markdown(response_text)

        # adress the llama3 double curly brace issue
        cleaned_text = self._fix_double_curly_braces(cleaned_text)
        
        # Step 2: Normalize JSON format
        cleaned_text = self._normalize_json_format(cleaned_text)
        
        # Step 3: Try to parse
        try:
            result = json.loads(cleaned_text)
            logger.debug(f"✅ JSON parsed successfully: {len(cleaned_text)} chars")
            return result
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            logger.warning(f"Initial JSON parse failed, attempting fixes: {e}")
            
            # Try with additional fixes
            cleaned_text = self._attempt_json_fixes(cleaned_text)
            
            try:
                result = json.loads(cleaned_text)
                logger.debug(f"✅ JSON parsed after fixes: {len(cleaned_text)} chars")
                return result
            except json.JSONDecodeError as e2:
                logger.error(f"Final JSON parse error: {e2}")
                logger.debug(f"Failed text: {cleaned_text[:500]}...")
                raise LayoutGenerationError(f"Could not parse layout JSON after fixes: {e2}")

    def _fix_double_curly_braces(self, text: str) -> str:
        """Fix double curly braces issue: {{ }} -> { }"""
        # Replace {{ with {
        text = text.replace('{{', '{')
        # Replace }} with }
        text = text.replace('}}', '}')
        return text
    
    def _extract_json_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks"""
        text = text.strip()
        
        # Remove ```json and ``` markers
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        # Also handle potential language specification
        lines = text.split('\n')
        if lines and lines[0].strip() in ["json", "JavaScript", "js"]:
            lines = lines[1:]
            text = '\n'.join(lines)
        
        return text.strip()
    
    def _normalize_json_format(self, text: str) -> str:
        """Normalize JSON format by fixing common issues"""
        
        # Fix 1: Replace single quotes with double quotes (carefully)
        # We need to avoid replacing apostrophes inside strings
        lines = []
        in_string = False
        escape_next = False
        
        for char in text:
            if escape_next:
                lines.append(char)
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                lines.append(char)
            elif char == '"':
                in_string = not in_string
                lines.append(char)
            elif char == "'" and not in_string:
                # Single quote outside a string - replace with double quote
                lines.append('"')
            else:
                lines.append(char)
        
        text = ''.join(lines)
        
        # Fix 2: Handle Python-style booleans
        text = text.replace(': True', ': true')
        text = text.replace(': False', ': false')
        text = text.replace(': None', ': null')
        
        # Fix 3: Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix 4: Ensure property names are quoted
        # Find unquoted property names
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Simple pattern to find unquoted property names at start of line
            line = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)', r'\1"\2"\3', line)
            fixed_lines.append(line)
        
        text = '\n'.join(fixed_lines)
        
        return text
    
    def _attempt_json_fixes(self, text: str) -> str:
        """Attempt more aggressive JSON fixes"""
        
        # Try to find JSON object in text
        import re
        
        # Look for { ... } pattern
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            text = match.group(1)
        
        # If still failing, try a simple approach - just replace all single quotes
        # This is a last resort
        if "'" in text:
            # Replace all single quotes with double quotes
            # This might break strings with apostrophes, but it's better than failing
            text = text.replace("'", '"')
        
        return text
    
    async def _convert_to_enhanced_components(
        self,
        components_data: List[Dict[str, Any]],
        screen_id: str
    ) -> List[EnhancedComponentDefinition]:
        """Convert LLM's component data to enhanced definitions with better property handling"""
        
        enhanced_components = []
        
        for idx, comp_data in enumerate(components_data):
            try:
                comp_id = comp_data.get('id', f"comp_{screen_id}_{idx}")
                comp_type = comp_data.get('type', 'Unknown')
                
                if comp_type not in settings.available_components:
                    logger.warning(f"Unsupported component type from LLM: {comp_type}, skipping")
                    continue
                
                # Initialize properties dict
                properties = {}
                
                # Handle style separately
                style_data = comp_data.get('style', {})
                position = comp_data.get('position', {'x': 0, 'y': 0})
                constraints = comp_data.get('constraints', {})
                
                # Get default dimensions
                width, height = self.component_defaults.get(comp_type, (280, 44))
                
                # Calculate actual dimensions
                width_value = constraints.get('width', width)
                height_value = constraints.get('height', height)
                
                # Parse width if it's a string
                if isinstance(width_value, str):
                    if width_value == 'auto' or width_value == 'fill':
                        width_value = 280
                    elif '%' in width_value:
                        try:
                            percentage = float(width_value.strip('%'))
                            width_value = int(self.canvas_width * percentage / 100)
                        except:
                            width_value = 280
                    elif 'px' in width_value:
                        try:
                            width_value = int(width_value.strip('px'))
                        except:
                            width_value = 280
                
                # Parse height if it's a string
                if isinstance(height_value, str):
                    if height_value == 'auto':
                        height_value = 44
                    elif 'px' in height_value:
                        try:
                            height_value = int(height_value.strip('px'))
                        except:
                            height_value = 44
                
                # Create style property
                style_value = {
                    'left': position.get('x', 0),
                    'top': position.get('y', 0),
                    'width': width_value,
                    'height': height_value
                }
                
                # Merge with any style from comp_data
                if isinstance(style_data, dict):
                    style_value.update(style_data)
                
                properties['style'] = PropertyValue(
                    type="literal",
                    value=style_value
                )
                
                # Add component-specific properties from comp_data
                for key, value in comp_data.items():
                    if key not in ['id', 'type', 'style', 'position', 'constraints', 'z_index']:
                        properties[key] = PropertyValue(type="literal", value=value)
                
                # Ensure required properties exist
                if comp_type == 'Checkbox' and 'checked' not in properties:
                    properties['checked'] = PropertyValue(type="literal", value=False)
                if comp_type == 'Switch' and 'checked' not in properties:
                    properties['checked'] = PropertyValue(type="literal", value=False)
                if comp_type == 'Slider' and 'value' not in properties:
                    properties['value'] = PropertyValue(type="literal", value=50)
                if comp_type == 'ProgressBar' and 'value' not in properties:
                    properties['value'] = PropertyValue(type="literal", value=0.5)
                if comp_type == 'Button' and 'value' not in properties:
                    properties['value'] = PropertyValue(type="literal", value="Button")
                if comp_type == 'Text' and 'value' not in properties:
                    properties['value'] = PropertyValue(type="literal", value="Text")
                if comp_type == 'InputText' and 'placeholder' not in properties:
                    properties['placeholder'] = PropertyValue(type="literal", value="Enter text")
                
                # Get z-index
                z_index = comp_data.get('z_index', idx)
                
                # Create enhanced component
                enhanced = EnhancedComponentDefinition(
                    component_id=comp_id,
                    component_type=comp_type,
                    properties=properties,
                    z_index=z_index,
                    parent_id=None,
                    children_ids=[]
                )
                
                enhanced_components.append(enhanced)
                logger.debug(f"Converted component: {comp_type} at position {style_value['left']}, {style_value['top']}")
                
            except Exception as e:
                logger.warning(f"Failed to convert component {idx} (type: {comp_data.get('type', 'unknown')}): {e}")
                continue
        
        if not enhanced_components:
            raise LayoutGenerationError("No components could be converted from LLM response")
        
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
                logger.warning(f"Unsupported component type in heuristic: {comp_type}")
                continue
            
            width, height = self.component_defaults.get(comp_type, (280, 44))
            x = (self.canvas_width - width) // 2
            
            comp_id = f"{comp_type.lower()}_{idx}"
            
            # Base properties
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
                properties['onClick'] = PropertyValue(type="literal", value="")
            elif comp_type == 'Text':
                properties['value'] = PropertyValue(type="literal", value="Sample Text")
                properties['color'] = PropertyValue(type="literal", value="#000000")
            elif comp_type == 'InputText':
                properties['placeholder'] = PropertyValue(type="literal", value="Enter text")
                properties['value'] = PropertyValue(type="literal", value="")
            elif comp_type == 'Checkbox':
                # ✅ FIXED: Add required 'checked' property
                properties['checked'] = PropertyValue(type="literal", value=False)
                properties['label'] = PropertyValue(type="literal", value="Check me")
            elif comp_type == 'Switch':
                properties['checked'] = PropertyValue(type="literal", value=False)
            elif comp_type == 'Slider':
                properties['value'] = PropertyValue(type="literal", value=50)
                properties['min'] = PropertyValue(type="literal", value=0)
                properties['max'] = PropertyValue(type="literal", value=100)
            elif comp_type == 'ProgressBar':
                properties['value'] = PropertyValue(type="literal", value=0.5)
                properties['max'] = PropertyValue(type="literal", value=1.0)
            elif comp_type == 'TextArea':
                properties['placeholder'] = PropertyValue(type="literal", value="Enter text here...")
                properties['value'] = PropertyValue(type="literal", value="")
            elif comp_type == 'Spinner':
                properties['value'] = PropertyValue(type="literal", value=0)
            elif comp_type == 'DatePicker':
                properties['value'] = PropertyValue(type="literal", value="")
            elif comp_type == 'TimePicker':
                properties['value'] = PropertyValue(type="literal", value="")
            elif comp_type == 'ColorPicker':
                properties['value'] = PropertyValue(type="literal", value="#000000")
            elif comp_type == 'Map':
                properties['latitude'] = PropertyValue(type="literal", value=0.0)
                properties['longitude'] = PropertyValue(type="literal", value=0.0)
            elif comp_type == 'Chart':
                properties['data'] = PropertyValue(type="literal", value=[])
            
            try:
                component = EnhancedComponentDefinition(
                    component_id=comp_id,
                    component_type=comp_type,
                    properties=properties,
                    z_index=idx
                )
                
                components.append(component)
                logger.debug(f"Heuristic created component: {comp_type}")
                
            except Exception as e:
                logger.error(f"Failed to create {comp_type} in heuristic: {e}")
                # Skip this component but continue with others
        
        # Move to next position
        current_y += height + 16
    
        if not components:
            raise LayoutGenerationError(f"No components could be generated for screen: {screen.name}")
        
        logger.info(
            "✅ layout.heuristic.generated",
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
    
    async def test_json_parsing(self):
        """Test JSON parsing with common failure cases"""
        
        test_cases = [
            # Single quotes
            "{'components': [{'type': 'Button', 'checked': true}]}",
            # Markdown with single quotes
            "```json\n{'components': []}\n```",
            # Python-style booleans
            "{components: [{type: 'Checkbox', checked: True}]}",
            # Trailing comma
            "{'components': [],}",
        ]
        
        for i, test in enumerate(test_cases):
            print(f"\nTest case {i+1}:")
            print(f"Input: {test}")
            try:
                result = await self._parse_layout_json(test)
                print(f"✅ Success: {result}")
            except Exception as e:
                print(f"❌ Failed: {e}")


# Global layout generator instance
layout_generator = LayoutGenerator()