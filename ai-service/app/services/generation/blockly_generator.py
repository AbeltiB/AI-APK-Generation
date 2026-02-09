"""
Blockly Generator - Phase 3
Uses LLM Orchestrator (Llama3 → Heuristic fallback)

Generates visual programming blocks for app logic.
"""
import json
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas.architecture import ArchitectureDesign
from app.models.schemas.layout import EnhancedLayoutDefinition
from app.models.prompts import prompts
from app.services.generation.blockly_validator import blockly_validator
from app.llm.orchestrator import LLMOrchestrator
from app.llm.base import LLMMessage
from app.utils.logging import get_logger, log_context, trace_async

logger = get_logger(__name__)


class BlocklyGenerationError(Exception):
    """Base exception for Blockly generation errors"""
    pass


class BlocklyGenerator:
    """
    Phase 3 Blockly Generator using LLM Orchestrator.
    
    Generation Flow:
    1. 🎯 Try Llama3 via orchestrator
    2. 🔄 Retry with corrections if needed
    3. 🛡️ Fall back to heuristic if all retries fail
    4. ✅ Validate result
    
    Features:
    - Llama3 as primary LLM
    - Automatic heuristic template fallback
    - Comprehensive validation
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
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'heuristic_fallbacks': 0,
            'llama3_successes': 0
        }
        
        logger.info(
            "blockly.generator.initialized",
            extra={
                "llm_provider": "llama3",
                "heuristic_fallback_enabled": True
            }
        )
    
    @trace_async("blockly.generation")
    async def generate(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Dict[str, Any]:
        """
        Generate Blockly blocks for application.
        
        Args:
            architecture: Complete architecture design
            layouts: Map of screen_id -> layout
            
        Returns:
            Blockly definition dictionary
            
        Raises:
            BlocklyGenerationError: If generation fails
        """
        self.stats['total_requests'] += 1
        
        with log_context(operation="blockly_generation"):
            logger.info(
                "🧩 blockly.generation.started",
                extra={
                    "screens": len(architecture.screens) if architecture and hasattr(architecture, 'screens') else 0,
                    "layouts": len(layouts) if layouts else 0
                }
            )
            
            # Validate inputs
            if not architecture:
                logger.error("🚨 blockly.generation.architecture_missing")
                return self._create_empty_blockly("No architecture provided")
            
            if not layouts:
                logger.warning("⚠️ blockly.generation.no_layouts")
            
            # Initialize blockly structure with defaults
            blockly = None
            used_heuristic = False
            generation_metadata = {
                'generation_method': 'unknown',
                'provider': 'unknown',
                'tokens_used': 0,
                'api_duration_ms': 0,
                'used_heuristic': False,
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z"
            }
            
            try:
                # Try LLM first
                try:
                    blockly, llm_metadata = await self._generate_with_llm(
                        architecture=architecture,
                        layouts=layouts
                    )
                    
                    if blockly and isinstance(blockly, dict):
                        generation_metadata.update(llm_metadata)
                        generation_metadata['generation_method'] = 'llm'
                        self.stats['llama3_successes'] += 1
                        logger.info(
                            "✅ blockly.llm.success",
                            extra={
                                "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                                "provider": llm_metadata.get('provider', 'llama3')
                            }
                        )
                    else:
                        raise BlocklyGenerationError("LLM generated invalid blockly structure")
                        
                except Exception as llm_error:
                    logger.warning(
                        "⚠️ blockly.llm.failed",
                        extra={"error": str(llm_error)[:200]},
                        exc_info=False  # Don't log full traceback for expected failures
                    )
                    
                    # Fall back to heuristic
                    logger.info("🛡️ blockly.fallback.initiating")
                    
                    try:
                        blockly = await self._generate_heuristic_blockly(
                            architecture=architecture,
                            layouts=layouts
                        )
                        
                        if blockly and isinstance(blockly, dict):
                            generation_metadata.update({
                                'generation_method': 'heuristic',
                                'provider': 'heuristic',
                                'fallback_reason': str(llm_error)[:200],
                                'used_heuristic': True
                            })
                            
                            used_heuristic = True
                            self.stats['heuristic_fallbacks'] += 1
                            
                            logger.info(
                                "✅ blockly.heuristic.success",
                                extra={
                                    "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                                    "variables": len(blockly.get('variables', []))
                                }
                            )
                        else:
                            raise BlocklyGenerationError("Heuristic generated invalid blockly structure")
                            
                    except Exception as heuristic_error:
                        logger.error(
                            "❌ blockly.heuristic.failed",
                            extra={"error": str(heuristic_error)},
                            exc_info=heuristic_error
                        )
                        
                        # Create minimal fallback blockly
                        blockly = self._create_minimal_blockly(architecture, layouts)
                        generation_metadata.update({
                            'generation_method': 'fallback',
                            'provider': 'fallback',
                            'fallback_reason': f"LLM: {llm_error}, Heuristic: {heuristic_error}",
                            'used_heuristic': True
                        })
                        
                        used_heuristic = True
                        self.stats['heuristic_fallbacks'] += 1
                        
                        logger.warning(
                            "⚠️ blockly.using_minimal_fallback",
                            extra={"reason": "Both LLM and heuristic failed"}
                        )
                
                # Ensure blockly is a valid dictionary
                if not blockly or not isinstance(blockly, dict):
                    logger.error("🚨 blockly.generation.invalid_output", extra={"type": type(blockly).__name__ if blockly else "None"})
                    blockly = self._create_empty_blockly("Generation returned invalid type")
                    generation_metadata['generation_method'] = 'emergency_fallback'
                
                # Add metadata to blockly structure
                if isinstance(blockly, dict):
                    blockly['metadata'] = generation_metadata
                    
                    # Ensure required structure exists
                    if 'blocks' not in blockly:
                        blockly['blocks'] = {'blocks': []}
                    elif not isinstance(blockly['blocks'], dict):
                        blockly['blocks'] = {'blocks': []}
                    
                    if 'variables' not in blockly:
                        blockly['variables'] = []
                    elif not isinstance(blockly['variables'], list):
                        blockly['variables'] = []
                    
                    if 'custom_blocks' not in blockly:
                        blockly['custom_blocks'] = []
                
                # Validate Blockly
                logger.info("🔍 blockly.validation.starting")
                
                try:
                    is_valid, warnings = await blockly_validator.validate(blockly)
                    
                    error_count = sum(1 for w in warnings if w.level == "error")
                    warning_count = sum(1 for w in warnings if w.level == "warning")
                    
                    if not is_valid:
                        logger.error(
                            "❌ blockly.validation.failed",
                            extra={
                                "errors": error_count,
                                "warnings": warning_count
                            }
                        )
                        # Add validation warnings to metadata
                        blockly['metadata']['validation_errors'] = error_count
                        blockly['metadata']['validation_warnings'] = warning_count
                    else:
                        logger.info(
                            "✅ blockly.validation.completed",
                            extra={
                                "warnings": warning_count,
                                "errors": error_count,
                                "used_heuristic": used_heuristic
                            }
                        )
                        
                except Exception as validation_error:
                    logger.warning(
                        "⚠️ blockly.validation.error",
                        extra={"error": str(validation_error)}
                    )
                    # Add validation error to metadata but don't fail
                    blockly['metadata']['validation_error'] = str(validation_error)
                
                self.stats['successful'] += 1
                
                logger.info(
                    "🎉 blockly.generation.completed",
                    extra={
                        "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                        "variables": len(blockly.get('variables', [])),
                        "used_heuristic": used_heuristic,
                        "method": generation_metadata.get('generation_method', 'unknown')
                    }
                )
                
                return blockly
                
            except Exception as e:
                logger.error(
                    "💥 blockly.generation.critical_error",
                    extra={
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    },
                    exc_info=e
                )
                
                self.stats['failed'] += 1
                
                # Always return a valid blockly structure, even if empty
                fallback_blockly = self._create_empty_blockly(f"Critical error: {str(e)[:200]}")
                fallback_blockly['metadata'] = {
                    'generation_method': 'critical_error_fallback',
                    'provider': 'fallback',
                    'error': str(e),
                    'generated_at': datetime.now(timezone.utc).isoformat() + "Z",
                    'used_heuristic': True
                }
                
                return fallback_blockly
    
    async def _generate_with_llm(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate Blockly using LLM orchestrator with retries"""
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "🔄 blockly.llm.attempt",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.max_retries
                    }
                )
                
                # Extract component events from layouts
                component_events = self._extract_component_events(layouts)
                
                # Format prompt
                system_prompt, user_prompt = prompts.BLOCKLY_GENERATE.format(
                    architecture=json.dumps(architecture.dict(), indent=2),
                    layout=json.dumps({
                        screen_id: layout.dict() 
                        for screen_id, layout in layouts.items()
                    }, indent=2),
                    component_events=json.dumps(component_events, indent=2)
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
                    "blockly.llm.response_received",
                    extra={
                        "response_length": len(response.content),
                        "api_duration_ms": api_duration,
                        "provider": response.provider.value
                    }
                )
                
                # Log first 500 chars for debugging
                if response.content and len(response.content) > 0:
                    logger.debug(
                        f"Raw LLM response (first 500 chars): {response.content[:500]}",
                        extra={"response_preview": response.content[:500]}
                    )
                
                # Parse response
                blockly_data = await self._parse_blockly_json(response.content)
                
                # Validate basic structure
                if not isinstance(blockly_data, dict):
                    raise BlocklyGenerationError(f"LLM response is not a dict: {type(blockly_data)}")
                
                # Build metadata
                metadata = {
                    'generation_method': 'llm',
                    'provider': response.provider.value,
                    'tokens_used': response.tokens_used,
                    'api_duration_ms': api_duration,
                    'attempt': attempt,
                    'response_length': len(response.content)
                }
                
                return blockly_data, metadata
                
            except Exception as e:
                last_error = e
                
                logger.warning(
                    f"⚠️ blockly.llm.retry",
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
                        "❌ blockly.llm.exhausted",
                        extra={
                            "total_attempts": attempt,
                            "final_error": str(last_error)[:500]
                        }
                    )
                    raise BlocklyGenerationError(f"All {attempt} retries failed: {last_error}")
        
        raise BlocklyGenerationError(f"All retries failed: {last_error}")
    
    async def _parse_blockly_json(self, response_text: str) -> Dict[str, Any]:
        """Parse Blockly JSON from LLM response"""
        
        if not response_text or response_text.strip() == "":
            raise BlocklyGenerationError("Empty response from LLM")
        
        # Clean the response text
        cleaned_text = response_text.strip()
        
        # Fix common JSON issues from logs: {{ -> { and }} -> }
        if cleaned_text.startswith("{{") and cleaned_text.endswith("}}"):
            logger.debug("Fixing double curly braces in JSON response")
            cleaned_text = cleaned_text[1:-1]  # Remove outer { and }
        
        # Remove markdown code blocks
        if cleaned_text.startswith("```"):
            lines = cleaned_text.split("\n")
            if lines[0].startswith("```json"):
                # Remove ```json and closing ```
                cleaned_text = "\n".join(lines[1:-1])
            elif lines[0].startswith("```"):
                # Remove any ``` blocks
                cleaned_text = "\n".join(lines[1:-1])
        
        # Try to parse JSON
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from malformed response
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Problematic JSON (first 1000 chars): {cleaned_text[:1000]}")
            
            # Try to find JSON object/array in the text
            try:
                # Look for { ... } pattern
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = cleaned_text[start_idx:end_idx + 1]
                    return json.loads(json_str)
            except:
                pass
            
            # Try to parse line by line to find valid JSON
            lines = cleaned_text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('{'):
                    # Try from this line to end
                    try:
                        return json.loads('\n'.join(lines[i:]))
                    except:
                        continue
            
            raise BlocklyGenerationError(f"Could not parse Blockly JSON: {e}. Response preview: {cleaned_text[:200]}")
    
    def _extract_component_events(
        self,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> List[Dict[str, str]]:
        """Extract component events from layouts"""
        
        events = []
        
        if not layouts:
            return events
            
        for screen_id, layout in layouts.items():
            if not hasattr(layout, 'components') or not layout.components:
                continue
                
            for component in layout.components:
                # Interactive components
                if component.component_type in ['Button', 'Switch', 'Checkbox']:
                    events.append({
                        'screen_id': screen_id,
                        'component_id': component.component_id,
                        'component_type': component.component_type,
                        'event': 'onPress' if component.component_type == 'Button' else 'onToggle'
                    })
                elif component.component_type in ['InputText', 'TextArea']:
                    events.append({
                        'screen_id': screen_id,
                        'component_id': component.component_id,
                        'component_type': component.component_type,
                        'event': 'onChange'
                    })
        
        return events
    
    async def _generate_heuristic_blockly(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Dict[str, Any]:
        """Generate Blockly using heuristic templates"""
        
        logger.info(
            "🛡️ blockly.heuristic.generating",
            extra={"screens": len(architecture.screens) if architecture and hasattr(architecture, 'screens') else 0}
        )
        
        # Extract component events
        component_events = self._extract_component_events(layouts)
        
        # Generate blocks
        blocks = []
        variables = []
        custom_blocks = []
        
        # Create event handlers for interactive components
        for idx, event_info in enumerate(component_events):
            block_id = f"event_{idx + 1}"
            
            # Create event block
            event_block = {
                'type': 'component_event',
                'id': block_id,
                'x': 20 + (idx * 200),
                'y': 20 + (idx * 100),
                'fields': {
                    'COMPONENT': event_info['component_id'],
                    'EVENT': event_info['event'],
                    'SCREEN': event_info['screen_id']
                },
                'next': None
            }
            
            blocks.append(event_block)
        
        # Add variables from state management
        if hasattr(architecture, 'state_management') and architecture.state_management:
            for idx, state in enumerate(architecture.state_management):
                variables.append({
                    'name': state.name,
                    'id': f"var_{idx + 1}",
                    'type': 'String' if 'text' in state.name.lower() else 'Number'
                })
        else:
            # Default variables
            variables = [
                {'name': 'user_input', 'id': 'var_1', 'type': 'String'},
                {'name': 'counter', 'id': 'var_2', 'type': 'Number'}
            ]
        
        # Add navigation blocks for multi-screen apps
        if hasattr(architecture, 'screens') and len(architecture.screens) > 1:
            nav_block = {
                'type': 'navigate_screen',
                'id': 'nav_1',
                'x': 20,
                'y': 300,
                'fields': {
                    'SCREEN': architecture.screens[0].screen_id if architecture.screens else 'home'
                }
            }
            blocks.append(nav_block)
        
        blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': blocks
            },
            'variables': variables,
            'custom_blocks': custom_blocks,
            'heuristic_generated': True
        }
        
        logger.info(
            "blockly.heuristic.generated",
            extra={
                "blocks": len(blocks),
                "variables": len(variables)
            }
        )
        
        return blockly
    
    def _create_minimal_blockly(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Dict[str, Any]:
        """Create minimal blockly structure when all else fails"""
        
        # Extract component events
        component_events = self._extract_component_events(layouts)
        
        blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': [
                    {
                        'type': 'app_start',
                        'id': 'start_1',
                        'x': 20,
                        'y': 20,
                        'fields': {'APP_NAME': architecture.app_name if hasattr(architecture, 'app_name') else 'App'}
                    }
                ]
            },
            'variables': [
                {'name': 'app_state', 'id': 'var_1', 'type': 'String'},
                {'name': 'user_data', 'id': 'var_2', 'type': 'String'}
            ],
            'custom_blocks': [],
            'minimal_fallback': True,
            'component_events_count': len(component_events)
        }
        
        return blockly
    
    def _create_empty_blockly(self, reason: str = "") -> Dict[str, Any]:
        """Create an empty but valid blockly structure"""
        
        return {
            'blocks': {
                'languageVersion': 0,
                'blocks': []
            },
            'variables': [],
            'custom_blocks': [],
            'empty_fallback': True,
            'fallback_reason': reason,
            'metadata': {
                'generation_method': 'empty_fallback',
                'provider': 'fallback',
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z"
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        total = self.stats['total_requests']
        
        return {
            **self.stats,
            'success_rate': (self.stats['successful'] / total * 100) if total > 0 else 0,
            'heuristic_fallback_rate': (self.stats['heuristic_fallbacks'] / total * 100) if total > 0 else 0,
            'llama3_success_rate': (self.stats['llama3_successes'] / total * 100) if total > 0 else 0
        }


# Global blockly generator instance
blockly_generator = BlocklyGenerator()