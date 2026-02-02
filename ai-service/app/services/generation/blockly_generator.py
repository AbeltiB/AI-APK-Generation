"""
Blockly Generator - Phase 3
Uses LLM Orchestrator (Llama3 → Heuristic fallback)

Generates visual programming blocks for app logic.
"""
import json
import asyncio
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate Blockly blocks for application.
        
        Args:
            architecture: Complete architecture design
            layouts: Map of screen_id -> layout
            
        Returns:
            Tuple of (blockly_definition, metadata)
            
        Raises:
            BlocklyGenerationError: If generation fails
        """
        self.stats['total_requests'] += 1
        
        with log_context(operation="blockly_generation"):
            logger.info(
                "🧩 blockly.generation.started",
                extra={
                    "screens": len(architecture.screens),
                    "layouts": len(layouts)
                }
            )
            
            # Try LLM first
            blockly = None
            metadata = {}
            used_heuristic = False
            
            try:
                blockly, metadata = await self._generate_with_llm(
                    architecture=architecture,
                    layouts=layouts
                )
                
                self.stats['llama3_successes'] += 1
                logger.info(
                    "✅ blockly.llm.success",
                    extra={
                        "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                        "provider": metadata.get('provider', 'llama3')
                    }
                )
                
            except Exception as llm_error:
                logger.warning(
                    "⚠️ blockly.llm.failed",
                    extra={"error": str(llm_error)},
                    exc_info=llm_error
                )
                
                # Fall back to heuristic
                logger.info("🛡️ blockly.fallback.initiating")
                
                try:
                    blockly = await self._generate_heuristic_blockly(
                        architecture=architecture,
                        layouts=layouts
                    )
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
                        "✅ blockly.heuristic.success",
                        extra={"blocks": len(blockly.get('blocks', {}).get('blocks', []))}
                    )
                    
                except Exception as heuristic_error:
                    logger.error(
                        "❌ blockly.heuristic.failed",
                        extra={"error": str(heuristic_error)},
                        exc_info=heuristic_error
                    )
                    
                    self.stats['failed'] += 1
                    raise BlocklyGenerationError(
                        f"Both LLM and heuristic generation failed. "
                        f"LLM: {llm_error}, Heuristic: {heuristic_error}"
                    )
            
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
                    # Don't fail - validation warnings are informational
                
                logger.info(
                    "✅ blockly.validation.completed",
                    extra={
                        "warnings": warning_count,
                        "used_heuristic": used_heuristic
                    }
                )
                
            except Exception as validation_error:
                logger.warning(
                    "⚠️ blockly.validation.error",
                    extra={"error": str(validation_error)}
                )
            
            # Update metadata
            metadata.update({
                'used_heuristic': used_heuristic,
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z"
            })
            
            self.stats['successful'] += 1
            
            logger.info(
                "🎉 blockly.generation.completed",
                extra={
                    "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                    "variables": len(blockly.get('variables', [])),
                    "used_heuristic": used_heuristic
                }
            )
            
            return blockly, metadata
    
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
                    f"🔄 blockly.llm.attempt",
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
                
                # Parse response
                blockly_data = await self._parse_blockly_json(response.content)
                
                # Build metadata
                metadata = {
                    'generation_method': 'llm',
                    'provider': response.provider.value,
                    'tokens_used': response.tokens_used,
                    'api_duration_ms': api_duration
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
                            "final_error": str(last_error)
                        }
                    )
                    raise last_error
        
        raise last_error or BlocklyGenerationError("All retries failed")
    
    async def _parse_blockly_json(self, response_text: str) -> Dict[str, Any]:
        """Parse Blockly JSON from LLM response"""
        
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
            raise BlocklyGenerationError(f"Could not parse Blockly JSON: {e}")
    
    def _extract_component_events(
        self,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> List[Dict[str, str]]:
        """Extract component events from layouts"""
        
        events = []
        
        for screen_id, layout in layouts.items():
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
            extra={"screens": len(architecture.screens)}
        )
        
        # Extract component events
        component_events = self._extract_component_events(layouts)
        
        # Generate blocks
        blocks = []
        variables = []
        
        # Create event handlers for interactive components
        for idx, event_info in enumerate(component_events):
            block_id = f"event_{idx + 1}"
            
            # Create event block
            event_block = {
                'type': 'component_event',
                'id': block_id,
                'fields': {
                    'COMPONENT': event_info['component_id'],
                    'EVENT': event_info['event']
                }
            }
            
            blocks.append(event_block)
        
        # Add variables from state management
        for idx, state in enumerate(architecture.state_management):
            variables.append({
                'name': state.name,
                'id': f"var_{idx + 1}",
                'type': ''
            })
        
        blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': blocks
            },
            'variables': variables,
            'custom_blocks': []
        }
        
        logger.info(
            "blockly.heuristic.generated",
            extra={
                "blocks": len(blocks),
                "variables": len(variables)
            }
        )
        
        return blockly
    
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