"""
AI Pipeline - Llama3 Version

9-stage pipeline for AI-powered app generation:
1. Rate Limit Check
2. Input Validation
3. Cache Check (Semantic)
4. Intent Analysis (Llama3)
5. Context Building
6. Architecture Generation (Llama3)
7. Layout Generation (Llama3)
8. Blockly Generation (Llama3)
9. Cache Save

All generation uses Llama3 as primary provider with heuristic fallback.
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.models.schemas.input_output import AIRequest, ProgressUpdate, ErrorResponse, CompleteResponse
from app.models.schemas.architecture import ArchitectureDesign
from app.models.schemas.layout import EnhancedLayoutDefinition
from app.core.messaging import queue_manager
from app.core.database import db_manager
from app.core.cache import cache_manager
from app.utils.logging import get_logger, log_context
from app.utils.rate_limiter import rate_limiter

# Import generators and analyzers
from app.services.analysis.intent_analyzer import intent_analyzer
from app.services.analysis.context_builder import context_builder
from app.services.generation.architecture_generator import architecture_generator
from app.services.generation.layout_generator import layout_generator
from app.services.generation.blockly_generator import blockly_generator
from app.services.generation.cache_manager import semantic_cache

logger = get_logger(__name__)


class PipelineStage:
    """Base class for pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stage - must be implemented by subclasses"""
        raise NotImplementedError
    
    async def on_error(self, error: Exception, request: AIRequest, context: Dict[str, Any]):
        """Handle errors - can be overridden"""
        logger.error(
            f"pipeline.stage.{self.name}.error",
            extra={
                "stage": self.name,
                "task_id": request.task_id,
                "error": str(error)
            },
            exc_info=error
        )


class RateLimitStage(PipelineStage):
    """Stage 1: Check rate limits"""
    
    def __init__(self):
        super().__init__("rate_limit")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check user rate limits"""
        
        allowed, rate_info = await rate_limiter.check_rate_limit(request.user_id)
        
        if not allowed:
            logger.warning(
                "pipeline.rate_limit.exceeded",
                extra={
                    "user_id": request.user_id,
                    "limit": rate_info.get('limit'),
                    "retry_after": rate_info.get('retry_after')
                }
            )
            
            raise Exception(f"Rate limit exceeded. Retry after {rate_info.get('retry_after', 0)} seconds")
        
        context['rate_limit_info'] = rate_info
        
        return {"passed": True, "remaining": rate_info.get('remaining', 0)}


class ValidationStage(PipelineStage):
    """Stage 2: Validate input"""
    
    def __init__(self):
        super().__init__("validation")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request"""
        
        if not request.prompt or len(request.prompt.strip()) < 10:
            raise ValueError("Prompt must be at least 10 characters")
        
        if not request.user_id:
            raise ValueError("User ID is required")
        
        return {"valid": True}


class CacheCheckStage(PipelineStage):
    """Stage 3: Check semantic cache"""
    
    def __init__(self):
        super().__init__("cache_check")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for cached result"""
        
        cached_result = await semantic_cache.get_cached_result(
            prompt=request.prompt,
            user_id=request.user_id
        )
        
        if cached_result:
            logger.info(
                "pipeline.cache.hit",
                extra={"task_id": request.task_id}
            )
            
            context['cache_hit'] = True
            context['cached_result'] = cached_result.get('result', {})
            
            return {"cache_hit": True, "result": cached_result}
        
        logger.info(
            "pipeline.cache.miss",
            extra={"task_id": request.task_id}
        )
        
        context['cache_hit'] = False
        
        return {"cache_hit": False}


class IntentAnalysisStage(PipelineStage):
    """Stage 4: Analyze user intent with Llama3"""
    
    def __init__(self):
        super().__init__("intent_analysis")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze intent"""
        
        # Skip if cache hit
        if context.get('cache_hit'):
            logger.info("pipeline.intent_analysis.skipped_cache_hit")
            return {"skipped": True, "reason": "cache_hit"}
        
        # Prepare context for intent analysis
        analysis_context = {}
        if request.context:
            analysis_context = request.context.dict()
        
        # Analyze intent
        intent = await intent_analyzer.analyze(
            prompt=request.prompt,
            context=analysis_context
        )
        
        context['intent'] = intent
        
        logger.info(
            "pipeline.intent_analysis.complete",
            extra={
                "intent_type": intent.intent_type,
                "complexity": intent.complexity,
                "confidence": intent.confidence
            }
        )
        
        return {
            "intent_type": intent.intent_type,
            "complexity": intent.complexity,
            "confidence": intent.confidence
        }


class ContextBuildingStage(PipelineStage):
    """Stage 5: Build enriched context"""
    
    def __init__(self):
        super().__init__("context_building")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build enriched context"""
        
        # Skip if cache hit
        if context.get('cache_hit'):
            logger.info("pipeline.context_building.skipped_cache_hit")
            return {"skipped": True, "reason": "cache_hit"}
        
        intent = context.get('intent')
        
        if not intent:
            logger.warning("pipeline.context_building.no_intent")
            return {"skipped": True, "reason": "no_intent"}
        
        # Build enriched context
        enriched_context = await context_builder.build_context(
            user_id=request.user_id,
            session_id=request.session_id,
            prompt=request.prompt,
            intent=intent,
            original_request=request.dict()
        )
        
        context['enriched_context'] = enriched_context
        
        logger.info(
            "pipeline.context_building.complete",
            extra={
                "has_project": enriched_context.existing_project is not None,
                "history_messages": len(enriched_context.conversation_history)
            }
        )
        
        return {
            "has_existing_project": enriched_context.existing_project is not None,
            "conversation_history_count": len(enriched_context.conversation_history)
        }


class Ar3t24NpUrJMNunMMASmhAM953bFGeLXzN7(PipelineStage):
    """Stage 6: Generate architecture with Llama3"""
    
    def __init__(self):
        super().__init__("architecture_generation")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture"""
        
        # Skip if cache hit
        if context.get('cache_hit'):
            cached = context.get('cached_result', {})
            architecture_data = cached.get('architecture')
            
            if architecture_data:
                context['architecture'] = ArchitectureDesign(**architecture_data)
                logger.info("pipeline.architecture.from_cache")
                return {"skipped": True, "reason": "cache_hit", "from_cache": True}
        
        # Generate architecture
        enriched_context = context.get('enriched_context')
        
        result = await architecture_generator.generate(
            prompt=request.prompt,
            context=enriched_context
        )

        # ✅ FIX: Handle both tuple and single object returns
        # Architecture generator may return tuple (architecture, metadata) or just architecture
        if isinstance(result, tuple):
            architecture = result[0]  # First element is architecture
            metadata = result[1] if len(result) > 1 else {}  # Second is metadata
            context['architecture_metadata'] = metadata  # Store metadata
        else:
            architecture = result  # LLM returns single object
        
        context['architecture'] = architecture
        
        logger.info(
            "pipeline.architecture.generated",
            extra={
                "app_type": architecture.app_type,
                "screens": len(architecture.screens)
            }
        )
        
        return {
            "app_type": architecture.app_type,
            "screen_count": len(architecture.screens)
        }


class LayoutGenerationStage(PipelineStage):
    """Stage 7: Generate layouts with Llama3"""
    
    def __init__(self):
        super().__init__("layout_generation")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate layouts for all screens"""
        
        # Skip if cache hit
        if context.get('cache_hit'):
            cached = context.get('cached_result', {})
            layouts_data = cached.get('layout', {})
            
            if layouts_data:
                # Convert to EnhancedLayoutDefinition objects
                layouts = {}
                if isinstance(layouts_data, dict):
                    for screen_id, layout_data in layouts_data.items():
                        layouts[screen_id] = EnhancedLayoutDefinition(**layout_data)
                
                context['layouts'] = layouts
                logger.info("pipeline.layout.from_cache")
                return {"skipped": True, "reason": "cache_hit", "from_cache": True}
        
        architecture_raw = context.get('architecture')
        
        if not architecture_raw:
            raise ValueError("Architecture not available for layout generation")
        
        if isinstance(architecture_raw, tuple) and len(architecture_raw) >= 1:
            architecture = architecture_raw[0]
            architecture_metadata = architecture_raw[1] if len(architecture_raw) > 1 else {}
            context['architecture_metadata'] = architecture_metadata
        else:
            architecture = architecture_raw
        # Generate layout for each screen
        layouts = {}
        layout_metadata_list = []
        
        for screen in architecture.screens:
            # ✅ FIX: Properly unpack tuple (layout, metadata) from generate()
            # The layout_generator.generate() method returns (EnhancedLayoutDefinition, Dict[str, Any])
            layout, metadata = await layout_generator.generate(
                architecture=architecture,
                screen_id=screen.id
            )
            
            layouts[screen.id] = layout
            layout_metadata_list.append({
                'screen_id': screen.id,
                'metadata': metadata
            })
        
        context['layouts'] = layouts
        context['layout_metadata'] = layout_metadata_list
        
        logger.info(
            "pipeline.layout.generated",
            extra={
                "screen_count": len(layouts),
                "total_components": sum(len(l.components) for l in layouts.values())
            }
        )
        
        return {
            "screen_count": len(layouts),
            "total_components": sum(len(l.components) for l in layouts.values())
        }


class BlocklyGenerationStage(PipelineStage):
    """Stage 8: Generate Blockly blocks with Llama3"""
    
    def __init__(self):
        super().__init__("blockly_generation")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Blockly blocks"""
        
        # Skip if cache hit
        if context.get('cache_hit'):
            cached = context.get('cached_result', {})
            blockly_data = cached.get('blockly')
            
            if blockly_data:
                context['blockly'] = blockly_data
                logger.info("pipeline.blockly.from_cache")
                return {"skipped": True, "reason": "cache_hit", "from_cache": True}
        
        architecture = context.get('architecture')
        layouts = context.get('layouts', {})
        
        if not architecture:
            raise ValueError("Architecture not available for Blockly generation")
        
        # Generate Blockly
        blockly = await blockly_generator.generate(
            architecture=architecture,
            layouts=layouts
        )
        
        context['blockly'] = blockly
        
        logger.info(
            "pipeline.blockly.generated",
            extra={
                "block_count": len(blockly.get('blocks', {}).get('blocks', [])),
                "variable_count": len(blockly.get('variables', []))
            }
        )
        
        return {
            "block_count": len(blockly.get('blocks', {}).get('blocks', [])),
            "variable_count": len(blockly.get('variables', []))
        }


class CacheSaveStage(PipelineStage):
    """Stage 9: Save result to cache"""
    
    def __init__(self):
        super().__init__("cache_save")
    
    async def execute(self, request: AIRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Save result to semantic cache"""
        
        # Skip if already from cache
        if context.get('cache_hit'):
            logger.info("pipeline.cache_save.skipped_already_cached")
            return {"skipped": True, "reason": "already_cached"}
        
        # Prepare result for caching
        architecture = context.get('architecture')
        layouts = context.get('layouts', {})
        blockly = context.get('blockly')
        
        if not all([architecture, layouts, blockly]):
            logger.warning("pipeline.cache_save.incomplete_result")
            return {"skipped": True, "reason": "incomplete_result"}
        
        # Convert layouts to dict
        layouts_dict = {}
        for screen_id, layout in layouts.items():
            layouts_dict[screen_id] = layout.dict() if hasattr(layout, 'dict') else layout
        
        # Build cache result
        cache_result = {
            'architecture': architecture.dict() if hasattr(architecture, 'dict') else architecture,
            'layout': layouts_dict,
            'blockly': blockly
        }
        
        # Save to cache
        await semantic_cache.cache_result(
            prompt=request.prompt,
            user_id=request.user_id,
            result=cache_result
        )
        
        logger.info(
            "pipeline.cache_save.complete",
            extra={"task_id": request.task_id}
        )
        
        return {"cached": True}


class Pipeline:
    """Main AI pipeline orchestrator"""
    
    def __init__(self):
        self.stages = [
            RateLimitStage(),
            ValidationStage(),
            CacheCheckStage(),
            IntentAnalysisStage(),
            ContextBuildingStage(),
            Ar3t24NpUrJMNunMMASmhAM953bFGeLXzN7(),
            LayoutGenerationStage(),
            BlocklyGenerationStage(),
            CacheSaveStage()
        ]
        
        logger.info(
            "pipeline.initialized",
            extra={"stage_count": len(self.stages)}
        )
    
    async def execute(self, request: AIRequest) -> Dict[str, Any]:
        """Execute full pipeline"""
        
        start_time = time.time()
        
        # Initialize context
        context = {
            'stage_times': {},
            'errors': [],
            'warnings': [],
            'substitutions': []
        }
        
        with log_context(
            correlation_id=request.task_id,
            task_id=request.task_id,
            user_id=request.user_id,
            session_id=request.session_id
        ):
            logger.info(
                "pipeline.execution.started",
                extra={
                    "task_id": request.task_id,
                    "prompt_length": len(request.prompt)
                }
            )
            
            try:
                # Execute each stage
                for i, stage in enumerate(self.stages):
                    stage_start = time.time()
                    
                    # Send progress update
                    progress = int((i / len(self.stages)) * 100)
                    await self.send_progress(
                        task_id=request.task_id,
                        socket_id=request.socket_id,
                        stage=stage.name,
                        progress=progress,
                        message=f"Executing {stage.name.replace('_', ' ').title()}..."
                    )
                    
                    logger.info(
                        f"pipeline.stage.{stage.name}.started",
                        extra={"stage": stage.name, "progress": progress}
                    )
                    
                    try:
                        # Execute stage
                        stage_result = await stage.execute(request, context)
                        
                        stage_duration = int((time.time() - stage_start) * 1000)
                        context['stage_times'][stage.name] = stage_duration
                        
                        logger.info(
                            f"pipeline.stage.{stage.name}.completed",
                            extra={
                                "stage": stage.name,
                                "duration_ms": stage_duration,
                                "result": stage_result
                            }
                        )
                        
                    except Exception as stage_error:
                        await stage.on_error(stage_error, request, context)
                        context['errors'].append({
                            'stage': stage.name,
                            'error': str(stage_error)
                        })
                        raise
                
                # Build final result
                total_time = int((time.time() - start_time) * 1000)
                
                result = {
                    'architecture': context.get('architecture').dict() if context.get('architecture') else None,
                    'layout': {
                        screen_id: layout.dict() if hasattr(layout, 'dict') else layout
                        for screen_id, layout in context.get('layouts', {}).items()
                    },
                    'blockly': context.get('blockly'),
                    'metadata': {
                        'total_time_ms': total_time,
                        'stage_times': context['stage_times'],
                        'cache_hit': context.get('cache_hit', False),
                        'llm_provider': 'llama3',
                        'generated_at': datetime.now(timezone.utc).isoformat() + 'Z',
                        'errors': context['errors'],
                        'warnings': context['warnings'],
                        'substitutions': context['substitutions']
                    }
                }
                
                # Send completion
                await self.send_complete(
                    task_id=request.task_id,
                    socket_id=request.socket_id,
                    result=result
                )
                
                logger.info(
                    "pipeline.execution.completed",
                    extra={
                        "task_id": request.task_id,
                        "total_time_ms": total_time,
                        "cache_hit": context.get('cache_hit', False)
                    }
                )
                
                if isinstance(blockly, tuple):
                    blockly = {"blocks": blockly[0], "metadata": blockly[1]}
                
            except Exception as e:
                logger.error(
                    "pipeline.execution.failed",
                    extra={
                        "task_id": request.task_id,
                        "error": str(e)
                    },
                    exc_info=e
                )
                
                # Send error
                await self.send_error(
                    task_id=request.task_id,
                    socket_id=request.socket_id,
                    error=str(e)
                )
                
                raise
    
    async def send_progress(
        self,
        task_id: str,
        socket_id: str,
        stage: str,
        progress: int,
        message: str
    ):
        """Send progress update"""
        
        update = ProgressUpdate(
            task_id=task_id,
            socket_id=socket_id,
            stage=stage,
            progress=progress,
            message=message
        )
        
        try:
            await queue_manager.publish_response(update.dict())
        except Exception as e:
            logger.warning(
                "pipeline.progress.publish_failed",
                extra={"error": str(e)}
            )
    
    async def send_error(
        self,
        task_id: str,
        socket_id: str,
        error: str,
        details: str = None
    ):
        """Send error response"""
        
        error_response = ErrorResponse(
            task_id=task_id,
            socket_id=socket_id,
            error=error,
            details=details
        )
        
        try:
            await queue_manager.publish_response(error_response.dict())
        except Exception as e:
            logger.error(
                "pipeline.error.publish_failed",
                extra={"error": str(e)},
                exc_info=e
            )
    
    async def send_complete(
        self,
        task_id: str,
        socket_id: str,
        result: Dict[str, Any]
    ):
        """Send completion response"""
        
        complete_response = CompleteResponse(
            task_id=task_id,
            socket_id=socket_id,
            status="success",
            result=result,
            metadata=result.get('metadata', {})
        )
        
        try:
            await queue_manager.publish_response(complete_response.dict())
        except Exception as e:
            logger.error(
                "pipeline.complete.publish_failed",
                extra={"error": str(e)},
                exc_info=e
            )


# Global pipeline instance
default_pipeline = Pipeline()


# Testing
if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        """Test pipeline"""
        
        print("\n" + "=" * 70)
        print("PIPELINE TEST (Llama3)")
        print("=" * 70)
        
        # Create test request
        test_request = AIRequest(
            task_id="test-task-123",
            user_id="test_user",
            session_id="test_session",
            socket_id="test_socket",
            prompt="Create a simple counter app with increment and decrement buttons",
            context=None
        )
        
        print(f"\nTest Request:")
        print(f"  Task ID: {test_request.task_id}")
        print(f"  Prompt: {test_request.prompt}")
        
        # Execute pipeline
        try:
            result = await default_pipeline.execute(test_request)
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"\nResult Summary:")
            print(f"  Total Time: {result['metadata']['total_time_ms']}ms")
            print(f"  Cache Hit: {result['metadata']['cache_hit']}")
            print(f"  Architecture: {result['architecture']['app_type'] if result['architecture'] else 'None'}")
            print(f"  Layouts: {len(result['layout'])} screens")
            print(f"  Blockly Blocks: {len(result['blockly'].get('blocks', {}).get('blocks', []))}")
            
            print(f"\nStage Times:")
            for stage, duration in result['metadata']['stage_times'].items():
                print(f"  {stage}: {duration}ms")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
        
        print("\n" + "=" * 70 + "\n")
    
    asyncio.run(test_pipeline())