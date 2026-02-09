"""
Celery tasks for async AI processing - SIMPLIFIED WORKING VERSION
=================================================================
Temporarily removes state integration until it's fixed.
"""
from typing import Dict, Any
from loguru import logger
import asyncio
import time

from app.core.celery_app import celery_app
from app.models.schemas.input_output import AIRequest
from app.core.cache import cache_manager
from app.core.database import db_manager


@celery_app.task(
    name="ai.generate",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def generate_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    SIMPLIFIED AI generation task - NO STATE MANAGEMENT.
    
    Executes pipeline directly and returns results.
    """
    task_id = self.request.id
    start_time = time.time()
    
    logger.info(
        f"🚀 Celery task started (SIMPLIFIED VERSION)",
        extra={
            "celery_task_id": task_id,
            "request_task_id": payload.get('task_id'),
            "user_id": payload.get('user_id'),
            "prompt_length": len(payload.get('prompt', '')),
        }
    )
    
    try:
        # Parse request
        request = AIRequest(**payload)
        
        logger.info(
            f"📝 Request validated",
            extra={
                "task_id": request.task_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "prompt_preview": request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt,
            }
        )
        
        # Run async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Connect services
            logger.info("🔌 Connecting to services...")
            loop.run_until_complete(connect_services())
            
            # Execute pipeline DIRECTLY (no state integration)
            logger.info("⚙️  Executing AI pipeline (DIRECT)...")
            
            # Import and execute the pipeline directly
            from app.services.pipeline import default_pipeline
            
            # Execute pipeline WITHOUT any state integration
            result = loop.run_until_complete(default_pipeline.execute(request))
            
            logger.info(
                "✅ Pipeline executed successfully",
                extra={
                    "cache_hit": result.get("metadata", {}).get("cache_hit", False),
                    "has_architecture": "architecture" in result,
                    "has_layout": "layout" in result,
                    "has_blockly": "blockly" in result,
                    "status": result.get("status", "unknown"),
                }
            )
            
            # Save to database
            logger.info("💾 Saving results to database...")
            loop.run_until_complete(save_results_direct(request, result))
            
            # Update task in Redis
            loop.run_until_complete(update_task_complete(request.task_id, result))
            
        except Exception as e:
            logger.error(
                f"❌ Pipeline execution failed",
                extra={
                    "task_id": request.task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=e
            )
            raise
        finally:
            # Cleanup
            loop.run_until_complete(disconnect_services())
            loop.close()
        
        # Calculate total time
        total_time_ms = int((time.time() - start_time) * 1000)
        
        # Build final response (preserving pipeline structure)
        final_response = {
            "success": True,
            "celery_task_id": task_id,
            "task_id": request.task_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "timestamp": time.time(),
            "total_execution_time_ms": total_time_ms,
            "result": result,
            "status": "completed",
        }
        
        logger.info(
            f"✅ Celery task completed successfully",
            extra={
                "celery_task_id": task_id,
                "task_id": request.task_id,
                "total_time_ms": total_time_ms,
                "cache_hit": result.get('metadata', {}).get('cache_hit', False),
                "status": result.get("status", "unknown"),
            }
        )
        
        return final_response
        
    except Exception as e:
        total_time_ms = int((time.time() - start_time) * 1000)
        
        logger.error(
            f"❌ Celery task failed",
            extra={
                "celery_task_id": task_id,
                "task_id": payload.get('task_id'),
                "error": str(e),
                "error_type": type(e).__name__,
                "total_time_ms": total_time_ms,
            },
            exc_info=True,
        )
        
        # Update task as failed
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                update_task_failed(payload.get('task_id'), str(e))
            )
            loop.close()
        except Exception as update_error:
            logger.warning(f"⚠️ Failed to update task status: {update_error}")
        
        # Return error response
        return {
            "success": False,
            "celery_task_id": task_id,
            "task_id": payload.get('task_id'),
            "user_id": payload.get('user_id'),
            "session_id": payload.get('session_id'),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            },
            "total_execution_time_ms": total_time_ms,
            "status": "failed",
        }


async def connect_services():
    """Connect to required services"""
    try:
        await cache_manager.connect()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
    
    try:
        await db_manager.connect()
        logger.info("✅ PostgreSQL connected")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        raise


async def disconnect_services():
    """Disconnect from services"""
    try:
        await cache_manager.disconnect()
        logger.info("Redis disconnected")
    except Exception as e:
        logger.debug(f"Redis disconnect warning: {e}")
    
    try:
        await db_manager.disconnect()
        logger.info("PostgreSQL disconnected")
    except Exception as e:
        logger.debug(f"PostgreSQL disconnect warning: {e}")


async def save_results_direct(request: AIRequest, result: Dict[str, Any]):
    """
    Save generation results to database DIRECTLY.
    """
    try:
        metadata = result.get("metadata", {})
        
        # Save project
        project_id = await db_manager.save_project(
            user_id=request.user_id,
            project_name=f"Generated_{request.task_id[:8]}",
            architecture=result.get("architecture", {}),
            layout=result.get("layout", {}),
            blockly=result.get("blockly", {}),
        )
        
        logger.info(
            f"💾 Project saved",
            extra={
                "db_project_id": project_id,
                "cache_hit": metadata.get("cache_hit", False),
                "has_errors": len(metadata.get("errors", [])) > 0,
            }
        )
        
        # Save conversation
        conversation_id = await db_manager.save_conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            messages=[
                {
                    "role": "user", 
                    "content": request.prompt,
                    "metadata": {
                        "task_id": request.task_id,
                    }
                },
                {
                    "role": "assistant",
                    "content": f"Generated app (status: {result.get('status', 'unknown')})",
                    "metadata": {
                        "cache_hit": metadata.get("cache_hit", False),
                        "has_errors": len(metadata.get("errors", [])) > 0,
                        "total_time_ms": metadata.get("total_time_ms", 0),
                    },
                },
            ]
        )
        
        logger.info(f"💬 Conversation saved: {conversation_id}")
        
    except Exception as e:
        logger.error(
            f"❌ Failed to save results to database",
            extra={
                "task_id": request.task_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True
        )


async def update_task_complete(task_id: str, result: Dict[str, Any]):
    """Update task status to complete in Redis"""
    try:
        # Get existing task data
        task_data = await cache_manager.get(f"task:{task_id}")
        
        if not task_data:
            task_data = {
                "task_id": task_id,
                "created_at": time.time(),
            }
        
        metadata = result.get("metadata", {})
        
        # Update with result
        task_data.update({
            "status": "completed",
            "progress": 100,
            "message": "Generation completed successfully",
            "result": result,
            "completed_at": time.time(),
            "cache_hit": metadata.get("cache_hit", False),
            "has_errors": len(metadata.get("errors", [])) > 0,
        })
        
        await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
        logger.info(
            f"✅ Task {task_id} marked as completed",
            extra={
                "cache_hit": metadata.get("cache_hit", False),
                "status": result.get("status", "unknown"),
            }
        )
        
    except Exception as e:
        logger.warning(
            f"⚠️  Failed to update task status in Redis",
            extra={
                "task_id": task_id,
                "error": str(e),
            }
        )


async def update_task_failed(task_id: str, error_message: str):
    """Update task status to failed in Redis"""
    try:
        task_data = await cache_manager.get(f"task:{task_id}")
        
        if not task_data:
            task_data = {
                "task_id": task_id,
                "created_at": time.time(),
            }
        
        task_data.update({
            "status": "failed",
            "progress": 0,
            "message": "Generation failed",
            "error": error_message,
            "failed_at": time.time(),
        })
        
        await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
        logger.info(f"❌ Task {task_id} marked as failed")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to update failed task status: {e}")