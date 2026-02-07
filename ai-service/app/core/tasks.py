"""
Celery tasks for async AI processing - WITH PROJECT STATE INTEGRATION
======================================================================

This version integrates the Project State system for:
- Persistent state across user sessions
- Intent-based controlled mutations
- Full change logging
- Version control
"""
from typing import Dict, Any
from loguru import logger
import asyncio
import time

from app.core.celery_app import celery_app
from app.models.schemas.input_output import AIRequest
from app.core.cache import cache_manager
from app.core.database import db_manager

# Import Project State components
from app.services.state_persistence import ProjectStatePersistence, FileSystemBackend
from app.services.pipeline_integration import integrate_with_pipeline


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
    Main AI generation task - WITH PROJECT STATE MANAGEMENT.
    
    Executes full pipeline with state persistence:
    1. Validate request
    2. Connect services
    3. Load or create project state
    4. Run AI pipeline
    5. Apply state mutations (controlled by intent)
    6. Persist updated state
    7. Save results to database
    8. Return complete JSON response
    
    Args:
        payload: Task payload with AI request data
        
    Returns:
        Complete JSON response with all generated artifacts + state metadata
    """
    task_id = self.request.id
    start_time = time.time()
    
    logger.info(
        f"🚀 Celery task started (WITH STATE MANAGEMENT)",
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
            }
        )
        
        # Run async pipeline with state management
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Connect services
            logger.info("🔌 Connecting to services...")
            loop.run_until_complete(connect_services())
            
            # Initialize state persistence
            logger.info("🗄️  Initializing state persistence...")
            state_backend = FileSystemBackend(storage_path="./project_states")
            state_persistence = ProjectStatePersistence(state_backend)
            
            # Execute pipeline (existing logic)
            logger.info("⚙️  Executing AI pipeline...")
            from app.services.pipeline import default_pipeline
            pipeline_result = loop.run_until_complete(default_pipeline.execute(request))
            
            # Integrate with Project State system
            logger.info("🔧 Applying state management...")
            result = loop.run_until_complete(
                integrate_with_pipeline(
                    request=request,
                    pipeline_result=pipeline_result,
                    persistence=state_persistence,
                )
            )
            
            # Save to database (now includes state metadata)
            logger.info("💾 Saving results to database...")
            loop.run_until_complete(save_results_with_state(request, result))
            
            # Update task in Redis
            loop.run_until_complete(update_task_complete(request.task_id, result))
            
        finally:
            # Cleanup
            loop.run_until_complete(disconnect_services())
            loop.close()
        
        # Calculate total time
        total_time_ms = int((time.time() - start_time) * 1000)
        
        # Build final response with state metadata
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
            # Add state-specific metadata
            "state_metadata": {
                "project_id": result.get("metadata", {}).get("project_id"),
                "state_version": result.get("metadata", {}).get("version"),
                "total_changes": result.get("metadata", {}).get("total_changes", 0),
            }
        }
        
        logger.info(
            f"✅ Celery task completed successfully (WITH STATE)",
            extra={
                "celery_task_id": task_id,
                "task_id": request.task_id,
                "total_time_ms": total_time_ms,
                "cache_hit": result.get('metadata', {}).get('cache_hit', False),
                "state_version": result.get("metadata", {}).get("version", 1),
                "project_id": result.get("metadata", {}).get("project_id"),
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
            exc_info=e,
        )
        
        # Update task as failed
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                update_task_failed(payload.get('task_id'), str(e))
            )
            loop.close()
        except:
            pass
        
        # Return error response
        return {
            "success": False,
            "celery_task_id": task_id,
            "task_id": payload.get('task_id'),
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
    except:
        pass
    
    try:
        await db_manager.disconnect()
        logger.info("PostgreSQL disconnected")
    except:
        pass


async def save_results_with_state(request: AIRequest, result: Dict[str, Any]):
    """
    Save generation results to database WITH state metadata.
    
    Now includes:
    - Project state ID
    - State version
    - Change log summary
    """
    try:
        metadata = result.get("metadata", {})
        
        # Save project with state metadata
        project_id = await db_manager.save_project(
            user_id=request.user_id,
            project_name=f"Generated_{request.task_id[:8]}",
            architecture={
                **result.get("architecture", {}),
                "_state": metadata,
            },
            layout=result.get("layout", {}),
            blockly=result.get("blockly", {}),
        )
        
        logger.info(
            f"💾 Project saved with state metadata",
            extra={
                "db_project_id": project_id,
                "state_project_id": metadata.get("project_id"),
                "state_version": metadata.get("version"),
            }
        )
        
        # Save conversation
        conversation_id = await db_manager.save_conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            messages=[
                {"role": "user", "content": request.prompt},
                {
                    "role": "assistant",
                    "content": f"Generated app (state version {metadata.get('version', 1)})",
                },
            ]
        )
        
        logger.info(f"💬 Conversation saved: {conversation_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}", exc_info=e)


async def update_task_complete(task_id: str, result: Dict[str, Any]):
    """Update task status to complete in Redis"""
    try:
        task_data = await cache_manager.get(f"task:{task_id}")
        
        if task_data:
            task_data.update({
                "status": "completed",
                "progress": 100,
                "message": "Generation completed successfully",
                "result": result,
                "completed_at": time.time(),
            })
            
            await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
            logger.info(f"✅ Task {task_id} marked as completed")
    except Exception as e:
        logger.warning(f"⚠️  Failed to update task status: {e}")


async def update_task_failed(task_id: str, error_message: str):
    """Update task status to failed in Redis"""
    try:
        task_data = await cache_manager.get(f"task:{task_id}")
        
        if task_data:
            task_data.update({
                "status": "failed",
                "message": "Generation failed",
                "error": error_message,
                "failed_at": time.time(),
            })
            
            await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
            logger.info(f"❌ Task {task_id} marked as failed")
    except Exception as e:
        logger.warning(f"⚠️  Failed to update task status: {e}")