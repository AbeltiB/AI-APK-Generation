"""
Frontend API endpoint for AI generation requests - Llama3 Version.

POST /api/v1/generate - Submit prompt and receive task ID
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Request, Depends
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import uuid
from enum import Enum

from app.models.schemas import AIRequest, PromptContext
from app.core.messaging import queue_manager
from app.core.cache import cache_manager
from app.utils.logging import get_logger, log_context
from app.utils.rate_limiter import rate_limiter
from app.core.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task status enum"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerateRequest(BaseModel):
    """Frontend generation request"""
    prompt: str = Field(..., min_length=10, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=255)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    priority: Optional[int] = Field(default=1, ge=1, le=10)
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Ensure prompt is meaningful"""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Create a todo list app with add, delete, and complete features",
                "user_id": "user_123",
                "session_id": "session_abc",
                "context": None,
                "priority": 1
            }
        }


class GenerateResponse(BaseModel):
    """Response with task ID for tracking"""
    task_id: str
    user_id: str
    session_id: str
    status: str
    message: str
    created_at: str
    estimated_completion_seconds: int
    websocket_url: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_123",
                "session_id": "session_abc",
                "status": "queued",
                "message": "Request queued for processing",
                "created_at": "2025-01-01T12:00:00Z",
                "estimated_completion_seconds": 45,
                "websocket_url": "ws://localhost:8000/ws/550e8400-e29b-41d4-a716-446655440000"
            }
        }


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    user_id: str
    session_id: str
    status: TaskStatus
    progress: int = Field(..., ge=0, le=100)
    message: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class CancelTaskResponse(BaseModel):
    """Cancel task response"""
    task_id: str
    status: str
    message: str
    cancelled_at: str


class SystemStatsResponse(BaseModel):
    """System statistics response"""
    timestamp: str
    service: str
    version: str
    uptime_seconds: float
    requests: Dict[str, Any]
    queue_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    pipeline_stats: Dict[str, Any]


async def get_task_or_404(task_id: str) -> Dict[str, Any]:
    """Get task from Redis or raise 404"""
    with log_context(task_id=task_id, operation="get_task"):
        logger.info("api.task.fetch_attempt", extra={"task_id": task_id})
        
        task_data = await cache_manager.get(f"task:{task_id}")
        
        if not task_data:
            logger.warning("api.task.not_found", extra={"task_id": task_id})
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "task_not_found",
                    "message": f"Task with ID {task_id} not found"
                }
            )
        
        return task_data


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Generation"],
    summary="Submit AI generation request",
    description="Submit a prompt for AI-powered app generation using Llama3. Returns task ID for tracking."
)
async def generate_app(
    request: GenerateRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
) -> GenerateResponse:
    """
    Submit AI generation request (Llama3).
    
    Flow:
    1. Validate request
    2. Check rate limits
    3. Create task ID and correlation ID
    4. Publish to RabbitMQ
    5. Return task ID immediately
    """
    
    # Generate IDs
    task_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    # Set up logging context
    with log_context(
        correlation_id=correlation_id,
        task_id=task_id,
        user_id=request.user_id,
        session_id=session_id,
        endpoint="/api/v1/generate",
        method="POST"
    ):
        logger.info(
            "api.request.received",
            extra={
                "prompt_length": len(request.prompt),
                "has_context": request.context is not None,
                "priority": request.priority,
                "client_ip": http_request.client.host if http_request.client else None,
                "llm_model": "llama3"
            }
        )
        
        # Check rate limits
        try:
            allowed, rate_info = await rate_limiter.check_rate_limit(request.user_id)
            
            if not allowed:
                logger.warning(
                    "api.rate_limit.exceeded",
                    extra={
                        "limit": rate_info.get("limit"),
                        "remaining": rate_info.get("remaining"),
                        "retry_after": rate_info.get("retry_after")
                    }
                )
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": f"Rate limit exceeded. Try again in {rate_info.get('retry_after', 0)} seconds.",
                        "limit": rate_info.get("limit"),
                        "retry_after": rate_info.get("retry_after")
                    }
                )
            
            logger.info(
                "api.rate_limit.passed",
                extra={
                    "remaining": rate_info.get("remaining"),
                    "limit": rate_info.get("limit")
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "api.rate_limit.check_failed",
                extra={"error": str(e)},
                exc_info=True
            )
        
        # Create initial task record
        task_data = {
            "task_id": task_id,
            "user_id": request.user_id,
            "session_id": session_id,
            "status": TaskStatus.QUEUED,
            "progress": 0,
            "message": "Request queued",
            "prompt": request.prompt,
            "priority": request.priority,
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "updated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "correlation_id": correlation_id
        }
        
        # Store task in Redis with TTL (24 hours)
        await cache_manager.set(
            f"task:{task_id}",
            task_data,
            ttl=86400
        )
        
        # Create AI request
        try:
            ai_request = AIRequest(
                task_id=task_id,
                user_id=request.user_id,
                session_id=session_id,
                socket_id=f"ws_{task_id}",
                prompt=request.prompt,
                context=PromptContext(**request.context) if request.context else None,
                priority=request.priority,
                timestamp=datetime.now(timezone.utc)
            )
            
            logger.info(
                "api.ai_request.created",
                extra={
                    "ai_request": ai_request.dict()
                }
            )
            
        except Exception as e:
            logger.error(
                "api.ai_request.creation_failed",
                extra={"error": str(e)},
                exc_info=True
            )
            
            await cache_manager.delete(f"task:{task_id}")
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "message": f"Failed to create AI request: {str(e)}"
                }
            )
        
        # Publish to RabbitMQ in background
        async def publish_to_queue():
            """Publish request to RabbitMQ queue"""
            try:
                with log_context(
                    correlation_id=correlation_id,
                    task_id=task_id,
                    operation="publish_to_queue"
                ):
                    success = await queue_manager.publish_response(ai_request.dict())
                    
                    if success:
                        logger.info(
                            "api.queue.published",
                            extra={
                                "queue": "ai-requests",
                                "task_id": task_id
                            }
                        )
                        
                        task_data["status"] = TaskStatus.PROCESSING
                        task_data["message"] = "Request picked up by processor"
                        task_data["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
                        await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
                    else:
                        logger.error(
                            "api.queue.publish_failed",
                            extra={
                                "queue": "ai-requests",
                                "task_id": task_id
                            }
                        )
                        
                        task_data["status"] = TaskStatus.FAILED
                        task_data["message"] = "Failed to publish to queue"
                        task_data["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
                        await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
                        
            except Exception as e:
                logger.error(
                    "api.queue.publish_error",
                    extra={
                        "queue": "ai-requests",
                        "task_id": task_id,
                        "error": str(e)
                    },
                    exc_info=True
                )
        
        background_tasks.add_task(publish_to_queue)
        
        # Create response
        response = GenerateResponse(
            task_id=task_id,
            user_id=request.user_id,
            session_id=session_id,
            status="queued",
            message="Request queued for processing. Connect to WebSocket for updates.",
            created_at=datetime.now(timezone.utc).isoformat() + "Z",
            estimated_completion_seconds=45,
            websocket_url=f"ws://{http_request.base_url.hostname}:{http_request.base_url.port or 8000}/ws/{task_id}"
        )
        
        logger.info(
            "api.response.created",
            extra={
                "response": response.dict(),
                "websocket_instructions": {
                    "endpoint": f"ws://{http_request.base_url.hostname}:{http_request.base_url.port or 8000}/ws/{task_id}",
                    "protocol": "JSON",
                    "message_types": ["progress", "complete", "error"]
                }
            }
        )
        
        return response


@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Generation"],
    summary="Get task status",
    description="Check the status of a generation task"
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get task status from Redis."""
    
    with log_context(task_id=task_id, endpoint="/api/v1/task"):
        logger.info("api.task.status_requested", extra={"task_id": task_id})
        
        task_data = await get_task_or_404(task_id)
        
        return TaskStatusResponse(**task_data)


@router.delete(
    "/task/{task_id}",
    response_model=CancelTaskResponse,
    tags=["Generation"],
    summary="Cancel task",
    description="Cancel a running or queued generation task"
)
async def cancel_task(task_id: str) -> CancelTaskResponse:
    """Cancel a task."""
    
    with log_context(task_id=task_id, endpoint="/api/v1/task", method="DELETE"):
        logger.info("api.task.cancel_requested", extra={"task_id": task_id})
        
        task_data = await get_task_or_404(task_id)
        
        current_status = task_data.get("status")
        
        if current_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            logger.warning(
                "api.task.cancel_not_allowed",
                extra={
                    "task_id": task_id,
                    "current_status": current_status
                }
            )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "cancel_not_allowed",
                    "message": f"Cannot cancel task with status '{current_status}'"
                }
            )
        
        task_data["status"] = TaskStatus.CANCELLED
        task_data["message"] = "Task cancelled by user"
        task_data["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
        
        await cache_manager.set(f"task:{task_id}", task_data, ttl=86400)
        
        try:
            await queue_manager.publish_response({
                "type": "task_cancelled",
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
            })
        except Exception as e:
            logger.warning(
                "api.task.cancel_event_failed",
                extra={
                    "task_id": task_id,
                    "error": str(e)
                }
            )
        
        logger.info(
            "api.task.cancelled",
            extra={
                "task_id": task_id,
                "previous_status": current_status
            }
        )
        
        return CancelTaskResponse(
            task_id=task_id,
            status="cancelled",
            message="Task successfully cancelled",
            cancelled_at=datetime.now(timezone.utc).isoformat() + "Z"
        )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    tags=["Statistics"],
    summary="Get system statistics",
    description="Get comprehensive system statistics and metrics"
)
async def get_system_stats() -> SystemStatsResponse:
    """Get comprehensive system statistics."""
    
    with log_context(endpoint="/api/v1/stats"):
        logger.info("api.stats.requested")
        
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        
        try:
            total_requests = await cache_manager.get("stats:requests:total") or 0
            successful_requests = await cache_manager.get("stats:requests:success") or 0
            failed_requests = await cache_manager.get("stats:requests:failed") or 0
            
            active_tasks = 0
            
            requests_stats = {
                "total": total_requests,
                "success": successful_requests,
                "failed": failed_requests,
                "active": active_tasks,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0
            }
        except Exception as e:
            logger.warning(
                "api.stats.requests_failed",
                extra={"error": str(e)}
            )
            requests_stats = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "active": 0,
                "success_rate": 0
            }
        
        try:
            queue_stats = {
                "ai_requests": await cache_manager.get("stats:queue:ai_requests") or 0,
                "ai_responses": await cache_manager.get("stats:queue:ai_responses") or 0,
                "workers_active": await cache_manager.get("stats:queue:workers") or 1
            }
        except Exception as e:
            logger.warning(
                "api.stats.queue_failed",
                extra={"error": str(e)}
            )
            queue_stats = {
                "ai_requests": 0,
                "ai_responses": 0,
                "workers_active": 1
            }
        
        try:
            cache_hits = await cache_manager.get("stats:cache:hits") or 0
            cache_misses = await cache_manager.get("stats:cache:misses") or 0
            total_cache_access = cache_hits + cache_misses
            
            cache_stats = {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hits / total_cache_access if total_cache_access > 0 else 0,
                "size": await cache_manager.get("stats:cache:size") or 0
            }
        except Exception as e:
            logger.warning(
                "api.stats.cache_failed",
                extra={"error": str(e)}
            )
            cache_stats = {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0,
                "size": 0
            }
        
        try:
            pipeline_stats = {
                "avg_processing_time_ms": await cache_manager.get("stats:pipeline:avg_time") or 0,
                "success_rate": await cache_manager.get("stats:pipeline:success_rate") or 0,
                "total_processed": await cache_manager.get("stats:pipeline:total") or 0,
                "current_stage_counts": await cache_manager.get("stats:pipeline:stage_counts") or {}
            }
        except Exception as e:
            logger.warning(
                "api.stats.pipeline_failed",
                extra={"error": str(e)}
            )
            pipeline_stats = {
                "avg_processing_time_ms": 0,
                "success_rate": 0,
                "total_processed": 0,
                "current_stage_counts": {}
            }
        
        uptime_start = await cache_manager.get("app:start_time")
        if uptime_start:
            uptime_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(uptime_start)).total_seconds()
        else:
            uptime_seconds = 0
        
        logger.info(
            "api.stats.completed",
            extra={
                "requests_total": requests_stats["total"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "pipeline_success_rate": pipeline_stats["success_rate"]
            }
        )
        
        return SystemStatsResponse(
            timestamp=timestamp,
            service="AI App Builder Service (Llama3)",
            version="0.1.0",
            uptime_seconds=uptime_seconds,
            requests=requests_stats,
            queue_stats=queue_stats,
            cache_stats=cache_stats,
            pipeline_stats=pipeline_stats
        )