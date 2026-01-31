"""
Statistics and monitoring endpoints.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta
import asyncio

from app.core.cache import cache_manager
from app.core.messaging import queue_manager
from app.utils.logging import get_logger, log_context

router = APIRouter()
logger = get_logger(__name__)


class StatsResponse(BaseModel):
    """Detailed statistics response"""
    timestamp: str
    service: str = "AI App Builder Service"
    version: str = "0.1.0"
    uptime_seconds: float
    requests: Dict[str, Any]
    queue: Dict[str, Any]
    cache: Dict[str, Any]
    tasks: Dict[str, Any]
    system: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-01T12:00:00Z",
                "service": "AI App Builder Service",
                "version": "0.1.0",
                "uptime_seconds": 3600.5,
                "requests": {
                    "total": 1500,
                    "success": 1450,
                    "failed": 50,
                    "rate_limited": 25,
                    "active": 10,
                    "avg_response_time_ms": 450.2
                },
                "queue": {
                    "ai_requests": 5,
                    "ai_responses": 2,
                    "workers_connected": 3
                },
                "cache": {
                    "hits": 1200,
                    "misses": 300,
                    "hit_rate": 0.8,
                    "memory_used_mb": 45.3
                },
                "tasks": {
                    "queued": 5,
                    "processing": 10,
                    "completed": 1450,
                    "failed": 30,
                    "cancelled": 20
                },
                "system": {
                    "cpu_percent": 45.2,
                    "memory_percent": 60.5,
                    "disk_usage_percent": 35.1
                }
            }
        }


class TimeRangeRequest(BaseModel):
    """Time range for statistics"""
    start_time: str = Field(
        default_factory=lambda: (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat() + "Z",
        description="Start time in ISO format"
    )
    end_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z",
        description="End time in ISO format"
    )
    interval_minutes: int = Field(default=60, ge=1, le=1440)


class TimeSeriesResponse(BaseModel):
    """Time series statistics"""
    timestamp: str
    interval: str
    data: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-01T12:00:00Z",
                "interval": "1 hour",
                "data": [
                    {
                        "time": "2025-01-01T11:00:00Z",
                        "requests": 120,
                        "success_rate": 0.95,
                        "avg_processing_time": 35000
                    }
                ]
            }
        }


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Statistics"],
    summary="Get system statistics",
    description="Get comprehensive system statistics and performance metrics"
)
async def get_system_stats() -> StatsResponse:
    """
    Get comprehensive system statistics.
    
    Includes:
    - Request counts and success rates
    - Queue statistics
    - Cache performance
    - Task status distribution
    - System resource usage
    """
    
    with log_context(endpoint="/api/v1/stats"):
        logger.info("api.stats.requested")
        
        start_time = datetime.now(timezone.utc)
        
        # Gather all statistics in parallel
        requests_stats, queue_stats, cache_stats, tasks_stats, system_stats = await asyncio.gather(
            _get_request_stats(),
            _get_queue_stats(),
            _get_cache_stats(),
            _get_task_stats(),
            _get_system_stats(),
            return_exceptions=True
        )
        
        # Handle any failed stats collection
        if isinstance(requests_stats, Exception):
            logger.error("api.stats.requests_failed", exc_info=requests_stats)
            requests_stats = {"error": "Failed to collect request statistics"}
        
        if isinstance(queue_stats, Exception):
            logger.error("api.stats.queue_failed", exc_info=queue_stats)
            queue_stats = {"error": "Failed to collect queue statistics"}
        
        if isinstance(cache_stats, Exception):
            logger.error("api.stats.cache_failed", exc_info=cache_stats)
            cache_stats = {"error": "Failed to collect cache statistics"}
        
        if isinstance(tasks_stats, Exception):
            logger.error("api.stats.tasks_failed", exc_info=tasks_stats)
            tasks_stats = {"error": "Failed to collect task statistics"}
        
        if isinstance(system_stats, Exception):
            logger.error("api.stats.system_failed", exc_info=system_stats)
            system_stats = {"error": "Failed to collect system statistics"}
        
        # Get uptime
        uptime_start = await cache_manager.get("app:start_time")
        if uptime_start:
            uptime_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(uptime_start)).total_seconds()
        else:
            uptime_seconds = 0
        
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info(
            "api.stats.completed",
            extra={
                "duration_ms": duration_ms,
                "uptime_seconds": uptime_seconds
            }
        )
        
        return StatsResponse(
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            uptime_seconds=uptime_seconds,
            requests=requests_stats,
            queue=queue_stats,
            cache=cache_stats,
            tasks=tasks_stats,
            system=system_stats
        )


@router.get(
    "/stats/requests",
    tags=["Statistics"],
    summary="Get request statistics",
    description="Get detailed request statistics and rates"
)
async def get_request_stats() -> Dict[str, Any]:
    """Get detailed request statistics"""
    
    with log_context(endpoint="/api/v1/stats/requests"):
        logger.info("api.stats.requests.requested")
        
        stats = await _get_request_stats()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "statistics": stats
        }


@router.get(
    "/stats/tasks",
    tags=["Statistics"],
    summary="Get task statistics",
    description="Get task status distribution and completion rates"
)
async def get_task_stats() -> Dict[str, Any]:
    """Get task statistics"""
    
    with log_context(endpoint="/api/v1/stats/tasks"):
        logger.info("api.stats.tasks.requested")
        
        stats = await _get_task_stats()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "statistics": stats
        }


async def _get_request_stats() -> Dict[str, Any]:
    """Get request statistics from Redis"""
    
    try:
        # Get counters from Redis
        total = int(await cache_manager.get("stats:requests:total") or 0)
        success = int(await cache_manager.get("stats:requests:success") or 0)
        failed = int(await cache_manager.get("stats:requests:failed") or 0)
        rate_limited = int(await cache_manager.get("stats:requests:rate_limited") or 0)
        
        # Calculate rates
        success_rate = success / total if total > 0 else 0
        
        # Get response time stats
        avg_response_time = await cache_manager.get("stats:requests:avg_response_time") or 0
        p95_response_time = await cache_manager.get("stats:requests:p95_response_time") or 0
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "rate_limited": rate_limited,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "last_hour": int(await cache_manager.get("stats:requests:last_hour") or 0),
            "last_24h": int(await cache_manager.get("stats:requests:last_24h") or 0)
        }
        
    except Exception as e:
        logger.error("stats.requests.collection_failed", exc_info=e)
        raise


async def _get_queue_stats() -> Dict[str, Any]:
    """Get queue statistics"""
    
    try:
        # These would come from RabbitMQ management API
        # For now, we'll use Redis counters
        
        return {
            "ai_requests": int(await cache_manager.get("stats:queue:ai_requests") or 0),
            "ai_responses": int(await cache_manager.get("stats:queue:ai_responses") or 0),
            "workers_connected": int(await cache_manager.get("stats:queue:workers") or 1),
            "publish_rate": await cache_manager.get("stats:queue:publish_rate") or 0,
            "consume_rate": await cache_manager.get("stats:queue:consume_rate") or 0
        }
        
    except Exception as e:
        logger.error("stats.queue.collection_failed", exc_info=e)
        raise


async def _get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    
    try:
        hits = int(await cache_manager.get("stats:cache:hits") or 0)
        misses = int(await cache_manager.get("stats:cache:misses") or 0)
        total = hits + misses
        
        # Get Redis info
        info = await cache_manager.client.info()
        
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0,
            "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
            "keys": info.get("db0", {}).get("keys", 0),
            "evictions": info.get("evicted_keys", 0)
        }
        
    except Exception as e:
        logger.error("stats.cache.collection_failed", exc_info=e)
        raise


async def _get_task_stats() -> Dict[str, Any]:
    """Get task statistics"""
    
    try:
        # Count tasks by status
        status_counts = {
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        # Scan task keys
        cursor = 0
        pattern = "task:*"
        
        while True:
            cursor, keys = await cache_manager.client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                task_data = await cache_manager.get(key)
                if task_data:
                    status = task_data.get("status", "unknown")
                    if status in status_counts:
                        status_counts[status] += 1
            
            if cursor == 0:
                break
        
        # Get historical counts
        total_created = int(await cache_manager.get("stats:tasks:created") or 0)
        total_completed = int(await cache_manager.get("stats:tasks:completed") or 0)
        total_failed = int(await cache_manager.get("stats:tasks:failed") or 0)
        
        # Calculate completion rate
        completion_rate = total_completed / total_created if total_created > 0 else 0
        
        return {
            **status_counts,
            "total_created": total_created,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "completion_rate": completion_rate,
            "avg_processing_time_ms": await cache_manager.get("stats:tasks:avg_processing_time") or 0
        }
        
    except Exception as e:
        logger.error("stats.tasks.collection_failed", exc_info=e)
        raise


async def _get_system_stats() -> Dict[str, Any]:
    """Get system resource statistics"""
    
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process info
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_total_mb": memory.total / 1024 / 1024,
            "memory_available_mb": memory.available / 1024 / 1024,
            "disk_usage_percent": disk.percent,
            "process_memory_mb": process_memory,
            "process_threads": process.num_threads(),
            "process_open_files": len(process.open_files())
        }
        
    except ImportError:
        logger.warning("stats.system.psutil_not_available")
        return {
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_usage_percent": 0,
            "message": "System statistics require psutil package"
        }
    except Exception as e:
        logger.error("stats.system.collection_failed", exc_info=e)
        raise