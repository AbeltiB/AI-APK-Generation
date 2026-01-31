"""
Task manager for tracking generation tasks.
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from app.core.cache import cache_manager
from app.utils.logging import get_logger, log_context

logger = get_logger(__name__)


class TaskManager:
    """Manages task lifecycle and tracking"""
    
    async def create_task(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        priority: int = 1
    ) -> Dict[str, Any]:
        """Create a new task"""
        
        task_id = str(uuid.uuid4())
        
        task_data = {
            "task_id": task_id,
            "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt,
            "priority": priority,
            "status": "queued",
            "progress": 0,
            "message": "Task created",
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "updated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "result": None,
            "error": None,
            "metrics": {
                "start_time": datetime.now(timezone.utc).isoformat() + "Z",
                "end_time": None,
                "total_time_ms": None
            }
        }
        
        # Store in Redis
        await cache_manager.set(
            f"task:{task_id}",
            task_data,
            expire=86400  # 24 hours
        )
        
        # Update statistics
        await self._update_stats("created")
        
        return task_data
    
    async def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update task status and data"""
        
        with log_context(task_id=task_id, operation="update_task"):
            
            # Get existing task
            task_data = await cache_manager.get(f"task:{task_id}")
            
            if not task_data:
                logger.warning("task.update.not_found", extra={"task_id": task_id})
                return False
            
            # Update fields
            if status:
                task_data["status"] = status
            if progress is not None:
                task_data["progress"] = progress
            if message:
                task_data["message"] = message
            if result is not None:
                task_data["result"] = result
            if error is not None:
                task_data["error"] = error
            if metrics:
                task_data["metrics"].update(metrics)
            
            task_data["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
            
            # Update Redis
            success = await cache_manager.set(
                f"task:{task_id}",
                task_data,
                expire=86400
            )
            
            if success:
                logger.info(
                    "task.update.success",
                    extra={
                        "task_id": task_id,
                        "status": status,
                        "progress": progress
                    }
                )
                
                # Update statistics based on status
                if status in ["completed", "failed", "cancelled"]:
                    await self._update_stats(status)
            
            return success
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return await cache_manager.get(f"task:{task_id}")
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete task (for cleanup)"""
        success = await cache_manager.delete(f"task:{task_id}")
        
        if success:
            logger.info("task.delete.success", extra={"task_id": task_id})
        
        return success
    
    async def get_user_tasks(self, user_id: str, limit: int = 50) -> list:
        """Get recent tasks for a user"""
        # Note: In production, you'd want to use Redis sorted sets or a database
        # This is a simplified implementation
        tasks = []
        
        # Scan for task keys (this is inefficient for large datasets)
        # Consider using a pattern like "task:user:{user_id}:{task_id}" for better performance
        cursor = 0
        pattern = f"task:*"
        
        while True:
            cursor, keys = await cache_manager.client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                task_data = await cache_manager.get(key)
                if task_data and task_data.get("user_id") == user_id:
                    tasks.append(task_data)
                    
                    if len(tasks) >= limit:
                        break
            
            if cursor == 0 or len(tasks) >= limit:
                break
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return tasks[:limit]
    
    async def cleanup_old_tasks(self, days: int = 7) -> int:
        """Clean up tasks older than specified days"""
        count = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        
        cursor = 0
        pattern = f"task:*"
        
        while True:
            cursor, keys = await cache_manager.client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                task_data = await cache_manager.get(key)
                if task_data:
                    created_at = datetime.fromisoformat(
                        task_data.get("created_at", "").replace("Z", "+00:00")
                    ).timestamp()
                    
                    if created_at < cutoff:
                        await cache_manager.delete(key)
                        count += 1
            
            if cursor == 0:
                break
        
        logger.info(
            "task.cleanup.completed",
            extra={
                "deleted_count": count,
                "older_than_days": days
            }
        )
        
        return count
    
    async def _update_stats(self, event_type: str):
        """Update statistics counters"""
        
        stats_key = f"stats:tasks:{event_type}"
        
        try:
            # Increment counter
            await cache_manager.client.incr(stats_key)
            
            # Set TTL if this is a new key (7 days)
            ttl = await cache_manager.client.ttl(stats_key)
            if ttl == -1:  # No TTL set
                await cache_manager.client.expire(stats_key, 604800)  # 7 days
                
        except Exception as e:
            logger.warning(
                "task.stats.update_failed",
                extra={"event_type": event_type, "error": str(e)}
            )


# Create singleton instance
task_manager = TaskManager()