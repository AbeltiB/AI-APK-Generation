"""
Enterprise-grade structured logging system.

Features:
- JSON formatted logs
- Correlation ID tracking
- Request tracing
- Performance metrics
- Error tracking
- Machine searchable
- Environment metadata
- Forensic analysis ready
"""
import sys
import json
import logging
import traceback
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from contextvars import ContextVar
from functools import wraps
import socket
import os

from loguru import logger as loguru_logger
from app.config import settings

# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
task_id_var: ContextVar[Optional[str]] = ContextVar('task_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class StructuredLogger:
    """
    Enterprise structured logger with correlation tracking.
    
    All logs are JSON formatted with:
    - Timestamp (ISO 8601)
    - Correlation ID (traces entire request)
    - Task ID (specific AI task)
    - User/Session IDs
    - Service metadata
    - Environment info
    - Performance metrics
    """
    
    def __init__(self, name: str):
        self.name = name
        self.hostname = socket.gethostname()
        self.service_name = settings.app_name
        self.service_version = settings.app_version
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.instance_id = os.getenv("INSTANCE_ID", self.hostname)
    
    def _get_base_context(self) -> Dict[str, Any]:
        """Get base logging context"""
        return {
            "@timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "environment": self.environment,
                "instance_id": self.instance_id,
                "hostname": self.hostname
            },
            "logger": {
                "name": self.name
            },
            "correlation": {
                "correlation_id": correlation_id_var.get(),
                "task_id": task_id_var.get(),
                "user_id": user_id_var.get(),
                "session_id": session_id_var.get()
            }
        }
    
    def _format_log(
        self,
        level: str,
        event: str,
        message: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Format log entry as JSON"""
        
        log_entry = self._get_base_context()
        
        log_entry.update({
            "level": level.upper(),
            "event": event,
            "message": message or event
        })
        
        # Add extra fields
        if extra:
            log_entry["data"] = extra
        
        # Add exception info
        if exc_info:
            log_entry["error"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "stacktrace": traceback.format_exc()
            }
        
        return log_entry
    
    def debug(self, event: str, message: str = None, extra: Dict = None, **kwargs):
        """Log debug message"""
        log_entry = self._format_log("DEBUG", event, message, extra)
        print(json.dumps(log_entry, default=str), file=sys.stdout)
    
    def info(self, event: str, message: str = None, extra: Dict = None, **kwargs):
        """Log info message"""
        log_entry = self._format_log("INFO", event, message, extra)
        print(json.dumps(log_entry, default=str), file=sys.stdout)
    
    def warning(self, event: str, message: str = None, extra: Dict = None, **kwargs):
        """Log warning message"""
        log_entry = self._format_log("WARNING", event, message, extra)
        print(json.dumps(log_entry, default=str), file=sys.stderr)
    
    def error(
        self,
        event: str,
        message: str = None,
        extra: Dict = None,
        exc_info: Exception = None,
        **kwargs
    ):
        """Log error message"""
        log_entry = self._format_log("ERROR", event, message, extra, exc_info)
        print(json.dumps(log_entry), file=sys.stderr)
    
    def critical(
        self,
        event: str,
        message: str = None,
        extra: Dict = None,
        exc_info: Exception = None,
        **kwargs
    ):
        """Log critical message"""
        log_entry = self._format_log("CRITICAL", event, message, extra, exc_info)
        print(json.dumps(log_entry, default=str), file=sys.stderr)
    
    def performance(
        self,
        event: str,
        duration_ms: float,
        extra: Dict = None
    ):
        """Log performance metric"""
        perf_data = {
            "performance": {
                "duration_ms": duration_ms,
                "duration_seconds": duration_ms / 1000
            }
        }
        
        if extra:
            perf_data.update(extra)
        
        log_entry = self._format_log("INFO", event, f"Performance: {duration_ms}ms", perf_data)
        print(json.dumps(log_entry, default=str), file=sys.stdout)


def get_logger(name: str) -> StructuredLogger:
    """
    Get structured logger for module.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("user.login", extra={"user_id": "123"})
    """
    return StructuredLogger(name)


class log_context:
    """
    Context manager for correlation tracking.
    
    Usage:
        with log_context(correlation_id="abc", task_id="123"):
            logger.info("processing.started")
    """
    
    def __init__(
        self,
        correlation_id: str = None,
        task_id: str = None,
        user_id: str = None,
        session_id: str = None,
        **kwargs
    ):
        self.correlation_id = correlation_id
        self.task_id = task_id
        self.user_id = user_id
        self.session_id = session_id
        self.extra_context = kwargs
        
        # Store previous values for restoration
        self.prev_correlation = None
        self.prev_task = None
        self.prev_user = None
        self.prev_session = None
    
    def __enter__(self):
        """Set context variables"""
        self.prev_correlation = correlation_id_var.get()
        self.prev_task = task_id_var.get()
        self.prev_user = user_id_var.get()
        self.prev_session = session_id_var.get()
        
        if self.correlation_id:
            correlation_id_var.set(self.correlation_id)
        if self.task_id:
            task_id_var.set(self.task_id)
        if self.user_id:
            user_id_var.set(self.user_id)
        if self.session_id:
            session_id_var.set(self.session_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context"""
        correlation_id_var.set(self.prev_correlation)
        task_id_var.set(self.prev_task)
        user_id_var.set(self.prev_user)
        session_id_var.set(self.prev_session)


def trace_async(event_prefix: str):
    """
    Decorator for tracing async functions.
    
    Usage:
        @trace_async("database.query")
        async def fetch_user(user_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            start_time = datetime.now(timezone.utc)
            
            logger.info(
                f"{event_prefix}.started",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.performance(
                    f"{event_prefix}.completed",
                    duration_ms=duration_ms,
                    extra={
                        "function": func.__name__,
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.error(
                    f"{event_prefix}.failed",
                    extra={
                        "function": func.__name__,
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__
                    },
                    exc_info=e
                )
                raise
        
        return wrapper
    return decorator


def trace_sync(event_prefix: str):
    """
    Decorator for tracing sync functions.
    
    Usage:
        @trace_sync("cache.get")
        def get_cached_value(key: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            start_time = datetime.now(timezone.utc)
            
            logger.info(
                f"{event_prefix}.started",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.performance(
                    f"{event_prefix}.completed",
                    duration_ms=duration_ms,
                    extra={
                        "function": func.__name__,
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.error(
                    f"{event_prefix}.failed",
                    extra={
                        "function": func.__name__,
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__
                    },
                    exc_info=e
                )
                raise
        
        return wrapper
    return decorator


# Example usage patterns for documentation
if __name__ == "__main__":
    # Initialize logger
    logger = get_logger("example")
    
    # Basic logging
    logger.info("app.started")
    
    # With context
    with log_context(correlation_id="req-123", user_id="user-456"):
        logger.info("request.processing", extra={"action": "create_app"})
        
        # Nested context inherits correlation_id
        with log_context(task_id="task-789"):
            logger.info("task.executing", extra={"step": 1})
    
    # Performance logging
    logger.performance("api.request", duration_ms=450.5, extra={"endpoint": "/generate"})
    
    # Error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("operation.failed", exc_info=e, extra={"operation": "test"})


"""
LOG EVENT NAMING CONVENTIONS:

Use dot notation: <domain>.<action>.<result>

Examples:
- api.request.received
- api.request.validated
- api.rate_limit.exceeded
- pipeline.stage.started
- pipeline.stage.completed
- pipeline.stage.failed
- database.query.executed
- database.connection.lost
- cache.get.hit
- cache.get.miss
- queue.message.published
- queue.message.consumed
- claude.api.called
- claude.api.succeeded
- claude.api.rate_limited

Searchable queries:
- All API events: event:api.*
- All failures: event:*.failed OR level:ERROR
- User journey: correlation.user_id:123
- Slow operations: performance.duration_ms:>5000
- Specific task: correlation.task_id:abc-123
"""