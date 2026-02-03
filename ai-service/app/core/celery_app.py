"""
Celery application configuration - Production Ready.
Uses Redis as broker and backend (NO RabbitMQ).
"""
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
from loguru import logger
import time

from app.config import settings


# Initialize Celery app
celery_app = Celery(
    "ai_app_builder",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer=settings.celery_task_serializer,
    accept_content=settings.celery_accept_content,
    result_serializer=settings.celery_result_serializer,
    
    # Timezone
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,
    
    # Task execution
    task_track_started=settings.celery_task_track_started,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_acks_late=settings.celery_task_acks_late,
    task_reject_on_worker_lost=settings.celery_task_reject_on_worker_lost,
    
    # Worker configuration
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    worker_max_tasks_per_child=settings.celery_worker_max_tasks_per_child,
    worker_disable_rate_limits=True,
    
    # Results
    result_expires=settings.celery_result_expires,
    result_persistent=True,
    
    # Connection retry
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    
    # Task routes (for future scaling)
    task_routes={
        'app.core.tasks.generate_task': {'queue': 'generation'},
        'app.core.tasks.*': {'queue': 'default'},
    },
    
    # Beat schedule (if needed in future)
    beat_schedule={},
)

# Task execution signals for logging
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extra):
    """Log task start"""
    logger.info(
        f"📋 Task started: {task.name}",
        extra={
            "task_id": task_id,
            "task_name": task.name,
            "args_count": len(args),
            "kwargs_count": len(kwargs),
        }
    )


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, **extra):
    """Log task completion"""
    logger.info(
        f"✅ Task completed: {task.name}",
        extra={
            "task_id": task_id,
            "task_name": task.name,
            "has_result": retval is not None,
        }
    )


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extra):
    """Log task failure"""
    logger.error(
        f"❌ Task failed: {task_id}",
        extra={
            "task_id": task_id,
            "exception": str(exception),
            "exception_type": type(exception).__name__,
        },
        exc_info=exception,
    )


logger.info("✅ Celery app initialized with Redis broker/backend")