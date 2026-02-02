"""
Celery application configuration.
"""
from celery import Celery
from loguru import logger

from app.config import settings


celery_app = Celery(
    "ai_app_builder",
    broker=settings.celery_broker_url,          # e.g. redis://redis:6379/0
    backend=settings.celery_result_backend,     # e.g. redis://redis:6379/1
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,

    broker_connection_retry_on_startup=True,
)

logger.info("✅ Celery app initialized")
