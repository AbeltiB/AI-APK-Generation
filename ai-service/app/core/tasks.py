"""
Celery tasks for async AI processing.
"""
from typing import Dict, Any
from loguru import logger

from app.core.celery_app import celery_app


@celery_app.task(
    name="ai.generate",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5},
    retry_backoff=True,
)
def generate_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic async task entrypoint.
    """
    task_id = self.request.id
    logger.info(f"🚀 Celery task started: {task_id}")

    # 🔴 PLACEHOLDER
    # Call your pipeline here:
    # result = pipeline.run(payload)

    result = {
        "task_id": task_id,
        "status": "success",
        "payload": payload,
    }

    logger.info(f"✅ Celery task completed: {task_id}")
    return result
