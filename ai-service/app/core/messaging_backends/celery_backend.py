import asyncio
from typing import Dict, Any, Callable
from loguru import logger

from app.core.messaging_backends.base import MessagingBackend
from app.core.tasks import generate_task
from celery import current_app  # Import celery current_app


class CeleryBackend(MessagingBackend):
    """
    Adapter that maps QueueManager semantics to Celery.
    """

    def __init__(self):
        self._connected = False

    async def connect(self) -> None:
        # Celery does not require explicit connection
        self._connected = True
        logger.info("✅ Celery backend connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Celery backend disconnected")

    async def publish_response(self, response: Dict[str, Any]) -> bool:
        try:
            # Determine message type and route appropriately
            message_type = self._get_message_type(response)
            
            if message_type == "progress":
                # Progress updates: just log, don't create tasks
                logger.debug(f"📊 Progress: {response.get('stage')} - {response.get('progress')}%")
                return True
            elif message_type == "result" or message_type == "error":
                # Results/errors: just log, don't create tasks
                logger.debug(f"📤 {message_type.title()}: {response.get('task_id')}")
                return True
            elif message_type == "generation":
                # Only actual generation requests should create tasks
                task = generate_task.delay(response)
                logger.debug(f"📤 Generation task enqueued: {task.id}")
                return True
            else:
                # Unknown type - log warning and don't create task
                logger.warning(f"⚠️ Unknown message type: {response}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Celery enqueue failed: {e}")
            return False

    def _get_message_type(self, response: Dict[str, Any]) -> str:
        """Determine the type of message based on content."""
        if 'stage' in response or 'progress' in response:
            return "progress"
        elif 'error' in response:
            return "error"
        elif 'success' in response:
            return "result"
        elif 'user_id' in response and 'prompt' in response:
            return "generation"
        else:
            return "unknown"

    async def consume(self, queue_name: str, callback: Callable):
        logger.warning(
            f"consume('{queue_name}') ignored — handled by Celery workers"
        )
        await asyncio.Future()

    @property
    def is_connected(self) -> bool:
        return self._connected