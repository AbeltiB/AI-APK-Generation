import asyncio
from typing import Dict, Any, Callable
from loguru import logger

from app.core.messaging_backends.base import MessagingBackend
from app.core.tasks import generate_task


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
            task = generate_task.delay(response)
            logger.debug(f"📤 Celery task enqueued: {task.id}")
            return True
        except Exception as e:
            logger.error(f"❌ Celery enqueue failed: {e}")
            return False

    async def consume(self, queue_name: str, callback: Callable):
        """
        No-op by design.
        Celery workers consume tasks, not the API process.
        """
        logger.warning(
            f"consume('{queue_name}') ignored — handled by Celery workers"
        )
        await asyncio.Future()  # preserve blocking contract

    @property
    def is_connected(self) -> bool:
        return self._connected
