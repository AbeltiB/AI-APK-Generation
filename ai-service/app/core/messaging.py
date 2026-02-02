"""
Messaging abstraction for async message handling.

Public API remains stable while backend implementation
can be swapped (Celery, RabbitMQ, etc).
"""
from typing import Dict, Any, Callable
from loguru import logger

from app.core.messaging_backends.celery_backend import CeleryBackend


class QueueManager:
    """
    Manages async messaging operations.

    Backend-agnostic adapter.
    """

    def __init__(self):
        self._backend = CeleryBackend()

    async def connect(self) -> None:
        logger.info("Initializing messaging backend...")
        await self._backend.connect()

    async def disconnect(self) -> None:
        await self._backend.disconnect()

    async def publish_response(self, response: Dict[str, Any]) -> bool:
        return await self._backend.publish_response(response)

    async def consume(
        self,
        queue_name: str,
        callback: Callable[[Dict[str, Any]], Any],
    ) -> None:
        await self._backend.consume(queue_name, callback)

    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected


# Global instance (unchanged)
queue_manager = QueueManager()
