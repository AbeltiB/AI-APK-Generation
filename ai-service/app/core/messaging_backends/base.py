from typing import Dict, Any, Callable


class MessagingBackend:
    async def connect(self) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError

    async def publish_response(self, response: Dict[str, Any]) -> bool:
        raise NotImplementedError

    async def consume(self, queue_name: str, callback: Callable):
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError
