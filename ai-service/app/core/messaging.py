"""
RabbitMQ queue manager for async message handling.

Manages connections, publishing, and consuming messages.
"""
import json
import asyncio
from typing import Callable, Dict, Any, Optional
from aio_pika import connect_robust, Message, Channel, Queue, Connection
from aio_pika.abc import AbstractRobustConnection, AbstractIncomingMessage
from loguru import logger

from app.config import settings
from app.utils.serialization import json_safe


class QueueManager:
    """
    Manages RabbitMQ connections and operations.
    
    Provides async interface for publishing and consuming messages.
    """
    
    def __init__(self):
        self.connection: Optional[AbstractRobustConnection] = None
        self.channel: Optional[Channel] = None
        self.request_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self._connected = False
    
    async def connect(self) -> None:
        """
        Establish connection to RabbitMQ.
        
        Creates connection pool, channel, and declares queues.
        """
        try:
            logger.info(f"Connecting to RabbitMQ: {settings.rabbitmq_url}")
            
            # Connect with automatic reconnection
            self.connection = await connect_robust(
                settings.rabbitmq_url,
                timeout=10,
                reconnect_interval=5
            )
            
            # Create channel
            self.channel = await self.connection.channel()
            
            # Set QoS (quality of service)
            await self.channel.set_qos(
                prefetch_count=settings.rabbitmq_prefetch_count
            )
            
            # Declare queues (idempotent - safe to call multiple times)
            self.request_queue = await self.channel.declare_queue(
                settings.rabbitmq_queue_ai_requests,
                durable=True  # Survive broker restart
            )
            
            self.response_queue = await self.channel.declare_queue(
                settings.rabbitmq_queue_ai_responses,
                durable=True
            )
            
            self._connected = True
            logger.info("✅ RabbitMQ connection established")
            logger.info(f"   Request queue: {settings.rabbitmq_queue_ai_requests}")
            logger.info(f"   Response queue: {settings.rabbitmq_queue_ai_responses}")
            
        except Exception as e:
            logger.error(f"❌ RabbitMQ connection failed: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """Close RabbitMQ connection gracefully."""
        if self.connection:
            await self.connection.close()
            self._connected = False
            logger.info("RabbitMQ connection closed")
    
    async def publish_response(self, response: Dict[str, Any]) -> bool:
        """
        Publish response message back to API Gateway.
        
        Args:
            response: Response data (will be JSON serialized)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            >>> await queue_manager.publish_response({
            ...     "task_id": "123",
            ...     "result": "success"
            ... })
        """
        if not self._connected or not self.channel:
            logger.error("Cannot publish: not connected to RabbitMQ")
            return False
        
        try:
            # Serialize to JSON
            body = json.dumps(response, default=json_safe).encode('utf-8')
            
            # Create message
            message = Message(
                body=body,
                content_type="application/json",
                delivery_mode=2  # Persistent (survive broker restart)
            )
            
            # Publish to response queue
            await self.channel.default_exchange.publish(
                message,
                routing_key=settings.rabbitmq_queue_ai_responses
            )
            
            logger.debug(f"📤 Published response: task_id={response.get('task_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish response: {e}")
            return False
    
    async def consume(
        self,
        queue_name: str,
        callback: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Start consuming messages from queue.
        
        Args:
            queue_name: Queue to consume from
            callback: Async function to process each message
            
        Example:
            >>> async def process_message(data: dict):
            ...     print(f"Received: {data}")
            >>> 
            >>> await queue_manager.consume("ai-requests", process_message)
        """
        if not self._connected or not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")
        
        # Declare queue (idempotent)
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        logger.info(f"👂 Consuming from queue: {queue_name}")
        
        async def process_message(message: AbstractIncomingMessage):
            """Internal message processor with error handling"""
            async with message.process():
                try:
                    # Decode JSON
                    body = json.loads(message.body.decode('utf-8'))
                    logger.debug(f"📨 Message received from {queue_name}")
                    
                    # Call user callback
                    await callback(body)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Message will be nack'ed and potentially requeued
                    raise
        
        # Start consuming
        await queue.consume(process_message)
        
        logger.info(f"✅ Consumer started for {queue_name}")
        
        # Keep consuming forever
        await asyncio.Future()  # Never completes
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to RabbitMQ"""
        return self._connected


# Global queue manager instance
queue_manager = QueueManager()


if __name__ == "__main__":
    # Test queue manager
    import time
    
    async def test_queue():
        """Test queue operations"""
        print("\n" + "=" * 60)
        print("QUEUE MANAGER TEST")
        print("=" * 60)
        
        # Connect
        print("\n[1/3] Connecting to RabbitMQ...")
        await queue_manager.connect()
        print(f"✅ Connected: {queue_manager.is_connected}")
        
        # Publish test message
        print("\n[2/3] Publishing test message...")
        test_msg = {
            "task_id": "test-123",
            "message": "Hello from queue test",
            "timestamp": time.time()
        }
        success = await queue_manager.publish_response(test_msg)
        print(f"✅ Published: {success}")
        
        # Consume test (will run for 5 seconds then stop)
        print("\n[3/3] Testing consumer (5 seconds)...")
        
        received_messages = []
        
        async def test_callback(data: dict):
            """Test callback function"""
            received_messages.append(data)
            print(f"✅ Received message: {data.get('task_id', 'unknown')}")
        
        # Start consumer in background
        consumer_task = asyncio.create_task(
            queue_manager.consume(
                settings.rabbitmq_queue_ai_responses,
                test_callback
            )
        )
        
        # Wait a bit for messages
        await asyncio.sleep(5)
        
        # Cancel consumer
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        
        print(f"\n✅ Received {len(received_messages)} message(s)")
        
        # Disconnect
        await queue_manager.disconnect()
        
        print("\n" + "=" * 60)
        print("✅ Queue manager test complete!")
        print("=" * 60 + "\n")
    
    asyncio.run(test_queue())