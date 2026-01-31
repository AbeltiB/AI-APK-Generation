"""
Redis cache manager for AI response caching.

Provides async interface to Redis with automatic serialization/deserialization.
"""
import json
from typing import Any, Optional
import redis.asyncio as redis
from loguru import logger

from app.config import settings


class CacheManager:
    """
    Manages Redis caching operations.
    
    Handles connection management, serialization, and error handling
    for caching AI responses and other data.
    """
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> None:
        """
        Establish connection to Redis.
        
        Creates a connection pool and tests the connection.
        """
        try:
            self.client = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10 #TODO: Make configurable if needed
            )
            
            # Test connection
            await self.client.ping()
            self._connected = True
            
            logger.info(f"✅ Redis cache connected: {settings.redis_url}")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self._connected = False
            logger.info("Redis cache disconnected")
    
    async def get(self, key: str) -> Optional[dict]:
        """
        Get cached value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value as dict, or None if not found
            
        Example:
            >>> cache = CacheManager()
            >>> await cache.connect()
            >>> value = await cache.get("my_key")
        """
        if not self._connected or not self.client:
            logger.warning("Cache not connected, skipping get")
            return None
        
        try:
            cached = await self.client.get(key)
            
            if cached:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(cached)
            
            logger.debug(f"Cache MISS: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set cache value with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (default: from settings)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            >>> await cache.set("my_key", {"data": "value"}, ttl=3600)
        """
        logger.debug(f"Redis SET key={key}, ttl={ttl}")


        if not self._connected or not self.client:
            logger.warning("Cache not connected, skipping set")
            return False
        
        try:
            ttl = ttl or settings.redis_cache_ttl
            serialized = json.dumps(value)
            
            await self.client.setex(key, ttl, serialized)
            
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete cached value.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False otherwise
        """
        if not self._connected or not self.client:
            return False
        
        try:
            result = await self.client.delete(key)
            if result:
                logger.debug(f"Cache DELETE: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not self._connected or not self.client:
            return False
        
        try:
            result = await self.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "ai:*" for all AI cache keys)
            
        Returns:
            Number of keys deleted
            
        Example:
            >>> # Clear all AI response caches
            >>> deleted = await cache.clear_pattern("ai:response:*")
        """
        if not self._connected or not self.client:
            return 0
        
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cache cleared {deleted} keys matching '{pattern}'")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear pattern error for '{pattern}': {e}")
            return 0


# Global cache instance
cache_manager = CacheManager()


if __name__ == "__main__":
    # Test cache manager
    import asyncio
    
    async def test_cache():
        """Test cache operations"""
        print("\n" + "=" * 60)
        print("CACHE MANAGER TEST")
        print("=" * 60)
        
        # Connect
        await cache_manager.connect()
        
        # Test set/get
        test_data = {"message": "Hello from cache!", "count": 42}
        await cache_manager.set("test_key", test_data, ttl=60)
        
        retrieved = await cache_manager.get("test_key")
        print(f"\n✅ Stored: {test_data}")
        print(f"✅ Retrieved: {retrieved}")
        
        # Test exists
        exists = await cache_manager.exists("test_key")
        print(f"✅ Key exists: {exists}")
        
        # Test delete
        deleted = await cache_manager.delete("test_key")
        print(f"✅ Deleted: {deleted}")
        
        exists_after = await cache_manager.exists("test_key")
        print(f"✅ Key exists after delete: {exists_after}")
        
        # Disconnect
        await cache_manager.disconnect()
        
        print("\n" + "=" * 60)
        print("✅ Cache manager test complete!")
        print("=" * 60)
    
    asyncio.run(test_cache())