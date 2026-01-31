"""
Rate Limiter - Prevent abuse and manage API quotas.

Uses Redis for distributed rate limiting across multiple service instances.
"""
import time
from loguru import logger
from typing import Optional, Tuple, Dict, Any

from app.core.cache import cache_manager
from app.config import settings


class RateLimiter:
    """
    Fixed window rate limiter using Redis.
    
    Features:
    - Per-user rate limiting
    - Configurable limits
    - Distributed across instances
    - Automatic token refill
    """
    
    def __init__(self):
        self.prefix = "rate_limit:"
        self.window_seconds = 3600  # 1 hour window TODO: Make configurable if needed
    
    async def check_rate_limit(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            limit: Optional custom limit (defaults to settings)
            
        Returns:
            Tuple of (allowed, info_dict)
            - allowed: True if request is allowed
            - info_dict: Rate limit information
        """
        if not settings.rate_limit_enabled:
            return True, {'limited': False, 'reason': 'Rate limiting disabled'}
        
        limit = limit or settings.rate_limit_requests_per_hour
        
        try:
            key = f"{self.prefix}{user_id}"
            
            # Get current count
            current_data = await cache_manager.get(key)
            
            if current_data is None:
                # First request - initialize
                data = {
                    'count': 1,
                    'reset_at': int(time.time()) + self.window_seconds,
                    'limit': limit
                }
                await cache_manager.set(key, data, ttl=self.window_seconds)
                
                logger.debug(f"Rate limit initialized for {user_id}: 1/{limit}")
                
                return True, {
                    'limited': False,
                    'remaining': limit - 1,
                    'limit': limit,
                    'reset_at': data['reset_at']
                }
            
            # Check if window has expired
            current_time = int(time.time())
            if current_time >= current_data['reset_at']:
                # Window expired - reset
                data = {
                    'count': 1,
                    'reset_at': current_time + self.window_seconds,
                    'limit': limit
                }
                await cache_manager.set(key, data, ttl=self.window_seconds)
                
                logger.debug(f"Rate limit window reset for {user_id}")
                
                return True, {
                    'limited': False,
                    'remaining': limit - 1,
                    'limit': limit,
                    'reset_at': data['reset_at']
                }
            
            # Check if limit exceeded
            if current_data['count'] >= limit:
                logger.warning(f"Rate limit exceeded for {user_id}: {current_data['count']}/{limit}")
                
                return False, {
                    'limited': True,
                    'remaining': 0,
                    'limit': limit,
                    'reset_at': current_data['reset_at'],
                    'retry_after': current_data['reset_at'] - current_time
                }
            
            # Increment count
            current_data['count'] += 1
            await cache_manager.set(key, current_data, ttl=self.window_seconds)
            
            logger.debug(f"Rate limit check for {user_id}: {current_data['count']}/{limit}")
            
            return True, {
                'limited': False,
                'remaining': limit - current_data['count'],
                'limit': limit,
                'reset_at': current_data['reset_at']
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request if rate limiter fails
            return True, {'limited': False, 'error': str(e)}
    
    async def get_rate_limit_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get current rate limit information for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Rate limit info dictionary
        """
        try:
            key = f"{self.prefix}{user_id}"
            data = await cache_manager.get(key)
            
            if data is None:
                return {
                    'count': 0,
                    'remaining': settings.rate_limit_requests_per_hour,
                    'limit': settings.rate_limit_requests_per_hour,
                    'reset_at': int(time.time()) + self.window_seconds
                }
            
            return {
                'count': data['count'],
                'remaining': data['limit'] - data['count'],
                'limit': data['limit'],
                'reset_at': data['reset_at']
            }
            
        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {}
    
    async def reset_rate_limit(self, user_id: str) -> bool:
        """
        Reset rate limit for user (admin function).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if reset successfully
        """
        try:
            key = f"{self.prefix}{user_id}"
            success = await cache_manager.delete(key)
            
            if success:
                logger.info(f"Rate limit reset for {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Rate limit reset error: {e}")
            return False


# Global rate limiter instance
rate_limiter = RateLimiter()


if __name__ == "__main__":
    # Test rate limiter
    import asyncio
    
    async def test_rate_limiter():
        """Test rate limiter"""
        print("\n" + "=" * 60)
        print("RATE LIMITER TEST")
        print("=" * 60)
        
        # Connect
        await cache_manager.connect()
        
        # Reset for clean test
        await rate_limiter.reset_rate_limit("test_user_rate_limit")
        
        # Test 1: Normal requests
        print("\n[TEST 1] Normal requests (within limit)")
        
        for i in range(5):
            allowed, info = await rate_limiter.check_rate_limit(
                user_id="test_user_rate_limit",
                limit=10
            )
            print(f"   Request {i+1}: Allowed={allowed}, Remaining={info.get('remaining', 'N/A')}")
        
        # Test 2: Get info
        print("\n[TEST 2] Get rate limit info")
        
        info = await rate_limiter.get_rate_limit_info("test_user_rate_limit")
        print(f"   Count: {info.get('count', 0)}")
        print(f"   Remaining: {info.get('remaining', 0)}")
        print(f"   Limit: {info.get('limit', 0)}")
        
        # Test 3: Exceed limit
        print("\n[TEST 3] Exceed rate limit")
        
        # Make requests until limit exceeded
        for i in range(10):
            allowed, info = await rate_limiter.check_rate_limit(
                user_id="test_user_rate_limit",
                limit=10
            )
            
            if not allowed:
                print(f"   Request {i+6}: BLOCKED (limit exceeded)")
                print(f"   Retry after: {info.get('retry_after', 0)} seconds")
                break
        
        # Test 4: Reset
        print("\n[TEST 4] Reset rate limit")
        
        reset = await rate_limiter.reset_rate_limit("test_user_rate_limit")
        print(f"   Reset: {reset}")
        
        info = await rate_limiter.get_rate_limit_info("test_user_rate_limit")
        print(f"   New count: {info.get('count', 0)}")
        
        # Disconnect
        await cache_manager.disconnect()
        
        print("\n" + "=" * 60)
        print("✅ Rate limiter test complete!")
        print("=" * 60 + "\n")
    
    asyncio.run(test_rate_limiter())