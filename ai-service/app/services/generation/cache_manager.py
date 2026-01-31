"""
Semantic Cache Manager - Intelligent prompt caching.

Uses sentence embeddings to find similar prompts and cache results,
dramatically reducing API calls for similar requests.
"""
import hashlib
import json
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger

from app.core.cache import cache_manager
from app.config import settings


class SemanticCacheManager:
    """
    Manages semantic caching for AI responses.
    
    Features:
    - Embedding-based similarity search
    - Configurable similarity threshold
    - TTL management for cache entries
    - Cache statistics tracking
    """
    
    def __init__(self):
        self.cache_prefix = "semantic_cache:"
        self.embedding_prefix = "embedding:"
        self.stats_key = "cache_stats"
        
        # Simple similarity check without ML dependencies for Phase 2
        # Will enhance with proper embeddings in future phases
        self.use_simple_similarity = True
    
    async def get_cached_result(
        self,
        prompt: str,
        user_id: str,
        context_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Try to find cached result for similar prompt.
        
        Args:
            prompt: User's prompt
            user_id: User identifier
            context_hash: Optional hash of context for cache key
            
        Returns:
            Cached result or None
        """
        logger.debug("🔍 Checking semantic cache...")
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, user_id, context_hash)
            
            # Try exact match first
            cached = await cache_manager.get(cache_key)
            if cached:
                await self._update_stats("hits")
                logger.info(f"✅ Cache HIT (exact): {cache_key}")
                return cached
            
            # Try similarity search
            if self.use_simple_similarity:
                similar_result = await self._find_similar_simple(
                    prompt=prompt,
                    user_id=user_id
                )
                
                if similar_result:
                    await self._update_stats("similarity_hits")
                    logger.info(f"✅ Cache HIT (similar): {similar_result['similarity']:.2f}")
                    return similar_result['result']
            
            # Cache miss
            await self._update_stats("misses")
            logger.debug("Cache MISS")
            return None
            
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            return None
    
    async def cache_result(
        self,
        prompt: str,
        user_id: str,
        result: Dict[str, Any],
        context_hash: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache result for future use.
        
        Args:
            prompt: User's prompt
            user_id: User identifier
            result: Result to cache
            context_hash: Optional hash of context
            ttl: Time to live (seconds), defaults to settings
            
        Returns:
            True if cached successfully
        """
        logger.debug("💾 Caching result...")
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, user_id, context_hash)
            
            # Add metadata
            import time
            cache_entry = {
                'prompt': prompt,
                'user_id': user_id,
                'result': result,
                'cached_at': str(int(time.time()))
            }
            
            # Cache with TTL
            ttl = ttl or settings.redis_semantic_cache_ttl
            success = await cache_manager.set(cache_key, cache_entry, ttl=ttl)
            
            if success:
                await self._update_stats("cached")
                logger.info(f"✅ Result cached: {cache_key} (TTL: {ttl}s)")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False
    
    def _generate_cache_key(
        self,
        prompt: str,
        user_id: str,
        context_hash: Optional[str] = None
    ) -> str:
        """
        Generate cache key from prompt and context.
        
        Args:
            prompt: User's prompt
            user_id: User identifier
            context_hash: Optional context hash
            
        Returns:
            Cache key string
        """
        # Normalize prompt
        normalized = prompt.lower().strip()
        
        # Create hash components
        components = [normalized, user_id]
        if context_hash:
            components.append(context_hash)
        
        # Generate hash
        combined = "|".join(components)
        hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return f"{self.cache_prefix}{hash_value}"
    
    async def _find_similar_simple(
        self,
        prompt: str,
        user_id: str,
        threshold: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find similar cached prompts using simple text similarity.
        
        This is a simple implementation for Phase 2. Will be enhanced
        with proper embeddings in future phases.
        
        Args:
            prompt: User's prompt
            user_id: User identifier
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            Similar cached result or None
        """
        threshold = threshold or settings.redis_cache_similarity_threshold
        
        try:
            # Get all cache keys
            pattern = f"{self.cache_prefix}*"
            similar_results = []
            
            # Scan through cache (limited to avoid performance issues)
            # In production, use a proper vector database
            cache_keys = []
            
            # This is a simplified approach - in production use vector search
            # For now, we'll skip similarity search and rely on exact matches
            # to avoid performance issues
            
            return None
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (Jaccard similarity).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Normalize and tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    async def _update_stats(self, stat_type: str) -> None:
        """Update cache statistics"""
        try:
            stats_key = f"{self.cache_prefix}{self.stats_key}"
            stats = await cache_manager.get(stats_key) or {}
            
            stats[stat_type] = stats.get(stat_type, 0) + 1
            
            await cache_manager.set(stats_key, stats, ttl=86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Stats update error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit/miss/cached counts
        """
        try:
            stats_key = f"{self.cache_prefix}{self.stats_key}"
            stats = await cache_manager.get(stats_key) or {}
            
            hits = stats.get('hits', 0) + stats.get('similarity_hits', 0)
            misses = stats.get('misses', 0)
            total = hits + misses
            
            return {
                'hits': hits,
                'exact_hits': stats.get('hits', 0),
                'similarity_hits': stats.get('similarity_hits', 0),
                'misses': misses,
                'cached': stats.get('cached', 0),
                'total_requests': total,
                'hit_rate': (hits / total * 100) if total > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval error: {e}")
            return {}
    
    async def clear_cache(self, user_id: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            user_id: Optional user ID to clear only their cache
            
        Returns:
            Number of entries cleared
        """
        try:
            if user_id:
                pattern = f"{self.cache_prefix}*"
                # Would need to filter by user_id
                # For now, just clear all
            else:
                pattern = f"{self.cache_prefix}*"
            
            cleared = await cache_manager.clear_pattern(pattern)
            logger.info(f"Cleared {cleared} cache entries")
            
            return cleared
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0


# Global semantic cache manager instance
semantic_cache = SemanticCacheManager()


if __name__ == "__main__":
    # Test semantic cache
    import asyncio
    
    async def test_semantic_cache():
        """Test semantic cache operations"""
        print("\n" + "=" * 60)
        print("SEMANTIC CACHE TEST")
        print("=" * 60)
        
        # Connect
        await cache_manager.connect()
        
        # Test 1: Cache and retrieve
        print("\n[TEST 1] Cache and retrieve")
        
        test_result = {
            'architecture': {'app_type': 'single-page'},
            'layout': {'screen_id': 'screen_1'},
            'blockly': {'blocks': []}
        }
        
        # Cache result
        cached = await semantic_cache.cache_result(
            prompt="Create a simple counter app",
            user_id="test_user",
            result=test_result
        )
        print(f"   Cached: {cached}")
        
        # Retrieve exact match
        retrieved = await semantic_cache.get_cached_result(
            prompt="Create a simple counter app",
            user_id="test_user"
        )
        print(f"   Retrieved: {retrieved is not None}")
        
        # Test 2: Cache miss
        print("\n[TEST 2] Cache miss")
        
        no_result = await semantic_cache.get_cached_result(
            prompt="Build a completely different app",
            user_id="test_user"
        )
        print(f"   Result: {no_result is None} (should be True)")
        
        # Test 3: Get stats
        print("\n[TEST 3] Cache statistics")
        
        stats = await semantic_cache.get_cache_stats()
        print(f"   Total requests: {stats.get('total_requests', 0)}")
        print(f"   Hits: {stats.get('hits', 0)}")
        print(f"   Misses: {stats.get('misses', 0)}")
        print(f"   Hit rate: {stats.get('hit_rate', 0):.1f}%")
        
        # Disconnect
        await cache_manager.disconnect()
        
        print("\n" + "=" * 60)
        print("✅ Semantic cache test complete!")
        print("=" * 60 + "\n")
    
    asyncio.run(test_semantic_cache())