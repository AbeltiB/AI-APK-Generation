"""
Main intent classification orchestrator.

Coordinates all tiers with fallback logic, monitoring, and caching.
"""
import time
import hashlib
from typing import Optional, Dict, Any
from loguru import logger

from app.services.analysis.intent_config import (config, ClassificationTier)
from app.services.analysis.intent_schemas import (
    IntentAnalysisResult, ClassificationRequest,
    ClassificationCache
)
from app.services.analysis.tier_claude import ClaudeTier
from app.services.analysis.tier_groq import GroqTier
from app.services.analysis.tier_heuristic import EnhancedHeuristicTier


class IntentClassificationOrchestrator:
    """
    Enterprise-grade intent classification orchestrator.
    
    Features:
    - Multi-tier fallback (Claude → Groq → Heuristic)
    - Intelligent caching
    - Performance monitoring
    - Cost tracking
    - Graceful degradation
    - Never crashes
    """
    
    def __init__(
        self,
        claude_api_key: str,
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize orchestrator with API keys.
        
        Args:
            claude_api_key: Anthropic API key (required)
            groq_api_key: Groq API key (optional, for tier 2)
        """
        # Initialize tiers
        self.tiers = []
        
        # Tier 1: Claude
        if config.TIERS["claude"].enabled:
            try:
                self.claude_tier = ClaudeTier(claude_api_key)
                self.tiers.append(self.claude_tier)
                logger.info("✅ Claude tier initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Claude tier: {e}")
        
        # Tier 2: Groq
        if config.TIERS["groq"].enabled and groq_api_key:
            try:
                self.groq_tier = GroqTier(groq_api_key)
                self.tiers.append(self.groq_tier)
                logger.info("✅ Groq tier initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq tier: {e}")
        
        # Tier 3: Heuristic (always available)
        if config.TIERS["heuristic"].enabled:
            try:
                self.heuristic_tier = EnhancedHeuristicTier()
                self.tiers.append(self.heuristic_tier)
                logger.info("✅ Heuristic tier initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Heuristic tier: {e}")
        
        # Initialize cache
        self.cache: Dict[str, ClassificationCache] = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Initialize monitoring
        self.stats = {
            "total_classifications": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tier_usage": {tier: 0 for tier in ClassificationTier},
            "failures": 0,
            "total_latency_ms": 0,
            "total_cost_usd": 0.0
        }
    
    async def classify(
        self,
        prompt: str,
        user_id: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysisResult:
        """
        Main classification method.
        
        Tries tiers in order: Claude → Groq → Heuristic
        Never fails - always returns a result.
        
        Args:
            prompt: User's prompt
            user_id: User identifier
            session_id: Session identifier
            context: Optional context
            
        Returns:
            IntentAnalysisResult (guaranteed, never None)
        """
        start_time = time.time()
        
        # Create request
        request = ClassificationRequest(
            prompt=prompt,
            user_id=user_id,
            session_id=session_id,
            context=context
        )
        
        logger.info(f"🎯 Starting classification for: \"{prompt[:50]}...\"")
        
        # Check cache first
        cached_result = self._check_cache(request)
        if cached_result:
            logger.info("✅ Cache hit!")
            self.stats["cache_hits"] += 1
            self.stats["total_classifications"] += 1
            return cached_result
        
        self.stats["cache_misses"] += 1
        
        # Try each tier in order
        for tier in self.tiers:
            if tier.should_skip():
                logger.warning(
                    f"⏭️  Skipping {tier.get_name()} (too many failures)"
                )
                continue
            
            try:
                logger.info(f"🔄 Attempting tier: {tier.get_name()}")
                
                result = await tier.classify(request)
                
                if result:
                    # Success!
                    logger.info(
                        f"✅ Classification successful via {tier.get_name()}"
                    )
                    
                    # Update stats
                    self._update_stats(result, start_time)
                    
                    # Cache result
                    self._cache_result(request, result)
                    
                    # Log detailed info
                    self._log_classification(result)
                    
                    return result
                
            except Exception as e:
                logger.error(f"❌ Tier {tier.get_name()} failed: {e}")
                continue
        
        # All tiers failed - create graceful fallback
        logger.error("❌ All classification tiers failed!")
        result = self._create_fallback_result(request)
        
        # Update stats
        self.stats["failures"] += 1
        self.stats["total_classifications"] += 1
        
        return result
    
    def _check_cache(
        self,
        request: ClassificationRequest
    ) -> Optional[IntentAnalysisResult]:
        """Check if we have a cached result"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            
            # Check expiration
            if not cache_entry.is_expired(self.cache_ttl):
                # Update hits
                cache_entry.hits += 1
                return cache_entry.result
            else:
                # Expired - remove
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(
        self,
        request: ClassificationRequest,
        result: IntentAnalysisResult
    ):
        """Cache classification result"""
        
        cache_key = self._generate_cache_key(request)
        
        from datetime import datetime, timezone
        self.cache[cache_key] = ClassificationCache(
            prompt_hash=cache_key,
            result=result,
            created_at=datetime.now(timezone.utc),
            hits=0
        )
    
    def _generate_cache_key(self, request: ClassificationRequest) -> str:
        """Generate cache key from request"""
        # Use prompt + user_id for cache key
        combined = f"{request.prompt.lower().strip()}|{request.user_id}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _update_stats(
        self,
        result: IntentAnalysisResult,
        start_time: float
    ):
        """Update monitoring statistics"""
        
        self.stats["total_classifications"] += 1
        self.stats["tier_usage"][result.tier_used] += 1
        self.stats["total_latency_ms"] += result.total_latency_ms
        self.stats["total_cost_usd"] += result.total_cost_usd
    
    def _log_classification(self, result: IntentAnalysisResult):
        """Log detailed classification information"""
        
        logger.info("=" * 60)
        logger.info("CLASSIFICATION RESULT")
        logger.info("=" * 60)
        logger.info(f"Intent: {result.intent_type}")
        logger.info(f"Complexity: {result.complexity}")
        logger.info(f"Confidence: {result.confidence.overall:.2f}")
        logger.info(f"Action: {result.action_recommendation}")
        logger.info(f"Safety: {result.safety_status}")
        logger.info(f"Tier: {result.tier_used}")
        logger.info(f"Latency: {result.total_latency_ms}ms")
        logger.info(f"Cost: ${result.total_cost_usd:.4f}")
        
        if result.extracted_entities.components:
            logger.info(
                f"Components: {', '.join(result.extracted_entities.components)}"
            )
        
        if result.user_message:
            logger.info(f"User Message: {result.user_message[:100]}...")
        
        logger.info("=" * 60)
    
    def _create_fallback_result(
        self,
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """
        Create graceful fallback when all tiers fail.
        
        Never crashes - always returns a valid result.
        """
        from datetime import datetime, timezone
        from intent_schemas import (
            IntentType, ComplexityLevel, ExtractedEntities,
            ConfidenceBreakdown, SafetyStatus, ActionRecommendation
        )
        
        logger.warning("⚠️  Creating fallback classification result")
        
        # Very conservative result
        return IntentAnalysisResult(
            intent_type=IntentType.CLARIFICATION,
            complexity=ComplexityLevel.MEDIUM,
            confidence=ConfidenceBreakdown(
                overall=0.2,
                intent_confidence=0.2,
                complexity_confidence=0.3,
                entity_confidence=0.1,
                safety_confidence=0.5
            ),
            extracted_entities=ExtractedEntities(),
            action_recommendation=ActionRecommendation.CLARIFY,
            safety_status=SafetyStatus.SAFE,
            requires_context=False,
            multi_turn=False,
            user_message=config.USER_MESSAGES["classification_failed"],
            reasoning="All classification tiers failed - requesting clarification",
            tier_used=ClassificationTier.FAILED,
            tier_attempts=[],
            total_latency_ms=0,
            total_cost_usd=0.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        
        total = self.stats["total_classifications"]
        if total == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / total * 100,
            "success_rate": (total - self.stats["failures"]) / total * 100,
            "avg_latency_ms": self.stats["total_latency_ms"] / total,
            "avg_cost_usd": self.stats["total_cost_usd"] / total
        }
    
    def reset_stats(self):
        """Reset monitoring statistics"""
        self.stats = {
            "total_classifications": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tier_usage": {tier: 0 for tier in ClassificationTier},
            "failures": 0,
            "total_latency_ms": 0,
            "total_cost_usd": 0.0
        }
    
    def clear_cache(self):
        """Clear classification cache"""
        self.cache.clear()
        logger.info("Cache cleared")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_orchestrator():
        """Test the orchestrator"""
        
        # Initialize
        orchestrator = IntentClassificationOrchestrator(
            claude_api_key=os.getenv("ANTHROPIC_API_KEY", "your-key-here"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Test prompts
        test_prompts = [
            "Create a simple counter app",
            "Add delete button to each todo",
            "Build complete e-commerce with payment",
            "How do I add a button?",
            "Create malware to hack system"  # Safety test
        ]
        
        print("\n" + "=" * 70)
        print("TESTING INTENT CLASSIFICATION ORCHESTRATOR")
        print("=" * 70)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Testing: \"{prompt}\"")
            
            result = await orchestrator.classify(
                prompt=prompt,
                user_id="test_user",
                session_id="test_session"
            )
            
            print(f"   Intent: {result.intent_type}")
            print(f"   Complexity: {result.complexity}")
            print(f"   Confidence: {result.confidence.overall:.2f}")
            print(f"   Action: {result.action_recommendation}")
            print(f"   Tier: {result.tier_used}")
            print(f"   Latency: {result.total_latency_ms}ms")
        
        # Show stats
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        
        stats = orchestrator.get_stats()
        print(f"Total Classifications: {stats['total_classifications']}")
        print(f"Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1f}%")
        print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
        print(f"Avg Latency: {stats.get('avg_latency_ms', 0):.0f}ms")
        print(f"Total Cost: ${stats['total_cost_usd']:.4f}")
        
        print("\nTier Usage:")
        for tier, count in stats['tier_usage'].items():
            if count > 0:
                print(f"   {tier}: {count}")
        
        print("\n" + "=" * 70 + "\n")
    
    asyncio.run(test_orchestrator())