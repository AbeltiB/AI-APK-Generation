"""
app/llm/orchestrator.py
Enhanced smart LLM routing with JSON validation and fallback
"""
import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider
from .llama3_provider import Llama3Provider
from .heuristic_provider import HeuristicProvider


logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Enhanced orchestrator with:
    - JSON validation
    - Smart fallback based on response quality
    - Performance monitoring
    - Response caching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize providers
        try:
            self.primary_provider = Llama3Provider(config)
            self.primary_available = True
        except Exception as e:
            logger.error(f"Failed to initialize Llama3 provider: {e}")
            self.primary_available = False
            self.primary_provider = None
        
        self.fallback_provider = HeuristicProvider(config)
        
        # Enhanced failure tracking
        self.failure_threshold = config.get("failure_threshold", 3)
        self.failure_window = config.get("failure_window_minutes", 5)
        self.json_validation_failures = config.get("json_validation_threshold", 2)
        
        self.failure_count = 0
        self.json_failures = 0
        self.last_failure_time = None
        self.force_fallback = not self.primary_available
        
        # Performance tracking
        self.total_requests = 0
        self.primary_success = 0
        self.fallback_success = 0
        self.response_times = []
        
        # Response cache for common queries
        self.response_cache: Dict[str, tuple[LLMResponse, datetime]] = {}
        self.cache_ttl = config.get("cache_ttl_seconds", 300)
        
        logger.info(f"LLM Orchestrator initialized. Primary available: {self.primary_available}")
    
    def _generate_cache_key(self, messages: List[LLMMessage], **kwargs) -> str:
        """Generate cache key from request parameters"""
        import hashlib
        
        key_data = {
            "messages": [(msg.role, msg.content[:100]) for msg in messages],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens"),
            "provider": "llama3"
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            response, cached_time = self.response_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key[:8]}")
                return response
            else:
                del self.response_cache[cache_key]
        
        return None
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        force_provider: Optional[LLMProvider] = None,
        validate_json: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Enhanced generation with smart routing, JSON validation, and caching.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            force_provider: Force specific provider
            validate_json: Validate response is valid JSON
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse from successful provider
        """
        self.total_requests += 1
        start_time = datetime.now()
        
        # Check cache first (only for primary provider)
        cache_key = self._generate_cache_key(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        cached_response = self._get_cached_response(cache_key)
        if cached_response and not force_provider == LLMProvider.HEURISTIC:
            logger.info(f"Using cached response for request")
            return cached_response
        
        # Reset failure count if outside failure window
        self._check_failure_window()
        
        # Determine provider strategy
        use_primary = (
            self.primary_available and 
            not self.force_fallback and 
            force_provider != LLMProvider.HEURISTIC
        )
        
        if use_primary and force_provider != LLMProvider.HEURISTIC:
            # Try primary provider (Llama3) with JSON validation
            try:
                logger.info("Attempting generation with Llama3 (primary)")
                
                # Add JSON response hint if validating
                if validate_json:
                    kwargs["json_response"] = True
                
                response = await self.primary_provider.generate(
                    messages, temperature, max_tokens, **kwargs
                )
                
                # Validate JSON if required
                if validate_json and not response.is_valid_json:
                    logger.warning(f"Llama3 returned invalid JSON, attempting to fix...")
                    
                    # Try to extract JSON from response
                    if response.extracted_json:
                        logger.info("Successfully extracted JSON from response")
                    else:
                        self.json_failures += 1
                        raise ValueError(f"Invalid JSON response from Llama3")
                
                # Success - reset failure tracking
                self.failure_count = 0
                self.json_failures = 0
                self.force_fallback = False
                self.primary_success += 1
                
                # Cache successful response
                self.response_cache[cache_key] = (response, datetime.now())
                
                # Track performance
                response_time = (datetime.now() - start_time).total_seconds()
                self.response_times.append(response_time)
                
                logger.info(
                    f"Llama3 generation successful - "
                    f"tokens={response.tokens_used}, "
                    f"time={response_time:.2f}s, "
                    f"valid_json={response.is_valid_json}"
                )
                return response
                
            except Exception as e:
                logger.warning(f"Llama3 generation failed: {e}")
                self._record_failure()
                
                # Check if we should force fallback
                if self.json_failures >= self.json_validation_failures:
                    logger.error(
                        f"Multiple JSON validation failures ({self.json_failures}), "
                        "forcing fallback mode"
                    )
                    self.force_fallback = True
                
                # Fall through to fallback
        
        # Use fallback provider
        return await self._generate_with_fallback(
            messages, temperature, max_tokens, start_time, **kwargs
        )
    
    async def _generate_with_fallback(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        start_time: datetime,
        **kwargs
    ) -> LLMResponse:
        """Generate using fallback provider with enhanced logging"""
        logger.info("Using heuristic fallback provider")
        
        try:
            response = await self.fallback_provider.generate(
                messages, temperature, max_tokens, **kwargs
            )
            
            self.fallback_success += 1
            response_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Heuristic fallback successful - "
                f"time={response_time:.2f}s, "
                f"valid_json={response.is_valid_json}"
            )
            return response
            
        except Exception as e:
            logger.error(f"Heuristic fallback failed: {e}")
            
            # Even heuristic failed, return emergency response
            return LLMResponse(
                content=json.dumps({
                    "error": "All LLM providers failed",
                    "message": "Unable to generate response",
                    "timestamp": datetime.now().isoformat(),
                    "emergency": True
                }),
                provider=LLMProvider.HEURISTIC,
                tokens_used=0,
                finish_reason="emergency_fallback",
                model="emergency",
                metadata={"error": str(e)}
            )
    
    def _record_failure(self):
        """Record provider failure with enhanced tracking"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(
            f"Provider failure recorded: {self.failure_count}/{self.failure_threshold}, "
            f"JSON failures: {self.json_failures}"
        )
        
        if self.failure_count >= self.failure_threshold:
            self.force_fallback = True
            logger.error(
                f"Failure threshold reached ({self.failure_threshold}). "
                "Forcing fallback mode for 5 minutes."
            )
    
    def _check_failure_window(self):
        """Reset failure count if outside failure window"""
        if self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            window = timedelta(minutes=self.failure_window)
            
            if time_since_failure > window:
                logger.info("Failure window expired, resetting failure counts")
                self.failure_count = 0
                self.json_failures = 0
                self.force_fallback = False
                self.last_failure_time = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check of all providers"""
        health_status = {
            "llama3": await self.primary_provider.health_check() if self.primary_provider else False,
            "heuristic": await self.fallback_provider.health_check(),
            "orchestrator": True,
            "force_fallback": self.force_fallback,
            "primary_available": self.primary_available
        }
        
        # If primary is unhealthy, update force_fallback
        if not health_status["llama3"] and self.primary_available:
            self.force_fallback = True
        
        logger.info(f"Health check: {health_status}")
        return health_status
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed orchestrator status"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "primary_success": self.primary_success,
            "fallback_success": self.fallback_success,
            "failure_count": self.failure_count,
            "json_failures": self.json_failures,
            "force_fallback": self.force_fallback,
            "primary_available": self.primary_available,
            "cache_size": len(self.response_cache),
            "performance": {
                "avg_response_time": f"{avg_response_time:.2f}s",
                "success_rate": f"{(self.primary_success + self.fallback_success) / self.total_requests * 100:.1f}%",
                "primary_usage": f"{(self.primary_success / self.total_requests * 100):.1f}%"
            },
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "failure_threshold": self.failure_threshold,
            "failure_window_minutes": self.failure_window
        }
    
    def reset(self):
        """Manually reset orchestrator state"""
        logger.info("Manually resetting orchestrator")
        self.failure_count = 0
        self.json_failures = 0
        self.force_fallback = False
        self.last_failure_time = None
        
        if self.primary_provider and hasattr(self.primary_provider, 'reset_circuit'):
            self.primary_provider.reset_circuit()
    
    def clear_cache(self):
        """Clear response cache"""
        logger.info(f"Clearing cache with {len(self.response_cache)} entries")
        self.response_cache.clear()