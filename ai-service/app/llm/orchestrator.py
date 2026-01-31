"""
app/llm/orchestrator.py
Smart LLM routing and fallback orchestration
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider
from .llama3_provider import Llama3Provider
from .heuristic_provider import HeuristicProvider


logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates LLM providers with smart routing and fallback
    
    Flow: Llama3 (primary) → Heuristic (fallback)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize providers
        self.primary_provider = Llama3Provider(config)
        self.fallback_provider = HeuristicProvider(config)
        
        # Failure tracking
        self.failure_threshold = config.get("failure_threshold", 3)
        self.failure_window = config.get("failure_window_minutes", 5)
        self.failure_count = 0
        self.last_failure_time = None
        self.force_fallback = False
        
        logger.info("LLM Orchestrator initialized with Llama3 → Heuristic")
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        force_provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with smart routing and fallback
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            force_provider: Force specific provider (for testing)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse from successful provider
        """
        
        # Reset failure count if outside failure window
        self._check_failure_window()
        
        # Determine which provider to use
        if force_provider == LLMProvider.HEURISTIC:
            return await self._generate_with_fallback(messages, temperature, max_tokens, **kwargs)
        
        if force_provider == LLMProvider.LLAMA3 or not self.force_fallback:
            # Try primary provider (Llama3)
            try:
                logger.info("Attempting generation with Llama3")
                response = await self.primary_provider.generate(
                    messages, temperature, max_tokens, **kwargs
                )
                
                # Reset failure tracking on success
                self.failure_count = 0
                self.force_fallback = False
                
                logger.info(f"Llama3 generation successful - {response.tokens_used} tokens")
                return response
                
            except Exception as e:
                logger.warning(f"Llama3 generation failed: {e}")
                self._record_failure()
                
                # Fall through to fallback
        
        # Use fallback provider
        return await self._generate_with_fallback(messages, temperature, max_tokens, **kwargs)
    
    async def _generate_with_fallback(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate using fallback provider"""
        logger.info("Using heuristic fallback provider")
        
        try:
            response = await self.fallback_provider.generate(
                messages, temperature, max_tokens, **kwargs
            )
            logger.info("Heuristic fallback generation successful")
            return response
        except Exception as e:
            logger.error(f"Heuristic fallback failed: {e}")
            raise Exception("All LLM providers failed")
    
    def _record_failure(self):
        """Record provider failure and check threshold"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(
            f"Provider failure recorded: {self.failure_count}/{self.failure_threshold}"
        )
        
        if self.failure_count >= self.failure_threshold:
            self.force_fallback = True
            logger.error(
                f"Failure threshold reached ({self.failure_threshold}). "
                "Forcing fallback mode."
            )
    
    def _check_failure_window(self):
        """Reset failure count if outside failure window"""
        if self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            window = timedelta(minutes=self.failure_window)
            
            if time_since_failure > window:
                logger.info("Failure window expired, resetting failure count")
                self.failure_count = 0
                self.force_fallback = False
                self.last_failure_time = None
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_status = {
            "llama3": await self.primary_provider.health_check(),
            "heuristic": await self.fallback_provider.health_check(),
            "orchestrator": True
        }
        
        logger.info(f"Health check: {health_status}")
        return health_status
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "failure_count": self.failure_count,
            "force_fallback": self.force_fallback,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "failure_threshold": self.failure_threshold,
            "failure_window_minutes": self.failure_window
        }
    
    def reset_failures(self):
        """Manually reset failure tracking"""
        logger.info("Manually resetting failure tracking")
        self.failure_count = 0
        self.force_fallback = False
        self.last_failure_time = None