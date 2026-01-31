"""
app/llm/llama3_provider.py
Llama3 LLM provider implementation - Production Ready
"""
import httpx
import logging
from typing import List, Optional, Dict, Any

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider


logger = logging.getLogger(__name__)


class Llama3Provider(BaseLLMProvider):
    """
    Production-ready Llama3 LLM provider with:
    - Proper error handling
    - Retry logic
    - Request/response logging
    - Timeout management
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = LLMProvider.LLAMA3
        
        # Configuration with sensible defaults
        self.api_url = config.get(
            "llama3_api_url", 
            "https://fastchat.ideeza.com/v1/chat/completions"
        )
        self.model = config.get("llama3_model", "llama-3")
        self.timeout = config.get("timeout", 60.0)
        self.api_key = config.get("llama3_api_key")
        
        # Request configuration
        self.max_retries = config.get("max_retries", 2)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Validation
        if not self.api_url:
            raise ValueError("Llama3 API URL is required")
        
        logger.info(
            f"Llama3 provider initialized: model={self.model}, "
            f"timeout={self.timeout}s, max_retries={self.max_retries}"
        )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Llama3 API with retry logic.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
            
        Raises:
            Exception: If all retries fail
        """
        
        # Validate inputs
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if not 0 <= temperature <= 1:
            logger.warning(f"Temperature {temperature} out of range, clamping to [0,1]")
            temperature = max(0, min(1, temperature))
        
        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Build payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Build headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Log request (truncated)
        logger.debug(
            f"Llama3 request: model={self.model}, "
            f"messages={len(messages)}, temp={temperature}, "
            f"max_tokens={max_tokens or 'default'}"
        )
        
        # Retry loop
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._make_request(payload, headers, attempt)
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Llama3 timeout (attempt {attempt}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                
                # Don't retry on client errors (4xx)
                if 400 <= status_code < 500:
                    logger.error(
                        f"Llama3 client error {status_code}: {e.response.text[:200]}"
                    )
                    raise Exception(f"Llama3 client error: {status_code}")
                
                # Retry on server errors (5xx)
                logger.warning(
                    f"Llama3 server error {status_code} "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                
            except Exception as e:
                last_error = e
                logger.error(f"Llama3 unexpected error: {e}")
                
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
        
        # All retries failed
        logger.error(
            f"Llama3 generation failed after {self.max_retries} attempts: "
            f"{last_error}"
        )
        raise Exception(f"Llama3 generation failed: {last_error}")
    
    async def _make_request(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        attempt: int
    ) -> LLMResponse:
        """Make actual HTTP request to Llama3 API"""
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError("Invalid Llama3 response: missing choices")
            
            # Parse response following OpenAI format
            choice = data["choices"][0]
            
            if "message" not in choice or "content" not in choice["message"]:
                raise ValueError("Invalid Llama3 response: missing message/content")
            
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            
            # Extract usage info
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            
            # Log success
            logger.info(
                f"Llama3 success (attempt {attempt}): "
                f"tokens={tokens_used}, finish={finish_reason}"
            )
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model=self.model,
                metadata={
                    "usage": usage,
                    "id": data.get("id"),
                    "attempt": attempt,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens")
                }
            )
    
    async def health_check(self) -> bool:
        """
        Check if Llama3 API is available.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            logger.debug("Running Llama3 health check")
            
            # Simple health check with minimal payload
            test_messages = [LLMMessage(role="user", content="test")]
            
            await self.generate(
                messages=test_messages,
                max_tokens=5,
                temperature=0.0
            )
            
            logger.info("Llama3 health check: PASSED")
            return True
            
        except Exception as e:
            logger.warning(f"Llama3 health check: FAILED - {e}")
            return False
    
    def get_provider_type(self) -> LLMProvider:
        """Return provider type"""
        return self.provider_name
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get provider configuration summary (for debugging)"""
        return {
            "provider": self.provider_name.value,
            "model": self.model,
            "api_url": self.api_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "has_api_key": bool(self.api_key)
        }