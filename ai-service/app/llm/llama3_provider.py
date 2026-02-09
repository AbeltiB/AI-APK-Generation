"""
app/llm/llama3_provider.py
Production-ready Llama3 LLM provider with JSON response handling
"""
import httpx
import logging
import json
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider


logger = logging.getLogger(__name__)


class Llama3Provider(BaseLLMProvider):
    """
    Production-ready Llama3 LLM provider with:
    - Proper JSON response extraction
    - Enhanced error handling
    - Circuit breaker pattern
    - Request/response logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = LLMProvider.LLAMA3
        
        # Enhanced configuration with sensible defaults
        self.api_url = config.get(
            "llama3_api_url", 
            "https://fastchat.ideeza.com/v1/chat/completions"
        )
        self.model = config.get("llama3_model", "llama-3")
        self.api_key = config.get("llama3_api_key")
        
        # Request configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.request_timeout = config.get("request_timeout", 45.0)
        
        # Circuit breaker settings
        self.failure_count = 0
        self.max_failures = config.get("max_failures", 5)
        self.circuit_reset_time = config.get("circuit_reset_seconds", 300)
        self.circuit_tripped_time = None
        self.circuit_open = False
        
        # Request statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Validation
        if not self.api_url:
            raise ValueError("Llama3 API URL is required")
        
        if not self.api_key:
            logger.error("Llama3 API key is required but not provided")
            raise ValueError("Llama3 API key is required")
        
        logger.info(
            f"Llama3 provider initialized: model={self.model}, "
            f"timeout={self.request_timeout}s, max_retries={self.max_retries}, "
            f"api_url={self.api_url}"
        )
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_open:
            return True
        
        if self.circuit_tripped_time:
            time_since_trip = (datetime.now() - self.circuit_tripped_time).total_seconds()
            if time_since_trip > self.circuit_reset_time:
                logger.info("Circuit breaker resetting after timeout")
                self.circuit_open = False
                self.failure_count = 0
                self.circuit_tripped_time = None
                return True
        
        return False
    
    def _trip_circuit_breaker(self):
        """Trip the circuit breaker"""
        self.circuit_open = True
        self.circuit_tripped_time = datetime.now()
        logger.error(f"Circuit breaker tripped after {self.failure_count} failures")
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Llama3 API with enhanced error handling.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object with JSON extraction
        """
        self.total_requests += 1
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - Llama3 provider unavailable")
        
        # Validate inputs
        if not self.validate_messages(messages):
            raise ValueError("Invalid messages format")
        
        if not 0 <= temperature <= 2:  # Llama models can handle up to 2
            logger.warning(f"Temperature {temperature} out of range, clamping to [0,2]")
            temperature = max(0, min(2, temperature))
        
        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Build payload with enhanced parameters
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "max_tokens": max_tokens or self.max_tokens_default,
            "stream": False,
        }
        
        # Add stop sequences for JSON responses
        if kwargs.get("json_response", False):
            payload["stop"] = ["```", "```json"]
        
        # Build headers with authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "AppGenerator/1.0"
        }
        
        # Log request
        logger.debug(
            f"Llama3 request: model={self.model}, "
            f"messages={len(messages)}, temp={temperature}, "
            f"max_tokens={max_tokens or 'default'}, "
            f"json_mode={kwargs.get('json_response', False)}"
        )
        
        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._make_request(payload, headers, attempt)
                
                # Success - reset circuit breaker
                self.successful_requests += 1
                self.failure_count = 0
                
                logger.info(
                    f"Llama3 generation successful: tokens={response.tokens_used}, "
                    f"valid_json={response.is_valid_json}"
                )
                return response
                
            except httpx.TimeoutException as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                
                logger.warning(
                    f"Llama3 timeout (attempt {attempt}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue
                
            except httpx.HTTPStatusError as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                status_code = e.response.status_code
                
                if status_code == 401:
                    logger.error(
                        "Llama3 authentication failed - check API key",
                        extra={"api_url": self.api_url, "status_code": 401}
                    )
                    self._trip_circuit_breaker()
                    raise Exception("Llama3 authentication failed: Invalid API key")
                
                elif status_code == 429:
                    logger.warning(f"Llama3 rate limited (attempt {attempt}/{self.max_retries})")
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * 2)
                        continue
                
                elif 400 <= status_code < 500:
                    logger.error(f"Llama3 client error {status_code}: {e.response.text[:200]}")
                    # Don't retry on client errors
                    break
                
                else:  # Server errors (5xx)
                    logger.warning(
                        f"Llama3 server error {status_code} "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                        continue
            
            except Exception as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                logger.error(f"Llama3 unexpected error (attempt {attempt}): {e}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue
        
        # All retries failed
        if self.failure_count >= self.max_failures:
            self._trip_circuit_breaker()
        
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
        
        async with httpx.AsyncClient(
            timeout=self.request_timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        ) as client:
            start_time = datetime.now()
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response_time = (datetime.now() - start_time).total_seconds()
            
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError("Invalid Llama3 response: missing choices")
            
            choice = data["choices"][0]
            
            if "message" not in choice or "content" not in choice["message"]:
                raise ValueError("Invalid Llama3 response: missing message/content")
            
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            
            # Extract usage info
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            
            # Create response with JSON extraction
            llm_response = LLMResponse(
                content=content,
                provider=self.provider_name,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model=data.get("model", self.model),
                metadata={
                    "usage": usage,
                    "id": data.get("id"),
                    "attempt": attempt,
                    "response_time": response_time,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "model": data.get("model")
                }
            )
            
            # Log success
            logger.info(
                f"Llama3 success (attempt {attempt}): "
                f"tokens={tokens_used}, finish={finish_reason}, "
                f"time={response_time:.2f}s, valid_json={llm_response.is_valid_json}"
            )
            
            return llm_response
    
    async def health_check(self) -> bool:
        """
        Enhanced health check with circuit breaker awareness.
        
        Returns:
            True if healthy, False otherwise
        """
        if self.circuit_open:
            logger.warning("Llama3 health check: circuit breaker open")
            return False
        
        try:
            logger.debug("Running Llama3 health check")
            
            # Quick health check with minimal payload
            test_messages = [
                LLMMessage(role="system", content="You are a helpful assistant."),
                LLMMessage(role="user", content="Respond with 'OK'")
            ]
            
            response = await self._make_request(
                payload={
                    "model": self.model,
                    "messages": self.format_messages(test_messages),
                    "max_tokens": 5,
                    "temperature": 0.0
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                attempt=1
            )
            
            if response.content.strip().upper() == "OK":
                logger.info("Llama3 health check: PASSED")
                return True
            else:
                logger.warning("Llama3 health check: Unexpected response")
                return False
            
        except Exception as e:
            logger.warning(f"Llama3 health check: FAILED - {e}")
            self.failure_count += 1
            
            if self.failure_count >= self.max_failures:
                self._trip_circuit_breaker()
            
            return False
    
    def get_provider_type(self) -> LLMProvider:
        """Return provider type"""
        return self.provider_name
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            "provider": self.provider_name.value,
            "model": self.model,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "failure_count": self.failure_count,
            "circuit_open": self.circuit_open,
            "success_rate": (
                (self.successful_requests / self.total_requests * 100) 
                if self.total_requests > 0 else 0
            )
        }
    
    def reset_circuit(self):
        """Manually reset circuit breaker"""
        logger.info("Manually resetting circuit breaker")
        self.circuit_open = False
        self.failure_count = 0
        self.circuit_tripped_time = None