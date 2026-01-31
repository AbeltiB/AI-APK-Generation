"""
app/llm/base.py
Abstract base class for all LLM providers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    LLAMA3 = "llama3"
    HEURISTIC = "heuristic"


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    provider: LLMProvider
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMMessage:
    """Standardized message format"""
    role: str  # "system", "user", "assistant"
    content: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = None
        
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is available
        
        Returns:
            True if provider is healthy
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> LLMProvider:
        """Return provider type"""
        pass
    
    def format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert LLMMessage to provider-specific format"""
        return [{"role": msg.role, "content": msg.content} for msg in messages]