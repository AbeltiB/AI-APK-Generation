"""
app/llm/base.py
Abstract base class for all LLM providers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


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
    is_valid_json: bool = False
    extracted_json: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract JSON from content if present"""
        self._extract_json()
    
    def _extract_json(self):
        """Extract JSON from markdown or plain text"""
        content = self.content.strip()
        
        # Check if content is already valid JSON
        try:
            parsed = json.loads(content)
            self.extracted_json = parsed
            self.is_valid_json = True
            return
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        
        # Pattern for ```json ... ``` or ``` ... ```
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        if matches:
            try:
                parsed = json.loads(matches[0])
                self.extracted_json = parsed
                self.is_valid_json = True
                # Update content to just the JSON
                self.content = json.dumps(parsed, indent=2)
                return
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = content[json_start:json_end + 1]
            try:
                parsed = json.loads(json_str)
                self.extracted_json = parsed
                self.is_valid_json = True
                # Update content to just the JSON
                self.content = json.dumps(parsed, indent=2)
                return
            except json.JSONDecodeError:
                pass


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
        self.request_timeout = getattr(config, "request_timeout", 30.0)
        self.max_tokens_default = config.get("max_tokens_default", 2000)
        
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
    
    def validate_messages(self, messages: List[LLMMessage]) -> bool:
        """Validate message format"""
        if not messages:
            return False
        
        valid_roles = {"system", "user", "assistant"}
        for msg in messages:
            if msg.role not in valid_roles:
                return False
            if not isinstance(msg.content, str):
                return False
        
        return True