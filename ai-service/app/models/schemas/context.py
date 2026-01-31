"""
Context and intent analysis models.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class IntentAnalysis(BaseModel):
    """Result of intent analysis"""
    intent_type: Literal["new_app", "extend_app", "modify_app", "clarification", "help"]
    complexity: Literal["simple", "medium", "complex"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted components, actions, data types, etc."
    )
    requires_context: bool = Field(
        default=False,
        description="Whether existing project context is needed"
    )
    multi_turn: bool = Field(
        default=False,
        description="Whether this is part of a conversation"
    )


class EnrichedContext(BaseModel):
    """Enriched context for AI generation"""
    original_request: Dict[str, Any]
    intent_analysis: IntentAnalysis
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    existing_project: Optional[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptContext(BaseModel):
    """Schema for managing prompt templates and generation settings"""
    template_name: str = Field(..., description="The ID or name of the prompt template")
    system_prompt: str
    user_prompt: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    model_settings: Dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 2048}
    )