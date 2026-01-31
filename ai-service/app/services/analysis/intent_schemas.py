"""
Enhanced schemas for intent classification system.

Production-grade data models with validation and metadata.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum
from app.services.analysis.intent_config import ClassificationTier
class IntentType(str, Enum):
    """Types of user intents"""
    CREATE_APP = "create_app"
    MODIFY_APP = "modify_app"
    EXTEND_APP = "extend_app"
    BUG_FIX = "bug_fix"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    OTHER = "other"
class ComplexityLevel(str, Enum):
    """Complexity levels of requests"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"  

class SafetyStatus(str, Enum):
    """Safety classification for requests"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    UNSAFE = "unsafe"


class ActionRecommendation(str, Enum):
    """Recommended action based on classification"""
    PROCEED = "proceed"
    CLARIFY = "clarify"
    BLOCK_MODIFY = "block_modify"
    BLOCK_EXTEND = "block_extend"
    REJECT = "reject"


class TierMetrics(BaseModel):
    """Metrics for a classification tier attempt"""
    tier: ClassificationTier
    attempt_number: int
    success: bool
    latency_ms: int
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class ExtractedEntities(BaseModel):
    """Entities extracted from user prompt"""
    components: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    data_types: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    screens: List[str] = Field(default_factory=list)
    integrations: List[str] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    """Detailed confidence scoring breakdown"""
    overall: float = Field(..., ge=0.0, le=1.0)
    intent_confidence: float = Field(..., ge=0.0, le=1.0)
    complexity_confidence: float = Field(..., ge=0.0, le=1.0)
    entity_confidence: float = Field(..., ge=0.0, le=1.0)
    safety_confidence: float = Field(..., ge=0.0, le=1.0)


class IntentAnalysisResult(BaseModel):
    """
    Complete intent analysis result with all metadata.
    
    This is the main output of the intent classification system.
    Designed for both backend processing and frontend consumption.
    """
    
    # ========================================================================
    # CORE CLASSIFICATION
    # ========================================================================
    
    intent_type: IntentType
    """Classified intent type"""
    
    complexity: ComplexityLevel
    """Complexity level of the request"""
    
    confidence: ConfidenceBreakdown
    """Detailed confidence scores"""
    
    extracted_entities: ExtractedEntities
    """Entities extracted from prompt"""
    
    # ========================================================================
    # DECISION MAKING
    # ========================================================================
    
    action_recommendation: ActionRecommendation
    """Recommended action based on confidence"""
    
    safety_status: SafetyStatus
    """Safety classification"""
    
    requires_context: bool = False
    """Whether existing project context is needed"""
    
    multi_turn: bool = False
    """Whether this is part of ongoing conversation"""
    
    # ========================================================================
    # USER COMMUNICATION
    # ========================================================================
    
    user_message: Optional[str] = None
    """User-facing message (clarification, warning, etc.)"""
    
    suggested_clarifications: List[str] = Field(default_factory=list)
    """Specific questions to ask user"""
    
    reasoning: Optional[str] = None
    """Explanation of classification (for debugging)"""
    
    # ========================================================================
    # SYSTEM METADATA
    # ========================================================================
    
    tier_used: ClassificationTier
    """Which tier successfully classified"""
    
    tier_attempts: List[TierMetrics] = Field(default_factory=list)
    """Metrics from all tier attempts"""
    
    total_latency_ms: int
    """Total time taken for classification"""
    
    total_cost_usd: float = 0.0
    """Total estimated cost"""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """When classification occurred"""
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence_consistency(cls, v: ConfidenceBreakdown) -> ConfidenceBreakdown:
        """Ensure overall confidence is reasonable"""
        component_avg = (
            v.intent_confidence + 
            v.complexity_confidence + 
            v.entity_confidence + 
            v.safety_confidence
        ) / 4.0
        
        # Overall should be close to component average
        if abs(v.overall - component_avg) > 0.2:
            # Auto-adjust overall to match components
            v.overall = component_avg
        
        return v
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def is_high_confidence(self) -> bool:
        """Check if confidence is high enough to proceed"""
        from intent_config import config
        return self.confidence.overall >= config.CONFIDENCE.auto_accept
    
    def needs_clarification(self) -> bool:
        """Check if clarification is needed"""
        from intent_config import config
        return (
            config.CONFIDENCE.block_dangerous <= 
            self.confidence.overall < 
            config.CONFIDENCE.clarification
        )
    
    def is_dangerous_operation(self) -> bool:
        """Check if this is a potentially dangerous operation"""
        return self.intent_type in [IntentType.MODIFY_APP, IntentType.EXTEND_APP]
    
    def should_block(self) -> bool:
        """Check if operation should be blocked"""
        from intent_config import config
        
        # Block if unsafe
        if self.safety_status == SafetyStatus.UNSAFE:
            return True
        
        # Block dangerous operations with low confidence
        if self.is_dangerous_operation():
            return self.confidence.overall < config.CONFIDENCE.block_dangerous
        
        # Block if confidence too low
        return self.confidence.overall < config.CONFIDENCE.reject
    
    def get_tier_summary(self) -> Dict[str, Any]:
        """Get summary of tier attempts"""
        return {
            "tier_used": self.tier_used,
            "total_attempts": len(self.tier_attempts),
            "successful_tier": next(
                (m.tier for m in self.tier_attempts if m.success),
                None
            ),
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "tiers_tried": [m.tier for m in self.tier_attempts]
        }
    
    def to_frontend_response(self) -> Dict[str, Any]:
        """Convert to frontend-friendly format"""
        return {
            "intent": self.intent_type,
            "complexity": self.complexity,
            "confidence": self.confidence.overall,
            "action": self.action_recommendation,
            "message": self.user_message,
            "clarifications": self.suggested_clarifications,
            "entities": {
                "components": self.extracted_entities.components,
                "features": self.extracted_entities.features
            },
            "metadata": {
                "safe": self.safety_status == SafetyStatus.SAFE,
                "tier": self.tier_used,
                "latency_ms": self.total_latency_ms
            }
        }


class ClassificationRequest(BaseModel):
    """Request for intent classification"""
    prompt: str = Field(..., min_length=1, max_length=5000)
    user_id: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    force_tier: Optional[ClassificationTier] = None
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Clean and validate prompt"""
        return v.strip()


class ClassificationCache(BaseModel):
    """Cache entry for classification results"""
    prompt_hash: str
    result: IntentAnalysisResult
    created_at: datetime
    hits: int = 0
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > ttl_seconds