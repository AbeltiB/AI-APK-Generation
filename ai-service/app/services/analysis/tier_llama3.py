"""
Llama3 Tier - Intent classification using Llama3 LLM

Implements the tier interface for Llama3 intent classification.
"""
import time
import json
from typing import Dict, Any, Optional
import httpx

from app.services.analysis.intent_config import config, ClassificationTier
from app.services.analysis.intent_schemas import (
    IntentAnalysisResult, IntentType, ComplexityLevel,
    ExtractedEntities, ConfidenceBreakdown, SafetyStatus,
    ActionRecommendation, ClassificationRequest, TierMetrics
)
from app.services.analysis.tier_base import ClassificationTierBase
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Llama3Tier(ClassificationTierBase):
    """
    Llama3 LLM tier for intent classification.
    
    Uses Llama3 API to classify user intent with structured prompts.
    """
    
    def __init__(self, llama_config: Dict[str, Any]):
        super().__init__(
            tier=ClassificationTier.CLAUDE,  # Use CLAUDE enum value for primary tier
            retry_config=config.TIERS["claude"].retry_config
        )
        
        # Llama3 configuration
        self.api_url = llama_config.get('llama3_api_url')
        self.model = llama_config.get('llama3_model', 'llama-3-70b-instruct')
        self.api_key = llama_config.get('llama3_api_key')
        self.timeout = llama_config.get('timeout', 60.0)
        self.max_retries = llama_config.get('max_retries', 3)
        
        if not self.api_url:
            raise ValueError("Llama3 API URL is required")
        
        logger.info(
            "tier.llama3.initialized",
            extra={
                "model": self.model,
                "api_url": self.api_url,
                "timeout": self.timeout
            }
        )
    
    async def _classify_internal(
        self,
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Classify using Llama3 API"""
        
        start_time = time.time()
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(request.prompt, request.context)
        
        # Call Llama3 API
        try:
            response_data = await self._call_llama3_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse response
            result = self._parse_llama3_response(response_data, request)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            result.total_latency_ms = latency_ms
            
            # Estimate cost (approximate)
            tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
            result.total_cost_usd = (tokens_used / 1000) * config.TIERS["claude"].cost_per_1k_tokens
            
            logger.info(
                "tier.llama3.success",
                extra={
                    "latency_ms": latency_ms,
                    "tokens": tokens_used,
                    "intent": result.intent_type.value
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "tier.llama3.failed",
                extra={"error": str(e)},
                exc_info=e
            )
            raise
    
    async def _call_llama3_api(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """Call Llama3 API"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,  # Lower for more consistent classification
            "max_tokens": 2000
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for intent classification"""
        
        return """You are an expert intent classifier for a mobile app generation system.

Your task is to analyze user requests and classify them into:

INTENT TYPES:
- create_app: User wants to create a new application
- modify_app: User wants to change an existing app
- extend_app: User wants to add features to existing app
- bug_fix: User is reporting a bug or issue
- optimize_performance: User wants performance improvements
- other: General questions or unclear requests

COMPLEXITY LEVELS:
- simple: Basic apps with 1-3 components, single screen
- medium: Apps with 3-8 components, multiple screens or features
- complex: Advanced apps with authentication, API integration, complex logic

EXTRACTED ENTITIES:
- components: UI components mentioned (Button, Text, Input, etc.)
- actions: User actions (click, tap, swipe, etc.)
- data_types: Data entities (user, product, task, etc.)
- features: App features (login, search, payment, etc.)

SAFETY:
- safe: Normal, constructive request
- suspicious: Potentially problematic but unclear
- unsafe: Malicious, harmful, or inappropriate request

Respond ONLY with valid JSON in this exact format:
{
  "intent_type": "create_app",
  "complexity": "simple",
  "confidence": {
    "overall": 0.85,
    "intent_confidence": 0.9,
    "complexity_confidence": 0.8,
    "entity_confidence": 0.85,
    "safety_confidence": 1.0
  },
  "extracted_entities": {
    "components": ["Button", "Text"],
    "actions": ["click", "display"],
    "data_types": ["counter"],
    "features": ["increment", "decrement"]
  },
  "safety_status": "safe",
  "requires_context": false,
  "multi_turn": false,
  "reasoning": "Brief explanation of classification"
}"""
    
    def _build_user_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build user prompt with context"""
        
        user_prompt = f"Classify this user request:\n\n\"{prompt}\""
        
        if context:
            if context.get('has_existing_project'):
                user_prompt += "\n\nNote: User has an existing project in this session."
        
        return user_prompt
    
    def _parse_llama3_response(
        self,
        response_data: Dict[str, Any],
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Parse Llama3 API response into IntentAnalysisResult"""
        
        from datetime import datetime, timezone
        
        # Extract content
        content = response_data['choices'][0]['message']['content']
        
        # Parse JSON (strip markdown if present)
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(
                "tier.llama3.parse_failed",
                extra={"content": content[:200]},
                exc_info=e
            )
            # Fallback to heuristic-like response
            return self._create_fallback_result(request)
        
        # Map string values to enums
        intent_type_map = {
            "create_app": IntentType.CREATE_APP,
            "modify_app": IntentType.MODIFY_APP,
            "extend_app": IntentType.EXTEND_APP,
            "bug_fix": IntentType.BUG_FIX,
            "optimize_performance": IntentType.OPTIMIZE_PERFORMANCE,
            "other": IntentType.OTHER
        }
        
        complexity_map = {
            "simple": ComplexityLevel.SIMPLE,
            "medium": ComplexityLevel.MEDIUM,
            "complex": ComplexityLevel.COMPLEX
        }
        
        safety_map = {
            "safe": SafetyStatus.SAFE,
            "suspicious": SafetyStatus.SUSPICIOUS,
            "unsafe": SafetyStatus.UNSAFE
        }
        
        # Build result
        intent_type = intent_type_map.get(
            parsed.get('intent_type', 'other'),
            IntentType.OTHER
        )
        
        complexity = complexity_map.get(
            parsed.get('complexity', 'medium'),
            ComplexityLevel.MEDIUM
        )
        
        safety = safety_map.get(
            parsed.get('safety_status', 'safe'),
            SafetyStatus.SAFE
        )
        
        # Parse confidence
        conf_data = parsed.get('confidence', {})
        confidence = ConfidenceBreakdown(
            overall=conf_data.get('overall', 0.7),
            intent_confidence=conf_data.get('intent_confidence', 0.7),
            complexity_confidence=conf_data.get('complexity_confidence', 0.7),
            entity_confidence=conf_data.get('entity_confidence', 0.7),
            safety_confidence=conf_data.get('safety_confidence', 1.0)
        )
        
        # Parse entities
        entities_data = parsed.get('extracted_entities', {})
        entities = ExtractedEntities(
            components=entities_data.get('components', []),
            actions=entities_data.get('actions', []),
            data_types=entities_data.get('data_types', []),
            features=entities_data.get('features', []),
            screens=[],
            integrations=[]
        )
        
        # Determine action
        action = self._determine_action(intent_type, safety, confidence)
        
        return IntentAnalysisResult(
            intent_type=intent_type,
            complexity=complexity,
            confidence=confidence,
            extracted_entities=entities,
            action_recommendation=action,
            safety_status=safety,
            requires_context=parsed.get('requires_context', False),
            multi_turn=parsed.get('multi_turn', False),
            user_message=None,
            reasoning=parsed.get('reasoning', ''),
            tier_used=self.tier,
            tier_attempts=[],
            total_latency_ms=0,
            total_cost_usd=0.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _determine_action(
        self,
        intent_type: IntentType,
        safety: SafetyStatus,
        confidence: ConfidenceBreakdown
    ) -> ActionRecommendation:
        """Determine recommended action based on classification"""
        
        if safety == SafetyStatus.UNSAFE:
            return ActionRecommendation.REJECT
        
        if intent_type == IntentType.MODIFY_APP:
            if confidence.overall < config.CONFIDENCE.block_dangerous:
                return ActionRecommendation.BLOCK_MODIFY
        
        if intent_type == IntentType.EXTEND_APP:
            if confidence.overall < config.CONFIDENCE.block_dangerous:
                return ActionRecommendation.BLOCK_EXTEND
        
        if confidence.overall < config.CONFIDENCE.clarification:
            return ActionRecommendation.CLARIFY
        
        return ActionRecommendation.PROCEED
    
    def _create_fallback_result(
        self,
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Create fallback result when parsing fails"""
        
        from datetime import datetime, timezone
        
        return IntentAnalysisResult(
            intent_type=IntentType.OTHER,
            complexity=ComplexityLevel.MEDIUM,
            confidence=ConfidenceBreakdown(
                overall=0.5,
                intent_confidence=0.5,
                complexity_confidence=0.5,
                entity_confidence=0.5,
                safety_confidence=0.8
            ),
            extracted_entities=ExtractedEntities(),
            action_recommendation=ActionRecommendation.CLARIFY,
            safety_status=SafetyStatus.SAFE,
            requires_context=False,
            multi_turn=False,
            user_message="Could not fully understand request. Please provide more details.",
            reasoning="Llama3 response parsing failed",
            tier_used=self.tier,
            tier_attempts=[],
            total_latency_ms=0,
            total_cost_usd=0.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_name(self) -> str:
        return "llama3"