"""
Enhanced heuristic tier with fuzzy matching.

Tier 3 - Fast, reliable fallback using pattern matching and NLP techniques.
"""
from typing import List, Optional, Tuple, Set
from rapidfuzz import fuzz, process
from loguru import logger

from app.services.analysis.intent_config import config, ClassificationTier
from app.services.analysis.intent_schemas import (
    IntentAnalysisResult, IntentType, ComplexityLevel,
    ExtractedEntities, ConfidenceBreakdown, SafetyStatus,
    ActionRecommendation, ClassificationRequest
)
from app.services.analysis.tier_base import ClassificationTierBase


class EnhancedHeuristicTier(ClassificationTierBase):
    """
    Enhanced heuristic classification with fuzzy matching.
    
    Features:
    - Fuzzy string matching for typos
    - Component aliases recognition
    - Pattern-based classification
    - Complexity scoring
    - Safety detection
    """
    
    def __init__(self):
        super().__init__(
            tier=ClassificationTier.HEURISTIC,
            retry_config=config.TIERS["heuristic"].retry_config
        )
        self._prepare_patterns()
    
    def _prepare_patterns(self):
        """Prepare patterns for fast matching"""
        # Build flat keyword lists for fuzzy matching
        self.intent_keywords = {}
        for intent, patterns in config.INTENT_PATTERNS.items():
            keywords = patterns.get("keywords", [])
            phrases = patterns.get("phrases", [])
            self.intent_keywords[intent] = keywords + phrases
        
        # Build component aliases map
        self.component_aliases_flat = []
        for component, aliases in config.COMPONENT_ALIASES.items():
            self.component_aliases_flat.append(component)
            self.component_aliases_flat.extend(aliases)
    
    async def _classify_internal(
        self,
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Classify using enhanced heuristics"""
        
        prompt_lower = request.prompt.lower()
        words = prompt_lower.split()
        word_count = len(words)
        
        # Step 1: Classify intent
        intent_type, intent_confidence = self._classify_intent(
            prompt_lower, words
        )
        
        # Step 2: Classify complexity
        complexity, complexity_confidence = self._classify_complexity(
            prompt_lower, words, word_count
        )
        
        # Step 3: Extract entities
        entities, entity_confidence = self._extract_entities(
            prompt_lower, words
        )
        
        # Step 4: Check safety
        safety_status, safety_confidence = self._check_safety(
            prompt_lower, words
        )
        
        # Step 5: Calculate overall confidence
        overall_confidence = (
            intent_confidence * 0.35 +
            complexity_confidence * 0.20 +
            entity_confidence * 0.25 +
            safety_confidence * 0.20
        )
        
        # Build confidence breakdown
        confidence = ConfidenceBreakdown(
            overall=overall_confidence,
            intent_confidence=intent_confidence,
            complexity_confidence=complexity_confidence,
            entity_confidence=entity_confidence,
            safety_confidence=safety_confidence
        )
        
        # Determine action
        action = self._determine_action(intent_type, safety_status, confidence)
        
        # Generate message
        user_message = self._generate_user_message(
            action, intent_type, confidence
        )
        
        # Check if context required
        requires_context = intent_type in [
            IntentType.EXTEND_APP, IntentType.MODIFY_APP
        ]
        
        return IntentAnalysisResult(
            intent_type=intent_type,
            complexity=complexity,
            confidence=confidence,
            extracted_entities=entities,
            action_recommendation=action,
            safety_status=safety_status,
            requires_context=requires_context,
            multi_turn=False,
            user_message=user_message,
            reasoning="Classified using enhanced heuristic analysis",
            tier_used=self.tier,
            tier_attempts=[],
            total_latency_ms=0,
            total_cost_usd=0.0
        )
    
    def _classify_intent(
        self,
        prompt_lower: str,
        words: List[str]
    ) -> Tuple[IntentType, float]:
        """Classify intent with fuzzy matching"""
        
        scores = {}
        
        for intent_type, keywords in self.intent_keywords.items():
            score = 0.0
            matches = 0
            
            # Exact keyword matches
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1.0
                    matches += 1
            
            # Fuzzy matches for individual words
            for word in words:
                if len(word) >= 4:  # Only fuzzy match longer words
                    best_match = process.extractOne(
                        word,
                        keywords,
                        scorer=fuzz.ratio,
                        score_cutoff=80
                    )
                    if best_match:
                        score += 0.5
                        matches += 1
            
            # Normalize by number of keywords
            if len(keywords) > 0:
                scores[intent_type] = (score / len(keywords), matches)
            else:
                scores[intent_type] = (0.0, 0)
        
        # Get best match
        if not scores:
            return IntentType.CLARIFICATION, 0.3
        
        best_intent = max(scores.items(), key=lambda x: (x[1][0], x[1][1]))
        intent_type = best_intent[0]
        normalized_score = best_intent[1][0]
        match_count = best_intent[1][1]
        
        # Calculate confidence
        confidence = min(0.85, normalized_score * 0.8 + (match_count * 0.05))
        
        # Lower confidence if score too low
        if normalized_score < 0.1:
            return IntentType.CLARIFICATION, 0.3
        
        return intent_type, confidence
    
    def _classify_complexity(
        self,
        prompt_lower: str,
        words: List[str],
        word_count: int
    ) -> Tuple[ComplexityLevel, float]:
        """Classify complexity level"""
        
        scores = {
            ComplexityLevel.SIMPLE: 0.0,
            ComplexityLevel.MEDIUM: 0.0,
            ComplexityLevel.COMPLEX: 0.0
        }
        
        # Word count based scoring
        if word_count <= config.COMPLEXITY_INDICATORS[ComplexityLevel.SIMPLE]["max_words"]:
            scores[ComplexityLevel.SIMPLE] += 0.5
        elif word_count <= config.COMPLEXITY_INDICATORS[ComplexityLevel.MEDIUM]["max_words"]:
            scores[ComplexityLevel.MEDIUM] += 0.5
        else:
            scores[ComplexityLevel.COMPLEX] += 0.5
        
        # Keyword based scoring
        for level, indicators in config.COMPLEXITY_INDICATORS.items():
            keywords = indicators.get("keywords", [])
            for keyword in keywords:
                if keyword in prompt_lower:
                    scores[level] += 0.3
        
        # Component count estimation
        component_mentions = sum(
            1 for comp in self.component_aliases_flat
            if comp.lower() in prompt_lower
        )
        
        if component_mentions <= 3:
            scores[ComplexityLevel.SIMPLE] += 0.2
        elif component_mentions <= 8:
            scores[ComplexityLevel.MEDIUM] += 0.2
        else:
            scores[ComplexityLevel.COMPLEX] += 0.2
        
        # Get best match
        best_level = max(scores.items(), key=lambda x: x[1])
        complexity = best_level[0]
        score = best_level[1]
        
        # Normalize confidence
        confidence = min(0.75, score)
        
        return complexity, confidence
    
    def _extract_entities(
        self,
        prompt_lower: str,
        words: List[str]
    ) -> Tuple[ExtractedEntities, float]:
        """Extract entities using pattern matching and fuzzy matching"""
        
        components = set()
        actions = set()
        data_types = set()
        features = set()
        
        # Extract components with fuzzy matching
        for component, aliases in config.COMPONENT_ALIASES.items():
            all_terms = [component] + aliases
            for term in all_terms:
                if term.lower() in prompt_lower:
                    components.add(component.capitalize())
        
        # Fuzzy match for typos
        for word in words:
            if len(word) >= 5:
                match = process.extractOne(
                    word,
                    self.component_aliases_flat,
                    scorer=fuzz.ratio,
                    score_cutoff=85
                )
                if match:
                    # Find original component name
                    for comp, aliases in config.COMPONENT_ALIASES.items():
                        if match[0] == comp or match[0] in aliases:
                            components.add(comp.capitalize())
                            break
        
        # Extract actions
        action_keywords = [
            "click", "tap", "press", "swipe", "scroll",
            "input", "type", "enter", "select", "choose",
            "add", "delete", "update", "modify", "change"
        ]
        for action in action_keywords:
            if action in prompt_lower:
                actions.add(action)
        
        # Extract common data types
        data_keywords = [
            "todo", "task", "user", "product", "item",
            "post", "comment", "message", "note", "event"
        ]
        for data_type in data_keywords:
            if data_type in prompt_lower:
                data_types.add(data_type)
        
        # Extract features
        feature_keywords = [
            "authentication", "login", "signup", "auth",
            "search", "filter", "sort",
            "notification", "alert",
            "payment", "checkout",
            "profile", "settings"
        ]
        for feature in feature_keywords:
            if feature in prompt_lower:
                features.add(feature)
        
        # Calculate entity confidence
        total_found = len(components) + len(actions) + len(data_types) + len(features)
        if total_found == 0:
            entity_confidence = 0.2
        elif total_found <= 2:
            entity_confidence = 0.4
        elif total_found <= 5:
            entity_confidence = 0.6
        else:
            entity_confidence = 0.8
        
        return ExtractedEntities(
            components=list(components),
            actions=list(actions),
            data_types=list(data_types),
            features=list(features),
            screens=[],
            integrations=[]
        ), entity_confidence
    
    def _check_safety(
        self,
        prompt_lower: str,
        words: List[str]
    ) -> Tuple[SafetyStatus, float]:
        """Check for unsafe content"""
        
        unsafe_patterns = config.INTENT_PATTERNS.get(IntentType.UNSAFE, {})
        unsafe_keywords = unsafe_patterns.get("keywords", [])
        unsafe_phrases = unsafe_patterns.get("phrases", [])
        
        # Check for exact matches
        for keyword in unsafe_keywords:
            if keyword in prompt_lower:
                return SafetyStatus.UNSAFE, 0.95
        
        for phrase in unsafe_phrases:
            if phrase in prompt_lower:
                return SafetyStatus.UNSAFE, 0.98
        
        # Fuzzy check for variations
        for word in words:
            if len(word) >= 5:
                match = process.extractOne(
                    word,
                    unsafe_keywords,
                    scorer=fuzz.ratio,
                    score_cutoff=90
                )
                if match:
                    return SafetyStatus.SUSPICIOUS, 0.80
        
        return SafetyStatus.SAFE, 0.90
    
    def _determine_action(
        self,
        intent_type: IntentType,
        safety: SafetyStatus,
        confidence: ConfidenceBreakdown
    ) -> ActionRecommendation:
        """Determine recommended action"""
        if safety == SafetyStatus.UNSAFE:
            return ActionRecommendation.REJECT
        
        if intent_type == IntentType.MODIFY_APP:
            if confidence.overall < config.CONFIDENCE.block_dangerous:
                return ActionRecommendation.BLOCK_MODIFY
        
        if intent_type == IntentType.EXTEND_APP:
            if confidence.overall < config.CONFIDENCE.block_dangerous:
                return ActionRecommendation.BLOCK_EXTEND
        
        # Heuristic is less confident, so more likely to ask for clarification
        if confidence.overall < 0.75:
            return ActionRecommendation.CLARIFY
        
        return ActionRecommendation.PROCEED
    
    def _generate_user_message(
        self,
        action: ActionRecommendation,
        intent_type: IntentType,
        confidence: ConfidenceBreakdown
    ) -> Optional[str]:
        """Generate user-facing message"""
        if action == ActionRecommendation.PROCEED:
            return None
        
        if action == ActionRecommendation.REJECT:
            return config.USER_MESSAGES["unsafe_request"]
        
        if action == ActionRecommendation.BLOCK_MODIFY:
            return config.USER_MESSAGES["modify_blocked"]
        
        if action == ActionRecommendation.BLOCK_EXTEND:
            return config.USER_MESSAGES["extend_blocked"]
        
        if action == ActionRecommendation.CLARIFY:
            intent_guess = intent_type.value.replace("_", " ")
            return config.USER_MESSAGES["low_confidence_clarification"].format(
                intent_guess=intent_guess
            )
        
        return None
    
    def get_name(self) -> str:
        return "heuristic"
