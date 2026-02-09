"""
Intent Analyzer - Production Ready with Enhanced LLM Integration

Enhanced version that works with the upgraded LLM module and proper schema alignment.
"""
import json
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.llm.orchestrator import LLMOrchestrator
from app.llm.prompt_manager import PromptManager, PromptType, PromptVersion
from app.llm.base import LLMMessage
from app.utils.logging import get_logger, log_context
from app.config import settings

logger = get_logger(__name__)


class LlamaIntentAnalyzer:
    """
    Production intent analyzer using enhanced LLM module with:
    - Strict JSON schema enforcement
    - Better error handling
    - Response validation
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with LLM configuration
        
        Args:
            config: LLM configuration dictionary
        """
        # Initialize orchestrator with config
        self.orchestrator = LLMOrchestrator(config)
        self.prompt_manager = PromptManager(default_version=PromptVersion.V3)
        
        # Initialize intent cache
        self.intent_cache: Dict[str, tuple[Dict[str, Any], datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'llama3_success': 0,
            'heuristic_fallback': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'avg_response_time': 0.0
        }
        
        # Response time tracking
        self.response_times: List[float] = []
        
        logger.info(
            "Intent analyzer initialized",
            extra={
                "provider": "llama3",
                "fallback": "heuristic",
                "prompt_version": "v3"
            }
        )
    
    def _get_cache_key(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key from prompt and context"""
        import hashlib
        
        # Normalize prompt (lowercase, strip whitespace)
        normalized_prompt = prompt.lower().strip()
        
        # Include context in cache key if provided
        context_str = ""
        if context:
            # Sort context keys for consistent hashing
            sorted_context = json.dumps(context, sort_keys=True)
            context_str = f"|{sorted_context}"
        
        # Create hash
        combined = f"{normalized_prompt}{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_intent(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached intent if available and not expired"""
        if cache_key in self.intent_cache:
            intent_data, cached_time = self.intent_cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            
            if age < self.cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key[:8]}")
                return intent_data
            else:
                # Remove expired cache entry
                del self.intent_cache[cache_key]
        
        return None
    
    def _cache_intent(self, cache_key: str, intent_data: Dict[str, Any]):
        """Cache intent result"""
        self.intent_cache[cache_key] = (intent_data, datetime.now())
    
    async def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user prompt to determine intent with enhanced validation.
        
        Returns a standardized intent dictionary that can be used by downstream systems.
        
        Args:
            prompt: User's natural language request
            context: Optional context (conversation history, existing project)
            
        Returns:
            Dict with intent analysis including:
            - intent_type: "new_app", "extend_app", "modify_app", "clarification", "help"
            - app_type: "counter", "todo", "calculator", "weather", "notes", "generic"
            - complexity: "simple", "medium", "complex"
            - confidence: float 0-1
            - extracted_features: List of features/components
            - requires_context: bool
            - needs_clarification: bool
            - action: "proceed", "clarify", "reject"
        """
        self.stats['total_requests'] += 1
        start_time = datetime.now()
        
        with log_context(operation="intent_analysis"):
            logger.info(
                "Intent analysis started",
                extra={
                    "prompt_length": len(prompt),
                    "has_context": context is not None
                }
            )
            
            # Check cache first
            cache_key = self._get_cache_key(prompt, context)
            cached_intent = self._get_cached_intent(cache_key)
            if cached_intent:
                self.stats['cache_hits'] += 1
                logger.info("Intent cache hit")
                return cached_intent
            
            try:
                # Build messages using prompt manager
                messages = self._build_enhanced_messages(prompt, context)
                
                # Try Llama3 with enhanced settings
                try:
                    logger.info("Attempting intent analysis with Llama3")
                    
                    response = await self.orchestrator.generate(
                        messages=messages,
                        temperature=0.3,  # Low temperature for classification
                        max_tokens=800,
                        validate_json=True,
                        json_response=True  # Request JSON response
                    )
                    
                    # Validate and parse response
                    intent_data = self._parse_and_validate_response(response, prompt)
                    
                    if intent_data:
                        self.stats['llama3_success'] += 1
                        
                        # Calculate response time
                        response_time = (datetime.now() - start_time).total_seconds()
                        self.response_times.append(response_time)
                        self.stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)
                        
                        logger.info(
                            "Intent analysis successful",
                            extra={
                                "intent_type": intent_data.get('intent_type'),
                                "app_type": intent_data.get('app_type'),
                                "confidence": intent_data.get('confidence'),
                                "response_time": f"{response_time:.2f}s"
                            }
                        )
                        
                        # Cache successful result
                        self._cache_intent(cache_key, intent_data)
                        
                        return intent_data
                    else:
                        self.stats['validation_failures'] += 1
                        logger.warning("Intent validation failed, falling back")
                
                except Exception as e:
                    logger.warning(
                        "Llama3 intent analysis failed",
                        extra={"error": str(e)}
                    )
                
                # Fallback to enhanced heuristic
                logger.info("Using enhanced heuristic fallback")
                self.stats['heuristic_fallback'] += 1
                
                intent_data = self._enhanced_heuristic_analysis(prompt, context)
                
                # Cache heuristic result (shorter TTL)
                self._cache_intent(cache_key, intent_data)
                
                return intent_data
                
            except Exception as e:
                logger.error(
                    "Intent analysis failed completely",
                    extra={"error": str(e)},
                    exc_info=True
                )
                
                # Emergency fallback
                return self._emergency_fallback(prompt)
    
    def _build_enhanced_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Build enhanced messages for intent analysis"""
        
        # Use prompt manager to build messages
        variables = {
            "user_prompt": prompt,
            "has_context": context is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        messages = self.prompt_manager.build_messages(
            prompt_type=PromptType.INTENT_ANALYSIS,
            user_input=prompt,
            variables=variables,
            version=PromptVersion.V3
        )
        
        # Add context if available
        if context:
            context_message = f"\n\nContext Information:\n{json.dumps(context, indent=2)}"
            messages[-1].content += context_message  # Add to user message
        
        return messages
    
    def _parse_and_validate_response(
        self, 
        response: Any,  # Using Any since LLMResponse is from different module
        original_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse and validate LLM response with enhanced validation.
        
        Note: response should be an LLMResponse object from the upgraded LLM module
        """
        
        # Check if response has extracted_json (from upgraded LLM module)
        if hasattr(response, 'extracted_json') and response.extracted_json:
            intent_data = response.extracted_json
        else:
            # Fallback to parsing content
            content = response.content.strip()
            intent_data = self._extract_json_from_content(content)
        
        if not intent_data:
            logger.warning("No JSON extracted from response")
            return None
        
        # Enhanced validation
        validation_result = self._validate_intent_data(intent_data, original_prompt)
        
        if validation_result["is_valid"]:
            # Add metadata
            intent_data["metadata"] = {
                "source": "llama3",
                "confidence": intent_data.get("confidence", 0.7),
                "validated": True,
                "validation_score": validation_result["score"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Determine action based on confidence
            confidence = intent_data.get("confidence", 0.5)
            if confidence >= 0.8:
                intent_data["action"] = "proceed"
                intent_data["needs_clarification"] = False
            elif confidence >= 0.6:
                intent_data["action"] = "proceed"
                intent_data["needs_clarification"] = True
            elif confidence >= 0.4:
                intent_data["action"] = "clarify"
                intent_data["needs_clarification"] = True
            else:
                intent_data["action"] = "clarify"
                intent_data["needs_clarification"] = True
            
            return intent_data
        
        logger.warning(
            "Intent validation failed",
            extra={
                "errors": validation_result["errors"],
                "score": validation_result["score"]
            }
        )
        return None
    
    def _extract_json_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with multiple strategies"""
        
        # Clean content
        content = content.strip()
        
        # Strategy 1: Try parsing as pure JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        import re
        
        # Pattern for ```json ... ``` or ``` ... ```
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON object in text
        brace_start = content.find('{')
        brace_end = content.rfind('}')
        
        if brace_start != -1 and brace_end > brace_start:
            json_str = content[brace_start:brace_end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Fix common JSON issues
        fixed_content = self._fix_common_json_issues(content)
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _fix_common_json_issues(self, content: str) -> str:
        """Fix common JSON formatting issues"""
        
        # Remove trailing commas
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Fix unquoted keys
        content = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
        
        # Fix single quotes to double quotes
        content = re.sub(r"'([^']*)'", r'"\1"', content)
        
        return content
    
    def _validate_intent_data(
        self, 
        intent_data: Dict[str, Any], 
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Validate intent data with scoring system.
        
        Returns dict with:
        - is_valid: bool
        - score: float 0-1
        - errors: List of validation errors
        """
        
        errors = []
        score = 1.0
        penalty = 0.1  # Penalty per error
        
        # Required fields
        required_fields = ["intent_type", "complexity", "confidence"]
        for field in required_fields:
            if field not in intent_data:
                errors.append(f"Missing required field: {field}")
                score -= penalty
        
        # Validate intent_type
        valid_intent_types = ["new_app", "extend_app", "modify_app", "clarification", "help"]
        intent_type = intent_data.get("intent_type")
        if intent_type and intent_type not in valid_intent_types:
            errors.append(f"Invalid intent_type: {intent_type}")
            score -= penalty
        
        # Validate complexity
        valid_complexities = ["simple", "medium", "complex"]
        complexity = intent_data.get("complexity")
        if complexity and complexity not in valid_complexities:
            errors.append(f"Invalid complexity: {complexity}")
            score -= penalty
        
        # Validate confidence
        confidence = intent_data.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                errors.append("Confidence must be a number")
                score -= penalty
            elif confidence < 0 or confidence > 1:
                errors.append("Confidence must be between 0 and 1")
                score -= penalty
        
        # Check extracted_entities if present
        if "extracted_entities" in intent_data:
            entities = intent_data["extracted_entities"]
            if not isinstance(entities, dict):
                errors.append("extracted_entities must be a dictionary")
                score -= penalty
        
        # Validate against original prompt (basic coherence check)
        if intent_type == "new_app" and len(original_prompt.split()) < 3:
            errors.append("New app request seems too short for meaningful analysis")
            score -= penalty * 0.5
        
        return {
            "is_valid": len(errors) == 0,
            "score": max(0.0, score),  # Don't go below 0
            "errors": errors
        }
    
    def _enhanced_heuristic_analysis(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhanced heuristic analysis with app type detection.
        """
        prompt_lower = prompt.lower()
        
        # Detect app type
        app_type = self._detect_app_type(prompt_lower)
        
        # Detect intent type
        intent_type = self._detect_intent_type(prompt_lower, context)
        
        # Detect complexity
        complexity = self._detect_complexity(prompt_lower)
        
        # Extract features
        extracted_features = self._extract_features(prompt_lower)
        
        # Calculate confidence (heuristic has lower confidence)
        confidence = self._calculate_heuristic_confidence(
            prompt_lower, 
            intent_type, 
            extracted_features
        )
        
        # Determine if context is required
        requires_context = intent_type in ["extend_app", "modify_app"]
        
        logger.info(
            "Heuristic analysis complete",
            extra={
                "app_type": app_type,
                "intent_type": intent_type,
                "complexity": complexity,
                "confidence": confidence,
                "features_found": len(extracted_features)
            }
        )
        
        return {
            "intent_type": intent_type,
            "app_type": app_type,
            "complexity": complexity,
            "confidence": confidence,
            "extracted_features": extracted_features,
            "requires_context": requires_context,
            "needs_clarification": confidence < 0.7,
            "action": "proceed" if confidence >= 0.6 else "clarify",
            "metadata": {
                "source": "heuristic",
                "validated": False,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _detect_app_type(self, prompt_lower: str) -> str:
        """Detect what type of app is being requested"""
        
        app_patterns = {
            "counter": ["counter", "count", "increment", "decrement", "clicker"],
            "todo": ["todo", "task", "checklist", "to-do", "reminder"],
            "calculator": ["calculator", "calc", "calculate", "math", "arithmetic"],
            "weather": ["weather", "forecast", "temperature", "climate"],
            "notes": ["note", "memo", "notepad", "journal", "diary"],
            "shopping": ["shopping", "cart", "e-commerce", "store", "product"],
            "chat": ["chat", "messenger", "message", "conversation"],
            "music": ["music", "player", "audio", "song", "playlist"],
            "calendar": ["calendar", "schedule", "event", "appointment"],
            "fitness": ["fitness", "workout", "exercise", "tracker", "health"]
        }
        
        for app_type, keywords in app_patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return app_type
        
        return "generic"
    
    def _detect_intent_type(self, prompt_lower: str, context: Optional[Dict[str, Any]]) -> str:
        """Detect intent type from prompt and context"""
        
        # Check context first
        if context and context.get("has_existing_project"):
            # User has existing project, likely modifying or extending
            if any(word in prompt_lower for word in ["add", "also", "include", "plus"]):
                return "extend_app"
            elif any(word in prompt_lower for word in ["change", "modify", "update", "fix"]):
                return "modify_app"
        
        # Check for new app creation
        new_app_keywords = ["create", "build", "make", "generate", "new", "design"]
        if any(word in prompt_lower for word in new_app_keywords):
            return "new_app"
        
        # Check for clarification/help
        help_keywords = ["what", "how", "why", "explain", "help", "tell", "show"]
        if any(word in prompt_lower for word in help_keywords):
            return "clarification"
        
        # Default to new app
        return "new_app"
    
    def _detect_complexity(self, prompt_lower: str) -> str:
        """Detect complexity level"""
        
        word_count = len(prompt_lower.split())
        
        # Check for complexity indicators
        if any(word in prompt_lower for word in ["simple", "basic", "minimal", "quick"]):
            return "simple"
        
        if any(word in prompt_lower for word in ["complex", "advanced", "complete", "full", "comprehensive"]):
            return "complex"
        
        # Word count based complexity
        if word_count <= 10:
            return "simple"
        elif word_count <= 30:
            return "medium"
        else:
            return "complex"
    
    def _extract_features(self, prompt_lower: str) -> List[str]:
        """Extract features and components from prompt"""
        
        features = []
        
        # Component detection
        components = {
            "button": ["button", "btn", "click", "press", "tap"],
            "input": ["input", "text field", "textbox", "entry"],
            "text": ["text", "label", "heading", "title"],
            "list": ["list", "items", "collection", "grid"],
            "image": ["image", "picture", "photo", "icon"],
            "switch": ["switch", "toggle", "checkbox"],
            "slider": ["slider", "range", "seekbar"],
            "map": ["map", "location", "gps", "navigation"],
            "chart": ["chart", "graph", "plot", "visualization"],
            "video": ["video", "player", "youtube"],
        }
        
        for component, keywords in components.items():
            if any(keyword in prompt_lower for keyword in keywords):
                features.append(component)
        
        # Feature detection
        if any(word in prompt_lower for word in ["login", "signin", "authentication", "auth"]):
            features.append("authentication")
        
        if any(word in prompt_lower for word in ["database", "storage", "save", "persist"]):
            features.append("data_storage")
        
        if any(word in prompt_lower for word in ["api", "fetch", "request", "network"]):
            features.append("api_integration")
        
        if any(word in prompt_lower for word in ["notification", "alert", "push"]):
            features.append("notifications")
        
        if any(word in prompt_lower for word in ["search", "filter", "sort"]):
            features.append("search_filter")
        
        if any(word in prompt_lower for word in ["payment", "credit card", "purchase", "buy"]):
            features.append("payment")
        
        return list(set(features))  # Remove duplicates
    
    def _calculate_heuristic_confidence(
        self, 
        prompt_lower: str, 
        intent_type: str, 
        features: List[str]
    ) -> float:
        """Calculate confidence score for heuristic analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on prompt length
        word_count = len(prompt_lower.split())
        if word_count >= 5:
            confidence += 0.1
        if word_count >= 10:
            confidence += 0.1
        
        # Adjust based on specific keywords found
        specific_keywords = ["counter", "todo", "calculator", "weather", "notes"]
        if any(keyword in prompt_lower for keyword in specific_keywords):
            confidence += 0.2
        
        # Adjust based on features extracted
        if features:
            confidence += min(0.2, len(features) * 0.05)
        
        # Adjust based on intent type specificity
        if intent_type in ["new_app", "clarification"]:
            confidence += 0.1
        
        # Cap at 0.9 (heuristic can't be 100% confident)
        return min(0.9, confidence)
    
    def _emergency_fallback(self, prompt: str) -> Dict[str, Any]:
        """Emergency fallback when everything fails"""
        
        logger.error("Emergency intent fallback triggered")
        
        return {
            "intent_type": "clarification",
            "app_type": "generic",
            "complexity": "medium",
            "confidence": 0.3,
            "extracted_features": [],
            "requires_context": False,
            "needs_clarification": True,
            "action": "clarify",
            "metadata": {
                "source": "emergency",
                "validated": False,
                "timestamp": datetime.now().isoformat(),
                "error": "All analysis methods failed"
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        
        total = self.stats['total_requests']
        
        if total == 0:
            return self.stats
        
        stats = {
            **self.stats,
            'llama3_success_rate': (self.stats['llama3_success'] / total) * 100,
            'heuristic_fallback_rate': (self.stats['heuristic_fallback'] / total) * 100,
            'cache_hit_rate': (self.stats['cache_hits'] / total) * 100,
            'avg_response_time': self.stats['avg_response_time']
        }
        
        # Add cache size
        stats['cache_size'] = len(self.intent_cache)
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'llama3_success': 0,
            'heuristic_fallback': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'avg_response_time': 0.0
        }
        self.response_times = []
    
    def clear_cache(self):
        """Clear intent cache"""
        self.intent_cache.clear()
        logger.info("Intent cache cleared")


# Factory function for easier initialization
def create_intent_analyzer(config: Dict[str, Any]) -> LlamaIntentAnalyzer:
    """
    Factory function to create intent analyzer with proper configuration.
    
    Args:
        config: LLM configuration dictionary
    
    Returns:
        Initialized LlamaIntentAnalyzer instance
    """
    return LlamaIntentAnalyzer(config)


# Testing
if __name__ == "__main__":
    import asyncio
    
    async def test_analyzer():
        """Test the enhanced intent analyzer"""
        
        print("\n" + "=" * 70)
        print("ENHANCED INTENT ANALYZER TEST")
        print("=" * 70)
        
        # Test configuration
        test_config = {
            "llama3_api_key": "test-key",
            "llama3_model": "llama-3-70b-instruct",
            "llama3_api_url": "https://fastchat.ideeza.com/v1/chat/completions",
            "request_timeout": 30.0,
            "max_retries": 2
        }
        
        analyzer = create_intent_analyzer(test_config)
        
        test_cases = [
            {
                "prompt": "Create a simple todo list app with add and delete buttons",
                "context": None,
                "description": "Todo app creation"
            },
            {
                "prompt": "Add notification feature to my counter app",
                "context": {"has_existing_project": True},
                "description": "Extend existing app"
            },
            {
                "prompt": "Build a complex e-commerce app with payment gateway and user authentication",
                "context": None,
                "description": "Complex app creation"
            },
            {
                "prompt": "How do I add a search bar to my app?",
                "context": None,
                "description": "Clarification request"
            },
            {
                "prompt": "Make a weather app that shows forecast for multiple cities",
                "context": None,
                "description": "Weather app with features"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {test['description']}")
            print(f"Prompt: \"{test['prompt']}\"")
            
            try:
                result = await analyzer.analyze(
                    prompt=test['prompt'],
                    context=test['context']
                )
                
                print(f"   Intent Type: {result.get('intent_type')}")
                print(f"   App Type: {result.get('app_type')}")
                print(f"   Complexity: {result.get('complexity')}")
                print(f"   Confidence: {result.get('confidence'):.2f}")
                print(f"   Features: {', '.join(result.get('extracted_features', []))}")
                print(f"   Action: {result.get('action')}")
                print(f"   Needs Clarification: {result.get('needs_clarification')}")
                print(f"   Source: {result.get('metadata', {}).get('source')}")
                
            except Exception as e:
                print(f"   ERROR: {e}")
        
        # Show stats
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        
        stats = analyzer.get_stats()
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Llama3 Success: {stats['llama3_success']}")
        print(f"Heuristic Fallback: {stats['heuristic_fallback']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Size: {stats.get('cache_size', 0)}")
        
        if stats['total_requests'] > 0:
            print(f"Llama3 Success Rate: {stats.get('llama3_success_rate', 0):.1f}%")
            print(f"Heuristic Fallback Rate: {stats.get('heuristic_fallback_rate', 0):.1f}%")
            print(f"Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1f}%")
            print(f"Avg Response Time: {stats.get('avg_response_time', 0):.2f}s")
        
        print("\n" + "=" * 70 + "\n")
    
    asyncio.run(test_analyzer())

    # Global instance
intent_analyzer = LlamaIntentAnalyzer(config=settings.dict())

# Export for import
__all__ = ["LlamaIntentAnalyzer", "intent_analyzer"]