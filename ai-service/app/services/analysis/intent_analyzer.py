"""
Enhanced Intent Analyzer - Production Ready with Domain Awareness and Llama3
Integrates with existing codebase structure
"""
import json
import re
import asyncio
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import httpx

from app.utils.logging import get_logger, log_context
from app.config import settings

logger = get_logger(__name__)


class IntentType(str):
    """Types of user intents - Compatible with existing code"""
    CREATE_NEW = "new_app"
    MODIFY_EXISTING = "modify_app"
    EXTEND_FEATURES = "extend_app"
    BUG_FIX = "bug_fix"
    OPTIMIZE = "optimize_performance"
    CLARIFICATION = "clarification"
    HELP = "help"
    OTHER = "other"


class AppDomain(str):
    """Application domains - Matching existing enums"""
    PRODUCTIVITY = "productivity"
    ENTERTAINMENT = "entertainment"
    UTILITY = "utility"
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTH_FITNESS = "health_fitness"
    FINANCE = "finance"
    DEVELOPMENT = "development"
    IOT_HARDWARE = "iot_hardware"
    CREATIVE_MEDIA = "creative_media"
    DATA_SCIENCE = "data_science"
    CUSTOM = "custom"


class ComplexityLevel(str):
    """Complexity levels - Matching existing enums"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    SIMPLE_UI = "simple_ui"
    DATA_DRIVEN = "data_driven"
    INTEGRATED = "integrated"
    ENTERPRISE = "enterprise"
    HARDWARE = "hardware"
    AI_ML = "ai_ml"


class EnhancedLlama3IntentAnalyzer:
    """
    Production-ready intent analyzer that replaces LlamaIntentAnalyzer
    Enhanced with domain awareness and Llama3 integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with enhanced capabilities"""
        # Llama3 configuration from settings
        self.api_url = config.get('llama3_api_url')
        self.api_key = config.get('llama3_api_key')
        self.model = config.get('llama3_model', 'llama-3-70b-instruct')
        self.timeout = config.get('timeout', 60.0)
        self.max_retries = config.get('max_retries', 3)
        
        if not self.api_url:
            raise ValueError("Llama3 API URL is required")
        
        # Initialize cache and stats (compatible with existing code)
        self.intent_cache: Dict[str, tuple] = {}
        self.cache_ttl = 300  # 5 minutes
        
        self.stats = {
            'total_requests': 0,
            'llama3_success': 0,
            'heuristic_fallback': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'avg_response_time': 0.0,
            'domain_breakdown': {},
            'complexity_distribution': {
                'simple': 0, 'medium': 0, 'complex': 0,
                'simple_ui': 0, 'data_driven': 0, 'integrated': 0,
                'enterprise': 0, 'hardware': 0, 'ai_ml': 0
            }
        }
        
        self.response_times: List[float] = []
        
        logger.info(
            "Enhanced intent analyzer initialized",
            extra={
                "model": self.model,
                "api_url": self.api_url,
                "cache_ttl": self.cache_ttl
            }
        )
    
    # ====== COMPATIBLE METHODS (same interface as old analyzer) ======
    
    async def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main analysis method - Same interface as old LlamaIntentAnalyzer
        """
        return await self.analyze_intent(prompt, context)
    
    async def analyze_intent(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        force_llama3: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced analysis with domain awareness
        Returns same format as old analyzer for compatibility
        """
        start_time = datetime.now()
        self.stats['total_requests'] += 1
        
        with log_context(operation="intent_analysis"):
            logger.info(
                "Enhanced intent analysis started",
                extra={
                    "prompt_length": len(prompt),
                    "has_context": context is not None
                }
            )
            
            # Generate cache key
            cache_key = self._get_cache_key(prompt, context)
            
            # Check cache
            cached_intent = self._get_cached_intent(cache_key)
            if cached_intent:
                self.stats['cache_hits'] += 1
                logger.info("Intent cache hit")
                return cached_intent
            
            try:
                # Step 1: Detect domain
                domain_info = self._detect_domain(prompt, context)
                
                # Step 2: Extract requirements
                special_reqs = self._extract_requirements(prompt, domain_info)
                
                # Step 3: Try Llama3
                if force_llama3 or self._should_use_llama3(domain_info, special_reqs):
                    try:
                        llm_result = await self._analyze_with_llama3(
                            prompt, domain_info, special_reqs, context
                        )
                        
                        if llm_result and llm_result.get('confidence', 0) > 0.6:
                            self.stats['llama3_success'] += 1
                            
                            # Convert to compatible format
                            result = self._convert_to_compatible_format(llm_result, domain_info)
                            result['metadata']['source'] = 'llama3'
                            
                            # Cache and track
                            self._cache_intent(cache_key, result)
                            self._update_stats(domain_info, result)
                            
                            response_time = (datetime.now() - start_time).total_seconds()
                            self._track_response_time(response_time)
                            
                            logger.info(
                                "Intent analysis successful",
                                extra={
                                    "intent_type": result.get('intent_type'),
                                    "app_type": result.get('app_type'),
                                    "confidence": result.get('confidence'),
                                    "response_time": f"{response_time:.2f}s"
                                }
                            )
                            
                            return result
                        else:
                            self.stats['validation_failures'] += 1
                            logger.warning("LLM validation failed, falling back")
                    
                    except Exception as e:
                        logger.warning(
                            "Llama3 intent analysis failed",
                            extra={"error": str(e)}
                        )
                
                # Step 4: Fallback to heuristic
                logger.info("Using enhanced heuristic fallback")
                self.stats['heuristic_fallback'] += 1
                
                heuristic_result = self._enhanced_heuristic_analysis(
                    prompt, domain_info, special_reqs, context
                )
                
                # Convert to compatible format
                result = self._convert_to_compatible_format(heuristic_result, domain_info)
                result['metadata']['source'] = 'heuristic'
                
                # Cache and track
                self._cache_intent(cache_key, result)
                self._update_stats(domain_info, result)
                
                response_time = (datetime.now() - start_time).total_seconds()
                self._track_response_time(response_time)
                
                return result
                
            except Exception as e:
                logger.error(
                    "Intent analysis failed completely",
                    extra={"error": str(e)},
                    exc_info=True
                )
                
                return self._emergency_fallback(prompt)
    
    # ====== CACHE METHODS (compatible with existing) ======
    
    def _get_cache_key(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key from prompt and context"""
        import hashlib
        
        normalized_prompt = prompt.lower().strip()
        context_str = ""
        if context:
            sorted_context = json.dumps(context, sort_keys=True)
            context_str = f"|{sorted_context}"
        
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
                del self.intent_cache[cache_key]
        
        return None
    
    def _cache_intent(self, cache_key: str, intent_data: Dict[str, Any]):
        """Cache intent result"""
        self.intent_cache[cache_key] = (intent_data, datetime.now())
    
    # ====== DOMAIN DETECTION METHODS ======
    
    def _detect_domain(self, prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect domain and app type"""
        prompt_lower = prompt.lower()
        
        # Hardware detection
        hardware_types = {
            "drone": ["drone", "quadcopter", "uav", "flight", "fpv"],
            "3d_printer": ["3d printer", "filament", "gcode", "print"],
            "iot_device": ["iot", "sensor", "smart home", "device"]
        }
        
        for hw_type, keywords in hardware_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return {
                    "domain": AppDomain.IOT_HARDWARE,
                    "specific_type": hw_type,
                    "confidence": 0.8,
                    "is_specialized": True
                }
        
        # AI detection
        ai_types = {
            "image_to_3d": ["image to 3d", "3d model", "convert image"],
            "ai_model_trainer": ["train model", "predict", "classify", "ai"]
        }
        
        for ai_type, keywords in ai_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return {
                    "domain": AppDomain.CREATIVE_MEDIA,
                    "specific_type": ai_type,
                    "confidence": 0.8,
                    "is_specialized": True
                }
        
        # Standard domain detection
        domain_patterns = {
            AppDomain.PRODUCTIVITY: ["todo", "calendar", "notes", "task"],
            AppDomain.ENTERTAINMENT: ["music", "video", "game", "player"],
            AppDomain.UTILITY: ["calculator", "converter", "scanner"],
            AppDomain.FINANCE: ["budget", "expense", "bank", "money"],
            AppDomain.HEALTH_FITNESS: ["fitness", "workout", "health", "tracker"]
        }
        
        best_score = 0
        best_domain = AppDomain.CUSTOM
        best_type = "generic"
        
        for domain, keywords in domain_patterns.items():
            matches = sum(1 for kw in keywords if kw in prompt_lower)
            if matches > 0:
                score = matches / len(keywords)
                if score > best_score:
                    best_score = score
                    best_domain = domain
                    best_type = keywords[0] if keywords else "generic"
        
        return {
            "domain": best_domain,
            "specific_type": best_type,
            "confidence": min(0.9, 0.5 + best_score * 0.4),
            "is_specialized": best_domain in [AppDomain.IOT_HARDWARE, AppDomain.CREATIVE_MEDIA]
        }
    
    def _extract_requirements(self, prompt: str, domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specialized requirements"""
        prompt_lower = prompt.lower()
        
        requirements = {
            "needs_hardware": domain_info.get("is_specialized", False) and domain_info["domain"] == AppDomain.IOT_HARDWARE,
            "needs_ai_ml": domain_info.get("is_specialized", False) and domain_info["domain"] == AppDomain.CREATIVE_MEDIA,
            "needs_real_time": any(word in prompt_lower for word in ["real-time", "live", "stream", "control"]),
            "needs_3d": any(word in prompt_lower for word in ["3d", "three.js", "webgl"]),
            "special_apis": [],
            "complex_components": []
        }
        
        # Add specific APIs based on domain
        if requirements["needs_hardware"]:
            requirements["special_apis"] = ["bluetooth", "websockets", "serial"]
        
        if requirements["needs_ai_ml"]:
            requirements["special_apis"] = ["tensorflow.js", "webgl", "webworkers"]
        
        if requirements["needs_real_time"]:
            if "websockets" not in requirements["special_apis"]:
                requirements["special_apis"].append("websockets")
        
        if requirements["needs_3d"]:
            if "webgl" not in requirements["special_apis"]:
                requirements["special_apis"].append("webgl")
            if "three.js" not in requirements["special_apis"]:
                requirements["special_apis"].append("three.js")
        
        return requirements
    
    def _should_use_llama3(self, domain_info: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Determine if Llama3 should be used"""
        # Use Llama3 for specialized domains
        if domain_info["is_specialized"]:
            return True
        
        # Use Llama3 for complex requirements
        if any([requirements["needs_hardware"], requirements["needs_ai_ml"], 
                requirements["needs_real_time"], requirements["needs_3d"]]):
            return True
        
        # Use Llama3 for low confidence
        if domain_info["confidence"] < 0.7:
            return True
        
        return False
    
    # ====== LLAMA3 INTEGRATION ======
    
    async def _analyze_with_llama3(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        requirements: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze with Llama3 API"""
        system_prompt = self._build_llama3_prompt(domain_info, requirements)
        user_prompt = f"User request: {prompt}"
        
        if context:
            user_prompt += f"\nContext: {json.dumps(context, indent=2)}"
        
        try:
            response = await self._call_llama3_api(system_prompt, user_prompt)
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return self._parse_llama3_response(content)
                
        except Exception as e:
            logger.error(f"Llama3 API call failed: {e}")
        
        return None
    
    def _build_llama3_prompt(self, domain_info: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """Build system prompt for Llama3"""
        return f"""You are an intent classifier for an app generation system.

Analyze the user request and provide JSON response with:

1. intent_type: "new_app", "extend_app", "modify_app", "clarification", "help"
2. app_type: specific app category
3. complexity: "simple", "medium", "complex"
4. confidence: float 0-1
5. extracted_features: list of features/components
6. requires_context: boolean
7. needs_clarification: boolean
8. action: "proceed", "clarify", "reject"

Domain: {domain_info['domain']}
App Type: {domain_info['specific_type']}
Requirements: {json.dumps(requirements, indent=2)}

Respond ONLY with valid JSON, no additional text."""
    
    async def _call_llama3_api(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call Llama3 API"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Llama3 API error: {e}")
            return None
    
    def _parse_llama3_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse Llama3 response"""
        try:
            # Clean content
            content = content.strip()
            
            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None
    
    # ====== HEURISTIC ANALYSIS ======
    
    def _enhanced_heuristic_analysis(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        requirements: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced heuristic analysis"""
        prompt_lower = prompt.lower()
        
        # Detect intent type
        intent_type = self._detect_heuristic_intent(prompt_lower, context)
        
        # Detect app type
        app_type = self._detect_heuristic_app_type(prompt_lower, domain_info)
        
        # Detect complexity
        complexity = self._detect_heuristic_complexity(prompt_lower, requirements)
        
        # Extract features
        extracted_features = self._extract_heuristic_features(prompt_lower)
        
        # Calculate confidence
        confidence = self._calculate_heuristic_confidence(prompt_lower, intent_type, extracted_features)
        
        # Determine if context is required
        requires_context = intent_type in ["extend_app", "modify_app"]
        
        # Determine action
        action = "proceed" if confidence >= 0.6 else "clarify"
        
        return {
            "intent_type": intent_type,
            "app_type": app_type,
            "complexity": complexity,
            "confidence": confidence,
            "extracted_features": extracted_features,
            "requires_context": requires_context,
            "needs_clarification": confidence < 0.7,
            "action": action,
            "domain": domain_info["domain"],
            "specific_type": domain_info["specific_type"],
            "technical_requirements": requirements
        }
    
    def _detect_heuristic_intent(self, prompt_lower: str, context: Optional[Dict[str, Any]]) -> str:
        """Detect intent type"""
        if context and context.get("has_existing_project"):
            if any(word in prompt_lower for word in ["add", "also", "include", "plus"]):
                return "extend_app"
            elif any(word in prompt_lower for word in ["change", "modify", "update", "fix"]):
                return "modify_app"
        
        if any(word in prompt_lower for word in ["create", "build", "make", "generate", "new"]):
            return "new_app"
        
        if any(word in prompt_lower for word in ["what", "how", "why", "explain", "help"]):
            return "clarification"
        
        return "new_app"
    
    def _detect_heuristic_app_type(self, prompt_lower: str, domain_info: Dict[str, Any]) -> str:
        """Detect app type"""
        # Use domain-specific type if available
        if domain_info["is_specialized"]:
            return domain_info["specific_type"]
        
        # Detect common app types
        app_patterns = {
            "counter": ["counter", "count", "increment", "decrement"],
            "todo": ["todo", "task", "checklist"],
            "calculator": ["calculator", "calc", "calculate"],
            "weather": ["weather", "forecast", "temperature"],
            "notes": ["note", "memo", "notepad"],
            "generic": []  # Default
        }
        
        for app_type, keywords in app_patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return app_type
        
        return "generic"
    
    def _detect_heuristic_complexity(self, prompt_lower: str, requirements: Dict[str, Any]) -> str:
        """Detect complexity"""
        # Special domains are complex
        if requirements["needs_hardware"]:
            return "hardware"
        if requirements["needs_ai_ml"]:
            return "ai_ml"
        
        # Check for complexity indicators
        if any(word in prompt_lower for word in ["simple", "basic", "minimal"]):
            return "simple"
        
        if any(word in prompt_lower for word in ["complex", "advanced", "complete", "full"]):
            return "complex"
        
        # Word count based
        word_count = len(prompt_lower.split())
        if word_count <= 10:
            return "simple"
        elif word_count <= 30:
            return "medium"
        else:
            return "complex"
    
    def _extract_heuristic_features(self, prompt_lower: str) -> List[str]:
        """Extract features"""
        features = []
        
        components = {
            "button": ["button", "btn", "click", "press"],
            "input": ["input", "text field", "textbox"],
            "text": ["text", "label", "heading"],
            "list": ["list", "items", "grid"],
            "image": ["image", "picture", "photo"],
            "switch": ["switch", "toggle", "checkbox"],
            "slider": ["slider", "range"],
            "map": ["map", "location", "gps"],
            "chart": ["chart", "graph", "plot"],
            "video": ["video", "player"]
        }
        
        for component, keywords in components.items():
            if any(keyword in prompt_lower for keyword in keywords):
                features.append(component)
        
        # Special features
        if any(word in prompt_lower for word in ["login", "signin", "authentication"]):
            features.append("authentication")
        
        if any(word in prompt_lower for word in ["database", "storage", "save"]):
            features.append("data_storage")
        
        if any(word in prompt_lower for word in ["api", "fetch", "request"]):
            features.append("api_integration")
        
        return list(set(features))
    
    def _calculate_heuristic_confidence(
        self, 
        prompt_lower: str, 
        intent_type: str, 
        features: List[str]
    ) -> float:
        """Calculate confidence"""
        confidence = 0.5
        
        # Adjust based on prompt length
        word_count = len(prompt_lower.split())
        if word_count >= 5:
            confidence += 0.1
        if word_count >= 10:
            confidence += 0.1
        
        # Adjust based on specific keywords
        specific_keywords = ["counter", "todo", "calculator", "weather", "notes"]
        if any(keyword in prompt_lower for keyword in specific_keywords):
            confidence += 0.2
        
        # Adjust based on features
        if features:
            confidence += min(0.2, len(features) * 0.05)
        
        return min(0.9, confidence)
    
    # ====== FORMAT CONVERSION ======
    
    def _convert_to_compatible_format(self, result: Dict[str, Any], domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert enhanced result to compatible format"""
        compatible = {
            "intent_type": result.get("intent_type", "new_app"),
            "app_type": result.get("app_type", "generic"),
            "complexity": result.get("complexity", "medium"),
            "confidence": result.get("confidence", 0.5),
            "extracted_features": result.get("extracted_features", []),
            "requires_context": result.get("requires_context", False),
            "needs_clarification": result.get("needs_clarification", False),
            "action": result.get("action", "proceed"),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "validated": True,
                "domain": domain_info["domain"],
                "specific_type": domain_info["specific_type"],
                "is_specialized": domain_info["is_specialized"]
            }
        }
        
        # Add technical requirements if available
        if "technical_requirements" in result:
            compatible["technical_requirements"] = result["technical_requirements"]
        
        return compatible
    
    # ====== EMERGENCY FALLBACK ======
    
    def _emergency_fallback(self, prompt: str) -> Dict[str, Any]:
        """Emergency fallback"""
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
    
    # ====== STATISTICS AND MONITORING ======
    
    def _update_stats(self, domain_info: Dict[str, Any], result: Dict[str, Any]):
        """Update statistics"""
        complexity = result.get("complexity", "medium")
        
        # Update complexity distribution
        if complexity in self.stats['complexity_distribution']:
            self.stats['complexity_distribution'][complexity] += 1
        
        # Update domain breakdown
        domain = domain_info["domain"]
        if domain not in self.stats['domain_breakdown']:
            self.stats['domain_breakdown'][domain] = 0
        self.stats['domain_breakdown'][domain] += 1
    
    def _track_response_time(self, response_time: float):
        """Track response time"""
        self.response_times.append(response_time)
        self.stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        total = self.stats['total_requests']
        
        if total == 0:
            return self.stats
        
        stats = {
            **self.stats,
            'llama3_success_rate': (self.stats['llama3_success'] / total) * 100,
            'heuristic_fallback_rate': (self.stats['heuristic_fallback'] / total) * 100,
            'cache_hit_rate': (self.stats['cache_hits'] / total) * 100
        }
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'llama3_success': 0,
            'heuristic_fallback': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'avg_response_time': 0.0,
            'domain_breakdown': {},
            'complexity_distribution': {
                'simple': 0, 'medium': 0, 'complex': 0,
                'simple_ui': 0, 'data_driven': 0, 'integrated': 0,
                'enterprise': 0, 'hardware': 0, 'ai_ml': 0
            }
        }
        self.response_times = []
    
    def clear_cache(self):
        """Clear intent cache"""
        self.intent_cache.clear()
        logger.info("Intent cache cleared")


# ====== FACTORY FUNCTIONS (for compatibility) ======

def create_intent_analyzer(config: Dict[str, Any]) -> EnhancedLlama3IntentAnalyzer:
    """
    Factory function - Replaces old create_intent_analyzer
    """
    return EnhancedLlama3IntentAnalyzer(config)


# ====== GLOBAL INSTANCE ======

try:
    # Try to get settings from your config
    from app.config import settings
    
    # Check if settings have the required attributes
    config_dict = {}
    
    # Try to get each setting, with fallbacks
    try:
        config_dict['llama3_api_url'] = settings.LLAMA3_API_URL
    except AttributeError:
        config_dict['llama3_api_url'] = "https://fastchat.ideeza.com/v1/chat/completions"
    
    try:
        config_dict['llama3_api_key'] = settings.LLAMA3_API_KEY
    except AttributeError:
        config_dict['llama3_api_key'] = ""
    
    try:
        config_dict['llama3_model'] = settings.LLAMA3_MODEL
    except AttributeError:
        config_dict['llama3_model'] = "llama-3-70b-instruct"
    
    try:
        config_dict['timeout'] = settings.LLAMA3_TIMEOUT
    except AttributeError:
        config_dict['timeout'] = 30.0
    
    try:
        config_dict['max_retries'] = settings.LLAMA3_MAX_RETRIES
    except AttributeError:
        config_dict['max_retries'] = 2
    
    # Create analyzer with extracted config
    intent_analyzer = EnhancedLlama3IntentAnalyzer(config_dict)
    
except ImportError:
    # Settings module not found
    intent_analyzer = None
except Exception as e:
    # Any other error
    logger.warning(f"Failed to create global intent analyzer: {e}")
    intent_analyzer = None

# ====== EXPORTS ======

__all__ = ["EnhancedLlama3IntentAnalyzer", "create_intent_analyzer", "intent_analyzer"]