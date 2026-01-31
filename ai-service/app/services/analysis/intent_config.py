"""
Intent Classification System Configuration.

Centralized configuration for all intent classification components.
"""
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    """Intent types for user requests"""
    NEW_APP = "new_app"
    EXTEND_APP = "extend_app"
    MODIFY_APP = "modify_app"
    CLARIFICATION = "clarification"
    HELP = "help"
    UNSAFE = "unsafe"  # Dangerous/malicious requests


class ComplexityLevel(str, Enum):
    """Complexity levels for app generation"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ClassificationTier(str, Enum):
    """Classification tier used"""
    CLAUDE = "claude"
    GROQ = "groq"
    HEURISTIC = "heuristic"
    FAILED = "failed"


@dataclass
class RetryConfig:
    """Retry configuration for LLM tiers"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for decision making"""
    auto_accept: float = 0.85  # High confidence - proceed
    clarification: float = 0.70  # Medium - ask for clarification
    block_dangerous: float = 0.60  # Low - block modify/extend
    reject: float = 0.40  # Very low - reject request


@dataclass
class TierConfig:
    """Configuration for each classification tier"""
    name: str
    enabled: bool
    timeout: float
    retry_config: RetryConfig
    cost_per_1k_tokens: float  # For monitoring


class IntentClassificationConfig:
    """
    Master configuration for intent classification system.
    
    This centralizes all configuration to make the system easy to tune
    and maintain in production.
    """
    
    # ========================================================================
    # TIER CONFIGURATIONS
    # ========================================================================
    
    TIERS = {
        "claude": TierConfig(
            name="Claude (Anthropic)",
            enabled=True,
            timeout=15.0,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0
            ),
            cost_per_1k_tokens=0.003  # Approximate
        ),
        "groq": TierConfig(
            name="Groq Llama 3.1 8B",
            enabled=True,
            timeout=10.0,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=0.5,
                max_delay=5.0
            ),
            cost_per_1k_tokens=0.0  # Free tier
        ),
        "heuristic": TierConfig(
            name="Enhanced Heuristic",
            enabled=True,
            timeout=1.0,
            retry_config=RetryConfig(max_attempts=1),
            cost_per_1k_tokens=0.0
        )
    }
    
    # ========================================================================
    # CONFIDENCE THRESHOLDS
    # ========================================================================
    
    CONFIDENCE = ConfidenceThresholds()
    
    # ========================================================================
    # INTENT PATTERNS
    # ========================================================================
    
    # Keywords and patterns for heuristic classification
    INTENT_PATTERNS = {
        IntentType.NEW_APP: {
            "keywords": [
                "create", "build", "make", "generate", "new",
                "develop", "design", "construct", "start"
            ],
            "phrases": [
                "create a", "build a", "make a", "new app",
                "i want to create", "can you make"
            ],
            "aliases": {
                "todo": ["task", "checklist", "to-do", "reminder"],
                "calculator": ["calc", "math", "arithmetic"],
                "weather": ["forecast", "temperature", "climate"],
                "counter": ["count", "tally", "increment"],
            }
        },
        IntentType.EXTEND_APP: {
            "keywords": [
                "add", "extend", "include", "also", "plus",
                "additionally", "more", "append", "insert"
            ],
            "phrases": [
                "add a", "add feature", "also add", "include a",
                "i also want", "can you add"
            ]
        },
        IntentType.MODIFY_APP: {
            "keywords": [
                "change", "modify", "update", "fix", "replace",
                "alter", "edit", "adjust", "revise"
            ],
            "phrases": [
                "change the", "make it", "update the", "fix the",
                "replace with", "instead of"
            ]
        },
        IntentType.CLARIFICATION: {
            "keywords": [
                "what", "how", "why", "explain", "tell",
                "show", "help", "understand", "mean"
            ],
            "phrases": [
                "what is", "how do", "can you explain",
                "what does", "help me understand"
            ]
        },
        IntentType.HELP: {
            "keywords": [
                "help", "assist", "guide", "tutorial",
                "support", "documentation", "stuck"
            ],
            "phrases": [
                "i need help", "can you help", "how do i",
                "i'm stuck", "show me how"
            ]
        },
        IntentType.UNSAFE: {
            "keywords": [
                "hack", "exploit", "crack", "bypass", "steal",
                "malware", "virus", "phishing", "scam"
            ],
            "phrases": [
                "how to hack", "bypass security", "steal data",
                "create malware", "phishing page"
            ]
        }
    }
    
    # ========================================================================
    # COMPONENT PATTERNS
    # ========================================================================
    
    COMPONENT_ALIASES = {
        "button": ["btn", "click", "press", "tap"],
        "input": ["text field", "textbox", "entry", "field"],
        "text": ["label", "heading", "title", "paragraph"],
        "switch": ["toggle", "checkbox", "option"],
        "slider": ["range", "scale", "adjuster"],
        "list": ["listview", "items", "collection"],
        "image": ["picture", "photo", "icon"],
        "map": ["location", "gps", "navigation"],
        "chart": ["graph", "plot", "visualization"]
    }
    
    # ========================================================================
    # COMPLEXITY INDICATORS
    # ========================================================================
    
    COMPLEXITY_INDICATORS = {
        ComplexityLevel.SIMPLE: {
            "max_words": 15,
            "max_components": 3,
            "max_screens": 1,
            "keywords": ["simple", "basic", "minimal", "quick"]
        },
        ComplexityLevel.MEDIUM: {
            "max_words": 50,
            "max_components": 8,
            "max_screens": 3,
            "keywords": ["app", "application", "features", "multiple"]
        },
        ComplexityLevel.COMPLEX: {
            "max_words": float('inf'),
            "max_components": float('inf'),
            "max_screens": float('inf'),
            "keywords": [
                "advanced", "complete", "full", "comprehensive",
                "authentication", "api", "database", "backend",
                "payment", "integration", "analytics"
            ]
        }
    }
    
    # ========================================================================
    # USER MESSAGES
    # ========================================================================
    
    USER_MESSAGES = {
        "low_confidence_clarification": (
            "I understand you want to {intent_guess}, but I need a bit more "
            "information to ensure I build exactly what you're looking for. "
            "Could you provide more details about:\n"
            "- What specific features do you need?\n"
            "- What should the app do?\n"
            "- Any specific components or interactions?"
        ),
        "unsafe_request": (
            "I cannot help with that request as it appears to involve "
            "potentially harmful or malicious functionality. "
            "If you have a legitimate use case, please rephrase your request "
            "to focus on the constructive aspects."
        ),
        "extend_blocked": (
            "I'm not confident about extending the existing app based on "
            "your request. To avoid breaking anything, could you:\n"
            "1. Be more specific about what to add\n"
            "2. Describe where it should be added\n"
            "3. Explain how it should work"
        ),
        "modify_blocked": (
            "I'm not confident about modifying the existing app. "
            "To ensure I make the right changes, please specify:\n"
            "1. Exactly what needs to change\n"
            "2. What it should become\n"
            "3. Any specific requirements"
        ),
        "classification_failed": (
            "I'm having trouble understanding your request. "
            "Please try rephrasing it with more details about:\n"
            "- What you want to build\n"
            "- What features it should have\n"
            "- How users should interact with it"
        )
    }
    
    # ========================================================================
    # MONITORING & ALERTING
    # ========================================================================
    
    MONITORING = {
        "alert_on_tier_fallback": True,
        "alert_on_low_confidence": True,
        "alert_threshold_failures": 5,  # Alert after N consecutive failures
        "track_latency": True,
        "track_cost": True,
        "log_all_classifications": True
    }
    
    # ========================================================================
    # RATE LIMITING
    # ========================================================================
    
    RATE_LIMITS = {
        "claude_per_minute": 60,
        "groq_per_minute": 100,
        "per_user_per_hour": 100
    }


# Global config instance
config = IntentClassificationConfig()