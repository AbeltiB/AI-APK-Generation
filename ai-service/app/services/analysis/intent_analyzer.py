"""
Intent Analyzer - Llama3 Only Version

Analyzes user intent using Llama3 with heuristic fallback.
No Claude, no Groq - simplified two-tier system.
"""
from typing import Dict, Any, Optional
import json
import re

from app.llm.orchestrator import LLMOrchestrator
from app.llm.base import LLMMessage
from app.models.schemas.context import IntentAnalysis
from app.config import settings
from app.utils.logging import get_logger, log_context

logger = get_logger(__name__)


class LlamaIntentAnalyzer:
    """
    Production intent analyzer using Llama3.
    
    Two-tier fallback:
    1. Llama3 (primary)
    2. Heuristic (fallback)
    """
    
    def __init__(self):
        """Initialize with Llama3 orchestrator"""
        
        # Get LLM config from settings
        llm_config = settings.llm_config
        
        # Initialize orchestrator
        self.orchestrator = LLMOrchestrator(llm_config)
        
        # Statistics
        self.stats = {
            'total_classifications': 0,
            'llama3_success': 0,
            'heuristic_fallback': 0,
            'cache_hits': 0
        }
        
        logger.info(
            "intent_analyzer.initialized",
            extra={
                "provider": "llama3",
                "fallback": "heuristic"
            }
        )
    
    async def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysis:
        """
        Analyze user prompt to determine intent.
        
        Args:
            prompt: User's natural language request
            context: Optional context (conversation history, existing project)
            
        Returns:
            IntentAnalysis object
        """
        
        self.stats['total_classifications'] += 1
        
        with log_context(operation="intent_analysis"):
            logger.info(
                "intent_analysis.started",
                extra={
                    "prompt_length": len(prompt),
                    "has_context": context is not None
                }
            )
            
            try:
                # Build messages for Llama3
                messages = self._build_messages(prompt, context)
                
                # Try Llama3
                try:
                    response = await self.orchestrator.generate(
                        messages=messages,
                        temperature=0.3,  # Lower temperature for classification
                        max_tokens=500
                    )
                    
                    # Parse JSON response
                    intent_data = self._parse_llm_response(response.content)
                    
                    if intent_data:
                        self.stats['llama3_success'] += 1
                        
                        logger.info(
                            "intent_analysis.llama3_success",
                            extra={
                                "intent_type": intent_data.get('intent_type'),
                                "confidence": intent_data.get('confidence'),
                                "tokens_used": response.tokens_used
                            }
                        )
                        
                        return self._create_intent_analysis(intent_data)
                    
                except Exception as e:
                    logger.warning(
                        "intent_analysis.llama3_failed",
                        extra={"error": str(e)},
                        exc_info=e
                    )
                
                # Fallback to heuristic
                logger.info("intent_analysis.using_heuristic_fallback")
                self.stats['heuristic_fallback'] += 1
                
                return self._heuristic_analysis(prompt, context)
                
            except Exception as e:
                logger.error(
                    "intent_analysis.failed",
                    extra={"error": str(e)},
                    exc_info=e
                )
                
                # Emergency fallback
                return self._emergency_fallback(prompt)
    
    def _build_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> list[LLMMessage]:
        """Build messages for Llama3"""
        
        system_prompt = """You are an intent classifier for a mobile app builder.

Analyze the user's request and return ONLY a JSON object with this exact structure:

{
  "intent_type": "new_app" | "extend_app" | "modify_app" | "clarification" | "help",
  "complexity": "simple" | "medium" | "complex",
  "confidence": 0.0-1.0,
  "extracted_entities": {
    "components": ["Button", "Text", ...],
    "actions": ["click", "submit", ...],
    "data": ["todo", "user", ...],
    "features": ["authentication", "search", ...]
  },
  "requires_context": true | false,
  "multi_turn": true | false
}

Intent types:
- new_app: Creating a new application from scratch
- extend_app: Adding features to existing app
- modify_app: Changing existing functionality
- clarification: Asking questions or seeking help
- help: General help or guidance

Complexity levels:
- simple: Basic single-screen app (1-3 components)
- medium: Multi-feature app (4-8 components)
- complex: Advanced app (9+ components, multiple screens, integrations)

Return ONLY the JSON object, no other text."""
        
        user_content = f"User request: {prompt}"
        
        if context and context.get('has_existing_project'):
            user_content += "\n\nNote: User has an existing project in this session."
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content)
        ]
    
    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response"""
        
        try:
            # Remove markdown code blocks if present
            content = content.strip()
            
            # Remove ```json and ``` markers
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\s*\n', '', content)
                content = re.sub(r'\n```\s*$', '', content)
            
            # Parse JSON
            data = json.loads(content)
            
            # Validate required fields
            required = ['intent_type', 'complexity', 'confidence', 'extracted_entities']
            if not all(field in data for field in required):
                logger.warning(
                    "intent_analysis.invalid_json",
                    extra={"missing_fields": [f for f in required if f not in data]}
                )
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.warning(
                "intent_analysis.json_parse_error",
                extra={"error": str(e), "content": content[:200]}
            )
            return None
        except Exception as e:
            logger.error(
                "intent_analysis.parse_error",
                extra={"error": str(e)},
                exc_info=e
            )
            return None
    
    def _create_intent_analysis(self, data: Dict[str, Any]) -> IntentAnalysis:
        """Create IntentAnalysis from parsed data"""
        
        return IntentAnalysis(
            intent_type=data['intent_type'],
            complexity=data['complexity'],
            confidence=float(data['confidence']),
            extracted_entities=data.get('extracted_entities', {
                "components": [],
                "actions": [],
                "data": [],
                "features": []
            }),
            requires_context=data.get('requires_context', False),
            multi_turn=data.get('multi_turn', False)
        )
    
    def _heuristic_analysis(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> IntentAnalysis:
        """Heuristic fallback analysis"""
        
        prompt_lower = prompt.lower()
        
        # Detect intent type
        intent_type = "new_app"
        confidence = 0.6
        
        if any(word in prompt_lower for word in ['add', 'extend', 'also', 'include']):
            intent_type = "extend_app"
            confidence = 0.7
        elif any(word in prompt_lower for word in ['change', 'modify', 'update', 'fix', 'replace']):
            intent_type = "modify_app"
            confidence = 0.7
        elif any(word in prompt_lower for word in ['what', 'how', 'why', 'explain', 'help']):
            intent_type = "clarification"
            confidence = 0.8
        elif any(word in prompt_lower for word in ['create', 'build', 'make', 'new']):
            intent_type = "new_app"
            confidence = 0.8
        
        # Detect complexity
        word_count = len(prompt.split())
        
        if word_count <= 10:
            complexity = "simple"
        elif word_count <= 30:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # Extract components (basic keyword matching)
        components = []
        component_keywords = {
            'button': 'Button',
            'text': 'Text',
            'input': 'InputText',
            'list': 'List',
            'image': 'Image',
            'switch': 'Switch',
            'slider': 'Slider'
        }
        
        for keyword, component in component_keywords.items():
            if keyword in prompt_lower:
                components.append(component)
        
        # Extract actions
        actions = []
        action_keywords = ['click', 'tap', 'press', 'submit', 'save', 'delete', 'add', 'remove']
        actions = [action for action in action_keywords if action in prompt_lower]
        
        # Extract data types
        data_types = []
        data_keywords = ['todo', 'task', 'user', 'product', 'item', 'note', 'message']
        data_types = [data for data in data_keywords if data in prompt_lower]
        
        # Extract features
        features = []
        feature_keywords = ['login', 'auth', 'search', 'filter', 'sort', 'notification']
        features = [feature for feature in feature_keywords if feature in prompt_lower]
        
        logger.info(
            "intent_analysis.heuristic_complete",
            extra={
                "intent_type": intent_type,
                "complexity": complexity,
                "confidence": confidence
            }
        )
        
        return IntentAnalysis(
            intent_type=intent_type,
            complexity=complexity,
            confidence=confidence,
            extracted_entities={
                "components": components,
                "actions": actions,
                "data": data_types,
                "features": features
            },
            requires_context=intent_type in ['extend_app', 'modify_app'],
            multi_turn=False
        )
    
    def _emergency_fallback(self, prompt: str) -> IntentAnalysis:
        """Emergency fallback when everything fails"""
        
        logger.warning("intent_analysis.emergency_fallback")
        
        return IntentAnalysis(
            intent_type="clarification",
            complexity="medium",
            confidence=0.3,
            extracted_entities={
                "components": [],
                "actions": [],
                "data": [],
                "features": []
            },
            requires_context=False,
            multi_turn=False
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        
        total = self.stats['total_classifications']
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'llama3_success_rate': (self.stats['llama3_success'] / total) * 100,
            'heuristic_fallback_rate': (self.stats['heuristic_fallback'] / total) * 100
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_classifications': 0,
            'llama3_success': 0,
            'heuristic_fallback': 0,
            'cache_hits': 0
        }


# Global instance
intent_analyzer = LlamaIntentAnalyzer()


# Testing
if __name__ == "__main__":
    import asyncio
    
    async def test_analyzer():
        """Test intent analyzer"""
        
        print("\n" + "=" * 70)
        print("INTENT ANALYZER TEST (Llama3)")
        print("=" * 70)
        
        test_cases = [
            {
                "prompt": "Create a simple todo list app",
                "context": None
            },
            {
                "prompt": "Add delete buttons to each item",
                "context": {"has_existing_project": True}
            },
            {
                "prompt": "Build a complex e-commerce app with payment and authentication",
                "context": None
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: \"{test['prompt']}\"")
            
            result = await intent_analyzer.analyze(
                prompt=test['prompt'],
                context=test['context']
            )
            
            print(f"   Intent: {result.intent_type}")
            print(f"   Complexity: {result.complexity}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Components: {len(result.extracted_entities.get('components', []))}")
            print(f"   Requires Context: {result.requires_context}")
        
        # Show stats
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        
        stats = intent_analyzer.get_stats()
        print(f"Total: {stats['total_classifications']}")
        print(f"Llama3 Success: {stats['llama3_success']}")
        print(f"Heuristic Fallback: {stats['heuristic_fallback']}")
        
        if stats['total_classifications'] > 0:
            print(f"Llama3 Success Rate: {stats.get('llama3_success_rate', 0):.1f}%")
        
        print("\n" + "=" * 70 + "\n")
    
    asyncio.run(test_analyzer())