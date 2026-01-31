"""
app/llm/prompt_manager.py
Prompt template management and versioning
"""
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

from .base import LLMMessage


logger = logging.getLogger(__name__)


class PromptVersion(str, Enum):
    """Available prompt versions"""
    V1 = "v1"
    V2 = "v2"


class PromptType(str, Enum):
    """Types of prompts"""
    APP_GENERATION = "app_generation"
    CODE_GENERATION = "code_generation"
    DESCRIPTION = "description"
    OPTIMIZATION = "optimization"


class PromptManager:
    """
    Manages prompt templates with versioning
    """
    
    def __init__(self, default_version: PromptVersion = PromptVersion.V2):
        self.default_version = default_version
        self.templates = self._initialize_templates()
        logger.info(f"PromptManager initialized with default version: {default_version}")
    
    def _initialize_templates(self) -> Dict[PromptVersion, Dict[PromptType, str]]:
        """Initialize all prompt templates"""
        return {
            PromptVersion.V1: {
                PromptType.APP_GENERATION: self._get_app_generation_v1(),
                PromptType.CODE_GENERATION: self._get_code_generation_v1(),
                PromptType.DESCRIPTION: self._get_description_v1(),
            },
            PromptVersion.V2: {
                PromptType.APP_GENERATION: self._get_app_generation_v2(),
                PromptType.CODE_GENERATION: self._get_code_generation_v2(),
                PromptType.DESCRIPTION: self._get_description_v2(),
            }
        }
    
    def get_prompt(
        self,
        prompt_type: PromptType,
        variables: Optional[Dict[str, Any]] = None,
        version: Optional[PromptVersion] = None
    ) -> str:
        """
        Get prompt template with variable substitution
        
        Args:
            prompt_type: Type of prompt needed
            variables: Variables to substitute in template
            version: Specific version to use (defaults to default_version)
            
        Returns:
            Formatted prompt string
        """
        version = version or self.default_version
        
        if version not in self.templates:
            logger.warning(f"Version {version} not found, using default")
            version = self.default_version
        
        if prompt_type not in self.templates[version]:
            raise ValueError(f"Prompt type {prompt_type} not found in version {version}")
        
        template = self.templates[version][prompt_type]
        
        if variables:
            try:
                template = template.format(**variables)
            except KeyError as e:
                logger.error(f"Missing variable in prompt template: {e}")
                raise ValueError(f"Missing required variable: {e}")
        
        return template
    
    def build_messages(
        self,
        prompt_type: PromptType,
        user_input: str,
        variables: Optional[Dict[str, Any]] = None,
        version: Optional[PromptVersion] = None,
        system_override: Optional[str] = None
    ) -> List[LLMMessage]:
        """
        Build complete message list for LLM
        
        Args:
            prompt_type: Type of prompt
            user_input: User's input/request
            variables: Variables for template
            version: Prompt version
            system_override: Override system prompt
            
        Returns:
            List of LLMMessage objects
        """
        system_prompt = system_override or self.get_prompt(prompt_type, variables, version)
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_input)
        ]
    
    # ============ VERSION 1 TEMPLATES ============
    
    def _get_app_generation_v1(self) -> str:
        """V1 app generation prompt"""
        return """You are a mobile app generator. Generate a JSON structure for a mobile application based on the user's request.

The JSON should include:
- type: app type
- description: app description
- screens: array of screen objects with components
- theme: color scheme

Be creative and comprehensive in your design."""
    
    def _get_code_generation_v1(self) -> str:
        """V1 code generation prompt"""
        return """Generate production-ready code based on the specification provided.

Follow best practices:
- Clean, readable code
- Proper error handling
- Comments where necessary
- Efficient algorithms

Language: {language}"""
    
    def _get_description_v1(self) -> str:
        """V1 description prompt"""
        return """Analyze the following app specification and provide a clear, concise description of what the app does and its key features."""
    
    # ============ VERSION 2 TEMPLATES ============
    
    def _get_app_generation_v2(self) -> str:
        """V2 app generation prompt - Enhanced"""
        return """You are an expert mobile app architect specializing in intuitive, user-friendly applications.

Generate a comprehensive JSON structure for a mobile application based on the user's request.

Requirements:
1. Structure must include:
   - type: application category
   - description: clear app description
   - screens: array of screen definitions with components
   - dataModel: data structures (if applicable)
   - theme: color scheme and styling
   - navigation: navigation flow (if multi-screen)

2. Component types available:
   - text: display text with styling
   - button: interactive buttons with actions
   - input: text input fields
   - list: scrollable lists
   - card: information cards
   - chart: data visualizations
   - image: image components

3. Best practices:
   - Follow platform design guidelines
   - Ensure accessibility
   - Optimize for performance
   - Include proper state management
   - Define clear user flows

4. Output format: Valid JSON only, no additional text.

Generate a production-ready app specification."""
    
    def _get_code_generation_v2(self) -> str:
        """V2 code generation prompt - Enhanced"""
        return """You are a senior software engineer specializing in production-ready code.

Language: {language}
Framework: {framework}

Requirements:
1. Code Quality:
   - Follow language/framework best practices
   - Use proper design patterns
   - Include comprehensive error handling
   - Add clear, concise comments
   - Follow naming conventions

2. Production Readiness:
   - Optimize for performance
   - Consider edge cases
   - Include input validation
   - Handle async operations properly
   - Use proper typing (if applicable)

3. Structure:
   - Modular and maintainable
   - Testable components
   - Clear separation of concerns
   - DRY principle

Generate clean, production-ready code based on the specification provided."""
    
    def _get_description_v2(self) -> str:
        """V2 description prompt - Enhanced"""
        return """You are a technical writer creating user-facing documentation.

Analyze the app specification and provide:

1. Overview: High-level description (2-3 sentences)
2. Key Features: Bullet list of main features
3. User Flow: Brief description of primary user journey
4. Technical Highlights: Notable technical aspects

Keep it clear, concise, and user-friendly. Focus on benefits and functionality."""
    
    def get_available_versions(self) -> List[str]:
        """Get list of available prompt versions"""
        return [v.value for v in PromptVersion]
    
    def get_available_types(self) -> List[str]:
        """Get list of available prompt types"""
        return [t.value for t in PromptType]