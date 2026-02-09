"""
app/llm/prompt_manager.py
Enhanced prompt management with JSON formatting enforcement
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
    V3 = "v3"  # New version with strict JSON formatting


class PromptType(str, Enum):
    """Types of prompts"""
    ARCHITECTURE = "architecture"
    LAYOUT = "layout"
    BLOCKLY = "blockly"
    INTENT_ANALYSIS = "intent_analysis"
    CODE_GENERATION = "code_generation"
    DESCRIPTION = "description"


class PromptManager:
    """
    Enhanced prompt manager with JSON formatting enforcement
    """
    
    def __init__(self, default_version: PromptVersion = PromptVersion.V3):
        self.default_version = default_version
        self.templates = self._initialize_templates()
        logger.info(f"PromptManager initialized with default version: {default_version}")
    
    def _initialize_templates(self) -> Dict[PromptVersion, Dict[PromptType, str]]:
        """Initialize all prompt templates"""
        return {
            PromptVersion.V1: {
                PromptType.ARCHITECTURE: self._get_architecture_v1(),
                PromptType.LAYOUT: self._get_layout_v1(),
                PromptType.BLOCKLY: self._get_blockly_v1(),
            },
            PromptVersion.V2: {
                PromptType.ARCHITECTURE: self._get_architecture_v2(),
                PromptType.LAYOUT: self._get_layout_v2(),
                PromptType.BLOCKLY: self._get_blockly_v2(),
                PromptType.INTENT_ANALYSIS: self._get_intent_analysis_v2(),
            },
            PromptVersion.V3: {
                PromptType.ARCHITECTURE: self._get_architecture_v3(),
                PromptType.LAYOUT: self._get_layout_v3(),
                PromptType.BLOCKLY: self._get_blockly_v3(),
                PromptType.INTENT_ANALYSIS: self._get_intent_analysis_v3(),
                PromptType.CODE_GENERATION: self._get_code_generation_v3(),
                PromptType.DESCRIPTION: self._get_description_v3(),
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
                # Use safe formatting to avoid KeyError
                import string
                formatter = string.Formatter()
                field_names = [fname for _, fname, _, _ in formatter.parse(template) if fname]
                
                # Check all required variables are present
                for field in field_names:
                    if field not in variables:
                        logger.warning(f"Missing variable '{field}' in prompt variables")
                
                template = template.format(**{k: v for k, v in variables.items() if k in field_names})
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
        system_override: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> List[LLMMessage]:
        """
        Build complete message list for LLM with optional examples
        
        Args:
            prompt_type: Type of prompt
            user_input: User's input/request
            variables: Variables for template
            version: Prompt version
            system_override: Override system prompt
            examples: List of example conversations
            
        Returns:
            List of LLMMessage objects
        """
        system_prompt = system_override or self.get_prompt(prompt_type, variables, version)
        
        messages = [LLMMessage(role="system", content=system_prompt)]
        
        # Add examples if provided
        if examples:
            for example in examples:
                if "user" in example:
                    messages.append(LLMMessage(role="user", content=example["user"]))
                if "assistant" in example:
                    messages.append(LLMMessage(role="assistant", content=example["assistant"]))
        
        messages.append(LLMMessage(role="user", content=user_input))
        
        return messages
    
    # ============ VERSION 1 TEMPLATES ============
    
    def _get_architecture_v1(self) -> str:
        return """Generate a mobile app architecture in JSON format."""
    
    def _get_layout_v1(self) -> str:
        return """Generate a mobile app layout in JSON format."""
    
    def _get_blockly_v1(self) -> str:
        return """Generate Blockly blocks in JSON format."""
    
    # ============ VERSION 2 TEMPLATES ============
    
    def _get_architecture_v2(self) -> str:
        return """You are a mobile app architect. Generate a JSON structure for a mobile application based on the user's request.

The JSON should include:
- app_type: "single-page" or "multi-page"
- screens: array of screen objects
- navigation: navigation structure
- state_management: state variables
- data_flow: user interactions and API calls

Generate valid JSON only."""
    
    def _get_layout_v2(self) -> str:
        return """You are a UI/UX designer. Generate a JSON structure for a mobile app layout based on the specification.

The JSON should include:
- screenId: identifier of the screen
- layoutType: "flex" or "absolute"
- components: array of UI components with properties and constraints

Generate valid JSON only."""
    
    def _get_blockly_v2(self) -> str:
        return """You are a visual programming expert. Generate Blockly blocks configuration in JSON format.

The JSON should follow Blockly workspace format with:
- blocks: array of block definitions
- variables: variable declarations
- custom_blocks: custom block definitions

Generate valid JSON only."""
    
    def _get_intent_analysis_v2(self) -> str:
        return """Analyze the user's request and determine the intent. Return a JSON object with:
- intent: primary intent (e.g., "create_counter", "create_todo", "create_calculator")
- confidence: confidence score (0-1)
- features: array of requested features
- complexity: "simple", "medium", or "complex"

Generate valid JSON only."""
    
    # ============ VERSION 3 TEMPLATES (ENHANCED WITH JSON ENFORCEMENT) ============
    
    def _get_architecture_v3(self) -> str:
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

You are an expert mobile app architect. Generate a comprehensive JSON structure for a mobile application.

STRICT REQUIREMENTS:
1. Response must be valid JSON parsable by json.loads()
2. No additional text before or after the JSON
3. No markdown formatting (no ```json, no ```)
4. Follow this exact schema:

{
  "app_type": "single-page" | "multi-page",
  "screens": [
    {
      "id": "string",
      "name": "string",
      "purpose": "string",
      "components": ["string"],
      "navigation": ["string"]
    }
  ],
  "navigation": {
    "type": "string",
    "routes": [
      {
        "from": "string",
        "to": "string",
        "label": "string"
      }
    ]
  },
  "state_management": [
    {
      "name": "string",
      "type": "string",
      "scope": "string",
      "initial_value": "any"
    }
  ],
  "data_flow": {
    "user_interactions": ["string"],
    "api_calls": ["string"],
    "local_storage": ["string"]
  },
  "theme": {
    "primary_color": "#hex",
    "secondary_color": "#hex",
    "background_color": "#hex"
  }
}

IMPORTANT: Return ONLY the JSON object, nothing else."""
    
    def _get_layout_v3(self) -> str:
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

You are a senior UI/UX designer. Generate a detailed JSON structure for a mobile app layout.

STRICT REQUIREMENTS:
1. Response must be valid JSON parsable by json.loads()
2. No additional text before or after the JSON
3. No markdown formatting (no ```json, no ```)
4. Follow this exact schema:

{
  "screenId": "string",
  "layoutType": "flex" | "absolute",
  "backgroundColor": "#hex",
  "components": [
    {
      "id": "string",
      "type": "Text" | "Button" | "InputText" | "Image" | "List" | "Checkbox" | "Slider",
      "properties": {
        "value": "string",
        "fontSize": number,
        "fontWeight": "string",
        "color": "#hex"
      },
      "position": {
        "x": number,
        "y": number
      },
      "constraints": {
        "width": number | "string",
        "height": number,
        "marginTop": number,
        "marginBottom": number,
        "marginLeft": number | "string",
        "marginRight": number | "string",
        "alignment": "string"
      }
    }
  ]
}

IMPORTANT: Return ONLY the JSON object, nothing else."""
    
    def _get_blockly_v3(self) -> str:
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

You are a visual programming expert. Generate Blockly blocks configuration.

STRICT REQUIREMENTS:
1. Response must be valid JSON parsable by json.loads()
2. No additional text before or after the JSON
3. No markdown formatting (no ```json, no ```)
4. Follow this exact schema:

{
  "blocks": {
    "languageVersion": 0,
    "blocks": [
      {
        "type": "string",
        "id": "string",
        "fields": {
          "COMPONENT": "string",
          "EVENT": "string"
        },
        "inputs": {
          "DO": {
            "block": {
              "type": "string",
              "id": "string",
              "fields": {
                "STATE_VAR": "string",
                "OPERATION": "string",
                "VALUE": "string"
              }
            }
          }
        }
      }
    ]
  },
  "variables": [
    {
      "name": "string",
      "id": "string",
      "type": "string"
    }
  ],
  "custom_blocks": []
}

IMPORTANT: Return ONLY the JSON object, nothing else."""
    
    def _get_intent_analysis_v3(self) -> str:
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

Analyze the user's app creation request and determine the intent.

STRICT REQUIREMENTS:
1. Response must be valid JSON parsable by json.loads()
2. No additional text before or after the JSON
3. No markdown formatting (no ```json, no ```)
4. Follow this exact schema:

{
  "intent": "counter" | "todo" | "calculator" | "notes" | "weather" | "generic",
  "confidence": number,
  "features": ["string"],
  "complexity": "simple" | "medium" | "complex",
  "screens_required": number,
  "requires_api": boolean,
  "requires_storage": boolean
}

IMPORTANT: Return ONLY the JSON object, nothing else."""
    
    def _get_code_generation_v3(self) -> str:
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

Generate code based on the specification.

STRICT REQUIREMENTS:
1. Response must be valid JSON parsable by json.loads()
2. No additional text before or after the JSON
3. No markdown formatting (no ```json, no ```)
4. Follow this exact schema:

{
  "language": "string",
  "framework": "string",
  "files": [
    {
      "filename": "string",
      "content": "string"
    }
  ]
}

IMPORTANT: Return ONLY the JSON object, nothing else."""
    
    def _get_description_v3(self) -> str:
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

Generate app description based on the specification.

STRICT REQUIREMENTS:
1. Response must be valid JSON parsable by json.loads()
2. No additional text before or after the JSON
3. No markdown formatting (no ```json, no ```)
4. Follow this exact schema:

{
  "app_name": "string",
  "description": "string",
  "key_features": ["string"],
  "target_audience": "string",
  "complexity": "string"
}

IMPORTANT: Return ONLY the JSON object, nothing else."""
    
    def get_available_versions(self) -> List[str]:
        """Get list of available prompt versions"""
        return [v.value for v in PromptVersion]
    
    def get_available_types(self) -> List[str]:
        """Get list of available prompt types"""
        return [t.value for t in PromptType]
    
    def validate_prompt_response(self, response: str, expected_type: PromptType) -> bool:
        """Validate that response matches expected format"""
        import json
        
        try:
            data = json.loads(response)
            
            # Basic JSON validation
            if not isinstance(data, dict):
                return False
            
            # Type-specific validation could be added here
            return True
            
        except json.JSONDecodeError:
            return False
    
    def _get_domain_aware_architecture_v3(self) -> str:
        """V3 architecture prompt that handles ANY app type"""
        return """IMPORTANT: You MUST respond with valid JSON only. Do NOT include any markdown, code blocks, explanations, or additional text.

    You are an expert application architect capable of designing ANY type of mobile application.

    STRICT REQUIREMENTS:
    1. Response must be valid JSON parsable by json.loads()
    2. No additional text before or after the JSON
    3. No markdown formatting (no ```json, no ```)
    4. Analyze the app type and domain to provide appropriate architecture
    5. Follow this schema:

    {
    "app_type": "string (descriptive name of the app)",
    "domain": "productivity|entertainment|utility|business|education|health_fitness|finance|development|iot_hardware|creative_media|data_science|custom",
    "complexity": "simple_ui|data_driven|integrated|enterprise|hardware|ai_ml",
    "architecture": {
        "pattern": "string (recommended architecture pattern)",
        "screens": [
        {
            "id": "string",
            "name": "string",
            "purpose": "string",
            "key_components": ["string"],
            "special_requirements": ["string"]
        }
        ],
        "state_management": {
        "approach": "string",
        "key_states": [
            {
            "name": "string",
            "type": "string",
            "scope": "global|screen|component"
            }
        ]
        },
        "data_flow": {
        "sources": ["string"],
        "transformations": ["string"],
        "destinations": ["string"]
        },
        "special_considerations": ["string"]
    },
    "technical_requirements": {
        "apis_needed": ["string"],
        "permissions_required": ["string"],
        "platform_considerations": ["string"],
        "risk_factors": ["string"]
    },
    "recommended_components": ["string"],
    "estimated_effort": "small|medium|large|very_large"
    }

    SPECIAL DOMAIN HANDLING:
    - For hardware/IoT apps: Include real-time communication, device state management
    - For AI/ML apps: Include data processing pipelines, model management
    - For real-time apps: Include WebSockets, live updates, connection management
    - For 3D/visualization apps: Include WebGL/Three.js considerations
    - For enterprise apps: Include security, scalability, multi-user considerations

    IMPORTANT: Return ONLY the JSON object, nothing else. Analyze the app type and provide domain-appropriate architecture."""