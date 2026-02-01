"""
Models package - Unified schema and prompt system.

Exports:
- schemas: All data models and validation
- prompts: All prompt templates
"""

from .schemas import (
    # Core types
    PropertyValue,
    ComponentStyle,
    COMPONENT_PROPERTY_SCHEMAS,
    
    # Input/Output
    AIRequest,
    ProgressUpdate,
    ErrorResponse,
    CompleteResponse,
    
    # Architecture
    ArchitectureDesign,
    ScreenDefinition,
    NavigationStructure,
    StateDefinition,
    DataFlowDiagram,
    
    # Components
    BaseComponentProperties,
    ButtonProperties,
    InputTextProperties,
    TextProperties,
    SwitchProperties,
    CheckboxProperties,
    SliderProperties,
    EnhancedComponentDefinition,
    
    # Layout
    Position,
    LayoutConstraints,
    EnhancedLayoutDefinition,
    
    # Blockly
    BlocklyDefinition,
    BlocklyBlock,
    BlocklyWorkspace,
    EnhancedBlocklyDefinition,
    
    # Context
    PromptContext,
    IntentAnalysis,
    EnrichedContext,
    
    # Helper functions
    create_component,
    validate_layout,
    validate_component,
    get_component_schema,
)

from .prompts import (
    PromptTemplate,
    PromptLibrary,
    PromptType,
    PromptVersion,

)

__all__ = [
    # Schemas
    'PropertyValue',
    'ComponentStyle',
    'COMPONENT_PROPERTY_SCHEMAS',
    'AIRequest',
    'ProgressUpdate',
    'ErrorResponse',
    'CompleteResponse',
    'ArchitectureDesign',
    'ScreenDefinition',
    'NavigationStructure',
    'StateDefinition',
    'DataFlowDiagram',
    'BaseComponentProperties',
    'ButtonProperties',
    'InputTextProperties',
    'TextProperties',
    'SwitchProperties',
    'CheckboxProperties',
    'SliderProperties',
    'EnhancedComponentDefinition',
    'Position',
    'LayoutConstraints',
    'EnhancedLayoutDefinition',
    'BlocklyDefinition',
    'BlocklyBlock',
    'BlocklyWorkspace',
    'EnhancedBlocklyDefinition',
    'PromptContext',
    'IntentAnalysis',
    'EnrichedContext',
    'create_component',
    'validate_layout',
    'validate_component',
    'get_component_schema',
    
    # Prompts
    'PromptTemplate',
    'PromptLibrary',
    'PromptType',
    'PromptVersion',
    'prompts',
]