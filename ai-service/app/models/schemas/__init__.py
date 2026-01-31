"""
Unified schema system for AI app builder.

This module provides all data models, validation, and helper functions
for the complete app generation pipeline.
"""

from .core import (
    PropertyValue,
    ComponentStyle,
    COMPONENT_PROPERTY_SCHEMAS,
)

from .input_output import (
    AIRequest,
    ProgressUpdate,
    ErrorResponse,
    CompleteResponse,
)

from .architecture import (
    ArchitectureDesign,
    ScreenDefinition,
    NavigationStructure,
    StateDefinition,
    DataFlowDiagram,
)

from .components import (
    BaseComponentProperties,
    ButtonProperties,
    InputTextProperties,
    TextProperties,
    SwitchProperties,
    CheckboxProperties,
    SliderProperties,
    EnhancedComponentDefinition,
)

from .layout import (
    Position,
    LayoutConstraints,
    EnhancedLayoutDefinition,
)

from .blockly import (
    BlocklyDefinition,
    BlocklyBlock,
    BlocklyWorkspace,
    EnhancedBlocklyDefinition,
)

from .context import (
    PromptContext,
    IntentAnalysis,
    EnrichedContext,
)

from .validation import (
    create_component,
    validate_layout,
    validate_component,
    get_component_schema,
    validate_color,
    validate_bounds,
    check_collisions,
)

__all__ = [
    # Core types
    'PropertyValue',
    'ComponentStyle',
    'COMPONENT_PROPERTY_SCHEMAS',
    
    # Input/Output
    'AIRequest',
    'ProgressUpdate',
    'ErrorResponse',
    'CompleteResponse',
    
    # Architecture
    'ArchitectureDesign',
    'ScreenDefinition',
    'NavigationStructure',
    'StateDefinition',
    'DataFlowDiagram',
    
    # Components
    'BaseComponentProperties',
    'ButtonProperties',
    'InputTextProperties',
    'TextProperties',
    'SwitchProperties',
    'CheckboxProperties',
    'SliderProperties',
    'EnhancedComponentDefinition',
    
    # Layout
    'Position',
    'LayoutConstraints',
    'EnhancedLayoutDefinition',
    
    # Blockly
    'BlocklyDefinition',
    'BlocklyBlock',
    'BlocklyWorkspace',
    'EnhancedBlocklyDefinition',
    
    # Context
    'PromptContext',
    'IntentAnalysis',
    'EnrichedContext',
    
    # Validation helpers
    'create_component',
    'validate_layout',
    'validate_component',
    'get_component_schema',
    'validate_color',
    'validate_bounds',
    'check_collisions',
]