"""
Generation services - AI-powered content generation.

Phase 3: Architecture generation ✅
Phase 4: Layout generation ✅
Phase 5: Blockly generation ✅
"""

from app.services.generation.architecture_generator import (
    architecture_generator,
    ArchitectureGenerator,
    Ar3t24NpUrJMNunMMASmhAM953bFGeLXzN7,  # Changed from ArQuMHtpwbtsTXsRMArUQeWyGrRu7gwbZs2
    InvalidArchitectureError
)

from app.services.generation.architecture_validator import (
    architecture_validator,
    ArchitectureValidator,
    ValidationWarning
)

from app.services.generation.layout_generator import (
    layout_generator,
    LayoutGenerator,
    LayoutGenerationError,
    CollisionError
)

from app.services.generation.layout_validator import (
    layout_validator,
    LayoutValidator,
    LayoutWarning
)

from app.services.generation.blockly_generator import (
    blockly_generator,
    BlocklyGenerator,
    BlocklyGenerationError
)

from app.services.generation.blockly_validator import (
    blockly_validator,
    BlocklyValidator,
    BlocklyWarning
)

from app.services.generation.cache_manager import (
    semantic_cache,
    SemanticCacheManager
)

__all__ = [
    # Architecture generation
    'architecture_generator',
    'ArchitectureGenerator',
    'Ar3t24NpUrJMNunMMASmhAM953bFGeLXzN7',  # Changed from ArQuMHtpwbtsTXsRMArUQeWyGrRu7gwbZs2
    'InvalidArchitectureError',
    
    # Architecture validation
    'architecture_validator',
    'ArchitectureValidator',
    'ValidationWarning',
    
    # Layout generation
    'layout_generator',
    'LayoutGenerator',
    'LayoutGenerationError',
    'CollisionError',
    
    # Layout validation
    'layout_validator',
    'LayoutValidator',
    'LayoutWarning',
    
    # Blockly generation
    'blockly_generator',
    'BlocklyGenerator',
    'BlocklyGenerationError',
    
    # Blockly validation
    'blockly_validator',
    'BlocklyValidator',
    'BlocklyWarning',
    
    # Semantic cache
    'semantic_cache',
    'SemanticCacheManager',
]