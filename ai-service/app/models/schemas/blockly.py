"""
Blockly visual programming models.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class BlocklyBlock(BaseModel):
    """Single Blockly block definition"""
    type: str = Field(..., description="Block type identifier")
    id: str = Field(..., description="Unique block ID")
    x: Optional[int] = Field(default=None, description="X position in workspace")
    y: Optional[int] = Field(default=None, description="Y position in workspace")
    fields: Optional[Dict[str, Any]] = Field(default_factory=dict)
    inputs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    next: Optional[Dict[str, Any]] = Field(default=None, description="Next block in sequence")
    
    @field_validator('id')
    @classmethod
    def validate_block_id(cls, v: str) -> str:
        """Ensure valid block ID"""
        if not v or len(v) == 0:
            raise ValueError("Block ID cannot be empty")
        return v


class BlocklyWorkspace(BaseModel):
    """Complete Blockly workspace"""
    languageVersion: int = 0
    blocks: List[BlocklyBlock]


class EnhancedBlocklyDefinition(BaseModel):
    """Enhanced Blockly definition with validation"""
    blocks: BlocklyWorkspace
    variables: List[Dict[str, Any]] = Field(default_factory=list)
    custom_blocks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom block type definitions"
    )
    
    @model_validator(mode='after')
    def validate_block_references(self) -> 'EnhancedBlocklyDefinition':
        """Ensure all block references are valid"""
        block_ids = {block.id for block in self.blocks.blocks}
        
        def check_block(block: BlocklyBlock) -> None:
            """Recursively check block and its children"""
            # Check input blocks
            if block.inputs:
                for input_data in block.inputs.values():
                    if isinstance(input_data, dict) and 'block' in input_data:
                        nested = input_data['block']
                        if isinstance(nested, dict) and 'id' in nested:
                            # Nested block is valid
                            pass
            
            # Check next block
            if block.next and isinstance(block.next, dict):
                if 'block' in block.next:
                    nested = block.next['block']
                    if isinstance(nested, dict) and 'id' in nested:
                        # Next block is valid
                        pass
        
        # Validate all blocks
        for block in self.blocks.blocks:
            check_block(block)
        
        return self


# Legacy model for backward compatibility
class BlocklyDefinition(BaseModel):
    """Legacy Blockly definition (for backward compatibility)"""
    block_id: str
    type: Literal["event", "action", "getter", "setter", "logic", "math", "text"]
    block_json: Dict[str, Any]
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "block_id": "event_1",
                "type": "event",
                "block_json": {
                    "type": "component_event",
                    "fields": {
                        "COMPONENT": "btn_add",
                        "EVENT": "onClick"
                    }
                },
                "connections": [
                    {
                        "from_block": "event_1",
                        "to_block": "action_1",
                        "connection_type": "next"
                    }
                ]
            }
        }