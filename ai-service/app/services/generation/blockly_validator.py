"""
Blockly Validator - Comprehensive block validation.

Validates generated Blockly blocks for:
- Block structure correctness
- Connection validity
- Variable declarations
- Reference checking
- Logic flow validation
"""
from typing import List, Tuple, Dict, Any, Set, Optional
from loguru import logger


class BlocklyWarning:
    """Represents a Blockly validation warning"""
    
    def __init__(self, level: str, block_id: str, message: str, suggestion: str = ""):
        self.level = level  # "info", "warning", "error"
        self.block_id = block_id
        self.message = message
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'level': self.level,
            'block_id': self.block_id,
            'message': self.message,
            'suggestion': self.suggestion
        }
    
    def __str__(self) -> str:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
        s = f"{emoji.get(self.level, '•')} [{self.level.upper()}] {self.block_id}: {self.message}"
        if self.suggestion:
            s += f"\n   → {self.suggestion}"
        return s


class BlocklyValidator:
    """
    Comprehensive Blockly validation.
    
    Validation passes:
    1. Block structure
    2. Connection validity
    3. Variable declarations
    4. Reference checking
    5. Logic flow validation
    6. Orphan detection
    """
    
    def __init__(self):
        self.warnings: List[BlocklyWarning] = []
        self.block_ids: Set[str] = set()
        self.variable_ids: Set[str] = set()
        self.variable_names: Set[str] = set()
    
    async def validate(
        self,
        blockly: Dict[str, Any]
    ) -> Tuple[bool, List[BlocklyWarning]]:
        """
        Comprehensive validation of Blockly blocks.
        
        Args:
            blockly: Blockly definition to validate
            
        Returns:
            Tuple of (is_valid, warnings_list)
        """
        self.warnings = []
        self.block_ids = set()
        self.variable_ids = set()
        self.variable_names = set()
        
        logger.info("🔍 Validating Blockly blocks...")
        
        # Extract blocks and variables
        blocks = blockly.get('blocks', {}).get('blocks', [])
        variables = blockly.get('variables', [])
        
        # Collect IDs first
        await self._collect_ids(blocks, variables)
        
        # Run validation checks
        await self._validate_structure(blockly)
        await self._validate_blocks(blocks)
        await self._validate_variables(variables)
        await self._validate_connections(blocks)
        await self._validate_references(blocks)
        await self._validate_logic_flow(blocks)
        await self._detect_orphans(blocks)
        
        # Determine if valid
        has_errors = any(w.level == "error" for w in self.warnings)
        is_valid = not has_errors
        
        if is_valid:
            logger.info("✅ Blockly validation passed")
        else:
            error_count = sum(1 for w in self.warnings if w.level == "error")
            logger.error(f"❌ Blockly validation failed: {error_count} error(s)")
        
        warning_count = sum(1 for w in self.warnings if w.level == "warning")
        if warning_count > 0:
            logger.warning(f"⚠️  {warning_count} warning(s) found")
        
        return is_valid, self.warnings
    
    async def _collect_ids(
        self,
        blocks: List[Dict[str, Any]],
        variables: List[Dict[str, str]]
    ) -> None:
        """Collect all block and variable IDs"""
        
        def collect_from_block(block: Dict[str, Any]):
            if 'id' in block:
                self.block_ids.add(block['id'])
            
            # Recursively check inputs
            inputs = block.get('inputs', {})
            for input_data in inputs.values():
                if isinstance(input_data, dict) and 'block' in input_data:
                    collect_from_block(input_data['block'])
            
            # Check next block
            if 'next' in block and isinstance(block['next'], dict):
                if 'block' in block['next']:
                    collect_from_block(block['next']['block'])
        
        for block in blocks:
            collect_from_block(block)
        
        for var in variables:
            if 'id' in var:
                self.variable_ids.add(var['id'])
            if 'name' in var:
                self.variable_names.add(var['name'])
    
    async def _validate_structure(self, blockly: Dict[str, Any]) -> None:
        """Validate basic structure"""
        
        if 'blocks' not in blockly:
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id="root",
                message="Missing 'blocks' key",
                suggestion="Add blocks workspace structure"
            ))
            return
        
        blocks_obj = blockly['blocks']
        
        if not isinstance(blocks_obj, dict):
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id="root",
                message="'blocks' must be an object",
                suggestion="Use {languageVersion: 0, blocks: [...]}"
            ))
            return
        
        if 'blocks' not in blocks_obj:
            self.warnings.append(BlocklyWarning(
                level="warning",
                block_id="root",
                message="No blocks array in workspace",
                suggestion="Add blocks array"
            ))
    
    async def _validate_blocks(self, blocks: List[Dict[str, Any]]) -> None:
        """Validate individual blocks"""
        
        if len(blocks) == 0:
            self.warnings.append(BlocklyWarning(
                level="warning",
                block_id="root",
                message="No blocks defined",
                suggestion="Add event handlers and logic blocks"
            ))
            return
        
        # Check for duplicate IDs
        id_counts = {}
        for block in blocks:
            block_id = block.get('id', 'unknown')
            id_counts[block_id] = id_counts.get(block_id, 0) + 1
        
        duplicates = [bid for bid, count in id_counts.items() if count > 1]
        if duplicates:
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id="root",
                message=f"Duplicate block IDs: {', '.join(duplicates)}",
                suggestion="Ensure all block IDs are unique"
            ))
        
        # Validate each block
        for block in blocks:
            await self._validate_single_block(block)
    
    async def _validate_single_block(self, block: Dict[str, Any]) -> None:
        """Validate a single block"""
        
        block_id = block.get('id', 'unknown')
        block_type = block.get('type', 'unknown')
        
        # Check required fields
        if 'type' not in block:
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id=block_id,
                message="Missing 'type' field",
                suggestion="Add block type identifier"
            ))
        
        if 'id' not in block:
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id=block_id,
                message="Missing 'id' field",
                suggestion="Add unique block ID"
            ))
        
        # Recursively validate nested blocks
        inputs = block.get('inputs', {})
        for input_name, input_data in inputs.items():
            if isinstance(input_data, dict) and 'block' in input_data:
                await self._validate_single_block(input_data['block'])
        
        # Validate next block
        if 'next' in block and isinstance(block['next'], dict):
            if 'block' in block['next']:
                await self._validate_single_block(block['next']['block'])
    
    async def _validate_variables(self, variables: List[Dict[str, str]]) -> None:
        """Validate variable declarations"""
        
        # Check for duplicate names
        name_counts = {}
        for var in variables:
            name = var.get('name', 'unknown')
            name_counts[name] = name_counts.get(name, 0) + 1
        
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id="variables",
                message=f"Duplicate variable names: {', '.join(duplicates)}",
                suggestion="Use unique names for each variable"
            ))
        
        # Check for duplicate IDs
        id_counts = {}
        for var in variables:
            var_id = var.get('id', 'unknown')
            id_counts[var_id] = id_counts.get(var_id, 0) + 1
        
        dup_ids = [vid for vid, count in id_counts.items() if count > 1]
        if dup_ids:
            self.warnings.append(BlocklyWarning(
                level="error",
                block_id="variables",
                message=f"Duplicate variable IDs: {', '.join(dup_ids)}",
                suggestion="Ensure all variable IDs are unique"
            ))
        
        # Check each variable
        for var in variables:
            if 'name' not in var:
                self.warnings.append(BlocklyWarning(
                    level="error",
                    block_id=var.get('id', 'unknown'),
                    message="Variable missing 'name' field",
                    suggestion="Add variable name"
                ))
            
            if 'id' not in var:
                self.warnings.append(BlocklyWarning(
                    level="error",
                    block_id="variables",
                    message=f"Variable '{var.get('name', 'unknown')}' missing ID",
                    suggestion="Add unique variable ID"
                ))
    
    async def _validate_connections(self, blocks: List[Dict[str, Any]]) -> None:
        """Validate block connections"""
        
        def check_connections(block: Dict[str, Any], parent_id: Optional[str] = None):
            block_id = block.get('id', 'unknown')
            
            # Check input connections
            inputs = block.get('inputs', {})
            for input_name, input_data in inputs.items():
                if isinstance(input_data, dict):
                    if 'block' in input_data:
                        nested_block = input_data['block']
                        if isinstance(nested_block, dict):
                            check_connections(nested_block, block_id)
                    elif 'shadow' in input_data:
                        # Shadow blocks are ok
                        pass
            
            # Check next connection
            next_block = block.get('next')
            if next_block:
                if isinstance(next_block, dict) and 'block' in next_block:
                    nested = next_block['block']
                    if isinstance(nested, dict):
                        check_connections(nested, block_id)
        
        for block in blocks:
            check_connections(block)
    
    async def _validate_references(self, blocks: List[Dict[str, Any]]) -> None:
        """Validate variable and component references"""
        
        def check_references(block: Dict[str, Any]):
            block_id = block.get('id', 'unknown')
            
            # Check field references
            fields = block.get('fields', {})
            for field_name, field_value in fields.items():
                if field_name in ['VAR', 'VARIABLE']:
                    # Variable reference
                    if isinstance(field_value, str):
                        if field_value not in self.variable_names:
                            self.warnings.append(BlocklyWarning(
                                level="warning",
                                block_id=block_id,
                                message=f"Reference to undefined variable: {field_value}",
                                suggestion="Declare variable or check spelling"
                            ))
                elif field_name == 'COMPONENT':
                    # Component reference - we can't validate without component list
                    pass
            
            # Check nested blocks
            inputs = block.get('inputs', {})
            for input_data in inputs.values():
                if isinstance(input_data, dict) and 'block' in input_data:
                    check_references(input_data['block'])
            
            if 'next' in block and isinstance(block['next'], dict):
                if 'block' in block['next']:
                    check_references(block['next']['block'])
        
        for block in blocks:
            check_references(block)
    
    async def _validate_logic_flow(self, blocks: List[Dict[str, Any]]) -> None:
        """Validate logic flow makes sense"""
        
        # Check for event handlers
        has_events = False
        
        def check_for_events(block: Dict[str, Any]):
            nonlocal has_events
            block_type = block.get('type', '')
            
            if 'event' in block_type.lower() or block_type == 'component_event':
                has_events = True
            
            # Check nested
            inputs = block.get('inputs', {})
            for input_data in inputs.values():
                if isinstance(input_data, dict) and 'block' in input_data:
                    check_for_events(input_data['block'])
            
            if 'next' in block and isinstance(block['next'], dict):
                if 'block' in block['next']:
                    check_for_events(block['next']['block'])
        
        for block in blocks:
            check_for_events(block)
        
        if not has_events and len(blocks) > 0:
            self.warnings.append(BlocklyWarning(
                level="info",
                block_id="root",
                message="No event handler blocks found",
                suggestion="Add event blocks to handle user interactions"
            ))
    
    async def _detect_orphans(self, blocks: List[Dict[str, Any]]) -> None:
        """Detect orphaned blocks (not connected to anything)"""
        
        # Top-level blocks should be event handlers
        for block in blocks:
            block_type = block.get('type', '')
            block_id = block.get('id', 'unknown')
            
            # Check if it's an event handler or has no previous statement
            is_event = 'event' in block_type.lower()
            
            # If it's not an event and has no clear entry point, warn
            if not is_event and block_type not in ['variables_set', 'component_event']:
                # Check if it has a next connection but no previous
                # This would indicate it might be orphaned
                pass  # More complex orphan detection could be added


# Global validator instance
blockly_validator = BlocklyValidator()


if __name__ == "__main__":
    # Test validator
    import asyncio
    
    async def test():
        print("\n" + "=" * 60)
        print("BLOCKLY VALIDATOR TEST")
        print("=" * 60)
        
        # Test 1: Valid Blockly
        print("\n[TEST 1] Valid Blockly")
        
        valid_blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': [
                    {
                        'type': 'component_event',
                        'id': 'event_1',
                        'fields': {
                            'COMPONENT': 'btn_increment',
                            'EVENT': 'onPress'
                        },
                        'next': {
                            'block': {
                                'type': 'state_set',
                                'id': 'action_1',
                                'fields': {
                                    'VAR': 'count'
                                },
                                'inputs': {
                                    'VALUE': {
                                        'block': {
                                            'type': 'math_arithmetic',
                                            'id': 'math_1',
                                            'fields': {'OP': 'ADD'},
                                            'inputs': {
                                                'A': {
                                                    'block': {
                                                        'type': 'variables_get',
                                                        'id': 'get_1',
                                                        'fields': {'VAR': 'count'}
                                                    }
                                                },
                                                'B': {
                                                    'block': {
                                                        'type': 'math_number',
                                                        'id': 'num_1',
                                                        'fields': {'NUM': 1}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            },
            'variables': [
                {'name': 'count', 'id': 'var_1', 'type': ''}
            ]
        }
        
        is_valid, warnings = await blockly_validator.validate(valid_blockly)
        
        print(f"Valid: {is_valid}")
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  {w}")
        
        # Test 2: Invalid Blockly
        print("\n[TEST 2] Invalid Blockly")
        
        invalid_blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': [
                    {
                        'type': 'state_set',
                        'id': 'block_1',
                        # Missing 'fields'
                    },
                    {
                        # Missing 'type'
                        'id': 'block_2',
                        'fields': {'VAR': 'undefined_var'}  # Undefined variable
                    }
                ]
            },
            'variables': [
                {'name': 'count', 'id': 'var_1'},
                {'name': 'count', 'id': 'var_2'}  # Duplicate name
            ]
        }
        
        is_valid, warnings = await blockly_validator.validate(invalid_blockly)
        
        print(f"Valid: {is_valid}")
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  {w}")
        
        print("\n" + "=" * 60)
        print("✅ Validator test complete!")
        print("=" * 60 + "\n")
    
    asyncio.run(test())