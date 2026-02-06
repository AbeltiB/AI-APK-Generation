"""
Project State Resolver - Intent-Based Mutation Engine
======================================================

This module contains the CORE BUSINESS LOGIC for mutating project state.

All state mutations MUST go through this resolver. No exceptions.

Responsibilities:
1. Enforce intent-based mutation rules
2. Apply minimal diffs (don't overwrite entire sections)
3. Validate all changes
4. Create change log entries
5. Increment version
6. Reject illegal mutations

Design Principles:
- Fail fast on violations
- Explicit is better than implicit
- No silent changes
- Deterministic behavior
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from loguru import logger

from app.models.project_state import (
    ProjectState,
    IntentType,
    StateSection,
    ChangeAction,
    StateChange,
    Screen,
    Navigation,
    Component,
    Position,
    Size,
    BlocklyBlock,
)


# ============================================================================
# INTENT TO SECTION MAPPING (AUTHORIZATION MATRIX)
# ============================================================================

INTENT_ALLOWED_SECTIONS: Dict[IntentType, Set[StateSection]] = {
    IntentType.CREATE_NEW_APP: {
        StateSection.FOUNDATIONS,
        StateSection.ARCHITECTURE,
        StateSection.LAYOUT,
        StateSection.BLOCKLY,
        StateSection.METADATA,
    },
    IntentType.MODIFY_FOUNDATION: {
        StateSection.FOUNDATIONS,
    },
    IntentType.UPDATE_FEATURE: {
        StateSection.ARCHITECTURE,
        StateSection.LAYOUT,
        StateSection.BLOCKLY,
    },
    IntentType.REGENERATE_LAYOUT: {
        StateSection.LAYOUT,
    },
    IntentType.ASK_ABOUT_APP: set(),  # Read-only, no mutations allowed
}


# ============================================================================
# EXCEPTIONS
# ============================================================================

class StateResolverError(Exception):
    """Base exception for state resolution errors"""
    pass


class IllegalMutationError(StateResolverError):
    """Raised when attempting unauthorized mutation"""
    pass


class ValidationError(StateResolverError):
    """Raised when proposed changes fail validation"""
    pass


class SectionNotAllowedError(IllegalMutationError):
    """Raised when intent doesn't permit mutating a section"""
    pass


class FoundationMutationError(IllegalMutationError):
    """Raised when attempting to modify foundations without proper intent"""
    pass


# ============================================================================
# STATE RESOLVER
# ============================================================================

class ProjectStateResolver:
    """
    Enforces controlled mutation of project state based on classified intent.
    
    This is the GATEKEEPER. All mutations flow through here.
    """
    
    def __init__(self):
        self.mutation_rules = INTENT_ALLOWED_SECTIONS
    
    def resolve_and_update(
        self,
        current_state: ProjectState,
        intent: IntentType,
        proposed_changes: Dict[str, Any],
        actor: str,
        reason: str = "AI-generated update"
    ) -> ProjectState:
        """
        Main entry point: Apply proposed changes to state based on intent.
        
        Args:
            current_state: Current project state
            intent: Classified intent from user prompt
            proposed_changes: Changes proposed by LLM (structured dict)
            actor: Who is making this change (user_id)
            reason: Human-readable explanation
            
        Returns:
            Updated ProjectState with changes applied
            
        Raises:
            IllegalMutationError: If intent doesn't allow proposed mutation
            ValidationError: If proposed changes are invalid
        """
        logger.info(
            f"🔧 Resolving state mutation",
            extra={
                "intent": intent.value,
                "actor": actor,
                "current_version": current_state.metadata.version,
                "proposed_sections": list(proposed_changes.keys()),
            }
        )
        
        # Step 1: Validate intent allows mutation
        self._validate_intent_allows_mutation(intent)
        
        # Step 2: Validate sections being modified
        sections_to_modify = self._extract_sections(proposed_changes)
        self._validate_sections_allowed(intent, sections_to_modify)
        
        # Step 3: Check foundation mutation rules
        if StateSection.FOUNDATIONS in sections_to_modify:
            self._validate_foundation_mutation(intent)
        
        # Step 4: Apply changes with minimal diff
        updated_state = self._apply_changes(
            current_state=current_state,
            proposed_changes=proposed_changes,
            intent=intent,
            actor=actor,
            reason=reason,
        )
        
        # Step 5: Validate final state
        self._validate_final_state(updated_state)
        
        # Step 6: Increment version
        updated_state.increment_version(modified_by=actor)
        
        logger.info(
            f"✅ State mutation completed",
            extra={
                "intent": intent.value,
                "new_version": updated_state.metadata.version,
                "changes_logged": len(updated_state.change_log.changes),
            }
        )
        
        return updated_state
    
    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================
    
    def _validate_intent_allows_mutation(self, intent: IntentType) -> None:
        """Ensure intent is allowed to mutate state"""
        if intent == IntentType.ASK_ABOUT_APP:
            raise IllegalMutationError(
                f"Intent '{intent.value}' is read-only and cannot mutate state"
            )
    
    def _validate_sections_allowed(
        self, intent: IntentType, sections: Set[StateSection]
    ) -> None:
        """Ensure intent has permission to modify specified sections"""
        allowed_sections = self.mutation_rules.get(intent, set())
        
        for section in sections:
            if section not in allowed_sections:
                raise SectionNotAllowedError(
                    f"Intent '{intent.value}' is not allowed to modify section '{section.value}'. "
                    f"Allowed sections: {[s.value for s in allowed_sections]}"
                )
    
    def _validate_foundation_mutation(self, intent: IntentType) -> None:
        """Special validation for foundation mutations"""
        if intent != IntentType.MODIFY_FOUNDATION:
            raise FoundationMutationError(
                f"Foundations are IMMUTABLE. "
                f"Current intent '{intent.value}' cannot modify foundations. "
                f"Use intent='{IntentType.MODIFY_FOUNDATION.value}' to modify foundations."
            )
    
    def _validate_final_state(self, state: ProjectState) -> None:
        """Validate that final state is internally consistent"""
        try:
            # Pydantic validation
            ProjectState.model_validate(state.model_dump())
            
            # Custom business logic validation
            self._validate_screen_references(state)
            self._validate_component_references(state)
            
        except Exception as e:
            raise ValidationError(f"Final state validation failed: {e}")
    
    def _validate_screen_references(self, state: ProjectState) -> None:
        """Ensure all screen references are valid"""
        screen_ids = set(state.architecture.screens.keys())
        
        # Check navigation references
        for nav in state.architecture.navigation:
            if nav.from_screen not in screen_ids:
                raise ValidationError(
                    f"Navigation references non-existent screen: {nav.from_screen}"
                )
            if nav.to_screen not in screen_ids:
                raise ValidationError(
                    f"Navigation references non-existent screen: {nav.to_screen}"
                )
        
        # Check component screen references
        for component in state.layout.components.values():
            if component.screen_id not in screen_ids:
                raise ValidationError(
                    f"Component {component.component_id} references non-existent screen: {component.screen_id}"
                )
    
    def _validate_component_references(self, state: ProjectState) -> None:
        """Ensure component IDs don't collide"""
        component_ids = list(state.layout.components.keys())
        if len(component_ids) != len(set(component_ids)):
            raise ValidationError("Duplicate component IDs detected")
    
    # ========================================================================
    # CHANGE APPLICATION METHODS
    # ========================================================================
    
    def _extract_sections(self, proposed_changes: Dict[str, Any]) -> Set[StateSection]:
        """Extract which sections are being modified"""
        sections = set()
        
        for key in proposed_changes.keys():
            try:
                sections.add(StateSection(key))
            except ValueError:
                # Ignore unknown keys (could be metadata)
                pass
        
        return sections
    
    def _apply_changes(
        self,
        current_state: ProjectState,
        proposed_changes: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> ProjectState:
        """
        Apply proposed changes to state using minimal diff strategy.
        
        This is where the actual mutation happens.
        """
        # Create a deep copy to avoid mutating original
        updated_state = current_state.model_copy(deep=True)
        
        # Apply changes section by section
        for section_key, section_changes in proposed_changes.items():
            if section_key == "foundations":
                self._apply_foundation_changes(
                    updated_state, section_changes, intent, actor, reason
                )
            elif section_key == "architecture":
                self._apply_architecture_changes(
                    updated_state, section_changes, intent, actor, reason
                )
            elif section_key == "layout":
                self._apply_layout_changes(
                    updated_state, section_changes, intent, actor, reason
                )
            elif section_key == "blockly":
                self._apply_blockly_changes(
                    updated_state, section_changes, intent, actor, reason
                )
        
        return updated_state
    
    def _apply_foundation_changes(
        self,
        state: ProjectState,
        changes: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Apply changes to foundations (minimal diff)"""
        for field_name, new_value in changes.items():
            if hasattr(state.foundations, field_name):
                old_value = getattr(state.foundations, field_name)
                
                if old_value != new_value:
                    # Create change log entry
                    change = StateChange(
                        actor=actor,
                        intent=intent,
                        action=ChangeAction.UPDATE,
                        section=StateSection.FOUNDATIONS,
                        path=f"foundations.{field_name}",
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    )
                    state.change_log.append(change)
                    
                    # Apply change
                    setattr(state.foundations, field_name, new_value)
                    
                    logger.debug(f"Updated foundations.{field_name}: {old_value} → {new_value}")
    
    def _apply_architecture_changes(
        self,
        state: ProjectState,
        changes: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Apply changes to architecture (screens and navigation)"""
        
        # Handle screens
        if "screens" in changes:
            screens_changes = changes["screens"]
            
            for screen_id, screen_data in screens_changes.items():
                if screen_id in state.architecture.screens:
                    # Update existing screen (minimal diff)
                    self._update_screen(
                        state, screen_id, screen_data, intent, actor, reason
                    )
                else:
                    # Create new screen
                    self._create_screen(
                        state, screen_id, screen_data, intent, actor, reason
                    )
        
        # Handle navigation
        if "navigation" in changes:
            # For navigation, we replace the entire list
            # (diff-ing navigation paths is complex)
            old_nav = state.architecture.navigation
            new_nav = [Navigation(**nav) for nav in changes["navigation"]]
            
            change = StateChange(
                actor=actor,
                intent=intent,
                action=ChangeAction.REPLACE,
                section=StateSection.ARCHITECTURE,
                path="architecture.navigation",
                old_value=len(old_nav),
                new_value=len(new_nav),
                reason=reason,
            )
            state.change_log.append(change)
            
            state.architecture.navigation = new_nav
            
            logger.debug(f"Replaced navigation: {len(old_nav)} → {len(new_nav)} paths")
        
        # Handle other architecture fields
        for field_name in ["state_management", "data_persistence"]:
            if field_name in changes:
                old_value = getattr(state.architecture, field_name)
                new_value = changes[field_name]
                
                if old_value != new_value:
                    change = StateChange(
                        actor=actor,
                        intent=intent,
                        action=ChangeAction.UPDATE,
                        section=StateSection.ARCHITECTURE,
                        path=f"architecture.{field_name}",
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    )
                    state.change_log.append(change)
                    setattr(state.architecture, field_name, new_value)
    
    def _update_screen(
        self,
        state: ProjectState,
        screen_id: str,
        screen_data: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Update existing screen with minimal diff"""
        screen = state.architecture.screens[screen_id]
        
        for field_name, new_value in screen_data.items():
            if hasattr(screen, field_name):
                old_value = getattr(screen, field_name)
                
                if old_value != new_value:
                    change = StateChange(
                        actor=actor,
                        intent=intent,
                        action=ChangeAction.UPDATE,
                        section=StateSection.ARCHITECTURE,
                        path=f"architecture.screens.{screen_id}.{field_name}",
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    )
                    state.change_log.append(change)
                    setattr(screen, field_name, new_value)
    
    def _create_screen(
        self,
        state: ProjectState,
        screen_id: str,
        screen_data: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Create new screen"""
        screen = Screen(screen_id=screen_id, **screen_data)
        state.architecture.screens[screen_id] = screen
        
        change = StateChange(
            actor=actor,
            intent=intent,
            action=ChangeAction.CREATE,
            section=StateSection.ARCHITECTURE,
            path=f"architecture.screens.{screen_id}",
            old_value=None,
            new_value=screen.model_dump(),
            reason=reason,
        )
        state.change_log.append(change)
        
        logger.debug(f"Created new screen: {screen_id}")
    
    def _apply_layout_changes(
        self,
        state: ProjectState,
        changes: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Apply changes to layout (components)"""
        
        # Handle components
        if "components" in changes:
            components_changes = changes["components"]
            
            for component_id, component_data in components_changes.items():
                if component_id in state.layout.components:
                    # Update existing component
                    self._update_component(
                        state, component_id, component_data, intent, actor, reason
                    )
                else:
                    # Create new component
                    self._create_component(
                        state, component_id, component_data, intent, actor, reason
                    )
        
        # Handle layout system settings
        for field_name in ["layout_system", "responsive"]:
            if field_name in changes:
                old_value = getattr(state.layout, field_name)
                new_value = changes[field_name]
                
                if old_value != new_value:
                    change = StateChange(
                        actor=actor,
                        intent=intent,
                        action=ChangeAction.UPDATE,
                        section=StateSection.LAYOUT,
                        path=f"layout.{field_name}",
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    )
                    state.change_log.append(change)
                    setattr(state.layout, field_name, new_value)
    
    def _update_component(
        self,
        state: ProjectState,
        component_id: str,
        component_data: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Update existing component with minimal diff"""
        component = state.layout.components[component_id]
        
        for field_name, new_value in component_data.items():
            if field_name in ["position", "size"]:
                # Handle nested objects
                if field_name == "position":
                    old_pos = component.position
                    new_pos = Position(**new_value)
                    if old_pos.x != new_pos.x or old_pos.y != new_pos.y:
                        change = StateChange(
                            actor=actor,
                            intent=intent,
                            action=ChangeAction.UPDATE,
                            section=StateSection.LAYOUT,
                            path=f"layout.components.{component_id}.position",
                            old_value=old_pos.model_dump(),
                            new_value=new_pos.model_dump(),
                            reason=reason,
                        )
                        state.change_log.append(change)
                        component.position = new_pos
                elif field_name == "size":
                    old_size = component.size
                    new_size = Size(**new_value)
                    if old_size.width != new_size.width or old_size.height != new_size.height:
                        change = StateChange(
                            actor=actor,
                            intent=intent,
                            action=ChangeAction.UPDATE,
                            section=StateSection.LAYOUT,
                            path=f"layout.components.{component_id}.size",
                            old_value=old_size.model_dump(),
                            new_value=new_size.model_dump(),
                            reason=reason,
                        )
                        state.change_log.append(change)
                        component.size = new_size
            else:
                # Handle simple fields
                if hasattr(component, field_name):
                    old_value = getattr(component, field_name)
                    
                    if old_value != new_value:
                        change = StateChange(
                            actor=actor,
                            intent=intent,
                            action=ChangeAction.UPDATE,
                            section=StateSection.LAYOUT,
                            path=f"layout.components.{component_id}.{field_name}",
                            old_value=old_value,
                            new_value=new_value,
                            reason=reason,
                        )
                        state.change_log.append(change)
                        setattr(component, field_name, new_value)
    
    def _create_component(
        self,
        state: ProjectState,
        component_id: str,
        component_data: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Create new component"""
        # Handle nested Position and Size
        if "position" in component_data:
            component_data["position"] = Position(**component_data["position"])
        if "size" in component_data:
            component_data["size"] = Size(**component_data["size"])
        
        component = Component(component_id=component_id, **component_data)
        state.layout.components[component_id] = component
        
        change = StateChange(
            actor=actor,
            intent=intent,
            action=ChangeAction.CREATE,
            section=StateSection.LAYOUT,
            path=f"layout.components.{component_id}",
            old_value=None,
            new_value=component.model_dump(),
            reason=reason,
        )
        state.change_log.append(change)
        
        logger.debug(f"Created new component: {component_id}")
    
    def _apply_blockly_changes(
        self,
        state: ProjectState,
        changes: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Apply changes to Blockly blocks"""
        
        # Handle blocks
        if "blocks" in changes:
            blocks_changes = changes["blocks"]
            
            for block_id, block_data in blocks_changes.items():
                if block_id in state.blockly.blocks:
                    # Update existing block
                    self._update_block(
                        state, block_id, block_data, intent, actor, reason
                    )
                else:
                    # Create new block
                    self._create_block(
                        state, block_id, block_data, intent, actor, reason
                    )
    
    def _update_block(
        self,
        state: ProjectState,
        block_id: str,
        block_data: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Update existing Blockly block"""
        block = state.blockly.blocks[block_id]
        
        for field_name, new_value in block_data.items():
            if hasattr(block, field_name):
                old_value = getattr(block, field_name)
                
                if old_value != new_value:
                    change = StateChange(
                        actor=actor,
                        intent=intent,
                        action=ChangeAction.UPDATE,
                        section=StateSection.BLOCKLY,
                        path=f"blockly.blocks.{block_id}.{field_name}",
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    )
                    state.change_log.append(change)
                    setattr(block, field_name, new_value)
    
    def _create_block(
        self,
        state: ProjectState,
        block_id: str,
        block_data: Dict[str, Any],
        intent: IntentType,
        actor: str,
        reason: str,
    ) -> None:
        """Create new Blockly block"""
        if "position" in block_data:
            block_data["position"] = Position(**block_data["position"])
        
        block = BlocklyBlock(block_id=block_id, **block_data)
        state.blockly.blocks[block_id] = block
        
        change = StateChange(
            actor=actor,
            intent=intent,
            action=ChangeAction.CREATE,
            section=StateSection.BLOCKLY,
            path=f"blockly.blocks.{block_id}",
            old_value=None,
            new_value=block.model_dump(),
            reason=reason,
        )
        state.change_log.append(change)
        
        logger.debug(f"Created new Blockly block: {block_id}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def resolve_and_update_state(
    state: ProjectState,
    intent: IntentType,
    proposed_changes: Dict[str, Any],
    actor: str = "system",
    reason: str = "AI-generated update"
) -> ProjectState:
    """
    Convenience function for resolving state mutations.
    
    This is the main public API.
    """
    resolver = ProjectStateResolver()
    return resolver.resolve_and_update(
        current_state=state,
        intent=intent,
        proposed_changes=proposed_changes,
        actor=actor,
        reason=reason,
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from app.models.project_state import ProjectState
    
    # Create initial state
    state = ProjectState.create_new(
        app_name="Counter App",
        app_description="Simple counter",
        created_by="user_123",
    )
    
    print(f"Initial state: {state}")
    print(f"Version: {state.metadata.version}")
    
    # Example 1: Add a screen
    print("\n--- Example 1: Add Screen ---")
    changes = {
        "architecture": {
            "screens": {
                "main_screen": {
                    "screen_name": "Main",
                    "screen_type": "main",
                    "is_entry_point": True,
                    "description": "Main counter screen",
                }
            }
        }
    }
    
    state = resolve_and_update_state(
        state=state,
        intent=IntentType.UPDATE_FEATURE,
        proposed_changes=changes,
        actor="user_123",
        reason="Initial screen setup",
    )
    
    print(f"After adding screen: version={state.metadata.version}")
    print(f"Screens: {list(state.architecture.screens.keys())}")
    
    # Example 2: Try to modify foundation (should fail without proper intent)
    print("\n--- Example 2: Try Illegal Foundation Mutation ---")
    try:
        illegal_changes = {
            "foundations": {
                "app_name": "New Name"
            }
        }
        
        state = resolve_and_update_state(
            state=state,
            intent=IntentType.UPDATE_FEATURE,  # Wrong intent!
            proposed_changes=illegal_changes,
            actor="user_123",
        )
    except FoundationMutationError as e:
        print(f"Correctly blocked: {e}")
    
    # Example 3: Modify foundation with correct intent
    print("\n--- Example 3: Legal Foundation Mutation ---")
    foundation_changes = {
        "foundations": {
            "primary_color": "#FF0000"
        }
    }
    
    state = resolve_and_update_state(
        state=state,
        intent=IntentType.MODIFY_FOUNDATION,
        proposed_changes=foundation_changes,
        actor="user_123",
        reason="User changed theme color",
    )
    
    print(f"After foundation change: version={state.metadata.version}")
    print(f"Primary color: {state.foundations.primary_color}")
    print(f"Change log entries: {len(state.change_log.changes)}")