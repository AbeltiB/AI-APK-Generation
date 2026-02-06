"""
Project State Data Model - Single Source of Truth
==================================================

This module defines the authoritative schema for AI-generated app design state.
All state must conform to this schema. No exceptions.

Design Principles:
- Explicit fields only (no untyped dicts)
- Strongly typed with Pydantic
- Serializable to/from JSON
- Schema versioned for backward compatibility
- Immutable foundations by default
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import uuid


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class IntentType(str, Enum):
    """Allowed intent types that can mutate project state"""
    CREATE_NEW_APP = "create_new_app"
    MODIFY_FOUNDATION = "modify_foundation"
    UPDATE_FEATURE = "update_feature"
    REGENERATE_LAYOUT = "regenerate_layout"
    ASK_ABOUT_APP = "ask_about_app"


class StateSection(str, Enum):
    """Sections of the project state"""
    FOUNDATIONS = "foundations"
    ARCHITECTURE = "architecture"
    LAYOUT = "layout"
    BLOCKLY = "blockly"
    METADATA = "metadata"


class ChangeAction(str, Enum):
    """Types of changes that can be applied"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    REPLACE = "replace"


# Current schema version - increment on breaking changes
CURRENT_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# CHANGE LOG MODELS
# ============================================================================

class StateChange(BaseModel):
    """
    Immutable record of a single state mutation.
    Every change to the project state MUST create one of these.
    """
    model_config = ConfigDict(frozen=True)
    
    change_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor: str = Field(description="Who/what made this change (user_id or 'system')")
    intent: IntentType = Field(description="Intent that triggered this change")
    action: ChangeAction = Field(description="Type of change applied")
    section: StateSection = Field(description="Which section was modified")
    path: str = Field(description="JSON path to changed field (e.g., 'layout.components.button_1.position.x')")
    old_value: Optional[Any] = Field(default=None, description="Previous value (null for CREATE)")
    new_value: Optional[Any] = Field(default=None, description="New value (null for DELETE)")
    reason: str = Field(description="Human-readable explanation of why this change was made")
    
    def __str__(self) -> str:
        return f"{self.action.value} {self.path}: {self.old_value} → {self.new_value} ({self.reason})"


class ChangeLog(BaseModel):
    """Collection of all changes applied to a project state"""
    changes: List[StateChange] = Field(default_factory=list)
    
    def append(self, change: StateChange) -> None:
        """Add a new change to the log"""
        self.changes.append(change)
    
    def get_changes_by_section(self, section: StateSection) -> List[StateChange]:
        """Retrieve all changes for a specific section"""
        return [c for c in self.changes if c.section == section]
    
    def get_changes_by_intent(self, intent: IntentType) -> List[StateChange]:
        """Retrieve all changes triggered by a specific intent"""
        return [c for c in self.changes if c.intent == intent]


# ============================================================================
# FOUNDATION MODELS (IMMUTABLE BY DEFAULT)
# ============================================================================

class AppFoundations(BaseModel):
    """
    Core app identity and constraints.
    
    ⚠️ IMMUTABLE BY DEFAULT ⚠️
    Can only be changed with intent=MODIFY_FOUNDATION
    """
    app_name: str = Field(description="Human-readable app name")
    app_type: str = Field(description="Type of app (e.g., 'productivity', 'game', 'social')")
    target_platform: str = Field(default="android", description="Target platform")
    min_sdk_version: int = Field(default=21, description="Minimum Android SDK version")
    theme: str = Field(default="material", description="UI theme identifier")
    primary_color: str = Field(default="#6200EE", description="Primary brand color (hex)")
    accent_color: str = Field(default="#03DAC6", description="Accent color (hex)")
    app_description: str = Field(description="Brief description of app purpose")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('primary_color', 'accent_color')
    @classmethod
    def validate_color(cls, v: str) -> str:
        """Ensure colors are valid hex codes"""
        if not v.startswith('#') or len(v) != 7:
            raise ValueError(f"Invalid color format: {v}. Must be #RRGGBB")
        try:
            int(v[1:], 16)
        except ValueError:
            raise ValueError(f"Invalid hex color: {v}")
        return v


# ============================================================================
# ARCHITECTURE MODELS
# ============================================================================

class Screen(BaseModel):
    """Represents a single screen/activity in the app"""
    screen_id: str = Field(description="Unique screen identifier")
    screen_name: str = Field(description="Human-readable screen name")
    screen_type: str = Field(description="Type of screen (e.g., 'main', 'detail', 'settings')")
    is_entry_point: bool = Field(default=False, description="Is this the app's entry screen?")
    description: str = Field(description="Purpose of this screen")
    
    @field_validator('screen_id')
    @classmethod
    def validate_screen_id(cls, v: str) -> str:
        """Ensure screen IDs follow naming convention"""
        if not v.replace('_', '').isalnum():
            raise ValueError(f"screen_id must be alphanumeric with underscores: {v}")
        return v


class Navigation(BaseModel):
    """Defines navigation between screens"""
    from_screen: str = Field(description="Source screen ID")
    to_screen: str = Field(description="Destination screen ID")
    trigger: str = Field(description="What triggers this navigation (e.g., 'button_click', 'swipe')")
    transition: str = Field(default="slide", description="Transition animation")


class AppArchitecture(BaseModel):
    """High-level app structure - screens, navigation, data flow"""
    screens: Dict[str, Screen] = Field(default_factory=dict, description="All screens keyed by screen_id")
    navigation: List[Navigation] = Field(default_factory=list, description="Navigation graph")
    state_management: str = Field(default="local", description="State management approach")
    data_persistence: str = Field(default="shared_prefs", description="Data storage mechanism")
    
    @field_validator('screens')
    @classmethod
    def validate_screens(cls, v: Dict[str, Screen]) -> Dict[str, Screen]:
        """Ensure at least one entry point exists"""
        if v:  # Only validate if screens exist
            entry_points = [s for s in v.values() if s.is_entry_point]
            if not entry_points:
                # Auto-fix: make first screen the entry point
                first_screen = next(iter(v.values()))
                first_screen.is_entry_point = True
        return v


# ============================================================================
# LAYOUT MODELS
# ============================================================================

class Position(BaseModel):
    """2D position in layout coordinate system"""
    x: float = Field(ge=0, description="X coordinate (0-1 normalized or pixels)")
    y: float = Field(ge=0, description="Y coordinate (0-1 normalized or pixels)")


class Size(BaseModel):
    """Component dimensions"""
    width: float = Field(gt=0, description="Width in layout units")
    height: float = Field(gt=0, description="Height in layout units")


class Component(BaseModel):
    """Visual component in the layout"""
    component_id: str = Field(description="Unique component identifier")
    component_type: str = Field(description="Type (e.g., 'button', 'text', 'image', 'container')")
    screen_id: str = Field(description="Which screen this component belongs to")
    position: Position = Field(description="Component position")
    size: Size = Field(description="Component size")
    z_index: int = Field(default=0, description="Stacking order")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Component-specific properties")
    
    @field_validator('component_id')
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        """Ensure component IDs follow naming convention"""
        if not v.replace('_', '').isalnum():
            raise ValueError(f"component_id must be alphanumeric with underscores: {v}")
        return v


class AppLayout(BaseModel):
    """Visual layout of all components across all screens"""
    components: Dict[str, Component] = Field(default_factory=dict, description="All components keyed by component_id")
    layout_system: str = Field(default="constraint", description="Layout engine (e.g., 'constraint', 'linear', 'frame')")
    responsive: bool = Field(default=True, description="Whether layout adapts to screen size")
    
    def get_components_for_screen(self, screen_id: str) -> List[Component]:
        """Retrieve all components belonging to a specific screen"""
        return [c for c in self.components.values() if c.screen_id == screen_id]


# ============================================================================
# BLOCKLY MODELS
# ============================================================================

class BlocklyBlock(BaseModel):
    """Represents a single Blockly visual programming block"""
    block_id: str = Field(description="Unique block identifier")
    block_type: str = Field(description="Type of block (e.g., 'event', 'action', 'logic', 'variable')")
    block_category: str = Field(description="Category (e.g., 'events', 'ui', 'data', 'control')")
    position: Position = Field(description="Position in visual editor")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Block-specific properties")
    children: List[str] = Field(default_factory=list, description="IDs of child blocks")
    parent: Optional[str] = Field(default=None, description="ID of parent block")


class AppBlockly(BaseModel):
    """Visual programming logic using Blockly"""
    blocks: Dict[str, BlocklyBlock] = Field(default_factory=dict, description="All blocks keyed by block_id")
    blockly_version: str = Field(default="1.0.0", description="Blockly schema version")
    
    def get_root_blocks(self) -> List[BlocklyBlock]:
        """Get all top-level blocks (no parent)"""
        return [b for b in self.blocks.values() if b.parent is None]


# ============================================================================
# METADATA
# ============================================================================

class ProjectMetadata(BaseModel):
    """Metadata about the project state itself"""
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = Field(default=1, description="State version - increments on every mutation")
    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version for compatibility")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(description="User ID who created this project")
    last_modified_by: str = Field(description="User ID who last modified this project")
    tags: List[str] = Field(default_factory=list, description="User-defined tags")
    is_template: bool = Field(default=False, description="Is this a reusable template?")


# ============================================================================
# PROJECT STATE (ROOT MODEL)
# ============================================================================

class ProjectState(BaseModel):
    """
    The authoritative, single source of truth for an AI-generated app design.
    
    This is the ROOT model. All other models are composed within it.
    
    Rules:
    1. Foundations are IMMUTABLE by default
    2. All mutations must go through the Resolver
    3. Every mutation creates a ChangeLog entry
    4. State must always be valid (validated on load/save)
    5. Version increments on every mutation
    """
    
    # Core sections
    foundations: AppFoundations = Field(description="Immutable app identity (unless intent=MODIFY_FOUNDATION)")
    architecture: AppArchitecture = Field(default_factory=AppArchitecture, description="App structure")
    layout: AppLayout = Field(default_factory=AppLayout, description="Visual layout")
    blockly: AppBlockly = Field(default_factory=AppBlockly, description="Visual programming blocks")
    
    # State management
    metadata: ProjectMetadata = Field(description="Project metadata")
    change_log: ChangeLog = Field(default_factory=ChangeLog, description="Immutable change history")
    
    def increment_version(self, modified_by: str) -> None:
        """
        Increment state version and update metadata.
        MUST be called after every mutation.
        """
        self.metadata.version += 1
        self.metadata.updated_at = datetime.utcnow()
        self.metadata.last_modified_by = modified_by
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)"""
        return self.model_dump(mode='json')
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectState":
        """Deserialize from dictionary with validation"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ProjectState":
        """Deserialize from JSON string with validation"""
        return cls.model_validate_json(json_str)
    
    @classmethod
    def create_new(
        cls,
        app_name: str,
        app_description: str,
        created_by: str,
        app_type: str = "general",
    ) -> "ProjectState":
        """
        Factory method to create a new, empty project state.
        Use this for intent=CREATE_NEW_APP
        """
        return cls(
            foundations=AppFoundations(
                app_name=app_name,
                app_description=app_description,
                app_type=app_type,
            ),
            metadata=ProjectMetadata(
                created_by=created_by,
                last_modified_by=created_by,
            ),
        )
    
    def __str__(self) -> str:
        return (
            f"ProjectState(id={self.metadata.project_id}, "
            f"name={self.foundations.app_name}, "
            f"version={self.metadata.version}, "
            f"screens={len(self.architecture.screens)}, "
            f"components={len(self.layout.components)})"
        )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_state_schema(state: ProjectState) -> bool:
    """
    Validate that state conforms to current schema.
    Raises ValidationError if invalid.
    """
    try:
        # Pydantic automatically validates on instantiation
        # This function exists for explicit validation calls
        ProjectState.model_validate(state.model_dump())
        return True
    except Exception as e:
        raise ValueError(f"State validation failed: {e}")


def check_schema_compatibility(state_schema_version: str) -> bool:
    """
    Check if state schema version is compatible with current version.
    
    Returns:
        True if compatible, False if migration needed
    """
    # For now, only exact version match is supported
    # In future, implement semantic versioning compatibility
    return state_schema_version == CURRENT_SCHEMA_VERSION


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a new project state
    state = ProjectState.create_new(
        app_name="My Counter App",
        app_description="A simple counter application",
        created_by="user_123",
        app_type="productivity",
    )
    
    print("Created new project state:")
    print(state)
    print(f"\nJSON Schema Version: {state.metadata.schema_version}")
    print(f"State Version: {state.metadata.version}")
    
    # Serialize to JSON
    json_output = state.to_json()
    print(f"\nSerialized to JSON ({len(json_output)} bytes)")
    
    # Deserialize back
    restored_state = ProjectState.from_json(json_output)
    print(f"\nRestored from JSON: {restored_state}")
    
    # Validate
    is_valid = validate_state_schema(restored_state)
    print(f"Validation passed: {is_valid}")