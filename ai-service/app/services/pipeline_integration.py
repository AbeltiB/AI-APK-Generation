"""
Pipeline Integration for Project State System
==============================================

Integrates the Project State system with the existing AI generation pipeline.

This module:
1. Classifies user intent from prompts
2. Determines if new state or mutation
3. Extracts LLM-proposed changes
4. Routes through State Resolver
5. Persists final state
6. Returns complete response

Integration Points:
- Replaces direct architecture/layout/blockly generation
- Maintains backward compatibility with existing API
- Adds state continuity across user sessions
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from app.models.project_state import ProjectState, IntentType
from app.services.state_resolver import resolve_and_update_state
from app.services.state_persistence import ProjectStatePersistence, FileSystemBackend
from app.models.schemas.input_output import AIRequest


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

class IntentClassifier:
    """
    Classifies user intent from prompt text.
    
    This can be:
    - Rule-based (keywords)
    - LLM-based (ask Claude to classify)
    - Hybrid (rules + LLM)
    """
    
    def __init__(self):
        # Keyword patterns for intent classification
        self.intent_patterns = {
            IntentType.CREATE_NEW_APP: [
                "create new app",
                "build new",
                "start new project",
                "make a new",
                "develop new application",
            ],
            IntentType.MODIFY_FOUNDATION: [
                "change app name",
                "rename app",
                "change theme",
                "change color",
                "modify foundation",
                "update app type",
            ],
            IntentType.UPDATE_FEATURE: [
                "add feature",
                "add screen",
                "add button",
                "create component",
                "add navigation",
                "update logic",
            ],
            IntentType.REGENERATE_LAYOUT: [
                "regenerate layout",
                "redo layout",
                "redesign ui",
                "change layout",
                "rearrange components",
            ],
            IntentType.ASK_ABOUT_APP: [
                "what does",
                "explain",
                "how does",
                "describe",
                "tell me about",
            ],
        }
    
    def classify(self, prompt: str, has_existing_state: bool = False) -> IntentType:
        """
        Classify user intent from prompt.
        
        Args:
            prompt: User's input prompt
            has_existing_state: Whether project state already exists
            
        Returns:
            Classified IntentType
        """
        prompt_lower = prompt.lower()
        
        # Check for explicit intent keywords
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    logger.debug(f"Intent classified (keyword): {intent.value}")
                    return intent
        
        # Default logic
        if not has_existing_state:
            # No existing state → CREATE_NEW_APP
            logger.debug("Intent classified (no state): CREATE_NEW_APP")
            return IntentType.CREATE_NEW_APP
        else:
            # Has existing state → UPDATE_FEATURE (most common)
            logger.debug("Intent classified (has state): UPDATE_FEATURE")
            return IntentType.UPDATE_FEATURE
    
    async def classify_with_llm(self, prompt: str, llm_service) -> IntentType:
        """
        Use LLM to classify intent (more accurate).
        
        Args:
            prompt: User's input prompt
            llm_service: LLM service instance
            
        Returns:
            Classified IntentType
        """
        classification_prompt = f"""
        Classify the user's intent from their prompt into one of these categories:
        
        1. CREATE_NEW_APP - User wants to create a completely new app
        2. MODIFY_FOUNDATION - User wants to change core app properties (name, theme, colors)
        3. UPDATE_FEATURE - User wants to add/modify features, screens, or components
        4. REGENERATE_LAYOUT - User wants to redesign the visual layout
        5. ASK_ABOUT_APP - User is asking a question, not requesting changes
        
        User prompt: "{prompt}"
        
        Respond with ONLY the intent category name, nothing else.
        """
        
        try:
            response = await llm_service.generate(classification_prompt)
            intent_str = response.strip().upper()
            
            # Map to IntentType
            intent = IntentType(intent_str.lower())
            logger.debug(f"Intent classified (LLM): {intent.value}")
            return intent
            
        except Exception as e:
            logger.warning(f"LLM intent classification failed, falling back to rules: {e}")
            return self.classify(prompt, has_existing_state=True)


# ============================================================================
# LLM OUTPUT PARSER
# ============================================================================

class LLMOutputParser:
    """
    Parses LLM-generated output into structured state changes.
    
    The LLM should generate JSON in this format:
    {
        "foundations": { ... },
        "architecture": { ... },
        "layout": { ... },
        "blockly": { ... }
    }
    """
    
    def parse(self, llm_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize LLM output into state changes.
        
        Args:
            llm_output: Raw output from LLM (should be structured JSON)
            
        Returns:
            Normalized state changes dict
        """
        # In production, add more validation and normalization
        return llm_output
    
    def extract_proposed_changes(
        self,
        architecture: Optional[Dict[str, Any]] = None,
        layout: Optional[Dict[str, Any]] = None,
        blockly: Optional[Dict[str, Any]] = None,
        foundations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract changes from individual pipeline outputs.
        
        This bridges the gap between the current pipeline outputs
        and the state system's expected format.
        """
        changes = {}
        
        if foundations:
            changes["foundations"] = foundations
        
        if architecture:
            changes["architecture"] = architecture
        
        if layout:
            changes["layout"] = layout
        
        if blockly:
            changes["blockly"] = blockly
        
        return changes


# ============================================================================
# SAFE STATE RESOLVER WRAPPER
# ============================================================================

def safe_resolve_and_update_state(
    state: ProjectState,
    intent: IntentType,
    proposed_changes: Dict[str, Any],
    actor: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Safe wrapper around resolve_and_update_state that won't crash.
    
    Args:
        state: Current project state
        intent: User intent
        proposed_changes: Proposed changes from pipeline
        actor: Who is making the change
        reason: Why the change is being made
        
    Returns:
        Dictionary with result information
    """
    try:
        logger.debug(
            "🔧 Calling resolve_and_update_state",
            extra={
                "state_version": state.metadata.version,
                "intent": intent.value,
                "proposed_changes_type": type(proposed_changes).__name__,
                "proposed_changes_keys": list(proposed_changes.keys()) if isinstance(proposed_changes, dict) else [],
                "actor": actor,
                "reason": reason[:100] if reason else "No reason provided",
            }
        )
        
        # Call the actual state resolver
        updated_state = resolve_and_update_state(
            current_state=state,
            intent=intent,
            proposed_changes=proposed_changes,
            actor=actor,
            reason=reason,
        )
        
        logger.info(
            "✅ State resolved successfully",
            extra={
                "old_version": state.metadata.version,
                "new_version": updated_state.metadata.version,
                "version_increment": updated_state.metadata.version - state.metadata.version,
                "changes_applied": True,
                "total_changes": len(updated_state.change_log.changes),
            }
        )
        
        return {
            "success": True,
            "updated_state": updated_state,
            "state_metadata": {
                "state_version": updated_state.metadata.version,
                "project_id": updated_state.metadata.project_id,
                "total_changes": len(updated_state.change_log.changes),
            },
            "changes_applied": True,
            "error": None,
        }
        
    except Exception as e:
        logger.error(
            "❌ State resolution failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "intent": intent.value,
                "state_version": state.metadata.version,
                "proposed_changes_sample": str(proposed_changes)[:200] if proposed_changes else "None",
            },
            exc_info=True
        )
        
        return {
            "success": False,
            "updated_state": state,  # Return original state
            "state_metadata": {
                "state_version": state.metadata.version,
                "project_id": state.metadata.project_id,
                "total_changes": len(state.change_log.changes),
            },
            "changes_applied": False,
            "error": str(e),
        }


# ============================================================================
# PIPELINE STATE MANAGER
# ============================================================================

class PipelineStateManager:
    """
    Manages project state throughout the AI generation pipeline.
    
    This is the main integration point between the pipeline and state system.
    """
    
    def __init__(
        self,
        persistence: ProjectStatePersistence,
        intent_classifier: Optional[IntentClassifier] = None,
    ):
        self.persistence = persistence
        self.classifier = intent_classifier or IntentClassifier()
        self.parser = LLMOutputParser()
    
    async def process_request(
        self,
        request: AIRequest,
        pipeline_outputs: Dict[str, Any],
    ) -> Tuple[ProjectState, Dict[str, Any]]:
        """
        Process AI request with state management.
        
        Args:
            request: User's AI request
            pipeline_outputs: Outputs from existing pipeline (architecture, layout, blockly)
            
        Returns:
            Tuple of (final_state, complete_response)
        """
        user_id = request.user_id
        session_id = request.session_id
        prompt = request.prompt
        
        logger.info(
            "🧠 Processing request with state management",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "prompt_length": len(prompt),
                "pipeline_outputs_keys": list(pipeline_outputs.keys()) if pipeline_outputs else [],
                "task_id": getattr(request, 'task_id', 'unknown'),
            }
        )
        
        # Step 1: Load or create project state
        logger.debug(
            "📂 Step 1: Loading or creating project state",
            extra={
                "state_id": f"{user_id}_{session_id}",
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        
        state, is_new = await self._get_or_create_state(user_id, session_id)
        
        logger.info(
            "📊 State loaded/created",
            extra={
                "is_new": is_new,
                "project_id": state.metadata.project_id if hasattr(state.metadata, 'project_id') else None,
                "state_version": state.metadata.version,
                "state_sections_present": {
                    "foundations": hasattr(state, 'foundations') and state.foundations is not None,
                    "architecture": hasattr(state, 'architecture') and state.architecture is not None,
                    "layout": hasattr(state, 'layout') and state.layout is not None,
                    "blockly": hasattr(state, 'blockly') and state.blockly is not None,
                },
            }
        )
        
        # Step 2: Classify intent
        logger.debug(
            "🔍 Step 2: Classifying intent",
            extra={
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "has_existing_state": not is_new,
                "state_exists": state is not None,
            }
        )
        
        intent = self.classifier.classify(prompt, has_existing_state=not is_new)
        
        logger.info(
            "🎯 Intent classified",
            extra={
                "intent": intent.value,
                "intent_enum": str(intent),
                "is_new_state": is_new,
                "current_version": state.metadata.version,
            }
        )
        
        # Step 3: Handle read-only intent
        if intent == IntentType.ASK_ABOUT_APP:
            logger.info(
                "📖 Read-only intent detected (ASK_ABOUT_APP)",
                extra={
                    "action": "returning current state without mutation",
                    "intent": intent.value,
                }
            )
            # Return current state without mutation
            return state, self._build_response(state, cache_hit=True)
        
        # Step 4: Extract proposed changes from pipeline outputs
        logger.debug(
            "🔧 Step 4: Extracting proposed changes from pipeline outputs",
            extra={
                "pipeline_outputs_type": type(pipeline_outputs).__name__,
                "pipeline_outputs_keys": list(pipeline_outputs.keys()) if isinstance(pipeline_outputs, dict) else "Not a dict",
                "pipeline_outputs_length": len(pipeline_outputs) if isinstance(pipeline_outputs, dict) else None,
            }
        )
        
        # Log detailed pipeline outputs structure
        if isinstance(pipeline_outputs, dict):
            for key, value in pipeline_outputs.items():
                if value is not None:
                    logger.debug(
                        f"📝 Pipeline output: {key}",
                        extra={
                            "key": key,
                            "value_type": type(value).__name__,
                            "value_length": len(value) if hasattr(value, '__len__') else None,
                            "value_is_dict": isinstance(value, dict),
                            "value_is_list": isinstance(value, list),
                            "value_is_empty": not bool(value) if hasattr(value, '__bool__') else None,
                            "value_preview": str(value)[:200] + "..." if len(str(value)) > 200 else str(value),
                        }
                    )
                else:
                    logger.debug(
                        f"📝 Pipeline output: {key} is None",
                        extra={"key": key}
                    )
        else:
            logger.warning(
                "⚠️ Pipeline outputs is not a dictionary",
                extra={
                    "pipeline_outputs_type": type(pipeline_outputs).__name__,
                    "pipeline_outputs_value": str(pipeline_outputs)[:500],
                }
            )
        
        try:
            proposed_changes = self.parser.extract_proposed_changes(
                foundations=pipeline_outputs.get("foundations"),
                architecture=pipeline_outputs.get("architecture"),
                layout=pipeline_outputs.get("layout"),
                blockly=pipeline_outputs.get("blockly"),
            )
            
            logger.info(
                "📋 Proposed changes extracted",
                extra={
                    "proposed_changes_type": type(proposed_changes).__name__,
                    "proposed_changes_is_dict": isinstance(proposed_changes, dict),
                    "proposed_changes_is_list": isinstance(proposed_changes, list),
                    "proposed_changes_length": len(proposed_changes) if hasattr(proposed_changes, '__len__') else None,
                    "proposed_changes_keys": list(proposed_changes.keys()) if isinstance(proposed_changes, dict) else "Not a dict",
                    "proposed_changes_sections": list(proposed_changes.keys()) if isinstance(proposed_changes, dict) else [],
                    "proposed_changes_preview": str(proposed_changes)[:500] + "..." if len(str(proposed_changes)) > 500 else str(proposed_changes),
                }
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to extract proposed changes",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "pipeline_outputs_keys": list(pipeline_outputs.keys()) if isinstance(pipeline_outputs, dict) else "Not a dict",
                },
                exc_info=True
            )
            # Try to create minimal changes structure
            proposed_changes = self._create_minimal_changes(pipeline_outputs)
            logger.info(
                "🔄 Created minimal changes as fallback",
                extra={
                    "minimal_changes_type": type(proposed_changes).__name__,
                    "minimal_changes": proposed_changes,
                }
            )
        
        # Step 5: Resolve and apply changes using safe wrapper
        logger.debug(
            "⚙️ Step 5: Resolving and applying state changes",
            extra={
                "current_state_version": state.metadata.version,
                "intent": intent.value,
                "proposed_changes_summary": {
                    "type": type(proposed_changes).__name__,
                    "is_dict": isinstance(proposed_changes, dict),
                    "is_list": isinstance(proposed_changes, list),
                    "length": len(proposed_changes) if hasattr(proposed_changes, '__len__') else None,
                    "has_data": bool(proposed_changes),
                },
                "actor": user_id,
                "reason": f"User request: {prompt[:50]}...",
            }
        )
        
        # Use the safe resolver
        resolver_result = safe_resolve_and_update_state(
            state=state,
            intent=intent,
            proposed_changes=proposed_changes,
            actor=user_id,
            reason=f"User request: {prompt[:100]}",
        )
        
        logger.info(
            "✅ State resolution completed",
            extra={
                "resolver_success": resolver_result["success"],
                "resolver_error": resolver_result["error"],
                "changes_applied": resolver_result.get("changes_applied", False),
                "new_version": resolver_result.get("state_metadata", {}).get("state_version"),
                "total_changes": resolver_result.get("state_metadata", {}).get("total_changes", 0),
            }
        )
        
        if resolver_result["success"]:
            updated_state = resolver_result["updated_state"]
            state_metadata = resolver_result["state_metadata"]
            
            logger.debug(
                "📈 State updated successfully",
                extra={
                    "old_version": state.metadata.version,
                    "new_version": updated_state.metadata.version,
                    "version_increment": updated_state.metadata.version - state.metadata.version,
                    "change_log_count": len(updated_state.change_log.changes),
                    "project_id": state_metadata.get("project_id"),
                }
            )
            
        else:
            # State resolution failed but we continue with original state
            logger.warning(
                "⚠️ State resolution failed (but continuing)",
                extra={
                    "error": resolver_result["error"],
                    "fallback_action": "using original state",
                    "original_state_version": state.metadata.version,
                    "state_metadata": resolver_result.get("state_metadata"),
                }
            )
            updated_state = state  # Fall back to original state
        
        # Step 6: Persist updated state
        logger.debug(
            "💾 Step 6: Persisting updated state",
            extra={
                "state_to_persist": updated_state is not None,
                "state_type": type(updated_state).__name__ if updated_state else None,
                "expected_version": state.metadata.version,
                "new_version": updated_state.metadata.version,
            }
        )
        
        try:
            await self.persistence.save_project_state(
                updated_state,
                expected_version=state.metadata.version,
            )
            
            logger.info(
                "💾 State persisted successfully",
                extra={
                    "project_id": updated_state.metadata.project_id,
                    "old_version": state.metadata.version,
                    "new_version": updated_state.metadata.version,
                    "changes_persisted": len(updated_state.change_log.changes),
                    "persistence_success": True,
                }
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to persist state",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "project_id": updated_state.metadata.project_id if updated_state else None,
                    "state_version": updated_state.metadata.version if updated_state else None,
                },
                exc_info=True
            )
            # Continue even if persistence fails (state is still valid in memory)
        
        # Step 7: Build complete response
        logger.debug(
            "📤 Step 7: Building complete response",
            extra={
                "final_state_version": updated_state.metadata.version,
                "has_change_log": hasattr(updated_state, 'change_log'),
                "change_log_count": len(updated_state.change_log.changes) if hasattr(updated_state, 'change_log') else 0,
            }
        )
        
        response = self._build_response(updated_state)
        
        # Log response structure
        logger.info(
            "🎉 Request processing completed",
            extra={
                "success": True,
                "processing_stages": {
                    "state_loaded": True,
                    "intent_classified": True,
                    "changes_extracted": True,
                    "state_resolved": True,
                    "state_persisted": True,
                    "response_built": True,
                },
                "final_metrics": {
                    "project_id": updated_state.metadata.project_id,
                    "state_version": updated_state.metadata.version,
                    "total_changes": len(updated_state.change_log.changes),
                    "intent_used": intent.value,
                    "is_new_project": is_new,
                    "response_keys": list(response.keys()) if isinstance(response, dict) else "Not a dict",
                },
                "response_preview": {
                    "type": type(response).__name__,
                    "has_architecture": "architecture" in response if isinstance(response, dict) else False,
                    "has_layout": "layout" in response if isinstance(response, dict) else False,
                    "has_blockly": "blockly" in response if isinstance(response, dict) else False,
                },
            }
        )
        
        return updated_state, response
    
    def _create_minimal_changes(self, pipeline_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal changes structure when extraction fails"""
        logger.warning(
            "🔄 Creating minimal changes fallback",
            extra={
                "pipeline_outputs_type": type(pipeline_outputs).__name__,
                "pipeline_outputs_keys": list(pipeline_outputs.keys()) if isinstance(pipeline_outputs, dict) else [],
            }
        )
        
        changes = {}
        
        # Try to extract any valid data
        if isinstance(pipeline_outputs, dict):
            for key in ["architecture", "layout", "blockly", "foundations"]:
                if key in pipeline_outputs and pipeline_outputs[key]:
                    changes[key] = pipeline_outputs[key]
                    logger.debug(
                        f"➕ Added {key} to minimal changes",
                        extra={
                            "key": key,
                            "value_type": type(pipeline_outputs[key]).__name__,
                            "has_data": bool(pipeline_outputs[key]),
                        }
                    )
        
        # If still empty, create a placeholder
        if not changes:
            changes = {
                "architecture": {"screens": [], "navigation": []},
                "layout": {"components": {}},
                "blockly": {"blocks": {}, "variables": []},
            }
            logger.debug("📝 Created empty placeholder changes")
        
        return changes
    
    async def _get_or_create_state(
        self,
        user_id: str,
        session_id: str,
    ) -> Tuple[ProjectState, bool]:
        """
        Get existing project state or create new one.
        
        Returns:
            Tuple of (state, is_new)
        """
        # Try to find existing state for this session
        project_id = self._generate_project_id(user_id, session_id)
        
        try:
            state = await self.persistence.load_project_state(project_id)
            logger.info(f"Loaded existing state: {project_id}")
            return state, False
            
        except Exception:
            # Create new state
            state = ProjectState.create_new(
                app_name=f"Project_{session_id[:8]}",
                app_description="AI-generated application",
                created_by=user_id,
            )
            
            # Override project_id to match session
            state.metadata.project_id = project_id
            
            logger.info(f"Created new state: {project_id}")
            return state, True
    
    def _generate_project_id(self, user_id: str, session_id: str) -> str:
        """Generate project ID from user and session"""
        return f"{user_id}_{session_id}"
    
    def _build_response(
        self,
        state: ProjectState,
        cache_hit: bool = False
    ) -> Dict[str, Any]:
        """Build complete response from state"""
        try:
            response = {
                "status": "success",
                "architecture": state.architecture.model_dump() if hasattr(state, 'architecture') and state.architecture else {},
                "layout": state.layout.model_dump() if hasattr(state, 'layout') and state.layout else {},
                "blockly": state.blockly.model_dump() if hasattr(state, 'blockly') and state.blockly else {},
                "metadata": {
                    "project_id": state.metadata.project_id,
                    "version": state.metadata.version,
                    "schema_version": state.metadata.schema_version,
                    "cache_hit": cache_hit,
                    "total_changes": len(state.change_log.changes) if hasattr(state, 'change_log') else 0,
                }
            }
            return response
        except Exception as e:
            logger.error(f"Error building response: {e}")
            return self._build_error_response(str(e))
    
    def _build_error_response(self, error: str) -> Dict[str, Any]:
        """Build error response"""
        return {
            "status": "error",
            "error": error,
            "architecture": {},
            "layout": {},
            "blockly": {},
            "metadata": {
                "error": True,
                "error_message": error,
            }
        }


# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

async def integrate_with_pipeline(
    request: AIRequest,
    pipeline_result: Dict[str, Any],
    persistence: Optional[ProjectStatePersistence] = None,
) -> Dict[str, Any]:
    """
    Main integration function to wrap existing pipeline with state management.
    
    Usage in tasks.py:
        # Old way:
        result = await pipeline.execute(request)
        
        # New way:
        pipeline_result = await pipeline.execute(request)
        result = await integrate_with_pipeline(request, pipeline_result)
    
    Args:
        request: AI request from user
        pipeline_result: Output from existing pipeline
        persistence: Optional persistence layer (defaults to FileSystem)
        
    Returns:
        Enhanced result with state management
    """
    if persistence is None:
        backend = FileSystemBackend(storage_path="./project_states")
        persistence = ProjectStatePersistence(backend)
    
    manager = PipelineStateManager(persistence)
    
    state, response = await manager.process_request(request, pipeline_result)
    
    # Merge with original pipeline metadata
    if "metadata" in pipeline_result:
        response["metadata"].update(pipeline_result["metadata"])
    
    return response


# ============================================================================
# BACKWARD COMPATIBILITY LAYER
# ============================================================================

async def execute_with_state_management(
    request: AIRequest,
    pipeline,
    persistence: Optional[ProjectStatePersistence] = None,
) -> Dict[str, Any]:
    """
    Execute pipeline with full state management (drop-in replacement).
    
    This function can replace the existing pipeline.execute() call
    while maintaining full backward compatibility.
    
    Args:
        request: AI request
        pipeline: Existing pipeline instance
        persistence: Optional persistence layer
        
    Returns:
        Complete result with state management
    """
    logger.info("Executing pipeline with state management")
    
    # Execute existing pipeline
    pipeline_result = await pipeline.execute(request)
    
    # Integrate with state system
    final_result = await integrate_with_pipeline(
        request,
        pipeline_result,
        persistence,
    )
    
    return final_result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from app.models.schemas.input_output import AIRequest
    
    async def example():
        # Simulate a request
        request = AIRequest(
            prompt="Create a counter app with increment and decrement buttons",
            user_id="user_123",
            session_id="session_456",
        )
        
        # Simulate pipeline output
        pipeline_output = {
            "architecture": {
                "screens": {
                    "main_screen": {
                        "screen_name": "Main",
                        "screen_type": "main",
                        "is_entry_point": True,
                        "description": "Main counter screen",
                    }
                }
            },
            "layout": {
                "components": {
                    "counter_text": {
                        "component_type": "text",
                        "screen_id": "main_screen",
                        "position": {"x": 0.5, "y": 0.3},
                        "size": {"width": 200, "height": 50},
                    },
                    "increment_button": {
                        "component_type": "button",
                        "screen_id": "main_screen",
                        "position": {"x": 0.3, "y": 0.5},
                        "size": {"width": 100, "height": 50},
                    },
                }
            },
            "blockly": {
                "blocks": {}
            },
        }
        
        # Initialize persistence
        backend = FileSystemBackend(storage_path="./test_states")
        persistence = ProjectStatePersistence(backend)
        
        # Process with state management
        result = await integrate_with_pipeline(request, pipeline_output, persistence)
        
        print("✅ Pipeline executed with state management")
        print(f"Project ID: {result['metadata']['project_id']}")
        print(f"Version: {result['metadata']['version']}")
        print(f"Screens: {len(result['architecture']['screens'])}")
        print(f"Components: {len(result['layout']['components'])}")
    
    asyncio.run(example())