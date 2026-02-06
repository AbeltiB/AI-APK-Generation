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
            f"Processing request with state management",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "prompt_length": len(prompt),
            }
        )
        
        # Step 1: Load or create project state
        state, is_new = await self._get_or_create_state(user_id, session_id)
        
        # Step 2: Classify intent
        intent = self.classifier.classify(prompt, has_existing_state=not is_new)
        
        logger.info(
            f"Intent classified: {intent.value}",
            extra={
                "is_new_state": is_new,
                "current_version": state.metadata.version,
            }
        )
        
        # Step 3: Handle read-only intent
        if intent == IntentType.ASK_ABOUT_APP:
            # Return current state without mutation
            return state, self._build_response(state, cache_hit=True)
        
        # Step 4: Extract proposed changes from pipeline outputs
        proposed_changes = self.parser.extract_proposed_changes(
            foundations=pipeline_outputs.get("foundations"),
            architecture=pipeline_outputs.get("architecture"),
            layout=pipeline_outputs.get("layout"),
            blockly=pipeline_outputs.get("blockly"),
        )
        
        # Step 5: Resolve and apply changes
        try:
            updated_state = resolve_and_update_state(
                state=state,
                intent=intent,
                proposed_changes=proposed_changes,
                actor=user_id,
                reason=f"User request: {prompt[:100]}",
            )
        except Exception as e:
            logger.error(f"State resolution failed: {e}", exc_info=e)
            # Return original state on error
            return state, self._build_error_response(str(e))
        
        # Step 6: Persist updated state
        await self.persistence.save_project_state(
            updated_state,
            expected_version=state.metadata.version,
        )
        
        logger.info(
            f"✅ State updated and persisted",
            extra={
                "project_id": updated_state.metadata.project_id,
                "old_version": state.metadata.version,
                "new_version": updated_state.metadata.version,
                "changes": len(updated_state.change_log.changes),
            }
        )
        
        # Step 7: Build complete response
        response = self._build_response(updated_state)
        
        return updated_state, response
    
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
        return {
            "architecture": state.architecture.model_dump(),
            "layout": state.layout.model_dump(),
            "blockly": state.blockly.model_dump(),
            "metadata": {
                "project_id": state.metadata.project_id,
                "version": state.metadata.version,
                "schema_version": state.metadata.schema_version,
                "cache_hit": cache_hit,
                "total_changes": len(state.change_log.changes),
            }
        }
    
    def _build_error_response(self, error: str) -> Dict[str, Any]:
        """Build error response"""
        return {
            "error": error,
            "architecture": {},
            "layout": {},
            "blockly": {},
            "metadata": {
                "error": True,
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