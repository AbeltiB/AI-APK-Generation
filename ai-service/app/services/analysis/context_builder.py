"""
Enhanced Context Builder - Fixed version with proper error handling

Loads and structures all relevant context for AI generation:
- Conversation history
- Existing projects
- User preferences
- Intent analysis results
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.core.database import db_manager
from app.models.schemas import IntentAnalysis, EnrichedContext
from app.utils.logging import get_logger, log_context

logger = get_logger(__name__)


class ContextRelevanceScore:
    """Calculate confidence that a project is relevant to current request."""
    
    @staticmethod
    def calculate(
        project: Dict[str, Any],
        user_id: str,
        session_id: str,
        intent: Any
    ) -> float:
        """Calculate relevance score (0.0 to 1.0)."""
        score = 0.0
        
        # CRITICAL: Ownership verification (mandatory)
        if project.get('user_id') != user_id:
            return 0.0  # Wrong user - NEVER return
        
        # Extract requires_context from intent (handling both dict and object)
        if isinstance(intent, dict):
            requires_context = intent.get('requires_context', False)
        else:
            requires_context = getattr(intent, 'requires_context', False)

        # Session match (highest weight)
        project_metadata = project.get('metadata', {})
        if project_metadata.get('session_id') == session_id:
            score += 0.6  # Same session = very relevant
        
        # Recency (within last hour)
        updated_at = project.get('updated_at')
        if updated_at:
            from datetime import datetime, timezone
            age_hours = (datetime.now(timezone.utc) - updated_at).total_seconds() / 3600
            if age_hours < 1:
                score += 0.3
            elif age_hours < 24:
                score += 0.1
        
        # Intent match
        if requires_context:
            score += 0.1
        
        return min(score, 1.0)


class ContextBuilder:
    """Builds enriched context for AI generation."""
    
    # Minimum confidence threshold for using existing projects
    MIN_CONFIDENCE_THRESHOLD = 0.5
    
    async def build_context(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        intent: Any,
        original_request: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> EnrichedContext:
        """Build comprehensive enriched context."""
        
        with log_context(operation="context_building"):
            #Extract intent fields may be handle both
            if isinstance(intent, dict):
                intent_type = intent.get('intent_type', 'unknown')
                complexity = intent.get('complexity', 'unknown')
                confidence = intent.get('confidence', 0.0)
                requires_context = intent.get('requires_context', False) 
            else:
                intent_type = getattr(intent, 'intent_type', 'unknown')
                complexity = getattr(intent, 'complexity', 'unknown')
                confidence = getattr(intent, 'confidence', 0.0)
                requires_context = getattr(intent, 'requires_context', False)
            
            logger.info(
                "context.building.started",
                extra={
                    "user_id": user_id,
                    "session_id": session_id,
                    "requires_context": requires_context,
                    "explicit_project_id": project_id is not None,
                    "intent_type": intent_type,
                    "complexity": complexity
                }
            )
            
            try:
                from app.models.schemas import IntentAnalysis
                intent_obj = IntentAnalysis(
                    intent_type=intent_type,
                    complexity=complexity,
                    confidence=confidence,
                    requires_context=requires_context,
                    # Add other required fields...
                )
            except ImportError:
                # If IntentAnalysis model doesn't exist, use the dict
                intent_obj = intent

            context = EnrichedContext(
                original_request=original_request,
                intent_analysis=intent_obj,
                conversation_history=[],
                existing_project=None,
                user_preferences={},
                timestamp=datetime.now(timezone.utc)
            )
            
            # Load conversation history
            try:
                context.conversation_history = await self._load_conversation_history(
                    user_id=user_id,
                    session_id=session_id,
                    limit=10
                )
            except Exception as e:
                logger.warning(
                    "context.history.load_failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
                context.conversation_history = []
            
            # Load existing project with strict validation
            if (isinstance(intent, dict) and intent.get('requires_context')) or project_id:
                try:
                    context.existing_project = await self._load_existing_project_safe(
                        user_id=user_id,
                        session_id=session_id,
                        intent=intent,
                        explicit_project_id=project_id
                    )
                    
                    if context.existing_project:
                        logger.info(
                            "context.project.loaded",
                            extra={
                                "project_id": context.existing_project.get('project_id'),
                                "confidence": context.existing_project.get('_confidence', 0.0)
                            }
                        )
                    elif intent.requires_context:
                        logger.warning(
                            "context.project.missing",
                            extra={
                                "user_id": user_id,
                                "session_id": session_id,
                                "intent_type": intent.intent_type,
                                "complexity": intent.complexity
                            }
                        )
                except Exception as e:
                    logger.error(
                        "context.project.load_failed",
                        extra={"error": str(e)},
                        exc_info=True
                    )
                    context.existing_project = None
            
            # Load user preferences
            try:
                context.user_preferences = await self._load_user_preferences(user_id)
            except Exception as e:
                logger.warning(
                    "context.preferences.load_failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
                context.user_preferences = self._get_default_preferences()
            
            logger.info(
                "context.building.completed",
                extra={
                    "history_messages": len(context.conversation_history),
                    "has_project": context.existing_project is not None,
                    "preferences_loaded": len(context.user_preferences) > 0
                }
            )
            
            return context
    
    async def _load_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Load recent conversation history."""
        
        logger.debug(
            "context.history.loading",
            extra={"user_id": user_id, "session_id": session_id}
        )
        
        try:
            conversations = await db_manager.get_conversation_history(
                user_id=user_id,
                session_id=session_id,
                limit=limit
            )
            
            if conversations:
                logger.debug(
                    "context.history.loaded",
                    extra={"count": len(conversations)}
                )
                return conversations
            else:
                logger.debug("context.history.empty")
                return []
                
        except Exception as e:
            logger.error(
                "context.history.load_error",
                extra={"error": str(e)},
                exc_info=True
            )
            return []
    
    async def _load_existing_project_safe(
        self,
        user_id: str,
        session_id: str,
        intent: IntentAnalysis,
        explicit_project_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load existing project with strict validation.
        
        FIXED: This method was missing and causing the NoneType error.
        """
        
        logger.debug(
            "context.project.loading",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "explicit_project_id": explicit_project_id
            }
        )
        
        try:
            # Case 1: Explicit project ID provided (highest priority)
            if explicit_project_id:
                project = await db_manager.get_project(explicit_project_id)
                
                if not project:
                    logger.warning(
                        "context.project.not_found",
                        extra={"project_id": explicit_project_id}
                    )
                    return None
                
                # CRITICAL: Verify ownership
                if project.get('user_id') != user_id:
                    logger.error(
                        "context.project.ownership_violation",
                        extra={
                            "project_id": explicit_project_id,
                            "requested_by": user_id,
                            "owned_by": project.get('user_id')
                        }
                    )
                    return None
                
                # Add confidence metadata
                project['_confidence'] = 1.0
                project['_match_reason'] = 'explicit_project_id'
                
                logger.info(
                    "context.project.loaded_explicit",
                    extra={"project_id": explicit_project_id}
                )
                
                return project
            
            # Case 2: Match by session_id
            projects = await db_manager.get_user_projects(
                user_id=user_id,
                limit=5
            )
            
            if not projects:
                logger.debug("context.project.none_found")
                return None
            
            # Find projects matching session with confidence scoring
            session_matches = []
            
            for project in projects:
                confidence = ContextRelevanceScore.calculate(
                    project=project,
                    user_id=user_id,
                    session_id=session_id,
                    intent=intent
                )
                
                if confidence >= self.MIN_CONFIDENCE_THRESHOLD:
                    project['_confidence'] = confidence
                    project['_match_reason'] = 'session_match'
                    session_matches.append(project)
            
            if not session_matches:
                logger.warning(
                    "context.project.no_confident_match",
                    extra={
                        "candidates": len(projects),
                        "threshold": self.MIN_CONFIDENCE_THRESHOLD
                    }
                )
                return None
            
            # Return highest confidence match
            best_match = max(session_matches, key=lambda p: p['_confidence'])
            
            logger.info(
                "context.project.loaded_session",
                extra={
                    "project_id": best_match.get('id'),
                    "confidence": best_match['_confidence'],
                    "candidates_evaluated": len(projects)
                }
            )
            
            return best_match
            
        except Exception as e:
            logger.error(
                "context.project.load_error",
                extra={"error": str(e)},
                exc_info=True
            )
            return None
    
    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences."""
        
        logger.debug(
            "context.preferences.loading",
            extra={"user_id": user_id}
        )
        
        try:
            preferences = await db_manager.get_user_preferences(user_id)
            
            if preferences:
                logger.debug(
                    "context.preferences.loaded",
                    extra={"count": len(preferences)}
                )
                return preferences
            else:
                logger.debug("context.preferences.using_defaults")
                return self._get_default_preferences()
                
        except Exception as e:
            logger.warning(
                "context.preferences.load_error",
                extra={"error": str(e)},
                exc_info=True
            )
            return self._get_default_preferences()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences."""
        return {
            "theme": "light",
            "component_style": "detailed",
            "preferred_colors": {
                "primary": "#007AFF",
                "secondary": "#5856D6",
                "background": "#FFFFFF",
                "text": "#000000"
            },
            "layout_style": "modern",
            "enable_animations": True
        }
    
    def format_context_for_prompt(self, context: EnrichedContext) -> str:
        """Format enriched context into a string for LLM prompts."""
        
        parts = []
        
        # Intent information
        parts.append(f"**Intent:** {context.intent_analysis.intent_type}")
        parts.append(f"**Complexity:** {context.intent_analysis.complexity}")
        
        # Extracted entities
        if context.intent_analysis.extracted_entities:
            entities = context.intent_analysis.extracted_entities
            if entities.get('components'):
                parts.append(f"**Requested Components:** {', '.join(entities['components'])}")
            if entities.get('features'):
                parts.append(f"**Requested Features:** {', '.join(entities['features'])}")
        
        # Conversation history
        if context.conversation_history:
            recent = context.conversation_history[-3:]  # Last 3 messages
            history_str = "\n".join([
                f"  - {msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}"
                for msg in recent
            ])
            parts.append(f"**Recent Conversation:**\n{history_str}")
        
        # Existing project
        if context.existing_project:
            proj = context.existing_project
            confidence = proj.get('_confidence', 0.0)
            match_reason = proj.get('_match_reason', 'unknown')
            
            parts.append(f"**Existing Project:** {proj.get('project_name', 'Unnamed')}")
            parts.append(f"  - Confidence: {confidence:.2f} (matched by {match_reason})")
            
            if proj.get('architecture'):
                arch = proj['architecture']
                parts.append(f"  - Type: {arch.get('app_type', 'unknown')}")
                parts.append(f"  - Screens: {len(arch.get('screens', []))}")
        
        # User preferences
        if context.user_preferences:
            prefs = context.user_preferences
            parts.append(f"**User Preferences:**")
            if 'theme' in prefs:
                parts.append(f"  - Theme: {prefs['theme']}")
            if 'component_style' in prefs:
                parts.append(f"  - Style: {prefs['component_style']}")
        
        return "\n".join(parts)


# Global context builder instance
context_builder = ContextBuilder()