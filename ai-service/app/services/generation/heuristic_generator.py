"""
Enhanced Heuristic Architecture Generator

Generates valid architectures deterministically when Claude API fails.
100% schema-compliant with proper validation support.
"""
from typing import Dict, Any
from uuid import uuid4

from app.models.schemas import (
    ArchitectureDesign,
    ScreenDefinition,
    NavigationStructure,
    StateDefinition,
    DataFlowDiagram
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class HeuristicArchitectureGenerator:
    """
    Deterministic fallback architecture generator.
    
    Features:
    - ✅ 100% schema compliant
    - ✅ Pattern-based app detection
    - ✅ Always produces valid output
    - ✅ Comprehensive logging
    - ✅ Template-based generation
    """
    
    def __init__(self):
        # Common app patterns
        self.patterns = {
            'counter': ['counter', 'count', 'increment', 'decrement'],
            'todo': ['todo', 'task', 'checklist', 'to-do', 'list'],
            'calculator': ['calculator', 'calc', 'math', 'arithmetic'],
            'timer': ['timer', 'stopwatch', 'countdown', 'clock'],
            'notes': ['note', 'notes', 'memo', 'text'],
            'weather': ['weather', 'temperature', 'forecast'],
            'contacts': ['contact', 'address', 'phone']
        }
        
        logger.info(
            "🛡️ heuristic.generator.initialized",
            extra={"patterns": list(self.patterns.keys())}
        )
    
    async def generate(self, prompt: str) -> ArchitectureDesign:
        """
        Generate architecture from prompt using pattern matching.
        
        Args:
            prompt: User's prompt
            
        Returns:
            Valid ArchitectureDesign (always succeeds)
        """
        
        prompt_lower = prompt.lower()
        
        logger.info(
            "🛡️ heuristic.generation.started",
            extra={"prompt_length": len(prompt)}
        )
        
        # Detect app type
        app_type = self._detect_app_type(prompt_lower)
        
        logger.debug(
            "heuristic.pattern.detected",
            extra={"pattern": app_type}
        )
        
        # Generate based on pattern
        if app_type == 'counter':
            architecture = self._counter_app()
        elif app_type == 'todo':
            architecture = self._todo_app()
        elif app_type == 'calculator':
            architecture = self._calculator_app()
        elif app_type == 'timer':
            architecture = self._timer_app()
        elif app_type == 'notes':
            architecture = self._notes_app()
        elif app_type == 'weather':
            architecture = self._weather_app()
        elif app_type == 'contacts':
            architecture = self._contacts_app()
        else:
            architecture = self._generic_app(prompt)
        
        logger.info(
            "✅ heuristic.generation.completed",
            extra={
                "app_type": architecture.app_type,
                "screens": len(architecture.screens),
                "pattern_used": app_type
            }
        )
        
        return architecture
    
    def _detect_app_type(self, prompt_lower: str) -> str:
        """Detect app type from prompt using keyword matching"""
        
        scores = {}
        
        for app_type, keywords in self.patterns.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                scores[app_type] = score
        
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            logger.debug(
                "heuristic.pattern.matched",
                extra={
                    "pattern": best_match[0],
                    "score": best_match[1],
                    "all_scores": scores
                }
            )
            return best_match[0]
        
        logger.debug("heuristic.pattern.no_match")
        return 'generic'
    
    # ========================================================================
    # APP TEMPLATES (Schema-compliant)
    # ========================================================================
    
    def _counter_app(self) -> ArchitectureDesign:
        """Counter app template"""
        
        return ArchitectureDesign(
            app_type="single-page",
            screens=[
                ScreenDefinition(
                    id="main_screen",
                    name="Counter",
                    purpose="Display counter value and control buttons",
                    components=["Text", "Button", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[]
            ),
            state_management=[
                StateDefinition(
                    name="count",
                    type="local-state",
                    scope="screen",
                    initial_value=0
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["increment", "decrement", "reset"],
                api_calls=[],
                local_storage=[]
            )
        )
    
    def _todo_app(self) -> ArchitectureDesign:
        """Todo list app template"""
        
        return ArchitectureDesign(
            app_type="single-page",
            screens=[
                ScreenDefinition(
                    id="todo_screen",
                    name="Todo List",
                    purpose="Manage todo items with add, complete, and delete",
                    components=["InputText", "Button", "Text", "Checkbox", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[]
            ),
            state_management=[
                StateDefinition(
                    name="todos",
                    type="local-state",
                    scope="screen",
                    initial_value=[]
                ),
                StateDefinition(
                    name="newTodoText",
                    type="local-state",
                    scope="screen",
                    initial_value=""
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["add_todo", "toggle_complete", "delete_todo"],
                api_calls=[],
                local_storage=["todos"]
            )
        )
    
    def _calculator_app(self) -> ArchitectureDesign:
        """Calculator app template"""
        
        return ArchitectureDesign(
            app_type="single-page",
            screens=[
                ScreenDefinition(
                    id="calc_screen",
                    name="Calculator",
                    purpose="Perform basic arithmetic operations",
                    components=["Text", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[]
            ),
            state_management=[
                StateDefinition(
                    name="display",
                    type="local-state",
                    scope="screen",
                    initial_value="0"
                ),
                StateDefinition(
                    name="currentOperation",
                    type="local-state",
                    scope="screen",
                    initial_value=None
                ),
                StateDefinition(
                    name="previousValue",
                    type="local-state",
                    scope="screen",
                    initial_value=0
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["number_input", "operation_select", "calculate"],
                api_calls=[],
                local_storage=[]
            )
        )
    
    def _timer_app(self) -> ArchitectureDesign:
        """Timer/stopwatch app template"""
        
        return ArchitectureDesign(
            app_type="single-page",
            screens=[
                ScreenDefinition(
                    id="timer_screen",
                    name="Timer",
                    purpose="Timer with start, stop, and reset controls",
                    components=["Text", "Button", "Button", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[]
            ),
            state_management=[
                StateDefinition(
                    name="seconds",
                    type="local-state",
                    scope="screen",
                    initial_value=0
                ),
                StateDefinition(
                    name="isRunning",
                    type="local-state",
                    scope="screen",
                    initial_value=False
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["start", "stop", "reset"],
                api_calls=[],
                local_storage=[]
            )
        )
    
    def _notes_app(self) -> ArchitectureDesign:
        """Notes app template"""
        
        return ArchitectureDesign(
            app_type="multi-page",
            screens=[
                ScreenDefinition(
                    id="notes_list",
                    name="Notes List",
                    purpose="Display list of notes",
                    components=["Button", "Text"],
                    navigation=["note_detail"]
                ),
                ScreenDefinition(
                    id="note_detail",
                    name="Note Detail",
                    purpose="View and edit note content",
                    components=["TextArea", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[
                    {"from": "notes_list", "to": "note_detail", "label": "View Note"}
                ]
            ),
            state_management=[
                StateDefinition(
                    name="notes",
                    type="local-state",
                    scope="global",
                    initial_value=[]
                ),
                StateDefinition(
                    name="currentNote",
                    type="local-state",
                    scope="screen",
                    initial_value=None
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["create_note", "edit_note", "delete_note"],
                api_calls=[],
                local_storage=["notes"]
            )
        )
    
    def _weather_app(self) -> ArchitectureDesign:
        """Weather app template"""
        
        return ArchitectureDesign(
            app_type="single-page",
            screens=[
                ScreenDefinition(
                    id="weather_screen",
                    name="Weather",
                    purpose="Display current weather information",
                    components=["Text", "InputText", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[]
            ),
            state_management=[
                StateDefinition(
                    name="location",
                    type="local-state",
                    scope="screen",
                    initial_value=""
                ),
                StateDefinition(
                    name="weatherData",
                    type="async-state",
                    scope="screen",
                    initial_value=None
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["search_location"],
                api_calls=["fetch_weather"],
                local_storage=["last_location"]
            )
        )
    
    def _contacts_app(self) -> ArchitectureDesign:
        """Contacts app template"""
        
        return ArchitectureDesign(
            app_type="multi-page",
            screens=[
                ScreenDefinition(
                    id="contacts_list",
                    name="Contacts",
                    purpose="Display list of contacts",
                    components=["Text", "Button"],
                    navigation=["contact_detail"]
                ),
                ScreenDefinition(
                    id="contact_detail",
                    name="Contact Detail",
                    purpose="View and edit contact information",
                    components=["InputText", "InputText", "InputText", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[
                    {"from": "contacts_list", "to": "contact_detail", "label": "View Contact"}
                ]
            ),
            state_management=[
                StateDefinition(
                    name="contacts",
                    type="local-state",
                    scope="global",
                    initial_value=[]
                ),
                StateDefinition(
                    name="currentContact",
                    type="local-state",
                    scope="screen",
                    initial_value=None
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["add_contact", "edit_contact", "delete_contact"],
                api_calls=[],
                local_storage=["contacts"]
            )
        )
    
    def _generic_app(self, prompt: str) -> ArchitectureDesign:
        """Generic app template for unrecognized patterns"""
        
        return ArchitectureDesign(
            app_type="single-page",
            screens=[
                ScreenDefinition(
                    id="main_screen",
                    name="Main Screen",
                    purpose=f"Main screen for: {prompt[:100]}",
                    components=["Text", "Button"],
                    navigation=[]
                )
            ],
            navigation=NavigationStructure(
                type="stack",
                routes=[]
            ),
            state_management=[
                StateDefinition(
                    name="data",
                    type="local-state",
                    scope="screen",
                    initial_value=None
                )
            ],
            data_flow=DataFlowDiagram(
                user_interactions=["interact"],
                api_calls=[],
                local_storage=[]
            )
        )


# Global instance
heuristic_architecture_generator = HeuristicArchitectureGenerator()