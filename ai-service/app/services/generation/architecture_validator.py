"""
Enhanced Architecture Validator with Heuristic Awareness

Validates architectures from both Claude and heuristic generators.
Provides detailed feedback with structured logging.
"""
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from app.config import settings
from app.models.schemas import ArchitectureDesign, ScreenDefinition
from app.utils.logging import get_logger, log_context

logger = get_logger(__name__)


class ValidationWarning:
    """Validation warning/error"""
    
    def __init__(self, level: str, component: str, message: str, suggestion: str = ""):
        self.level = level  # "info", "warning", "error"
        self.component = component
        self.message = message
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'level': self.level,
            'component': self.component,
            'message': self.message,
            'suggestion': self.suggestion
        }
    
    def __str__(self) -> str:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
        s = f"{emoji.get(self.level, '•')} [{self.level.upper()}] {self.component}: {self.message}"
        if self.suggestion:
            s += f"\n   💡 {self.suggestion}"
        return s


class ArchitectureValidator:
    """
    Comprehensive architecture validator.
    
    Validates architectures from any source (Claude or heuristic).
    Provides detailed feedback with actionable suggestions.
    
    Validation passes:
    1. Schema validation (Pydantic)
    2. Component availability
    3. Navigation integrity
    4. State management
    5. Performance considerations
    6. UX best practices
    """
    
    def __init__(self):
        self.warnings: List[ValidationWarning] = []
        self.available_components = set(settings.available_components)
        
        # Validation stats
        self.stats = {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'heuristic_validated': 0,
            'claude_validated': 0
        }
        
        logger.info(
            "🔍 validator.initialized",
            extra={
                "available_components": len(self.available_components)
            }
        )
    
    async def validate(
        self,
        architecture: ArchitectureDesign,
        source: str = "unknown"
    ) -> Tuple[bool, List[ValidationWarning]]:
        """
        Validate architecture comprehensively.
        
        Args:
            architecture: Architecture to validate
            source: Source of architecture ("claude" or "heuristic")
            
        Returns:
            Tuple of (is_valid, warnings_list)
        """
        
        self.warnings = []
        self.stats['total_validations'] += 1
        
        if source == "heuristic":
            self.stats['heuristic_validated'] += 1
        elif source == "claude":
            self.stats['claude_validated'] += 1
        
        with log_context(operation="architecture_validation", source=source):
            logger.info(
                "🔍 validation.started",
                extra={
                    "app_type": architecture.app_type,
                    "screens": len(architecture.screens),
                    "source": source
                }
            )
            
            # Run all validation checks
            await self._validate_screens(architecture)
            await self._validate_components(architecture)
            await self._validate_navigation(architecture)
            await self._validate_state_management(architecture)
            await self._validate_performance(architecture)
            await self._validate_user_experience(architecture)
            
            # Special validation for heuristic architectures
            if source == "heuristic":
                await self._validate_heuristic_specific(architecture)
            
            # Determine if architecture is valid
            has_errors = any(w.level == "error" for w in self.warnings)
            is_valid = not has_errors
            
            if is_valid:
                self.stats['passed'] += 1
            else:
                self.stats['failed'] += 1
            
            # Log results
            error_count = sum(1 for w in self.warnings if w.level == "error")
            warning_count = sum(1 for w in self.warnings if w.level == "warning")
            info_count = sum(1 for w in self.warnings if w.level == "info")
            
            if is_valid:
                logger.info(
                    "✅ validation.passed",
                    extra={
                        "warnings": warning_count,
                        "infos": info_count,
                        "source": source
                    }
                )
            else:
                logger.error(
                    "❌ validation.failed",
                    extra={
                        "errors": error_count,
                        "warnings": warning_count,
                        "source": source
                    }
                )
            
            # Log detailed warnings if present
            if self.warnings:
                logger.debug(
                    "validation.warnings",
                    extra={
                        "total": len(self.warnings),
                        "by_level": {
                            "error": error_count,
                            "warning": warning_count,
                            "info": info_count
                        },
                        "details": [
                            {
                                "level": w.level,
                                "component": w.component,
                                "message": w.message
                            }
                            for w in self.warnings[:10]  # First 10
                        ]
                    }
                )
            
            return is_valid, self.warnings
    
    async def _validate_screens(self, architecture: ArchitectureDesign) -> None:
        """Validate screen definitions"""
        
        if len(architecture.screens) == 0:
            self.warnings.append(ValidationWarning(
                level="error",
                component="screens",
                message="No screens defined",
                suggestion="Add at least one screen to the architecture"
            ))
            return
        
        # Check for duplicate screen IDs
        screen_ids = [s.id for s in architecture.screens]
        duplicates = [id for id in screen_ids if screen_ids.count(id) > 1]
        
        if duplicates:
            self.warnings.append(ValidationWarning(
                level="error",
                component="screens",
                message=f"Duplicate screen IDs: {set(duplicates)}",
                suggestion="Ensure all screen IDs are unique"
            ))
        
        # Check individual screens
        for screen in architecture.screens:
            if not screen.purpose or len(screen.purpose.strip()) < 10:
                self.warnings.append(ValidationWarning(
                    level="warning",
                    component=f"screen:{screen.id}",
                    message=f"Screen '{screen.name}' has unclear purpose",
                    suggestion="Add a clear description (at least 10 characters)"
                ))
            
            if len(screen.components) == 0:
                self.warnings.append(ValidationWarning(
                    level="warning",
                    component=f"screen:{screen.id}",
                    message=f"Screen '{screen.name}' has no components",
                    suggestion="Add UI components to make the screen functional"
                ))
        
        # Warn about too many screens
        if len(architecture.screens) > 10:
            self.warnings.append(ValidationWarning(
                level="warning",
                component="screens",
                message=f"Large number of screens ({len(architecture.screens)})",
                suggestion="Consider simplifying or using tab/drawer navigation"
            ))
    
    async def _validate_components(self, architecture: ArchitectureDesign) -> None:
        """Validate component usage"""
        
        all_components = []
        for screen in architecture.screens:
            all_components.extend(screen.components)
        
        # Check for unsupported components
        for component in set(all_components):
            if component not in self.available_components:
                self.warnings.append(ValidationWarning(
                    level="error",
                    component="components",
                    message=f"Unsupported component: '{component}'",
                    suggestion=f"Use one of: {', '.join(sorted(self.available_components))}"
                ))
        
        # Check for reasonable component diversity
        unique_components = set(all_components)
        
        if len(unique_components) == 1 and len(all_components) > 1:
            self.warnings.append(ValidationWarning(
                level="info",
                component="components",
                message=f"App uses only one component type: {unique_components.pop()}",
                suggestion="Consider adding more component types for richer UI"
            ))
        
        # Check for common UI patterns
        has_input = any(c in all_components for c in ['InputText', 'TextArea'])
        has_button = 'Button' in all_components
        
        if has_input and not has_button:
            self.warnings.append(ValidationWarning(
                level="warning",
                component="components",
                message="App has input fields but no buttons",
                suggestion="Add buttons for form submission or actions"
            ))
    
    async def _validate_navigation(self, architecture: ArchitectureDesign) -> None:
        """Validate navigation structure"""
        
        screen_ids = {s.id for s in architecture.screens}
        
        # Validate navigation routes
        for route in architecture.navigation.routes:
            from_screen = route.get('from')
            to_screen = route.get('to')
            
            if from_screen and from_screen not in screen_ids:
                self.warnings.append(ValidationWarning(
                    level="error",
                    component="navigation",
                    message=f"Route from non-existent screen: {from_screen}",
                    suggestion=f"Valid screens: {', '.join(sorted(screen_ids))}"
                ))
            
            if to_screen and to_screen not in screen_ids:
                self.warnings.append(ValidationWarning(
                    level="error",
                    component="navigation",
                    message=f"Route to non-existent screen: {to_screen}",
                    suggestion=f"Valid screens: {', '.join(sorted(screen_ids))}"
                ))
        
        # Check for orphaned screens (multi-page apps only)
        if len(architecture.screens) > 1:
            reachable = {architecture.screens[0].id}
            
            for route in architecture.navigation.routes:
                from_screen = route.get('from')
                to_screen = route.get('to')
                if from_screen in reachable and to_screen:
                    reachable.add(to_screen)
            
            # Check screen navigation lists
            for screen in architecture.screens:
                if screen.id in reachable:
                    for nav_target in screen.navigation:
                        if nav_target in screen_ids:
                            reachable.add(nav_target)
            
            orphaned = screen_ids - reachable
            if orphaned:
                self.warnings.append(ValidationWarning(
                    level="warning",
                    component="navigation",
                    message=f"Unreachable screens: {', '.join(sorted(orphaned))}",
                    suggestion="Add navigation routes to make these screens accessible"
                ))
    
    async def _validate_state_management(self, architecture: ArchitectureDesign) -> None:
        """Validate state management"""
        
        if len(architecture.state_management) == 0:
            self.warnings.append(ValidationWarning(
                level="info",
                component="state",
                message="No state management defined",
                suggestion="Consider if the app needs to maintain any state"
            ))
            return
        
        # Check for duplicate state names
        state_names = [s.name for s in architecture.state_management]
        duplicates = [name for name in state_names if state_names.count(name) > 1]
        
        if duplicates:
            self.warnings.append(ValidationWarning(
                level="error",
                component="state",
                message=f"Duplicate state variable names: {set(duplicates)}",
                suggestion="Use unique names for each state variable"
            ))
        
        # Check state scope consistency
        for state in architecture.state_management:
            if state.scope == "component" and state.type == "global-state":
                self.warnings.append(ValidationWarning(
                    level="error",
                    component="state",
                    message=f"Inconsistent state '{state.name}': component-scoped but global-state",
                    suggestion="Change scope to 'global' or type to 'local-state'"
                ))
    
    async def _validate_performance(self, architecture: ArchitectureDesign) -> None:
        """Validate performance considerations"""
        
        total_components = sum(len(s.components) for s in architecture.screens)
        
        if total_components > 100:
            self.warnings.append(ValidationWarning(
                level="warning",
                component="performance",
                message=f"High total component count ({total_components})",
                suggestion="Consider simplifying the UI or using pagination"
            ))
        
        for screen in architecture.screens:
            if len(screen.components) > 20:
                self.warnings.append(ValidationWarning(
                    level="warning",
                    component=f"screen:{screen.id}",
                    message=f"Screen '{screen.name}' has many components ({len(screen.components)})",
                    suggestion="Consider breaking into multiple screens"
                ))
    
    async def _validate_user_experience(self, architecture: ArchitectureDesign) -> None:
        """Validate UX considerations"""
        
        # Check for input validation
        has_inputs = any(
            any(c in screen.components for c in ['InputText', 'TextArea'])
            for screen in architecture.screens
        )
        
        if has_inputs:
            has_input_state = any(
                'input' in s.name.lower() or 'text' in s.name.lower() or 'value' in s.name.lower()
                for s in architecture.state_management
            )
            
            if not has_input_state:
                self.warnings.append(ValidationWarning(
                    level="info",
                    component="ux",
                    message="App has inputs but no obvious input state",
                    suggestion="Add state variables to store user input"
                ))
    
    async def _validate_heuristic_specific(self, architecture: ArchitectureDesign) -> None:
        """
        Special validation for heuristic-generated architectures.
        
        Heuristic architectures are valid by design, but we add
        informational warnings about their limitations.
        """
        
        logger.debug(
            "validation.heuristic.checking",
            extra={"app_type": architecture.app_type}
        )
        
        self.warnings.append(ValidationWarning(
            level="info",
            component="generation",
            message="Architecture generated using heuristic fallback",
            suggestion="Template-based architecture. Consider refining the prompt for better results."
        ))
        
        # Check if it's a generic template
        if any(screen.name == "Main Screen" for screen in architecture.screens):
            self.warnings.append(ValidationWarning(
                level="info",
                component="generation",
                message="Generic template used (pattern not recognized)",
                suggestion="Use more specific keywords in your prompt (e.g., 'counter', 'todo', 'calculator')"
            ))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.stats['total_validations']
        
        return {
            **self.stats,
            'pass_rate': (self.stats['passed'] / total * 100) if total > 0 else 0,
            'heuristic_rate': (self.stats['heuristic_validated'] / total * 100) if total > 0 else 0
        }


# Global validator instance
architecture_validator = ArchitectureValidator()