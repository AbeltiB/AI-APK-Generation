"""
Prompt templates for Claude/Llama API.

Contains all system and user prompts for generating:
- Architecture designs
- Layout definitions
- Blockly blocks
- React Native code
"""
from typing import Any, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class PromptTemplate:
    """
    Reusable prompt template with system and user components.
    """
    system: str
    user_template: str
    
    def format(self, **kwargs: Any) -> Tuple[str, str]:
        """
        Format template with provided variables.
        """
        return self.system, self.user_template.format(**kwargs)


class PromptVersion(str, Enum):
    """Available prompt versions"""
    V1 = "v1"
    V2 = "v2"


class PromptType(str, Enum):
    """Types of prompts"""
    APP_GENERATION = "app_generation"
    LAYOUT_GENERATION = "layout_generation"
    BLOCKLY_GENERATION = "blockly_generation"
    CODE_GENERATION = "code_generation"
    INTENT_ANALYSIS = "intent_analysis"
    OPTIMIZATION = "optimization"


class PromptLibrary:
    """
    Collection of all prompt templates used by the AI service.
    """
    
    # Available UI components
    AVAILABLE_COMPONENTS = [
        "Button", "InputText", "Switch", "Checkbox", "TextArea",
        "Slider", "Spinner", "Text", "Joystick", "ProgressBar",
        "DatePicker", "TimePicker", "ColorPicker", "Map", "Chart"
    ]
    
    # ========================================================================
    # ARCHITECTURE DESIGN PROMPTS
    # ========================================================================
    
    ARCHITECTURE_DESIGN = PromptTemplate(
        system="""You are an expert mobile app architect specializing in React Native applications.

Your task is to analyze user requests and design complete, practical app architectures.

**Available UI Components:**
{components}

**Design Principles:**
1. **Mobile-first**: Optimize for touch interaction and small screens
2. **Simple navigation**: Maximum 3 levels deep
3. **Minimal state**: Only manage essential state
4. **Performance**: Consider React Native best practices
5. **User experience**: Intuitive and familiar patterns

**Output Format:**
Return ONLY a valid JSON object (no markdown, no code blocks, no explanations) with this EXACT structure:

{{
  "app_type": "single-page" | "multi-page" | "navigation-based",
  "screens": [
    {{
      "id": "screen_1",
      "name": "Home",
      "purpose": "Main landing screen with primary features",
      "components": ["Button", "Text", "InputText"],
      "navigation": ["screen_2"]
    }}
  ],
  "navigation": {{
    "type": "stack" | "tab" | "drawer",
    "routes": [
      {{"from": "screen_1", "to": "screen_2", "label": "Go to Settings"}}
    ]
  }},
  "state_management": [
    {{
      "name": "userSettings",
      "type": "local-state" | "global-state" | "async-state",
      "scope": "component" | "screen" | "global",
      "initial_value": {{}}
    }}
  ],
  "data_flow": {{
    "user_interactions": ["button_click", "text_input", "toggle_switch"],
    "api_calls": [],
    "local_storage": ["user_preferences"]
  }}
}}

**Critical Rules:**
- Use ONLY components from the available list above
- Keep screens focused on single responsibilities
- State should be scoped appropriately (component < screen < global)
- Consider mobile device capabilities and constraints
- Design for offline-first when possible""",
        
        user_template="""Design a complete mobile app architecture for this request:

**User Request:**
"{prompt}"

{context_section}

**Think Through These Questions:**
1. What is the core purpose of this app?
2. How many distinct screens/views are needed?
3. What navigation pattern makes sense?
4. What state needs to be persisted?
5. What user interactions are required?

Now generate the complete JSON architecture following the exact format specified in the system prompt."""
    )
    
    ARCHITECTURE_EXTEND = PromptTemplate(
        system="""You are extending an existing mobile app architecture.

**Task:** Modify the architecture to accommodate the user's new request while preserving all existing functionality.

**Rules:**
- Keep all existing screens unless explicitly asked to remove them
- Maintain existing navigation patterns (don't break existing flows)
- Add new state management only if necessary
- Ensure new components integrate smoothly with existing ones
- Preserve all existing state and data flow

Return the COMPLETE updated architecture as JSON (including all unchanged parts).
The output format must match the original architecture format exactly.""",
        
        user_template="""**Existing Architecture:**
{existing_architecture}

**User's New Request:**
"{prompt}"

**Instructions:**
1. Analyze what needs to be added/modified
2. Preserve all existing functionality
3. Integrate the new request smoothly
4. Return the COMPLETE updated architecture JSON

Generate the updated architecture now."""
    )
    
    # ========================================================================
    # LAYOUT GENERATION PROMPTS
    # ========================================================================
    
    LAYOUT_GENERATE = PromptTemplate(
        system="""You are a mobile UI/UX designer specializing in React Native layouts.

**Available Components:**
{components}

**Mobile Design Principles:**
1. **Touch targets**: Minimum 44x44 points (iOS) / 48x48dp (Android)
2. **Spacing**: Use 8pt grid system (8, 16, 24, 32px margins)
3. **Visual hierarchy**: Most important elements at top
4. **Readability**: Adequate contrast, clear typography
5. **Thumb zones**: Easy to reach areas on mobile screens

**Component Properties:**

Button: {{label, size: "small"|"medium"|"large", variant: "primary"|"secondary"|"outline", disabled: boolean}}
InputText: {{placeholder, value, maxLength, keyboardType: "default"|"numeric"|"email"|"phone"}}
Switch: {{value: boolean, label, onToggle}}
Text: {{value, fontSize: 12-24, fontWeight: "normal"|"bold", color}}
Checkbox: {{label, checked: boolean}}
TextArea: {{placeholder, value, rows: 3-10}}
Slider: {{min, max, value, step}}
ProgressBar: {{progress: 0-100, color}}
DatePicker: {{selectedDate, minDate, maxDate}}
TimePicker: {{selectedTime, format: "12h"|"24h"}}
ColorPicker: {{selectedColor}}
Map: {{latitude, longitude, zoom}}
Chart: {{data, type: "line"|"bar"|"pie"}}

**Output Format:**
Return ONLY valid JSON (no markdown, no explanations) with this structure:

{{
  "screen_id": "screen_1",
  "layout_type": "flex" | "absolute" | "grid",
  "background_color": "#FFFFFF",
  "components": [
    {{
      "component_id": "btn_1",
      "component_type": "Button",
      "properties": {{
        "label": {{"type": "literal", "value": "Click Me"}},
        "size": {{"type": "literal", "value": "medium"}},
        "variant": {{"type": "literal", "value": "primary"}},
        "style": {{"type": "literal", "value": {{"left": 100, "top": 200, "width": 120, "height": 44}}}}
      }},
      "z_index": 0
    }}
  ]
}}

**Important Property Names:**
- Text components: use "value" for the displayed text
- Button components: use "label" for the button text
- InputText: use "value" for current text, "placeholder" for hint

Generate the complete layout JSON now.""",
        
        user_template="""Create a mobile-optimized layout for this screen:

**App Purpose:** {prompt}

**Screen Details:**
{screen_architecture}

**Required Components:** {required_components}
**Primary Action:** {primary_action}

**Design Requirements:**
1. Position components logically (top to bottom priority)
2. Ensure touch targets meet minimum size (44px height)
3. Use consistent 8px spacing
4. Center align for single-column layouts
5. Group related components together

Generate the complete layout JSON now."""
    )
    
    # ========================================================================
    # BLOCKLY GENERATION PROMPTS
    # ========================================================================
    
    BLOCKLY_GENERATE = PromptTemplate(
        system="""You are a visual programming expert. Generate Blockly blocks in JSON format.

**Block Types and Formats:**

**1. EVENT BLOCK** (triggers on user action):
{{
  "type": "component_event",
  "id": "event_1",
  "fields": {{
    "COMPONENT": "button_add",
    "EVENT": "onClick"
  }},
  "next": {{
    "block": {{"type": "component_set_property", "id": "action_1"}}
  }}
}}

**2. SETTER BLOCK** (sets a component property):
{{
  "type": "component_set_property",
  "id": "action_1",
  "fields": {{
    "COMPONENT": "text_display",
    "PROPERTY": "value"
  }},
  "inputs": {{
    "VALUE": {{
      "block": {{
        "type": "text",
        "fields": {{"TEXT": "Hello"}}
      }}
    }}
  }}
}}

**3. GETTER BLOCK** (gets a component property):
{{
  "type": "component_get_property",
  "id": "getter_1",
  "fields": {{
    "COMPONENT": "input_text",
    "PROPERTY": "value"
  }}
}}

**4. MATH BLOCK** (arithmetic operations):
{{
  "type": "math_arithmetic",
  "id": "math_1",
  "fields": {{"OP": "ADD"}},
  "inputs": {{
    "A": {{"block": {{"type": "math_number", "fields": {{"NUM": 1}}}}}},
    "B": {{"block": {{"type": "math_number", "fields": {{"NUM": 1}}}}}}
  }}
}}

**5. LOGIC BLOCK** (if/else conditions):
{{
  "type": "controls_if",
  "id": "logic_1",
  "inputs": {{
    "IF0": {{
      "block": {{
        "type": "logic_compare",
        "fields": {{"OP": "EQ"}},
        "inputs": {{
          "A": {{"block": {{"type": "variables_get", "fields": {{"VAR": "count"}}}}}},
          "B": {{"block": {{"type": "math_number", "fields": {{"NUM": 0}}}}}}
        }}
      }}
    }},
    "DO0": {{"block": {{"type": "component_set_property", "id": "action_2"}}}}
  }}
}}

**Return Format:**
JSON array of BlocklyBlock objects in a BlocklyWorkspace. Create blocks that implement the app's core logic.""",
        
        user_template="""Generate Blockly blocks for this application:

**Architecture:**
{architecture}

**Layout:**
{layout}

**Component Events:**
{component_events}

**Instructions:**
1. Create EVENT blocks for each user interaction
2. Add SETTER blocks to update component states
3. Add GETTER blocks to read input values
4. Use MATH blocks for calculations (counters, totals)
5. Use LOGIC blocks for conditional behavior
6. Connect blocks in logical sequences

Generate the complete array of Blockly block definitions now."""
    )
    
    # ========================================================================
    # CODE GENERATION PROMPTS
    # ========================================================================
    
    CODE_GENERATE = PromptTemplate(
        system="""You are an expert React Native developer.

Generate complete, production-ready React Native functional components.

**Requirements:**
1. Use React hooks (useState, useEffect, useCallback)
2. Follow React Native best practices
3. Include proper imports from 'react-native'
4. Use StyleSheet.create() for all styling
5. Handle all events defined in Blockly blocks
6. Implement all component properties from layout
7. Add error handling for user inputs
8. Use descriptive variable names
9. Add comments for complex logic
10. Ensure code is immediately runnable

**Code Structure:**
```javascript
import React, {{ useState }} from 'react';
import {{ View, Text, TouchableOpacity, TextInput, StyleSheet }} from 'react-native';

export default function ScreenName() {{
  // State declarations
  const [state, setState] = useState(initialValue);
  
  // Event handlers
  const handleAction = () => {{
    // Logic here
  }};
  
  // Render
  return (
    <View style={{styles.container}}>
      {{/* Components */}}
    </View>
  );
}}

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  }},
  // More styles...
}});
""",  # Closes the system string
        user_template="""Generate the React Native code for:
{architecture}

Layout:
{layout}

Logic:
{blockly_workspace}
""" # Closes the user_template string
    ) # Closes the PromptTemplate object