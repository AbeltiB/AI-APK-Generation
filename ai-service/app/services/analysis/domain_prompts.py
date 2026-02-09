"""
Specialized prompts for complex domains like drones, 3D printing, AI, etc.
"""
from enum import Enum
from typing import Dict, List


class DomainPromptType(str, Enum):
    """Types of domain-specific prompts"""
    HARDWARE_CONTROL = "hardware_control"
    AI_PROCESSING = "ai_processing"
    REAL_TIME_DASHBOARD = "real_time_dashboard"
    DATA_VISUALIZATION = "data_visualization"
    DEVICE_MANAGEMENT = "device_management"


class DomainPromptManager:
    """Manager for domain-specific prompts"""
    
    @staticmethod
    def get_hardware_control_prompt(device_type: str) -> str:
        """Get prompt for hardware control applications"""
        
        prompts = {
            "drone": """You are designing a drone control application.

Key features needed:
1. Real-time flight controls (pitch, roll, yaw, throttle)
2. Video streaming from drone camera
3. GPS navigation and waypoint planning
4. Battery and telemetry monitoring
5. Flight status and alerts

Architecture considerations:
- Low-latency communication (WebSockets/WebRTC)
- Real-time data processing
- Connection state management
- Safety features (return-to-home, geofencing)

Design a responsive UI with:
- Joystick controls or touch gestures
- Map view with drone location
- Video feed display
- Telemetry dashboard
- Settings for flight parameters""",
            
            "3d_printer": """You are designing a 3D printer control application.

Key features needed:
1. File upload and management (G-code, STL files)
2. Print job control (start, pause, stop, resume)
3. Real-time temperature monitoring (nozzle, bed)
4. Print progress visualization
5. Manual printer controls (movement, temperature)

Architecture considerations:
- Serial/USB/network communication
- Real-time status updates
- File processing and slicing integration
- Error handling and recovery

Design an intuitive UI with:
- 3D model preview
- Temperature charts
- Control panel for manual operations
- Print job queue
- Printer settings and calibration""",
            
            "iot_device": """You are designing an IoT device control application.

Key features needed:
1. Device discovery and pairing
2. Real-time sensor data visualization
3. Device control and automation rules
4. Historical data logging
5. Alerts and notifications

Architecture considerations:
- Multiple communication protocols (MQTT, HTTP, WebSockets)
- Device state management
- Rule engine for automations
- Data persistence and synchronization

Design a dashboard UI with:
- Device cards with status indicators
- Real-time charts and graphs
- Automation rule editor
- Device settings panel
- Notification center"""
        }
        
        return prompts.get(device_type, """You are designing a hardware control application.

Provide architecture, layout, and logic for controlling a hardware device.
Focus on real-time communication, device state management, and user safety.""")
    
    @staticmethod
    def get_ai_processing_prompt(ai_type: str) -> str:
        """Get prompt for AI/ML processing applications"""
        
        prompts = {
            "image_to_3d": """You are designing an image-to-3D conversion application.

Key features needed:
1. Image upload and preprocessing
2. 3D model generation from 2D images
3. 3D model preview and manipulation
4. Model export in various formats (STL, OBJ, GLTF)
5. Processing progress tracking

Architecture considerations:
- Client-side or server-side AI processing
- Large file handling and processing
- Real-time progress updates
- 3D rendering performance

Design a creative UI with:
- Image upload and preview area
- 3D viewer with rotation/zoom controls
- Processing parameters panel
- Export options
- Progress indicator""",
            
            "ai_model_trainer": """You are designing an AI model training application.

Key features needed:
1. Dataset upload and preprocessing
2. Model configuration and training
3. Training progress monitoring
4. Model evaluation and visualization
5. Model export and deployment

Architecture considerations:
- Distributed training support
- GPU/CPU resource management
- Experiment tracking
- Model versioning

Design a data science UI with:
- Dataset preview and statistics
- Model configuration forms
- Training progress charts
- Evaluation metrics visualization
- Model deployment options"""
        }
        
        return prompts.get(ai_type, """You are designing an AI/ML application.

Provide architecture, layout, and logic for AI-powered processing.
Focus on data handling, processing pipelines, and result visualization.""")
    
    @staticmethod
    def get_real_time_dashboard_prompt() -> str:
        """Get prompt for real-time dashboard applications"""
        return """You are designing a real-time monitoring dashboard.

Key features needed:
1. Real-time data streaming and updates
2. Live charts and visualizations
3. Alert system for threshold breaches
4. Historical data comparison
5. Multi-view layouts for different data types

Architecture considerations:
- WebSocket connections for real-time data
- Data aggregation and processing
- State management for live updates
- Offline data caching

Design a responsive dashboard with:
- Real-time charts and gauges
- Alert notifications panel
- Data filtering and time range selection
- Multiple view layouts (grid, single, split)
- Export and sharing options"""
    
    @staticmethod
    def get_domain_architecture_prompt(domain: str, app_type: str) -> str:
        """Get architecture prompt for specific domain"""
        
        if "drone" in app_type.lower():
            return DomainPromptManager.get_hardware_control_prompt("drone")
        elif "printer" in app_type.lower() or "3d" in app_type.lower():
            return DomainPromptManager.get_hardware_control_prompt("3d_printer")
        elif "image" in app_type.lower() and "3d" in app_type.lower():
            return DomainPromptManager.get_ai_processing_prompt("image_to_3d")
        elif "ai" in app_type.lower() or "ml" in app_type.lower():
            return DomainPromptManager.get_ai_processing_prompt("ai_model_trainer")
        elif "real-time" in app_type.lower() or "dashboard" in app_type.lower():
            return DomainPromptManager.get_real_time_dashboard_prompt()
        
        # Default domain-aware prompt
        return f"""You are designing a {app_type} application in the {domain} domain.

Consider domain-specific requirements and best practices.
Provide architecture, UI design, and logic that addresses the unique needs of this domain."""