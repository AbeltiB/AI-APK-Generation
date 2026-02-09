"""
Domain-Aware Intent Configuration System
Supports ANY app type with dynamic categorization
"""
from enum import Enum
from typing import Dict, List, Set, Tuple
import re


class IntentCategory(str, Enum):
    """High-level intent categories"""
    CREATE_NEW = "create_new"
    MODIFY_EXISTING = "modify_existing"
    EXTEND_FEATURES = "extend_features"
    CLARIFY_REQUEST = "clarify_request"
    TECHNICAL_HELP = "technical_help"
    UNSUPPORTED = "unsupported"


class AppDomain(str, Enum):
    """Application domains/categories"""
    PRODUCTIVITY = "productivity"
    ENTERTAINMENT = "entertainment"
    UTILITY = "utility"
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTH_FITNESS = "health_fitness"
    FINANCE = "finance"
    DEVELOPMENT = "development"
    IOT_HARDWARE = "iot_hardware"
    CREATIVE_MEDIA = "creative_media"
    DATA_SCIENCE = "data_science"
    CUSTOM = "custom"


class ComplexityLevel(str, Enum):
    """Complexity levels with technical implications"""
    SIMPLE_UI = "simple_ui"          # Basic UI, no backend
    DATA_DRIVEN = "data_driven"      # Local data, simple logic
    INTEGRATED = "integrated"        # Multiple screens, APIs
    ENTERPRISE = "enterprise"        # Complex backend, auth, payments
    HARDWARE = "hardware"            # Device control, Bluetooth, USB
    AI_ML = "ai_ml"                  # Machine learning, computer vision


# Domain-Specific Pattern Database
DOMAIN_PATTERNS: Dict[AppDomain, Dict[str, List[str]]] = {
    AppDomain.PRODUCTIVITY: {
        "keywords": ["todo", "calendar", "notes", "reminder", "schedule", "task"],
        "components": ["list", "calendar", "input", "checkbox", "datepicker"],
        "features": ["sync", "notifications", "sharing", "search"]
    },
    
    AppDomain.ENTERTAINMENT: {
        "keywords": ["music", "video", "game", "stream", "player", "media"],
        "components": ["video_player", "audio_player", "carousel", "rating"],
        "features": ["playback", "streaming", "download", "playlists"]
    },
    
    AppDomain.UTILITY: {
        "keywords": ["calculator", "converter", "scanner", "qr", "barcode", "measure"],
        "components": ["input", "output", "button_grid", "result_display"],
        "features": ["calculation", "conversion", "scanning", "history"]
    },
    
    AppDomain.BUSINESS: {
        "keywords": ["inventory", "pos", "crm", "invoice", "payment", "ecommerce"],
        "components": ["table", "form", "chart", "invoice_template"],
        "features": ["payments", "inventory", "reporting", "multi_user"]
    },
    
    AppDomain.EDUCATION: {
        "keywords": ["learn", "course", "quiz", "flashcard", "tutorial", "language"],
        "components": ["quiz", "progress_tracker", "lesson_viewer", "scoreboard"],
        "features": ["progress_tracking", "scoring", "content_delivery"]
    },
    
    AppDomain.HEALTH_FITNESS: {
        "keywords": ["fitness", "workout", "health", "tracker", "diet", "nutrition"],
        "components": ["chart", "timer", "counter", "progress_bar", "calendar"],
        "features": ["tracking", "charts", "reminders", "goals"]
    },
    
    AppDomain.FINANCE: {
        "keywords": ["bank", "budget", "expense", "investment", "stock", "crypto"],
        "components": ["chart", "table", "calculator", "transaction_list"],
        "features": ["calculations", "charts", "security", "sync"]
    },
    
    AppDomain.DEVELOPMENT: {
        "keywords": ["code", "editor", "git", "terminal", "api", "debug"],
        "components": ["code_editor", "terminal", "file_browser", "output_panel"],
        "features": ["syntax_highlighting", "execution", "debugging", "git"]
    },
    
    AppDomain.IOT_HARDWARE: {
        "keywords": ["drone", "printer", "sensor", "bluetooth", "arduino", "raspberry"],
        "components": ["controls", "status_panel", "connection_indicator", "log_viewer"],
        "features": ["device_control", "real_time_updates", "connection_management", "firmware"]
    },
    
    AppDomain.CREATIVE_MEDIA: {
        "keywords": ["photo", "video", "edit", "design", "3d", "draw", "art"],
        "components": ["canvas", "toolbar", "preview", "layer_panel", "color_picker"],
        "features": ["editing", "filters", "export", "undo_redo"]
    },
    
    AppDomain.DATA_SCIENCE: {
        "keywords": ["data", "analyze", "visualize", "chart", "graph", "dataset"],
        "components": ["chart", "data_table", "filter_panel", "statistics_panel"],
        "features": ["data_import", "analysis", "visualization", "export"]
    }
}


# Hardware/Device Control Patterns
HARDWARE_PATTERNS = {
    "drone": {
        "keywords": ["drone", "quadcopter", "uav", "flight", "fly", "controller", "fpv"],
        "required_features": ["real_time_control", "video_stream", "telemetry", "gps"],
        "special_components": ["joystick_control", "map_view", "battery_indicator", "status_panel"],
        "complexity": ComplexityLevel.HARDWARE
    },
    
    "3d_printer": {
        "keywords": ["3d printer", "print", "filament", "gcode", "slicer", "extruder"],
        "required_features": ["file_upload", "print_control", "temperature_monitoring", "progress_tracking"],
        "special_components": ["model_preview", "temperature_chart", "control_panel", "file_manager"],
        "complexity": ComplexityLevel.HARDWARE
    },
    
    "iot_device": {
        "keywords": ["smart home", "iot", "sensor", "device", "automation", "control"],
        "required_features": ["device_pairing", "real_time_updates", "automation_rules", "notifications"],
        "special_components": ["device_list", "control_card", "automation_editor", "dashboard"],
        "complexity": ComplexityLevel.INTEGRATED
    }
}


# AI/ML Patterns
AI_PATTERNS = {
    "image_processing": {
        "keywords": ["convert image", "image to 3d", "3d model", "scan", "reconstruct", "point cloud"],
        "required_features": ["image_upload", "processing", "3d_preview", "export"],
        "special_components": ["image_uploader", "3d_viewer", "parameter_controls", "progress_indicator"],
        "complexity": ComplexityLevel.AI_ML
    },
    
    "machine_learning": {
        "keywords": ["train model", "predict", "classify", "detect", "recognize", "ai"],
        "required_features": ["data_upload", "model_training", "prediction", "results_visualization"],
        "special_components": ["data_uploader", "training_progress", "results_chart", "model_selector"],
        "complexity": ComplexityLevel.AI_ML
    }
}


# Technical Requirements by Domain
TECHNICAL_REQUIREMENTS = {
    ComplexityLevel.HARDWARE: {
        "apis_needed": ["bluetooth", "websockets", "serial", "usb"],
        "permissions": ["bluetooth", "location", "camera"],
        "platform_considerations": ["low_latency", "real_time", "background_operations"],
        "risk_level": "high"
    },
    
    ComplexityLevel.AI_ML: {
        "apis_needed": ["tensorflow.js", "webgl", "webworkers", "file_system"],
        "permissions": ["camera", "storage"],
        "platform_considerations": ["performance", "memory", "processing_power"],
        "risk_level": "medium"
    },
    
    ComplexityLevel.ENTERPRISE: {
        "apis_needed": ["rest_apis", "websockets", "database", "authentication"],
        "permissions": ["internet", "storage", "camera"],
        "platform_considerations": ["security", "scalability", "offline_support"],
        "risk_level": "medium"
    },
    
    ComplexityLevel.INTEGRATED: {
        "apis_needed": ["rest_apis", "local_storage", "camera", "location"],
        "permissions": ["internet", "storage"],
        "platform_considerations": ["network_status", "caching", "error_handling"],
        "risk_level": "low"
    }
}


# Domain to Template Mapping
DOMAIN_TEMPLATES = {
    AppDomain.IOT_HARDWARE: {
        "architecture_template": "hardware_control",
        "layout_template": "device_dashboard",
        "blockly_template": "hardware_events"
    },
    
    AppDomain.CREATIVE_MEDIA: {
        "architecture_template": "creative_tool",
        "layout_template": "editor_workspace",
        "blockly_template": "media_processing"
    },
    
    AppDomain.DATA_SCIENCE: {
        "architecture_template": "data_analysis",
        "layout_template": "data_dashboard",
        "blockly_template": "data_pipeline"
    },
    
    "custom": {
        "architecture_template": "custom_app",
        "layout_template": "custom_layout",
        "blockly_template": "custom_logic"
    }
}


class IntentConfig:
    """Dynamic intent configuration that adapts to ANY app type"""
    
    @staticmethod
    def detect_domain(prompt_lower: str) -> Tuple[AppDomain, str, float]:
        """
        Detect application domain and specific type with confidence
        
        Returns: (domain, specific_type, confidence)
        """
        # Check hardware/device patterns first
        for device_type, patterns in HARDWARE_PATTERNS.items():
            if any(keyword in prompt_lower for keyword in patterns["keywords"]):
                return (AppDomain.IOT_HARDWARE, device_type, 0.9)
        
        # Check AI/ML patterns
        for ai_type, patterns in AI_PATTERNS.items():
            if any(keyword in prompt_lower for keyword in patterns["keywords"]):
                return (AppDomain.CREATIVE_MEDIA, ai_type, 0.85)
        
        # Check domain patterns
        best_match = (AppDomain.CUSTOM, "generic", 0.5)
        best_score = 0
        
        for domain, patterns in DOMAIN_PATTERNS.items():
            keywords = patterns.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in prompt_lower)
            
            if matches > 0:
                score = matches / len(keywords) if keywords else 0
                if score > best_score:
                    best_score = score
                    specific_type = keywords[0] if keywords else "generic"
                    best_match = (domain, specific_type, min(0.9, 0.5 + score))
        
        return best_match
    
    @staticmethod
    def extract_special_requirements(prompt: str, domain: AppDomain, app_type: str) -> Dict:
        """Extract special requirements for complex app types"""
        requirements = {
            "needs_hardware": False,
            "needs_ai_ml": False,
            "needs_real_time": False,
            "needs_3d": False,
            "special_apis": [],
            "complex_components": []
        }
        
        prompt_lower = prompt.lower()
        
        # Hardware/device requirements
        if domain == AppDomain.IOT_HARDWARE:
            requirements["needs_hardware"] = True
            requirements["needs_real_time"] = True
            
            if app_type in HARDWARE_PATTERNS:
                reqs = HARDWARE_PATTERNS[app_type]
                requirements["special_apis"] = ["bluetooth", "websockets", "serial_api"]
                requirements["complex_components"] = reqs.get("special_components", [])
        
        # AI/ML requirements
        if domain == AppDomain.CREATIVE_MEDIA and app_type in AI_PATTERNS:
            requirements["needs_ai_ml"] = True
            reqs = AI_PATTERNS[app_type]
            requirements["special_apis"] = ["tensorflow.js", "webgl", "webworkers"]
            requirements["complex_components"] = reqs.get("special_components", [])
        
        # 3D requirements
        if any(word in prompt_lower for word in ["3d", "three.js", "webgl", "model", "mesh"]):
            requirements["needs_3d"] = True
            requirements["special_apis"].append("webgl")
            requirements["special_apis"].append("three.js")
        
        # Real-time requirements
        if any(word in prompt_lower for word in ["real-time", "live", "stream", "control", "telemetry"]):
            requirements["needs_real_time"] = True
            requirements["special_apis"].append("websockets")
        
        return requirements
    
    @staticmethod
    def get_complexity_level(prompt: str, domain: AppDomain, app_type: str) -> ComplexityLevel:
        """Determine complexity level based on domain and requirements"""
        
        # Check for known complex types
        if domain == AppDomain.IOT_HARDWARE:
            return ComplexityLevel.HARDWARE
        
        if domain == AppDomain.CREATIVE_MEDIA and app_type in AI_PATTERNS:
            return ComplexityLevel.AI_ML
        
        # Check for enterprise features
        enterprise_keywords = ["payment", "bank", "invoice", "crm", "erp", "multi-tenant"]
        if any(word in prompt.lower() for word in enterprise_keywords):
            return ComplexityLevel.ENTERPRISE
        
        # Check for integrations
        integration_keywords = ["api", "database", "backend", "sync", "cloud"]
        if sum(1 for word in integration_keywords if word in prompt.lower()) >= 2:
            return ComplexityLevel.INTEGRATED
        
        # Check for data features
        data_keywords = ["chart", "graph", "data", "analyze", "report"]
        if any(word in prompt.lower() for word in data_keywords):
            return ComplexityLevel.DATA_DRIVEN
        
        # Default to simple UI
        return ComplexityLevel.SIMPLE_UI
    
    @staticmethod
    def get_template_for_domain(domain: AppDomain, app_type: str) -> Dict:
        """Get appropriate templates for the domain"""
        if domain in DOMAIN_TEMPLATES:
            return DOMAIN_TEMPLATES[domain]
        
        # Check if it's a hardware type
        if app_type in HARDWARE_PATTERNS:
            return DOMAIN_TEMPLATES[AppDomain.IOT_HARDWARE]
        
        # Check if it's an AI type
        if app_type in AI_PATTERNS:
            return DOMAIN_TEMPLATES[AppDomain.CREATIVE_MEDIA]
        
        # Default to custom
        return DOMAIN_TEMPLATES["custom"]