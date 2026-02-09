"""
Domain-Aware Intent Analyzer - Handles ANY app type including hardware, AI, and specialized domains
"""
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from app.llm.orchestrator import LLMOrchestrator
from app.llm.prompt_manager import PromptManager, PromptType, PromptVersion
from app.llm.base import LLMMessage
from app.services.intent.intent_config import (
    IntentConfig, AppDomain, ComplexityLevel,
    HARDWARE_PATTERNS, AI_PATTERNS, DOMAIN_PATTERNS,
    TECHNICAL_REQUIREMENTS
)
from app.utils.logging import get_logger, log_context

logger = get_logger(__name__)


class SpecializedAppType(str, Enum):
    """Specialized app types that need custom handling"""
    DRONE_CONTROLLER = "drone_controller"
    PRINTER_CONTROL = "printer_control"
    IOT_DASHBOARD = "iot_dashboard"
    IMAGE_TO_3D = "image_to_3d"
    AI_MODEL_TRAINER = "ai_model_trainer"
    DATA_VISUALIZATION = "data_visualization"
    REAL_TIME_MONITOR = "real_time_monitor"
    CUSTOM_HARDWARE = "custom_hardware"


class DomainAwareIntentAnalyzer:
    """
    Advanced intent analyzer that understands:
    - Hardware control apps (drones, 3D printers, IoT)
    - AI/ML applications
    - Real-time systems
    - Specialized domains
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with enhanced capabilities
        
        Args:
            config: LLM configuration dictionary
        """
        self.orchestrator = LLMOrchestrator(config)
        self.prompt_manager = PromptManager(default_version=PromptVersion.V3)
        
        # Domain knowledge database
        self.domain_knowledge = self._load_domain_knowledge()
        
        # Cache for domain-specific analyses
        self.domain_cache: Dict[str, tuple[Dict[str, Any], datetime]] = {}
        self.cache_ttl = 600  # 10 minutes for domain analysis
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'hardware_apps': 0,
            'ai_ml_apps': 0,
            'real_time_apps': 0,
            'custom_domains': 0,
            'complexity_distribution': {
                'simple_ui': 0,
                'data_driven': 0,
                'integrated': 0,
                'enterprise': 0,
                'hardware': 0,
                'ai_ml': 0
            }
        }
        
        logger.info(
            "Domain-aware intent analyzer initialized",
            extra={
                "domains_supported": len(DOMAIN_PATTERNS),
                "hardware_types": len(HARDWARE_PATTERNS),
                "ai_types": len(AI_PATTERNS)
            }
        )
    
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain-specific knowledge and patterns"""
        return {
            "hardware_keywords": self._extract_all_keywords(HARDWARE_PATTERNS),
            "ai_keywords": self._extract_all_keywords(AI_PATTERNS),
            "domain_keywords": self._extract_all_keywords(DOMAIN_PATTERNS),
            "technical_terms": self._load_technical_vocabulary()
        }
    
    def _extract_all_keywords(self, patterns_dict: Dict) -> List[str]:
        """Extract all keywords from patterns dictionary"""
        keywords = []
        for patterns in patterns_dict.values():
            if isinstance(patterns, dict) and "keywords" in patterns:
                keywords.extend(patterns["keywords"])
        return list(set(keywords))
    
    def _load_technical_vocabulary(self) -> Dict[str, List[str]]:
        """Load technical vocabulary for specialized domains"""
        return {
            "drone_terms": [
                "pitch", "roll", "yaw", "throttle", "gimbal", "telemetry",
                "waypoint", "autopilot", "fpv", "gps", "altitude", "battery"
            ],
            "3d_printing_terms": [
                "extruder", "hotend", "bed", "filament", "gcode", "slicer",
                "infill", "supports", "raft", "brim", "layer_height", "nozzle"
            ],
            "ai_ml_terms": [
                "model", "training", "inference", "dataset", "epoch", "batch",
                "neural network", "convolution", "tensor", "prediction", "accuracy"
            ],
            "iot_terms": [
                "mqtt", "bluetooth", "wifi", "sensor", "actuator", "protocol",
                "firmware", "ota", "device_id", "pairing", "certificate"
            ]
        }
    
    async def analyze_with_domain(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced intent analysis with domain awareness
        
        Args:
            prompt: User's natural language request
            context: Optional context (user expertise, existing hardware, etc.)
            
        Returns:
            Enhanced intent analysis with domain-specific details
        """
        self.stats['total_analyses'] += 1
        start_time = datetime.now()
        
        with log_context(operation="domain_intent_analysis"):
            logger.info(
                "Domain-aware intent analysis started",
                extra={
                    "prompt": prompt[:100],
                    "context_keys": list(context.keys()) if context else []
                }
            )
            
            try:
                # Step 1: Detect domain and app type
                domain_info = self._detect_domain_and_type(prompt, context)
                
                # Step 2: Extract specialized requirements
                special_reqs = self._extract_specialized_requirements(prompt, domain_info)
                
                # Step 3: Determine complexity with domain awareness
                complexity = self._determine_domain_complexity(prompt, domain_info, special_reqs)
                
                # Step 4: Use LLM for nuanced understanding (for complex domains)
                if complexity in [ComplexityLevel.HARDWARE, ComplexityLevel.AI_ML, ComplexityLevel.ENTERPRISE]:
                    llm_analysis = await self._analyze_with_llm_domain(
                        prompt, domain_info, special_reqs, context
                    )
                    
                    if llm_analysis:
                        # Merge domain detection with LLM insights
                        result = self._merge_analyses(domain_info, llm_analysis, special_reqs)
                        self._update_stats(domain_info, complexity, special_reqs)
                        return result
                
                # Step 5: Build comprehensive result for simpler domains
                result = self._build_domain_result(prompt, domain_info, special_reqs, complexity)
                self._update_stats(domain_info, complexity, special_reqs)
                
                logger.info(
                    "Domain analysis complete",
                    extra={
                        "domain": domain_info["domain"],
                        "app_type": domain_info["specific_type"],
                        "complexity": complexity,
                        "hardware_required": special_reqs.get("needs_hardware", False),
                        "ai_required": special_reqs.get("needs_ai_ml", False)
                    }
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    "Domain analysis failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
                
                return self._emergency_domain_fallback(prompt)
    
    def _detect_domain_and_type(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect domain and specific app type with high precision"""
        prompt_lower = prompt.lower()
        
        # First, check for hardware/device control
        hardware_detection = self._detect_hardware_app(prompt_lower)
        if hardware_detection["is_hardware"]:
            return {
                "domain": AppDomain.IOT_HARDWARE,
                "specific_type": hardware_detection["type"],
                "confidence": hardware_detection["confidence"],
                "is_specialized": True,
                "detection_method": "hardware_patterns",
                "keywords_found": hardware_detection["keywords_found"]
            }
        
        # Check for AI/ML applications
        ai_detection = self._detect_ai_app(prompt_lower)
        if ai_detection["is_ai"]:
            return {
                "domain": AppDomain.CREATIVE_MEDIA,
                "specific_type": ai_detection["type"],
                "confidence": ai_detection["confidence"],
                "is_specialized": True,
                "detection_method": "ai_patterns",
                "keywords_found": ai_detection["keywords_found"]
            }
        
        # Check standard domains
        domain, specific_type, confidence = IntentConfig.detect_domain(prompt_lower)
        
        # Enhance with context if available
        if context and context.get("user_expertise"):
            expertise = context["user_expertise"].lower()
            if any(term in expertise for term in ["hardware", "iot", "electronics", "robotics"]):
                # User has hardware expertise, might be requesting hardware app
                if confidence < 0.7:
                    domain = AppDomain.IOT_HARDWARE
                    specific_type = "custom_hardware"
                    confidence = 0.6
        
        return {
            "domain": domain,
            "specific_type": specific_type,
            "confidence": confidence,
            "is_specialized": domain in [AppDomain.IOT_HARDWARE, AppDomain.CREATIVE_MEDIA],
            "detection_method": "domain_patterns",
            "keywords_found": []
        }
    
    def _detect_hardware_app(self, prompt_lower: str) -> Dict[str, Any]:
        """Detect hardware/device control applications"""
        best_match = {"is_hardware": False, "type": "generic", "confidence": 0.0, "keywords_found": []}
        best_score = 0
        
        for hw_type, patterns in HARDWARE_PATTERNS.items():
            keywords = patterns.get("keywords", [])
            found_keywords = [kw for kw in keywords if kw in prompt_lower]
            
            if found_keywords:
                score = len(found_keywords) / len(keywords)
                
                # Bonus for technical terms
                technical_bonus = 0
                if hw_type == "drone":
                    drone_terms = self.domain_knowledge["technical_terms"]["drone_terms"]
                    technical_bonus = sum(1 for term in drone_terms if term in prompt_lower) * 0.1
                elif hw_type == "3d_printer":
                    printer_terms = self.domain_knowledge["technical_terms"]["3d_printing_terms"]
                    technical_bonus = sum(1 for term in printer_terms if term in prompt_lower) * 0.1
                
                total_score = min(0.95, 0.5 + score * 0.4 + technical_bonus)
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = {
                        "is_hardware": True,
                        "type": hw_type,
                        "confidence": total_score,
                        "keywords_found": found_keywords
                    }
        
        return best_match
    
    def _detect_ai_app(self, prompt_lower: str) -> Dict[str, Any]:
        """Detect AI/ML applications"""
        best_match = {"is_ai": False, "type": "generic", "confidence": 0.0, "keywords_found": []}
        best_score = 0
        
        for ai_type, patterns in AI_PATTERNS.items():
            keywords = patterns.get("keywords", [])
            found_keywords = [kw for kw in keywords if kw in prompt_lower]
            
            if found_keywords:
                score = len(found_keywords) / len(keywords)
                
                # Bonus for AI technical terms
                ai_terms = self.domain_knowledge["technical_terms"]["ai_ml_terms"]
                technical_bonus = sum(1 for term in ai_terms if term in prompt_lower) * 0.1
                
                total_score = min(0.95, 0.5 + score * 0.4 + technical_bonus)
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = {
                        "is_ai": True,
                        "type": ai_type,
                        "confidence": total_score,
                        "keywords_found": found_keywords
                    }
        
        return best_match
    
    def _extract_specialized_requirements(
        self,
        prompt: str,
        domain_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract specialized requirements for the domain"""
        base_reqs = IntentConfig.extract_special_requirements(prompt, domain_info["domain"], domain_info["specific_type"])
        
        # Add domain-specific enhancements
        if domain_info["domain"] == AppDomain.IOT_HARDWARE:
            base_reqs.update(self._extract_hardware_requirements(prompt, domain_info["specific_type"]))
        elif domain_info["domain"] == AppDomain.CREATIVE_MEDIA and domain_info["specific_type"] in AI_PATTERNS:
            base_reqs.update(self._extract_ai_requirements(prompt, domain_info["specific_type"]))
        
        # Add real-time requirements detection
        if self._requires_real_time(prompt):
            base_reqs["needs_real_time"] = True
            if "websockets" not in base_reqs["special_apis"]:
                base_reqs["special_apis"].append("websockets")
        
        # Add 3D visualization requirements
        if self._requires_3d_visualization(prompt):
            base_reqs["needs_3d"] = True
            if "webgl" not in base_reqs["special_apis"]:
                base_reqs["special_apis"].append("webgl")
            if "three.js" not in base_reqs["special_apis"]:
                base_reqs["special_apis"].append("three.js")
        
        return base_reqs
    
    def _extract_hardware_requirements(self, prompt: str, hw_type: str) -> Dict[str, Any]:
        """Extract hardware-specific requirements"""
        requirements = {
            "communication_protocols": [],
            "control_interfaces": [],
            "monitoring_features": [],
            "safety_features": []
        }
        
        prompt_lower = prompt.lower()
        
        if hw_type == "drone":
            requirements["communication_protocols"] = ["bluetooth", "wifi", "mavlink"]
            requirements["control_interfaces"] = ["joystick", "touch_controls", "gestures"]
            requirements["monitoring_features"] = ["battery", "gps", "altitude", "video_feed"]
            requirements["safety_features"] = ["return_to_home", "geofence", "low_battery_warning"]
            
            if any(word in prompt_lower for word in ["autonomous", "waypoint", "mission"]):
                requirements["control_interfaces"].append("waypoint_planning")
        
        elif hw_type == "3d_printer":
            requirements["communication_protocols"] = ["serial", "usb", "network"]
            requirements["control_interfaces"] = ["manual_controls", "file_upload", "remote_monitoring"]
            requirements["monitoring_features"] = ["temperature", "progress", "filament_sensor"]
            requirements["safety_features"] = ["thermal_runaway", "emergency_stop"]
        
        elif hw_type == "iot_device":
            requirements["communication_protocols"] = ["mqtt", "http", "websockets"]
            requirements["control_interfaces"] = ["dashboard", "automation_rules", "schedules"]
            requirements["monitoring_features"] = ["real_time_status", "history", "alerts"]
            requirements["safety_features"] = ["authentication", "encryption"]
        
        return requirements
    
    def _extract_ai_requirements(self, prompt: str, ai_type: str) -> Dict[str, Any]:
        """Extract AI/ML-specific requirements"""
        requirements = {
            "processing_steps": [],
            "model_types": [],
            "input_formats": [],
            "output_formats": []
        }
        
        prompt_lower = prompt.lower()
        
        if ai_type == "image_to_3d":
            requirements["processing_steps"] = ["image_upload", "preprocessing", "3d_reconstruction", "postprocessing"]
            requirements["model_types"] = ["neural_radiance_field", "point_cloud_generator"]
            requirements["input_formats"] = ["jpg", "png", "webp"]
            requirements["output_formats"] = ["obj", "stl", "gltf", "ply"]
            
            if "multiple images" in prompt_lower or "360" in prompt_lower:
                requirements["processing_steps"].append("multi_view_stereo")
        
        elif ai_type == "ai_model_trainer":
            requirements["processing_steps"] = ["data_upload", "preprocessing", "training", "evaluation"]
            requirements["model_types"] = ["neural_network", "random_forest", "svm"]
            requirements["input_formats"] = ["csv", "json", "images", "text"]
            requirements["output_formats"] = ["model_file", "metrics", "visualizations"]
        
        return requirements
    
    def _requires_real_time(self, prompt: str) -> bool:
        """Check if application requires real-time capabilities"""
        real_time_indicators = [
            "real-time", "live", "stream", "control", "telemetry",
            "monitor", "dashboard", "instant", "immediate", "continuous",
            "updates", "feed", "track", "follow"
        ]
        
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in real_time_indicators)
    
    def _requires_3d_visualization(self, prompt: str) -> bool:
        """Check if application requires 3D visualization"""
        three_d_indicators = [
            "3d", "three.js", "webgl", "model", "mesh", "scene",
            "render", "visualize", "viewer", "rotate", "zoom",
            "perspective", "camera", "lighting", "texture"
        ]
        
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in three_d_indicators)
    
    def _determine_domain_complexity(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        special_reqs: Dict[str, Any]
    ) -> ComplexityLevel:
        """Determine complexity with domain awareness"""
        
        # Hardware apps are always complex
        if special_reqs.get("needs_hardware", False):
            return ComplexityLevel.HARDWARE
        
        # AI/ML apps are complex
        if special_reqs.get("needs_ai_ml", False):
            return ComplexityLevel.AI_ML
        
        # Real-time apps with 3D are complex
        if special_reqs.get("needs_real_time", False) and special_reqs.get("needs_3d", False):
            return ComplexityLevel.ENTERPRISE
        
        # Use standard complexity detection
        return IntentConfig.get_complexity_level(prompt, domain_info["domain"], domain_info["specific_type"])
    
    async def _analyze_with_llm_domain(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        special_reqs: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for nuanced understanding of complex domains"""
        
        try:
            # Build domain-specific prompt
            messages = self._build_domain_prompt(prompt, domain_info, special_reqs, context)
            
            response = await self.orchestrator.generate(
                messages=messages,
                temperature=0.4,
                max_tokens=1000,
                validate_json=True,
                json_response=True
            )
            
            if hasattr(response, 'extracted_json') and response.extracted_json:
                return response.extracted_json
            
            # Try to parse manually
            try:
                return json.loads(response.content.strip())
            except json.JSONDecodeError:
                # Extract JSON from content
                content = response.content.strip()
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
        except Exception as e:
            logger.warning(f"LLM domain analysis failed: {e}")
        
        return None
    
    def _build_domain_prompt(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        special_reqs: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Build domain-specific prompt for LLM"""
        
        domain = domain_info["domain"]
        app_type = domain_info["specific_type"]
        
        system_prompt = f"""You are a {domain.value} application architect specializing in {app_type} apps.

Analyze the user's request and provide detailed architectural guidance.

DOMAIN: {domain.value}
APP TYPE: {app_type}
SPECIAL REQUIREMENTS: {json.dumps(special_reqs, indent=2)}

USER REQUEST: "{prompt}"

Provide a JSON response with:
1. Core functionality description
2. Required technical components
3. Suggested architecture patterns
4. Potential challenges
5. Recommended tech stack

Respond with ONLY valid JSON, no additional text."""

        user_content = f"Analyze this {app_type} application request: {prompt}"
        
        if context:
            user_content += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content)
        ]
    
    def _merge_analyses(
        self,
        domain_info: Dict[str, Any],
        llm_analysis: Dict[str, Any],
        special_reqs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge domain detection with LLM insights"""
        
        return {
            "domain": domain_info["domain"].value,
            "specific_type": domain_info["specific_type"],
            "confidence": domain_info["confidence"],
            "is_specialized": domain_info["is_specialized"],
            "complexity": llm_analysis.get("complexity", "enterprise"),
            "technical_requirements": {
                **special_reqs,
                "llm_insights": llm_analysis
            },
            "architecture_hints": llm_analysis.get("architecture_patterns", []),
            "challenges": llm_analysis.get("potential_challenges", []),
            "recommended_tech": llm_analysis.get("recommended_tech_stack", []),
            "metadata": {
                "analysis_method": "llm_enhanced",
                "timestamp": datetime.now().isoformat(),
                "domain_detection_confidence": domain_info["confidence"]
            }
        }
    
    def _build_domain_result(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        special_reqs: Dict[str, Any],
        complexity: ComplexityLevel
    ) -> Dict[str, Any]:
        """Build comprehensive domain-aware result"""
        
        # Get templates for this domain
        templates = IntentConfig.get_template_for_domain(domain_info["domain"], domain_info["specific_type"])
        
        # Determine features based on domain
        features = self._extract_domain_features(prompt, domain_info, special_reqs)
        
        # Get technical requirements
        tech_reqs = TECHNICAL_REQUIREMENTS.get(complexity, {})
        
        return {
            "domain": domain_info["domain"].value,
            "specific_type": domain_info["specific_type"],
            "display_name": self._generate_display_name(domain_info["specific_type"]),
            "complexity": complexity.value,
            "confidence": domain_info["confidence"],
            "is_specialized": domain_info["is_specialized"],
            "features": features,
            "technical_requirements": {
                **special_reqs,
                "complexity_requirements": tech_reqs
            },
            "templates": templates,
            "architecture_guidance": self._get_architecture_guidance(domain_info, complexity, special_reqs),
            "component_recommendations": self._get_domain_components(domain_info, special_reqs),
            "metadata": {
                "analysis_method": domain_info["detection_method"],
                "timestamp": datetime.now().isoformat(),
                "keywords_found": domain_info.get("keywords_found", [])
            }
        }
    
    def _extract_domain_features(
        self,
        prompt: str,
        domain_info: Dict[str, Any],
        special_reqs: Dict[str, Any]
    ) -> List[str]:
        """Extract domain-specific features"""
        features = []
        
        # Domain-specific features
        domain = domain_info["domain"]
        app_type = domain_info["specific_type"]
        
        if domain == AppDomain.IOT_HARDWARE:
            features.extend([
                "device_control",
                "real_time_monitoring",
                "connection_management"
            ])
            
            if app_type == "drone":
                features.extend(["flight_control", "video_streaming", "gps_navigation"])
            elif app_type == "3d_printer":
                features.extend(["print_control", "temperature_monitoring", "file_management"])
        
        elif domain == AppDomain.CREATIVE_MEDIA and app_type in AI_PATTERNS:
            features.extend([
                "ai_processing",
                "data_transformation",
                "result_visualization"
            ])
            
            if app_type == "image_to_3d":
                features.extend(["image_processing", "3d_reconstruction", "model_export"])
        
        # Add features based on special requirements
        if special_reqs.get("needs_real_time", False):
            features.append("real_time_updates")
        
        if special_reqs.get("needs_3d", False):
            features.append("3d_visualization")
        
        return list(set(features))
    
    def _generate_display_name(self, app_type: str) -> str:
        """Generate user-friendly display name from app type"""
        # Convert snake_case to Title Case with spaces
        words = app_type.split('_')
        return ' '.join(word.capitalize() for word in words)
    
    def _get_architecture_guidance(
        self,
        domain_info: Dict[str, Any],
        complexity: ComplexityLevel,
        special_reqs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get architecture guidance for the domain"""
        
        guidance = {
            "pattern": "mvvm",  # Default pattern
            "state_management": "redux",
            "data_flow": "unidirectional",
            "special_considerations": []
        }
        
        # Domain-specific patterns
        if domain_info["domain"] == AppDomain.IOT_HARDWARE:
            guidance["pattern"] = "observer"
            guidance["state_management"] = "event_driven"
            guidance["data_flow"] = "bidirectional"
            guidance["special_considerations"].append("Handle connection state changes")
            guidance["special_considerations"].append("Implement reconnection logic")
        
        elif domain_info["domain"] == AppDomain.CREATIVE_MEDIA and domain_info["specific_type"] in AI_PATTERNS:
            guidance["pattern"] = "pipeline"
            guidance["state_management"] = "workflow"
            guidance["data_flow"] = "sequential"
            guidance["special_considerations"].append("Handle large file processing")
            guidance["special_considerations"].append("Implement progress tracking")
        
        # Complexity adjustments
        if complexity == ComplexityLevel.HARDWARE:
            guidance["special_considerations"].append("Low latency requirements")
            guidance["special_considerations"].append("Background task management")
        
        elif complexity == ComplexityLevel.AI_ML:
            guidance["special_considerations"].append("GPU/CPU intensive operations")
            guidance["special_considerations"].append("Memory management")
        
        return guidance
    
    def _get_domain_components(
        self,
        domain_info: Dict[str, Any],
        special_reqs: Dict[str, Any]
    ) -> List[str]:
        """Get recommended components for the domain"""
        components = []
        
        # Domain-specific components
        if domain_info["domain"] == AppDomain.IOT_HARDWARE:
            components.extend([
                "ConnectionStatus",
                "DeviceControls",
                "RealTimeChart",
                "LogViewer",
                "SettingsPanel"
            ])
            
            if domain_info["specific_type"] == "drone":
                components.extend(["JoystickControl", "MapView", "VideoStream"])
            elif domain_info["specific_type"] == "3d_printer":
                components.extend(["TemperatureChart", "GCodeViewer", "PrintProgress"])
        
        elif domain_info["domain"] == AppDomain.CREATIVE_MEDIA and domain_info["specific_type"] in AI_PATTERNS:
            components.extend([
                "FileUploader",
                "ProgressIndicator",
                "ResultsViewer",
                "ParameterControls",
                "ExportPanel"
            ])
            
            if domain_info["specific_type"] == "image_to_3d":
                components.extend(["ImagePreview", "3DViewer", "ModelControls"])
        
        # Add components based on special requirements
        if special_reqs.get("needs_3d", False):
            components.append("ThreeJSViewer")
        
        if special_reqs.get("needs_real_time", False):
            components.append("RealTimeDashboard")
        
        return list(set(components))
    
    def _update_stats(
        self,
        domain_info: Dict[str, Any],
        complexity: ComplexityLevel,
        special_reqs: Dict[str, Any]
    ):
        """Update statistics based on analysis"""
        
        # Update domain stats
        if special_reqs.get("needs_hardware", False):
            self.stats['hardware_apps'] += 1
        
        if special_reqs.get("needs_ai_ml", False):
            self.stats['ai_ml_apps'] += 1
        
        if special_reqs.get("needs_real_time", False):
            self.stats['real_time_apps'] += 1
        
        if domain_info["domain"] == AppDomain.CUSTOM:
            self.stats['custom_domains'] += 1
        
        # Update complexity distribution
        complexity_key = complexity.value
        if complexity_key in self.stats['complexity_distribution']:
            self.stats['complexity_distribution'][complexity_key] += 1
    
    def _emergency_domain_fallback(self, prompt: str) -> Dict[str, Any]:
        """Emergency fallback for domain analysis"""
        return {
            "domain": "custom",
            "specific_type": "generic",
            "display_name": "Custom Application",
            "complexity": "integrated",
            "confidence": 0.3,
            "is_specialized": False,
            "features": ["custom_functionality"],
            "technical_requirements": {
                "needs_hardware": False,
                "needs_ai_ml": False,
                "needs_real_time": False,
                "needs_3d": False,
                "special_apis": [],
                "complex_components": []
            },
            "templates": {
                "architecture_template": "custom_app",
                "layout_template": "custom_layout",
                "blockly_template": "custom_logic"
            },
            "metadata": {
                "analysis_method": "emergency_fallback",
                "timestamp": datetime.now().isoformat(),
                "error": "Domain analysis failed"
            }
        }
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Get domain-specific statistics"""
        total = self.stats['total_analyses']
        
        if total == 0:
            return self.stats
        
        stats = {
            **self.stats,
            'hardware_app_rate': (self.stats['hardware_apps'] / total) * 100,
            'ai_ml_app_rate': (self.stats['ai_ml_apps'] / total) * 100,
            'real_time_app_rate': (self.stats['real_time_apps'] / total) * 100,
            'custom_domain_rate': (self.stats['custom_domains'] / total) * 100,
            'complexity_breakdown': {
                k: (v / total) * 100 
                for k, v in self.stats['complexity_distribution'].items()
            }
        }
        
        return stats


# Factory function
def create_domain_aware_analyzer(config: Dict[str, Any]) -> DomainAwareIntentAnalyzer:
    """Create domain-aware intent analyzer"""
    return DomainAwareIntentAnalyzer(config)