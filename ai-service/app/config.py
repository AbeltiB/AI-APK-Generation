"""
Application configuration management using Pydantic Settings.
Loads configuration from environment variables and .env file.

LLAMA3 ONLY VERSION - All Anthropic/Claude/Groq references removed.
"""
import logging
import sys
from typing import Literal, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from functools import lru_cache
import os

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings - Llama3 Only Version"""
    
    # APPLICATION METADATA & RUNTIME SETTINGS
    app_name: str = "AI App Builder Service (Llama3)"
    app_version: str = "0.1.0"
    api_title: str = "AI Service API - Llama3"
    api_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # LLAMA3 API SETTINGS (PRIMARY AND ONLY LLM)
    llama3_api_url: str = "https://fastchat.ideeza.com/v1/chat/completions"
    llama3_model: str = "llama-3-70b-instruct"
    llama3_api_key: Optional[str] = None
    llama3_timeout: float = 60.0
    llama3_max_tokens: int = 4096
    llama3_temperature: float = 0.7
    llama3_max_retries: int = 3
    llama3_retry_delay: int = 2
    
    # LLM ORCHESTRATOR SETTINGS (Llama3 → Heuristic only)
    llm_primary_provider: Literal["llama3"] = "llama3"
    llm_fallback_enabled: bool = True
    llm_fallback_sequence: list[str] = ["llama3", "heuristic"]
    llm_failure_threshold: int = 3
    llm_failure_window_minutes: int = 5
    llm_health_check_interval: int = 60
    llm_default_temperature: float = 0.7
    llm_default_max_tokens: int = 4096
    
    # RABBITMQ MESSAGE QUEUE SETTINGS
    rabbitmq_url: str = "amqp://admin:password@localhost:5672"
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_queue_ai_requests: str = "ai-requests"
    rabbitmq_queue_ai_responses: str = "ai-responses"
    rabbitmq_queue_app_generation: str = "app-generation-queue"
    rabbitmq_prefetch_count: int = 1
    rabbitmq_heartbeat: int = 60
    rabbitmq_connection_timeout: int = 10
    
    # REDIS CACHE & SESSION SETTINGS
    redis_url: str = "redis://localhost:6379"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_cache_ttl: int = 86400
    redis_semantic_cache_ttl: int = 604800
    redis_session_ttl: int = 28800
    redis_cache_similarity_threshold: float = 0.95
    redis_pool_size: int = 10
    redis_socket_timeout: int = 5
    
    # POSTGRESQL DATABASE SETTINGS
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "appbuilder"
    postgres_user: str = "admin"
    postgres_password: str = "password"
    postgres_min_connections: int = 5
    postgres_max_connections: int = 20
    postgres_connection_timeout: int = 30
    postgres_statement_timeout: int = 30000
    postgres_pool_recycle: int = 3600
    
    # PROCESSING, RETRY & TIMEOUT SETTINGS
    max_retries: int = 3
    retry_delay: int = 2
    retry_backoff_factor: float = 1.5
    request_timeout: int = 30
    batch_processing_size: int = 10
    concurrent_workers: int = 4
    
    # RATE LIMITING & SECURITY SETTINGS
    rate_limit_enabled: bool = True
    rate_limit_requests_per_hour: int = 100
    rate_limit_requests_per_minute: int = 20
    rate_limit_window_size: int = 60
    rate_limit_storage_backend: Literal["redis", "memory"] = "redis"
    
    # UI/CANVAS SPECIFICATION SETTINGS
    canvas_width: int = 375
    canvas_height: int = 667
    canvas_safe_area_top: int = 44
    canvas_safe_area_bottom: int = 34
    canvas_background_color: str = "#FFFFFF"
    canvas_grid_size: int = 8
    canvas_snap_to_grid: bool = True
    
    # COMPONENT LIBRARY SETTINGS
    available_components: list[str] = [
        "Button", "InputText", "Switch", "Checkbox", "TextArea",
        "Slider", "Spinner", "Text", "Joystick", "ProgressBar",
        "DatePicker", "TimePicker", "ColorPicker", "Map", "Chart",
        "Dropdown", "RadioGroup", "Image", "Video", "Audio",
        "List", "Grid", "Card", "Modal", "TabView"
    ]
    
    min_touch_target_size: int = 44
    default_font_family: str = "San Francisco, Roboto, sans-serif"
    default_font_size: int = 16
    default_spacing_unit: int = 8
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary"""
        return {
            "primary_provider": self.llm_primary_provider,
            "fallback_enabled": self.llm_fallback_enabled,
            "fallback_sequence": self.llm_fallback_sequence,
            "failure_threshold": self.llm_failure_threshold,
            "failure_window_minutes": self.llm_failure_window_minutes,
            "default_temperature": self.llm_default_temperature,
            "default_max_tokens": self.llm_default_max_tokens,
            "llama3_api_url": self.llama3_api_url,
            "llama3_model": self.llama3_model,
        }
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
        validate_default=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()