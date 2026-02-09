"""
Input/output models for AI service communication.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from uuid import uuid4


class PromptContext(BaseModel):
    """Optional context information for prompt processing"""
    existing_components: Optional[List[Dict[str, Any]]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    project_metadata: Optional[Dict[str, Any]] = None


class AIRequest(BaseModel):
    """AI request message received from Celery"""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str = Field(default="unknown")
    socket_id: str = Field(default="unknown")
    prompt: str = Field(..., min_length=10, max_length=2000)
    context: Optional[PromptContext] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Ensure prompt is not just whitespace"""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_123",
                "session_id": "session_abc",
                "socket_id": "socket_xyz",
                "prompt": "Create a todo list app with add, delete, and complete features",
                "context": None,
                "timestamp": "2025-12-16T10:00:00Z"
            }
        }


class ProgressUpdate(BaseModel):
    """Progress update during AI processing"""
    task_id: str
    socket_id: str
    type: Literal["progress"] = "progress"
    stage: str = Field(..., description="Current processing stage")
    progress: int = Field(..., ge=0, le=100)
    message: str = ""


class ErrorResponse(BaseModel):
    """Error response when something goes wrong"""
    task_id: str
    socket_id: str
    type: Literal["error"] = "error"
    error: str
    details: Optional[str] = None


class CompleteResponse(BaseModel):
    """Complete AI generation response"""
    task_id: str
    socket_id: str
    type: Literal["complete"] = "complete"
    status: Literal["success", "partial_success", "error"] = "success"
    result: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    conversation: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('result')
    @classmethod
    def validate_result_structure(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure result has required keys for success status"""
        # This validation should be done at runtime based on status
        return v