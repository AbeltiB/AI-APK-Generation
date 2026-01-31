"""
Testing API routes for easy development and debugging.

Add this file: ai-service/app/api/v1/test_routes.py
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import json

from app.config import settings
from app.models.schemas import AIRequest
from app.core.messaging import queue_manager
from app.services.analysis.intent_analyzer import intent_analyzer
from app.services.generation.architecture_generator import architecture_generator
from app.services.generation.layout_generator import layout_generator
from app.services.generation.blockly_generator import blockly_generator

router = APIRouter()


class TestPromptRequest(BaseModel):
    """Test prompt request"""
    prompt: str = Field(..., min_length=10, max_length=2000)
    user_id: str = Field(default="test_user")
    session_id: str = Field(default="test_session")
    context: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Create a simple counter app with + and - buttons",
                "user_id": "test_user",
                "session_id": "test_session"
            }
        }


class StageTestRequest(BaseModel):
    """Test individual stage"""
    prompt: str = Field(..., min_length=10)
    stage: str = Field(..., description="intent | architecture | layout | blockly")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Create a todo list app",
                "stage": "intent"
            }
        }


@router.post("/test/prompt")
async def test_prompt(
    request: TestPromptRequest,
    background_tasks: BackgroundTasks
):
    """
    Test a prompt by sending it through RabbitMQ.
    
    This simulates the complete flow:
    1. Receives prompt
    2. Sends to RabbitMQ ai-requests queue
    3. Returns task ID for tracking
    
    Check logs and RabbitMQ management UI to see processing.
    """
    
    # Create AI request
    ai_request = AIRequest(
        user_id=request.user_id,
        session_id=request.session_id,
        socket_id=f"test_socket_{int(datetime.now(timezone.utc).timestamp())}",
        prompt=request.prompt,
        context=request.context
    )
    
    # Send to RabbitMQ (in background to return quickly)
    def send_to_queue():
        import asyncio
        asyncio.run(queue_manager.publish_response({
            **ai_request.dict(),
            "test_mode": True
        }))
    
    background_tasks.add_task(send_to_queue)
    
    return {
        "status": "queued",
        "task_id": ai_request.task_id,
        "message": "Request sent to processing queue",
        "check": {
            "logs": "Check AI service logs for processing details",
            "rabbitmq": f"http://localhost:15672 (admin/password)",
            "queue": "ai-responses"
        },
        "request": {
            "user_id": ai_request.user_id,
            "session_id": ai_request.session_id,
            "prompt": ai_request.prompt
        }
    }


@router.post("/test/stage/{stage}")
async def test_stage(stage: str, request: StageTestRequest):
    """
    Test individual pipeline stage in isolation.
    
    Stages:
    - intent: Intent classification
    - architecture: Architecture generation
    - layout: Layout generation
    - blockly: Blockly generation
    
    Returns detailed output including intermediate results.
    """
    
    result = {
        "stage": stage,
        "prompt": request.prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": None,
        "metadata": None,
        "logs": []
    }
    
    try:
        if stage == "intent":
            # Test intent analyzer
            result["logs"].append("Starting intent analysis...")
            
            intent = await intent_analyzer.analyze(
                prompt=request.prompt,
                context=None
            )
            
            result["output"] = {
                "intent_type": intent.intent_type,
                "complexity": intent.complexity,
                "confidence": intent.confidence,
                "extracted_entities": intent.extracted_entities,
                "requires_context": intent.requires_context,
                "multi_turn": intent.multi_turn
            }
            
            result["logs"].append(f"✅ Intent: {intent.intent_type}")
            result["logs"].append(f"✅ Complexity: {intent.complexity}")
            result["logs"].append(f"✅ Confidence: {intent.confidence:.2f}")
            
        elif stage == "architecture":
            # Test architecture generator
            result["logs"].append("Starting architecture generation...")
            
            architecture, metadata = await architecture_generator.generate(
                prompt=request.prompt,
                context=None
            )
            
            result["output"] = architecture.dict()
            result["metadata"] = metadata
            
            result["logs"].append(f"✅ App type: {architecture.app_type}")
            result["logs"].append(f"✅ Screens: {len(architecture.screens)}")
            result["logs"].append(f"✅ State vars: {len(architecture.state_management)}")
            
        elif stage == "layout":
            # Need architecture first
            result["logs"].append("Generating architecture first...")
            architecture, _ = await architecture_generator.generate(
                prompt=request.prompt,
                context=None
            )
            
            result["logs"].append("Starting layout generation...")
            
            # Generate layout for first screen
            screen_id = architecture.screens[0].id
            layout, metadata = await layout_generator.generate(
                architecture=architecture,
                screen_id=screen_id
            )
            
            result["output"] = layout.dict()
            result["metadata"] = metadata
            
            result["logs"].append(f"✅ Screen: {layout.screen_id}")
            result["logs"].append(f"✅ Components: {len(layout.components)}")
            
        elif stage == "blockly":
            # Need architecture and layout first
            result["logs"].append("Generating architecture...")
            architecture, _ = await architecture_generator.generate(
                prompt=request.prompt,
                context=None
            )
            
            result["logs"].append("Generating layout...")
            layouts = {}
            for screen in architecture.screens:
                layout, _ = await layout_generator.generate(
                    architecture=architecture,
                    screen_id=screen.id
                )
                layouts[screen.id] = layout
            
            result["logs"].append("Starting Blockly generation...")
            
            blockly, metadata = await blockly_generator.generate(
                architecture=architecture,
                layouts=layouts
            )
            
            result["output"] = blockly
            result["metadata"] = metadata
            
            result["logs"].append(f"✅ Blocks: {len(blockly['blocks']['blocks'])}")
            result["logs"].append(f"✅ Variables: {len(blockly['variables'])}")
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage: {stage}. Use: intent, architecture, layout, or blockly"
            )
        
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["logs"].append(f"❌ Error: {e}")
    
    return result


@router.get("/test/stats")
async def get_test_stats():
    """
    Get statistics from all generators.
    
    Useful for monitoring performance and costs.
    """
    
    # Get intent analyzer stats
    intent_stats = intent_analyzer.get_stats()
    
    # Get generator stats
    arch_stats = architecture_generator.get_statistics()
    layout_stats = layout_generator.get_statistics()
    blockly_stats = blockly_generator.get_statistics()
    
    return {
        "intent_analyzer": intent_stats,
        "architecture_generator": arch_stats,
        "layout_generator": layout_stats,
        "blockly_generator": blockly_stats,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.post("/test/complete")
async def test_complete_flow(request: TestPromptRequest):
    """
    Test complete flow synchronously (no RabbitMQ).
    
    Runs all stages and returns all intermediate outputs.
    
    ⚠️ This can take 30-60 seconds for all stages to complete.
    """
    
    result = {
        "prompt": request.prompt,
        "stages": {},
        "total_time_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    import time
    total_start = time.time()
    
    try:
        # Stage 1: Intent
        start = time.time()
        intent = await intent_analyzer.analyze(request.prompt, None)
        result["stages"]["intent"] = {
            "output": {
                "intent_type": intent.intent_type,
                "complexity": intent.complexity,
                "confidence": intent.confidence,
                "entities": intent.extracted_entities
            },
            "duration_ms": int((time.time() - start) * 1000)
        }
        
        # Stage 2: Architecture
        start = time.time()
        architecture, arch_meta = await architecture_generator.generate(
            request.prompt, None
        )
        result["stages"]["architecture"] = {
            "output": architecture.dict(),
            "metadata": arch_meta,
            "duration_ms": int((time.time() - start) * 1000)
        }
        
        # Stage 3: Layout
        start = time.time()
        layouts = {}
        for screen in architecture.screens:
            layout, layout_meta = await layout_generator.generate(
                architecture, screen.id
            )
            layouts[screen.id] = layout.dict()
        result["stages"]["layout"] = {
            "output": layouts if len(layouts) > 1 else list(layouts.values())[0],
            "duration_ms": int((time.time() - start) * 1000)
        }
        
        # Stage 4: Blockly
        start = time.time()
        from app.models.enhanced_schemas import EnhancedLayoutDefinition
        layout_objs = {
            sid: EnhancedLayoutDefinition(**ld) 
            for sid, ld in (layouts.items() if isinstance(layouts, dict) else [(architecture.screens[0].id, layouts)])
        }
        blockly, blockly_meta = await blockly_generator.generate(
            architecture, layout_objs
        )
        result["stages"]["blockly"] = {
            "output": blockly,
            "metadata": blockly_meta,
            "duration_ms": int((time.time() - start) * 1000)
        }
        
        result["total_time_ms"] = int((time.time() - total_start) * 1000)
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["total_time_ms"] = int((time.time() - total_start) * 1000)
    
    return result


@router.get("/test/config")
async def get_test_config():
    """Get current configuration for testing"""
    
    return {
        "app_name": settings.app_name,
        "debug": settings.debug,
        "anthropic_model": settings.anthropic_model,
        "rate_limiting": settings.rate_limit_enabled,
        "canvas": {
            "width": settings.canvas_width,
            "height": settings.canvas_height
        },
        "available_components": settings.available_components,
        "queues": {
            "requests": settings.rabbitmq_queue_ai_requests,
            "responses": settings.rabbitmq_queue_ai_responses
        }
    }


# New test endpoints for the API
@router.get("/test/api/generate")
async def test_api_generate_flow():
    """Test the complete API flow"""
    
    return {
        "test_cases": [
            {
                "method": "POST",
                "endpoint": "/api/v1/generate",
                "description": "Submit generation request",
                "example_body": {
                    "prompt": "Create a counter app with + and - buttons",
                    "user_id": "test_user_123",
                    "session_id": "test_session_456",
                    "priority": 1
                }
            },
            {
                "method": "GET",
                "endpoint": "/api/v1/task/{task_id}",
                "description": "Check task status",
                "example": "/api/v1/task/550e8400-e29b-41d4-a716-446655440000"
            },
            {
                "method": "DELETE",
                "endpoint": "/api/v1/task/{task_id}",
                "description": "Cancel task",
                "example": "/api/v1/task/550e8400-e29b-41d4-a716-446655440000"
            },
            {
                "method": "GET",
                "endpoint": "/api/v1/stats",
                "description": "Get system statistics"
            },
            {
                "method": "POST",
                "endpoint": "/api/v1/test/prompt",
                "description": "Test prompt (development only)"
            }
        ]
    }