"""
ai-service/app/api/v1/results.py

REST API endpoints for retrieving generation results.
Provides complete, consumable responses for frontend systems.
"""
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

from app.core.cache import cache_manager
from app.utils.logging import get_logger, log_context

router = APIRouter()
logger = get_logger(__name__)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class ResultStatus(str, Enum):
    """Result status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ComponentData(BaseModel):
    """Individual component in layout"""
    id: str
    type: str
    properties: Dict[str, Any]
    position: Dict[str, int]
    constraints: Dict[str, Any]


class ScreenLayout(BaseModel):
    """Layout for a single screen"""
    screen_id: str
    layout_type: str
    background_color: str
    components: List[ComponentData]


class BlocklyBlock(BaseModel):
    """Individual Blockly block"""
    type: str
    id: str
    fields: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Any]] = None
    next: Optional[Dict[str, Any]] = None


class GenerationMetadata(BaseModel):
    """Metadata about the generation process"""
    generated_at: str
    total_time_ms: int
    cache_hit: bool
    provider_used: str
    generation_method: str
    
    # Stage-specific timing
    intent_analysis_ms: Optional[int] = None
    architecture_generation_ms: Optional[int] = None
    layout_generation_ms: Optional[int] = None
    blockly_generation_ms: Optional[int] = None
    
    # Quality metrics
    validation_warnings: int
    substitutions_made: int
    heuristic_fallback_used: bool


class IntentInfo(BaseModel):
    """Intent classification information"""
    type: str
    complexity: str
    confidence: float
    requires_context: bool
    extracted_components: List[str] = Field(default_factory=list)
    extracted_features: List[str] = Field(default_factory=list)


class ArchitectureData(BaseModel):
    """Architecture design data"""
    app_type: str
    screens: List[Dict[str, Any]]
    navigation: Dict[str, Any]
    state_management: List[Dict[str, Any]]
    data_flow: Dict[str, Any]


class GenerationResult(BaseModel):
    """Complete generation result"""
    task_id: str
    user_id: str
    session_id: str
    status: ResultStatus
    
    # Original request
    prompt: str
    created_at: str
    completed_at: Optional[str] = None
    
    # Generated content
    architecture: Optional[ArchitectureData] = None
    layouts: Optional[Dict[str, ScreenLayout]] = None
    blockly: Optional[Dict[str, Any]] = None
    
    # Metadata
    metadata: GenerationMetadata
    intent: Optional[IntentInfo] = None
    
    # Warnings and errors
    warnings: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_123",
                "session_id": "session_abc",
                "status": "completed",
                "prompt": "Create a counter app with + and - buttons",
                "created_at": "2025-01-01T12:00:00Z",
                "completed_at": "2025-01-01T12:00:45Z",
                "architecture": {
                    "app_type": "single-page",
                    "screens": [],
                    "navigation": {},
                    "state_management": [],
                    "data_flow": {}
                },
                "metadata": {
                    "generated_at": "2025-01-01T12:00:45Z",
                    "total_time_ms": 45000,
                    "cache_hit": False,
                    "provider_used": "claude",
                    "generation_method": "llm",
                    "validation_warnings": 0,
                    "substitutions_made": 0,
                    "heuristic_fallback_used": False
                }
            }
        }


class ResultListItem(BaseModel):
    """Simplified result for list view"""
    task_id: str
    prompt: str
    status: ResultStatus
    created_at: str
    completed_at: Optional[str] = None
    app_type: Optional[str] = None


class ResultListResponse(BaseModel):
    """Response for list of results"""
    results: List[ResultListItem]
    total: int
    page: int
    page_size: int
    has_more: bool


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/results/{task_id}",
    # Temporarily remove response_model to bypass validation
    # response_model=GenerationResult,
    tags=["Results"],
    summary="Get complete generation result",
    description="Retrieve the complete generation result including architecture, layouts, and Blockly blocks"
)
async def get_result(task_id: str) -> Dict[str, Any]:
    """
    Get complete generation result by task ID.
    
    TEMPORARY: Returns raw data without strict validation
    """
    
    with log_context(task_id=task_id, endpoint="/api/v1/results", method="GET"):
        logger.info(
            "api.results.get.requested",
            extra={"task_id": task_id}
        )
        
        try:
            # Get task data from Redis
            task_key = f"task:{task_id}"
            task_data = await cache_manager.get(task_key)
            
            if not task_data:
                logger.warning(
                    "api.results.get.not_found",
                    extra={"task_id": task_id}
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "result_not_found",
                        "message": f"No result found for task ID: {task_id}",
                        "task_id": task_id
                    }
                )
            
            logger.info(
                "api.results.get.found",
                extra={
                    "task_id": task_id,
                    "status": task_data.get("status"),
                    "created_at": task_data.get("created_at")
                }
            )
            
            # TEMPORARY: Return raw data for debugging
            logger.debug(
                "api.results.get.raw_data",
                extra={
                    "task_id": task_id,
                    "raw_task_data": str(task_data)[:500],  # First 500 chars
                    "raw_result": str(task_data.get("result", {}))[:500] if task_data.get("result") else "No result"
                }
            )
            
            # Check status
            current_status = task_data.get("status", "pending")
            
            if current_status in ["queued", "processing"]:
                logger.info(
                    "api.results.get.still_processing",
                    extra={
                        "task_id": task_id,
                        "status": current_status,
                        "progress": task_data.get("progress", 0)
                    }
                )
                
                # Return basic status info
                return {
                    "task_id": task_id,
                    "status": current_status,
                    "message": "Still processing",
                    "progress": task_data.get("progress", 0),
                    "raw_data_available": False
                }
            
            # Get result data
            result_data = task_data.get("result", {})
            
            if not result_data and current_status == "completed":
                logger.error(
                    "api.results.get.missing_result",
                    extra={"task_id": task_id, "status": current_status}
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "result_incomplete",
                        "message": "Task marked as completed but result data is missing"
                    }
                )
            
            # TEMPORARY: Return raw result data without validation
            response = {
                "task_id": task_id,
                "status": current_status,
                "user_id": task_data.get("user_id", "unknown"),
                "session_id": task_data.get("session_id", "unknown"),
                "prompt": task_data.get("prompt", ""),
                "created_at": task_data.get("created_at", ""),
                "completed_at": task_data.get("updated_at"),
                "raw_result": result_data,
                "metadata": result_data.get("metadata", {}),
                "warnings": result_data.get("warnings", []),
                "errors": result_data.get("errors", []),
                "note": "TEMPORARY: Returning raw data without validation"
            }
            
            # Try to parse layout data manually for debugging
            if "layout" in result_data:
                layout_data = result_data["layout"]
                response["layout_debug"] = {
                    "type": type(layout_data).__name__,
                    "keys": list(layout_data.keys()) if isinstance(layout_data, dict) else "Not a dict",
                    "sample_component": layout_data.get("components", [{}])[0] if isinstance(layout_data, dict) and "components" in layout_data else "No components"
                }
            
            logger.info(
                "api.results.get.success_raw",
                extra={
                    "task_id": task_id,
                    "status": current_status,
                    "has_layout": "layout" in result_data,
                    "layout_type": type(result_data.get("layout")).__name__ if result_data.get("layout") else None
                }
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "api.results.get.error",
                extra={
                    "task_id": task_id,
                    "error": str(e)
                },
                exc_info=e
            )
            # Still return some info even on error
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "message": "Error occurred but returning partial data",
                "raw_task_data_available": bool(task_data),
                "raw_result_available": bool(task_data.get("result")) if task_data else False
            }

@router.get(
    "/results",
    response_model=ResultListResponse,
    tags=["Results"],
    summary="List generation results",
    description="Get a paginated list of generation results for the authenticated user"
)
async def list_results(
    user_id: str = Query(..., description="User ID to filter results"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    status: Optional[ResultStatus] = Query(None, description="Filter by status")
) -> ResultListResponse:
    """
    List generation results for a user.
    
    Frontend Usage:
    ```javascript
    const response = await fetch(
        `/api/v1/results?user_id=${userId}&page=1&page_size=20`
    );
    const { results, total, has_more } = await response.json();
    ```
    """
    
    with log_context(user_id=user_id, endpoint="/api/v1/results", method="GET"):
        logger.info(
            "api.results.list.requested",
            extra={
                "user_id": user_id,
                "page": page,
                "page_size": page_size,
                "status_filter": status
            }
        )
        
        try:
            # Scan for user's tasks
            pattern = f"task:*"
            results = []
            
            cursor = 0
            scanned = 0
            max_scan = 1000  # Limit scan for performance
            
            while True:
                cursor, keys = await cache_manager.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    task_data = await cache_manager.get(key)
                    
                    if not task_data:
                        continue
                    
                    # Filter by user_id
                    if task_data.get("user_id") != user_id:
                        continue
                    
                    # Filter by status if specified
                    if status and task_data.get("status") != status.value:
                        continue
                    
                    # Extract app_type from result
                    app_type = None
                    if "result" in task_data and "architecture" in task_data["result"]:
                        app_type = task_data["result"]["architecture"].get("app_type")
                    
                    results.append(ResultListItem(
                        task_id=task_data.get("task_id", key.split(":")[-1]),
                        prompt=task_data.get("prompt", "")[:100],  # Truncate
                        status=ResultStatus(task_data.get("status", "pending")),
                        created_at=task_data.get("created_at", ""),
                        completed_at=task_data.get("updated_at"),
                        app_type=app_type
                    ))
                    
                    scanned += 1
                    if scanned >= max_scan:
                        break
                
                if cursor == 0 or scanned >= max_scan:
                    break
            
            # Sort by created_at descending
            results.sort(
                key=lambda r: r.created_at,
                reverse=True
            )
            
            # Paginate
            total = len(results)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_results = results[start_idx:end_idx]
            has_more = end_idx < total
            
            logger.info(
                "api.results.list.success",
                extra={
                    "user_id": user_id,
                    "total_found": total,
                    "page": page,
                    "page_size": page_size,
                    "returned": len(page_results),
                    "has_more": has_more
                }
            )
            
            return ResultListResponse(
                results=page_results,
                total=total,
                page=page,
                page_size=page_size,
                has_more=has_more
            )
            
        except Exception as e:
            logger.error(
                "api.results.list.error",
                extra={
                    "user_id": user_id,
                    "error": str(e)
                },
                exc_info=e
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "internal_error",
                    "message": "An error occurred while listing results"
                }
            )


@router.delete(
    "/results/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Results"],
    summary="Delete generation result",
    description="Delete a generation result (soft delete - moves to archive)"
)
async def delete_result(task_id: str, user_id: str = Query(...)):
    """
    Delete a generation result.
    
    This performs a soft delete by archiving the result.
    """
    
    with log_context(task_id=task_id, user_id=user_id):
        logger.info(
            "api.results.delete.requested",
            extra={"task_id": task_id, "user_id": user_id}
        )
        
        try:
            # Get task to verify ownership
            task_key = f"task:{task_id}"
            task_data = await cache_manager.get(task_key)
            
            if not task_data:
                logger.warning(
                    "api.results.delete.not_found",
                    extra={"task_id": task_id}
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "not_found", "message": "Result not found"}
                )
            
            # Verify ownership
            if task_data.get("user_id") != user_id:
                logger.warning(
                    "api.results.delete.forbidden",
                    extra={
                        "task_id": task_id,
                        "requesting_user": user_id,
                        "owning_user": task_data.get("user_id")
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "forbidden", "message": "Not authorized"}
                )
            
            # Archive the result (move to archive key)
            archive_key = f"archive:{task_id}"
            await cache_manager.set(
                archive_key,
                {**task_data, "archived_at": datetime.now(timezone.utc).isoformat() + "Z"},
                ttl=86400 * 30  # 30 days in archive
            )
            
            # Delete original
            await cache_manager.delete(task_key)
            
            logger.info(
                "api.results.delete.success",
                extra={"task_id": task_id, "user_id": user_id}
            )
            
            return None
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "api.results.delete.error",
                extra={
                    "task_id": task_id,
                    "error": str(e)
                },
                exc_info=e
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "internal_error"}
            )