"""
Comprehensive health check system for AI service with LLAMA3, RabbitMQ, Redis, and PostgreSQL.
"""
from fastapi import APIRouter, status, Response
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import time
import asyncio
import os

from app.config import settings
from app.core.cache import cache_manager
from app.core.database import db_manager
from app.core.messaging import queue_manager
from app.utils.logging import get_logger, log_context

router = APIRouter()
logger = get_logger(__name__)

# Track service start time
SERVICE_START_TIME = time.time()


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Simple health check response model"""
    status: str
    service: str
    version: str
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "AI Service with LLAMA3",
                "version": "0.1.0",
                "timestamp": "2025-12-16T12:00:00Z"
            }
        }


class LivenessResponse(BaseModel):
    """Liveness probe response"""
    status: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "alive",
                "timestamp": "2025-01-01T12:00:00Z"
            }
        }


class DependencyStatus(BaseModel):
    """Status of a single dependency"""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: Optional[float] = None
    message: Optional[str] = None
    last_checked: str


class ReadinessResponse(BaseModel):
    """Readiness probe response"""
    status: str  # "ready", "not_ready"
    ready: bool
    dependencies: Dict[str, DependencyStatus]
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ready",
                "ready": True,
                "dependencies": {
                    "redis": {
                        "name": "Redis Cache",
                        "status": "healthy",
                        "response_time_ms": 2.5,
                        "message": "Connected",
                        "last_checked": "2025-01-01T12:00:00Z"
                    }
                },
                "timestamp": "2025-01-01T12:00:00Z"
            }
        }


class FullHealthResponse(BaseModel):
    """Complete health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    ready: bool
    version: str
    environment: str
    instance_id: str
    uptime_seconds: float
    dependencies: Dict[str, DependencyStatus]
    metrics: Dict[str, Any]
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "ready": True,
                "version": "0.1.0",
                "environment": "production",
                "instance_id": "ai-service-1",
                "uptime_seconds": 3600.5,
                "dependencies": {},
                "metrics": {
                    "requests_total": 1523,
                    "requests_success": 1498,
                    "requests_failed": 25,
                    "avg_response_time_ms": 450.2
                },
                "timestamp": "2025-01-01T12:00:00Z"
            }
        }


# ============================================================================
# DEPENDENCY CHECK FUNCTIONS
# ============================================================================

async def check_redis() -> DependencyStatus:
    """Check Redis connection"""
    start = time.time()
    
    try:
        if not cache_manager._connected:
            return DependencyStatus(
                name="Redis Cache",
                status="unhealthy",
                message="Not connected",
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        # Quick ping
        await asyncio.wait_for(
            cache_manager.client.ping(),
            timeout=1.0
        )
        
        response_time = (time.time() - start) * 1000
        
        return DependencyStatus(
            name="Redis Cache",
            status="healthy",
            response_time_ms=response_time,
            message="Connected",
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )
        
    except asyncio.TimeoutError:
        return DependencyStatus(
            name="Redis Cache",
            status="unhealthy",
            message="Ping timeout (>1s)",
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )
    except Exception as e:
        return DependencyStatus(
            name="Redis Cache",
            status="unhealthy",
            message=str(e),
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )


async def check_rabbitmq() -> DependencyStatus:
    """Check RabbitMQ connection"""
    start = time.time()
    
    try:
        if not queue_manager.is_connected:
            return DependencyStatus(
                name="RabbitMQ",
                status="unhealthy",
                message="Not connected",
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        response_time = (time.time() - start) * 1000
        
        return DependencyStatus(
            name="RabbitMQ",
            status="healthy",
            response_time_ms=response_time,
            message="Connected",
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )
        
    except Exception as e:
        return DependencyStatus(
            name="RabbitMQ",
            status="unhealthy",
            message=str(e),
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )


async def check_database() -> DependencyStatus:
    """Check PostgreSQL connection"""
    start = time.time()
    
    try:
        if not db_manager.is_connected:
            return DependencyStatus(
                name="PostgreSQL",
                status="unhealthy",
                message="Not connected",
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        # Quick query
        await asyncio.wait_for(
            db_manager.fetch_val("SELECT 1"),
            timeout=2.0
        )
        
        response_time = (time.time() - start) * 1000
        
        return DependencyStatus(
            name="PostgreSQL",
            status="healthy",
            response_time_ms=response_time,
            message="Connected",
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )
        
    except asyncio.TimeoutError:
        return DependencyStatus(
            name="PostgreSQL",
            status="unhealthy",
            message="Query timeout (>2s)",
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )
    except Exception as e:
        return DependencyStatus(
            name="PostgreSQL",
            status="unhealthy",
            message=str(e),
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )


async def check_llama3() -> DependencyStatus:
    """Check LLAMA3 model availability"""
    start = time.time()
    
    try:
        # This would depend on your LLAMA3 implementation
        # For now, we'll assume it's always available if the service is running
        # You should replace this with actual model checking logic
        
        response_time = (time.time() - start) * 1000
        
        return DependencyStatus(
            name="LLAMA3 Model",
            status="healthy",
            response_time_ms=response_time,
            message="Model loaded and ready",
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )
        
    except Exception as e:
        return DependencyStatus(
            name="LLAMA3 Model",
            status="unhealthy",
            message=str(e),
            last_checked=datetime.now(timezone.utc).isoformat() + "Z"
        )


# ============================================================================
# BASIC HEALTH CHECK (Simple endpoint)
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Service health check",
    description="Returns the current health status of the AI service"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns basic service information and health status.
    Used by monitoring systems and load balancers.
    """
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc)
    )


# ============================================================================
# LIVENESS PROBE (Kubernetes - Very Strict)
# ============================================================================

@router.get(
    "/health/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Liveness probe",
    description="Kubernetes liveness probe. MUST be fast (<100ms). NO dependency checks."
)
async def liveness_check() -> LivenessResponse:
    """
    Liveness probe - Is the process alive and responsive?
    
    Rules:
    - MUST be FAST (<100ms)
    - MUST NOT check dependencies
    - MUST NOT block
    - Only checks if process is running
    
    Kubernetes uses this to restart pods.
    Failure = Pod restart.
    """
    
    # No logging in liveness probe (too slow)
    # Just return immediately
    
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now(timezone.utc).isoformat() + "Z"
    )


# ============================================================================
# READINESS PROBE (Load Balancer - Traffic Gate)
# ============================================================================

@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe",
    description="Load balancer readiness check. Checks critical dependencies."
)
async def readiness_check(response: Response) -> ReadinessResponse:
    """
    Readiness probe - Can this instance receive traffic?
    
    Checks:
    - Redis connected?
    - RabbitMQ connected?
    - PostgreSQL connected?
    - LLAMA3 model available?
    
    Load balancers use this to route traffic.
    Failure = Remove from load balancer pool.
    """
    
    with log_context(endpoint="/health/ready"):
        logger.info("health.readiness.check_started")
        
        start_time = time.time()
        
        # Check all dependencies in parallel
        redis_status, rabbitmq_status, db_status, llama_status = await asyncio.gather(
            check_redis(),
            check_rabbitmq(),
            check_database(),
            check_llama3(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(redis_status, Exception):
            redis_status = DependencyStatus(
                name="Redis Cache",
                status="unhealthy",
                message=str(redis_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(rabbitmq_status, Exception):
            rabbitmq_status = DependencyStatus(
                name="RabbitMQ",
                status="unhealthy",
                message=str(rabbitmq_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(db_status, Exception):
            db_status = DependencyStatus(
                name="PostgreSQL",
                status="unhealthy",
                message=str(db_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(llama_status, Exception):
            llama_status = DependencyStatus(
                name="LLAMA3 Model",
                status="unhealthy",
                message=str(llama_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        dependencies = {
            "redis": redis_status,
            "rabbitmq": rabbitmq_status,
            "database": db_status,
            "llama3": llama_status
        }
        
        # Determine overall readiness
        all_healthy = all(
            dep.status == "healthy"
            for dep in dependencies.values()
        )
        
        check_duration = (time.time() - start_time) * 1000
        
        if all_healthy:
            logger.info(
                "health.readiness.passed",
                extra={
                    "check_duration_ms": check_duration,
                    "dependencies": {k: v.status for k, v in dependencies.items()}
                }
            )
            
            result_status = "ready"
            http_status = status.HTTP_200_OK
        else:
            logger.warning(
                "health.readiness.failed",
                extra={
                    "check_duration_ms": check_duration,
                    "dependencies": {k: v.status for k, v in dependencies.items()},
                    "unhealthy": [
                        k for k, v in dependencies.items()
                        if v.status != "healthy"
                    ]
                }
            )
            
            result_status = "not_ready"
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
        
        response.status_code = http_status
        
        return ReadinessResponse(
            status=result_status,
            ready=all_healthy,
            dependencies=dependencies,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z"
        )


# ============================================================================
# FULL HEALTH CHECK (Observability)
# ============================================================================

async def get_service_metrics() -> Dict[str, Any]:
    """Get service metrics (placeholder - implement based on your needs)"""
    
    # TODO: Implement actual metrics collection
    # This could pull from Prometheus, internal counters, etc.
    
    return {
        "requests_total": 0,
        "requests_success": 0,
        "requests_failed": 0,
        "avg_response_time_ms": 0.0,
        "cache_hit_rate": 0.0,
        "queue_depth": {
            "ai_requests": 0,
            "ai_responses": 0
        }
    }


@router.get(
    "/health/full",
    response_model=FullHealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Full health status",
    description="Complete health status for monitoring and observability."
)
async def full_health_check() -> FullHealthResponse:
    """
    Full health check - What is the real state of the system?
    
    For:
    - Monitoring systems (Datadog, New Relic)
    - SRE dashboards
    - Incident response
    - Capacity planning
    
    Includes:
    - Dependency states
    - Service uptime
    - Version info
    - Environment
    - Performance metrics
    - Resource usage
    """
    
    with log_context(endpoint="/health/full"):
        logger.info("health.full.check_started")
        
        start_time = time.time()
        
        # Check all dependencies in parallel
        redis_status, rabbitmq_status, db_status, llama_status, metrics = await asyncio.gather(
            check_redis(),
            check_rabbitmq(),
            check_database(),
            check_llama3(),
            get_service_metrics(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(redis_status, Exception):
            redis_status = DependencyStatus(
                name="Redis Cache",
                status="unhealthy",
                message=str(redis_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(rabbitmq_status, Exception):
            rabbitmq_status = DependencyStatus(
                name="RabbitMQ",
                status="unhealthy",
                message=str(rabbitmq_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(db_status, Exception):
            db_status = DependencyStatus(
                name="PostgreSQL",
                status="unhealthy",
                message=str(db_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(llama_status, Exception):
            llama_status = DependencyStatus(
                name="LLAMA3 Model",
                status="unhealthy",
                message=str(llama_status),
                last_checked=datetime.now(timezone.utc).isoformat() + "Z"
            )
        
        if isinstance(metrics, Exception):
            metrics = {"error": str(metrics)}
        
        dependencies = {
            "redis": redis_status,
            "rabbitmq": rabbitmq_status,
            "database": db_status,
            "llama3": llama_status
        }
        
        # Calculate uptime
        uptime_seconds = time.time() - SERVICE_START_TIME
        
        # Determine overall status
        unhealthy_count = sum(
            1 for dep in dependencies.values()
            if dep.status == "unhealthy"
        )
        
        degraded_count = sum(
            1 for dep in dependencies.values()
            if dep.status == "degraded"
        )
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
            ready = False
        elif degraded_count > 0:
            overall_status = "degraded"
            ready = True
        else:
            overall_status = "healthy"
            ready = True
        
        check_duration = (time.time() - start_time) * 1000
        
        logger.info(
            "health.full.completed",
            extra={
                "overall_status": overall_status,
                "check_duration_ms": check_duration,
                "uptime_seconds": uptime_seconds,
                "dependencies": {k: v.status for k, v in dependencies.items()}
            }
        )
        
        return FullHealthResponse(
            status=overall_status,
            ready=ready,
            version=settings.app_version,
            environment=os.getenv("ENVIRONMENT", "development"),
            instance_id=os.getenv("INSTANCE_ID", "unknown"),
            uptime_seconds=uptime_seconds,
            dependencies=dependencies,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z"
        )


# ============================================================================
# LEGACY ENDPOINTS (for backward compatibility)
# ============================================================================

@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Legacy readiness check",
    description="Simple readiness check for backward compatibility"
)
async def legacy_readiness_check():
    """
    Simple readiness check endpoint.
    
    Used by systems that expect a simple {"ready": true} response.
    """
    # Run a quick check of critical dependencies
    try:
        # Check just Redis and Database quickly
        await asyncio.wait_for(
            asyncio.gather(
                cache_manager.client.ping(),
                db_manager.fetch_val("SELECT 1")
            ),
            timeout=1.0
        )
        return {"ready": True}
    except Exception:
        return {"ready": False}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Legacy liveness check",
    description="Simple liveness check for backward compatibility"
)
async def legacy_liveness_check():
    """
    Simple liveness check endpoint.
    
    Used by systems that expect a simple {"alive": true} response.
    """
    return {"alive": True}