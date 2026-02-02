"""
Main FastAPI application - Updated with enterprise features.

New features:
1. REST API endpoint for frontend (/api/v1/generate)
2. Enterprise structured logging with correlation tracking
3. Multi-tier health checks (liveness, readiness, full)
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import time

from app.config import settings
from app.core.messaging import queue_manager
from app.core.cache import cache_manager
from app.core.database import db_manager
from app.models.schemas import AIRequest
from app.services.pipeline import default_pipeline
from app.api.v1 import stats
from app.api.v1 import results

# Import new structured logging
from app.utils.logging import get_logger, log_context

# Import routers
from app.api.v1 import health, generate

logger = get_logger(__name__)


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with structured logging"""
    
    correlation_id = str(uuid.uuid4())
    
    with log_context(correlation_id=correlation_id, operation="startup"):
        logger.info(
            "app.startup.started",
            extra={
                "service": settings.app_name,
                "version": settings.app_version,
                "debug": settings.debug
            }
        )
        
        # RabbitMQ
        try:
            await queue_manager.connect()
            logger.info("app.startup.rabbitmq.connected")
        except Exception as e:
            logger.critical("app.startup.rabbitmq.failed", exc_info=e)
            raise
        
        # Redis
        try:
            await cache_manager.connect()
            logger.info("app.startup.redis.connected")
        except Exception as e:
            logger.error("app.startup.redis.failed", exc_info=e)
            logger.warning("app.startup.redis.degraded_mode", message="Continuing without cache")
        
        # PostgreSQL
        try:
            await db_manager.connect()
            logger.info("app.startup.postgresql.connected")
        except Exception as e:
            logger.critical("app.startup.postgresql.failed", exc_info=e)
            raise
        
        # Start consumer
        logger.info("app.startup.consumer.starting")
        consumer_task = asyncio.create_task(consume_ai_requests())
        
        logger.info(
            "app.startup.completed",
            extra={"status": "ready"}
        )
        
        yield
        
        # Shutdown
        with log_context(correlation_id=str(uuid.uuid4()), operation="shutdown"):
            logger.info("app.shutdown.started")
            
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                logger.info("app.shutdown.consumer.cancelled")
            
            await queue_manager.disconnect()
            logger.info("app.shutdown.rabbitmq.disconnected")
            
            await cache_manager.disconnect()
            logger.info("app.shutdown.redis.disconnected")
            
            await db_manager.disconnect()
            logger.info("app.shutdown.postgresql.disconnected")
            
            logger.info("app.shutdown.completed")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered mobile app generation service with enterprise features",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE LOGGING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all HTTP requests with correlation tracking"""
    
    # Generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    
    start_time = time.time()
    
    with log_context(
        correlation_id=correlation_id,
        endpoint=request.url.path,
        method=request.method
    ):
        logger.info(
            "http.request.received",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        try:
            response = await call_next(request)
            
            duration_ms = (time.time() - start_time) * 1000
            
            logger.performance(
                "http.request.completed",
                duration_ms=duration_ms,
                extra={
                    "status_code": response.status_code,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "http.request.failed",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "duration_ms": duration_ms
                },
                exc_info=e
            )
            raise


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging"""
    
    logger.error(
        "app.exception.unhandled",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=exc
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "correlation_id": request.headers.get("X-Correlation-ID", "unknown")
        }
    )


# ============================================================================
# ROUTERS
# ============================================================================

# Health checks (multi-tier)
app.include_router(
    health.router,
    tags=["Health"]
)

# Generation endpoint (frontend API)
app.include_router(
    generate.router,
    prefix="/api/v1",
    tags=["Generation"]
)

# Results endpoint
app.include_router(
    results.router,
    prefix="/api/v1",
    tags=["Results"]
)

# Statistics endpoints
app.include_router(
    stats.router,
    prefix="/api/v1",
    tags=["Statistics"]
)

# Test routes (development only)
if settings.debug:
    from app.api.v1 import test_routes
    app.include_router(
        test_routes.router,
        prefix="/api/v1",
        tags=["Testing"]
    )

# ============================================================================
# RABBITMQ CONSUMER
# ============================================================================

async def consume_ai_requests():
    """Main consumer loop with structured logging"""
    
    logger.info(
        "consumer.started",
        extra={"queue": settings.rabbitmq_queue_ai_requests}
    )
    
    async def message_handler(message_body: dict):
        """Handle incoming AI request with correlation tracking"""
        
        try:
            request = AIRequest(**message_body)
            
            # Set up logging context for entire request processing
            with log_context(
                correlation_id=str(uuid.uuid4()),
                task_id=request.task_id,
                user_id=request.user_id,
                session_id=request.session_id
            ):
                logger.info(
                    "consumer.message.received",
                    extra={
                        "prompt_length": len(request.prompt),
                        "has_context": request.context is not None
                    }
                )
                
                # Send initial progress
                await default_pipeline.send_progress(
                    task_id=request.task_id,
                    socket_id=request.socket_id,
                    stage="analyzing",
                    progress=5,
                    message="Starting AI processing..."
                )
                
                try:
                    # Execute pipeline
                    result = await default_pipeline.execute(request)
                    
                    logger.info(
                        "consumer.message.completed",
                        extra={
                            "total_time_ms": result.get('total_time_ms', 0),
                            "cache_hit": result.get('cache_hit', False)
                        }
                    )
                    
                except Exception as pipeline_error:
                    logger.error(
                        "consumer.pipeline.failed",
                        exc_info=pipeline_error
                    )
                    
                    await default_pipeline.send_error(
                        task_id=request.task_id,
                        socket_id=request.socket_id,
                        error="Pipeline execution failed",
                        details=str(pipeline_error)
                    )
        
        except Exception as e:
            logger.error(
                "consumer.message.processing_failed",
                extra={"error_type": type(e).__name__},
                exc_info=e
            )
    
    try:
        await queue_manager.consume(
            queue_name=settings.rabbitmq_queue_ai_requests,
            callback=message_handler
        )
    except Exception as e:
        logger.critical(
            "consumer.failed",
            exc_info=e
        )
        raise


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": {
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "full": "/health/full"
        },
        "api": {
            "generate": "POST /api/v1/generate"
        }
    }


# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(
        "app.dev_server.starting",
        extra={
            "host": "0.0.0.0",
            "port": 8000,
            "reload": settings.debug
        }
    )
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )