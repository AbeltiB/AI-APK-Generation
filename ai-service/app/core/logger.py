"""
Logging configuration using Loguru.

Provides structured, colorized logging with automatic rotation and retention.
"""
import sys
from pathlib import Path
from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """
    Configure loguru logger with appropriate handlers and formatting.
    
    Development mode:
    - Colorized console output
    - DEBUG level
    - Detailed format with file:line info
    
    Production mode:
    - Console output (for container logs)
    - File output with rotation
    - INFO level
    - JSON-compatible format
    """
    
    # Remove default handler
    logger.remove()
    
    # Development format: colorized, detailed
    dev_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add file handler with rotation
    logger.add(
        "logs/detailed_{time}.log",
        format=dev_format,
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
        compression="zip"
    )
    
    # Console handler
    logger.add(
        sys.stdout,
        format=dev_format,
        level=settings.log_level,
        colorize=True
    )
    
    # Production format: structured, parseable
    prod_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Console handler (always enabled)
    logger.add(
        sys.stdout,
        format=dev_format if settings.debug else prod_format,
        level=settings.log_level,
        colorize=settings.debug,
        backtrace=settings.debug,
        diagnose=settings.debug,
    )
    
    # File handler (production only)
    if not settings.debug:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "ai-service.log",
            format=prod_format,
            level="INFO",
            rotation="500 MB",  # Rotate when file reaches 500MB
            retention="10 days",  # Keep logs for 10 days
            compression="zip",  # Compress rotated logs
            enqueue=True,  # Async logging
        )
    
    # Log startup info
    logger.info(f"Logging configured - Level: {settings.log_level}")
    logger.debug(f"Debug mode: {settings.debug}")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> from app.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting service")
    """
    return logger.bind(name=name)


if __name__ == "__main__":
    # Test logging configuration
    setup_logging()
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.success("This is a SUCCESS message")
    
    # Test structured logging
    logger.info("User action", user_id="user_123", action="login", success=True)