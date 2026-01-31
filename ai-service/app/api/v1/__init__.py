"""
API v1 endpoints.
"""

from .health import router as health_router
from .generate import router as generate_router
from .stats import router as stats_router
from .results import router as results_router

__all__ = ["health_router", "generate_router", "stats_router", "results_router"]