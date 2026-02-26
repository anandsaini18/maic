from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# ── Decorator Pattern ─────────────────────────────────────────────────────────

def require_model(func: Callable) -> Callable:
    """
    Decorator that blocks a route with HTTP 503 if the model isn't loaded yet.

    Without this, a request that arrives before startup finishes (or if load()
    failed) would hit model_manager.generate() with _model=None and produce a
    confusing AttributeError. This gives the client a clear, actionable error
    instead.

    Usage:
        @router.post("/v1/chat/completions")
        @require_model
        async def chat(...): ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Import here to avoid circular imports at module load time
        from app.core.model_manager import model_manager
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "Model is not loaded. Server may still be starting up.",
                        "type": "service_unavailable",
                    }
                },
            )
        return await func(*args, **kwargs)

    return wrapper


def timed(func: Callable) -> Callable:
    """
    Decorator that logs how long each request takes (wall-clock time in seconds).

    Timing is attached here at the decorator layer so the route handlers
    themselves stay focused on business logic. Logs at DEBUG level so it
    doesn't clutter normal output — set LOG_LEVEL=DEBUG to see it.

    Usage:
        @router.post("/v1/chat/completions")
        @timed
        async def chat(...): ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.debug("%s completed in %.3fs", func.__name__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.debug("%s failed after %.3fs", func.__name__, elapsed)
            raise

    return wrapper
