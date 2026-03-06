from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.adapters.openai_adapter import OpenAIAdapter
from app.api.decorators import require_model, timed
from app.core.config import default_strategy, settings
from app.core.model_manager import (
    KNOWN_MODEL_SIZES,
    TOKEN_REQUIRED_MODELS,
    TOTAL_RAM_GB,
    ModelLoadError,
    ModelTooLargeError,
    model_manager,
)
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
    SupportedModelList,
)

router = APIRouter()

# Track background download state so the UI can poll progress
_download_state: dict[str, str] = {}  # model_id → "downloading" | "done" | "error: ..."

# Semaphore: only one inference at a time (MLX model is not safe for concurrent use)
_inference_semaphore = asyncio.Semaphore(1)


# ── /v1/models ────────────────────────────────────────────────────────────────


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """Return the currently loaded model in OpenAI models-list format."""
    model_id = model_manager.model_id or settings.model_id
    return ModelList(data=[ModelCard(id=model_id)])


# ── /v1/models/supported ─────────────────────────────────────────────────────


@router.get("/v1/models/supported", response_model=SupportedModelList)
async def list_supported_models() -> SupportedModelList:
    """
    Return all known models with their RAM requirements and whether they are
    feasible on this machine (based on available unified memory).
    """
    return OpenAIAdapter.build_supported_models()


# ── /v1/chat/completions ──────────────────────────────────────────────────────


@router.post("/v1/chat/completions", response_model=None)
@require_model  # Decorator: returns 503 if model not loaded
@timed  # Decorator: logs wall-clock time per request
async def chat_completions(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint.

    - stream=false  → returns full ChatCompletionResponse JSON
    - stream=true   → returns SSE stream (text/event-stream)

    Concurrency: only one inference runs at a time (MLX model singleton).
    A second concurrent request receives HTTP 503 "model_busy" immediately.

    Patterns in play:
      Strategy  — default_strategy (swappable generation config)
      Facade    — model_manager.generate() hides all MLX internals
      Iterator  — TokenStream consumed by OpenAIAdapter
      Adapter   — OpenAIAdapter converts TokenStream ↔ OpenAI wire format
    """
    # Reject concurrent inference immediately rather than queuing unboundedly.
    # A client that wants to retry can simply re-send after a short delay.
    if not _inference_semaphore._value:  # non-blocking peek
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Another inference is already in progress. Retry shortly.",
                    "type": "model_busy",
                }
            },
        )

    # Override per-request generation params if provided
    def per_request_strategy(
        *, max_tokens: int, temperature: float, top_p: float
    ) -> dict[str, Any]:
        return default_strategy(
            max_tokens=req.max_tokens or max_tokens,
            temperature=req.temperature if req.temperature is not None else temperature,
            top_p=req.top_p if req.top_p is not None else top_p,
        )

    messages = OpenAIAdapter.messages_to_dicts(req.messages)

    async with _inference_semaphore:
        stream = model_manager.generate(messages, per_request_strategy)  # Facade

        if req.stream:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            return StreamingResponse(
                OpenAIAdapter.stream_to_sse(stream, req.model, chunk_id),  # Adapter
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Run blocking stream.collect() in a thread so MLX iteration doesn't
        # freeze the event loop during non-streaming responses.
        return await asyncio.to_thread(
            OpenAIAdapter.stream_to_response,
            stream,
            req.model,
            req.max_tokens,
        )


# ── Model management endpoints (for the UI) ─────────────────────────────────


class ModelActionRequest(BaseModel):
    model_id: str


@router.get("/v1/models/status")
async def models_status() -> dict[str, Any]:
    """
    Return full status for every known model: size, feasibility, download state,
    whether it's the currently active model, and if a HuggingFace token is needed.
    The UI polls this to render the model selector.

    Uses the module-level TOTAL_RAM_GB constant (computed once at startup)
    instead of calling psutil.virtual_memory() on every poll.
    """
    safe_limit = TOTAL_RAM_GB * 0.8

    models = []
    for mid, size in sorted(KNOWN_MODEL_SIZES.items(), key=lambda x: x[1]):
        downloaded = model_manager.is_downloaded(mid)
        models.append(
            {
                "id": mid,
                "name": mid.split("/")[-1],
                "size_gb": size,
                "feasible": size <= safe_limit,
                "downloaded": downloaded,
                "disk_gb": model_manager.disk_size_gb(mid) if downloaded else None,
                "active": model_manager.model_id == mid,
                "requires_token": mid in TOKEN_REQUIRED_MODELS,
                "downloading": _download_state.get(mid) == "downloading",
                "download_error": (
                    _download_state[mid].removeprefix("error: ")
                    if _download_state.get(mid, "").startswith("error:")
                    else None
                ),
            }
        )

    return {
        "available_ram_gb": round(TOTAL_RAM_GB, 1),
        "active_model": model_manager.model_id,
        "models": models,
    }


@router.post("/v1/models/download")
async def download_model(req: ModelActionRequest) -> dict[str, str]:
    """Trigger a background download for a model. Returns immediately."""
    mid = req.model_id
    if model_manager.is_downloaded(mid):
        return {"status": "already_downloaded"}

    if _download_state.get(mid) == "downloading":
        return {"status": "already_downloading"}

    def _bg_download() -> None:
        _download_state[mid] = "downloading"
        try:
            model_manager.download_model(mid)
            _download_state[mid] = "done"
        except Exception as exc:
            _download_state[mid] = f"error: {exc}"

    threading.Thread(target=_bg_download, daemon=True).start()
    return {"status": "started"}


@router.post("/v1/models/load")
async def load_model(req: ModelActionRequest) -> dict[str, str]:
    """
    Switch the active model. Downloads first if not already on disk.

    Runs model loading in a thread pool so the blocking MLX weight-loading
    (disk I/O + memory mapping) does not freeze the asyncio event loop.
    """
    mid = req.model_id
    if model_manager.model_id == mid:
        return {"status": "already_active"}

    try:
        await asyncio.to_thread(model_manager.load, mid)
    except ModelTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "loaded", "active_model": mid}


@router.delete("/v1/models/{model_id:path}")
async def delete_model(model_id: str) -> dict[str, str]:
    """Delete a downloaded model's weights from disk."""
    if not model_manager.is_downloaded(model_id):
        raise HTTPException(status_code=404, detail="Model not found on disk.")

    try:
        model_manager.delete_model(model_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"status": "deleted"}
