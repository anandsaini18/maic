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
from app.core.hub_fetcher import fetch_hub_models
from app.core.model_manager import (
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
    ModelsStatusResponse,
    ModelStatus,
    SupportedModelList,
)

router = APIRouter()


class DownloadTracker:
    """Tracks background download state for model IDs.

    States a model can be in: not tracked, "downloading", "done", or an error message.
    Written from daemon threads (_bg_download), read from async route handlers.
    Python's GIL makes individual dict reads/writes atomic, so no explicit lock is needed.
    """

    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    def is_downloading(self, model_id: str) -> bool:
        return self._state.get(model_id) == "downloading"

    def error(self, model_id: str) -> str | None:
        """Return the error message if the last download failed, else None."""
        state = self._state.get(model_id, "")
        return state.removeprefix("error: ") if state.startswith("error: ") else None

    def set_downloading(self, model_id: str) -> None:
        self._state[model_id] = "downloading"

    def set_done(self, model_id: str) -> None:
        self._state[model_id] = "done"

    def set_error(self, model_id: str, exc: Exception) -> None:
        self._state[model_id] = f"error: {exc}"


_downloads = DownloadTracker()

# Semaphore: only one inference at a time (MLX model is not safe for concurrent use)
_inference_semaphore = asyncio.Semaphore(1)


def _inference_busy() -> bool:
    """Non-blocking check whether the inference slot is currently occupied.

    asyncio.Semaphore has no public locked() method (unlike asyncio.Lock), so
    _value is read directly. This attribute has been stable across all CPython
    versions since asyncio was introduced (3.4+).
    """
    return _inference_semaphore._value == 0  # type: ignore[attr-defined]


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
    Return curated models with their RAM requirements and feasibility on this machine.
    Uses the static KNOWN_SIZES list from model_sizing for fast, offline-capable responses.
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
    if _inference_busy():
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


@router.get("/v1/models/status", response_model=ModelsStatusResponse)
async def models_status() -> ModelsStatusResponse:
    """
    Return full status for every model discovered from HuggingFace Hub.

    Each entry includes estimated size, RAM feasibility, and live local state
    (downloaded, active, downloading). The hub model list is cached for 5 minutes
    and falls back to 3 offline defaults when the Hub is unreachable.

    Uses the module-level TOTAL_RAM_GB constant (computed once at startup) to
    avoid psutil calls on every 3-second poll.
    """
    safe_limit = TOTAL_RAM_GB * 0.8
    hub_models = await fetch_hub_models()

    statuses = [
        ModelStatus(
            id=item.id,
            name=item.id.split("/")[-1],
            size_gb=item.size_gb,
            feasible=item.size_gb <= safe_limit,
            downloaded=(dl := model_manager.is_downloaded(item.id)),
            disk_gb=model_manager.disk_size_gb(item.id) if dl else None,
            active=model_manager.model_id == item.id,
            # item.gated is set from the HF API "gated" field; this is the
            # primary signal. HubModelCache also OR's in the static GATED_MODELS
            # frozenset as a safety-net, so requires_token is always correct.
            requires_token=item.gated,
            downloading=_downloads.is_downloading(item.id),
            download_error=_downloads.error(item.id),
            downloads=item.downloads,
        )
        for item in sorted(hub_models, key=lambda x: x.size_gb)
    ]

    return ModelsStatusResponse(
        available_ram_gb=round(TOTAL_RAM_GB, 1),
        active_model=model_manager.model_id,
        models=statuses,
    )


@router.post("/v1/models/download")
async def download_model(req: ModelActionRequest) -> dict[str, str]:
    """Trigger a background download for a model. Returns immediately."""
    mid = req.model_id
    if model_manager.is_downloaded(mid):
        return {"status": "already_downloaded"}

    if _downloads.is_downloading(mid):
        return {"status": "already_downloading"}

    def _bg_download() -> None:
        _downloads.set_downloading(mid)
        try:
            model_manager.download_model(mid)
            _downloads.set_done(mid)
        except Exception as exc:
            _downloads.set_error(mid, exc)

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
