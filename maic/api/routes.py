r"""FastAPI route handlers for the OpenAI-compatible API.

Defines all HTTP endpoints: chat completions (streaming and non-streaming),
model listing, model status polling, download/load/delete management, and
supported-model queries. Enforces single-inference concurrency via semaphore.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from maic.adapters.openai_adapter import OpenAIAdapter
from maic.api.decorators import require_model, timed
from maic.core.config import default_strategy, settings
from maic.core.hub_fetcher import fetch_hub_models
from maic.core.model_manager import (
    TOTAL_RAM_GB,
    ModelLoadError,
    ModelTooLargeError,
    model_manager,
)
from maic.schemas.openai import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ModelCard,
    ModelList,
    ModelsStatusResponse,
    ModelStatus,
    SupportedModelList,
    UsageInfo,
)

router = APIRouter()


def _model_tool_calling(mm: Any, model_id: str, downloaded: bool) -> bool | None:
    """Return tool_calling capability for a model, or None when undetermined.

    For the currently loaded model we trust the in-memory detection result.
    For downloaded-but-not-loaded models we read the tokenizer_config on disk.
    For not-yet-downloaded models we return None (capability unknown).
    """
    if not downloaded and mm.model_id != model_id:
        return None
    from maic.core.model_manager import ToolCallFormat

    fmt = mm.tool_call_format_for(model_id)
    return fmt not in (ToolCallFormat.NONE, ToolCallFormat.UNKNOWN)


class DownloadTracker:
    """Tracks background download state for model IDs.

    States a model can be in: not tracked, "downloading", "done", or an error message.
    Written from daemon threads (_bg_download), read from async route handlers.
    Python's GIL makes individual dict reads/writes atomic, so no explicit lock is needed.
    """

    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    def is_downloading(self, model_id: str) -> bool:
        r"""Return ``True`` if a background download is currently in progress for *model_id*."""
        return self._state.get(model_id) == "downloading"

    def error(self, model_id: str) -> str | None:
        """Return the error message if the last download failed, else None."""
        state = self._state.get(model_id, "")
        return state.removeprefix("error: ") if state.startswith("error: ") else None

    def set_downloading(self, model_id: str) -> None:
        r"""Mark *model_id* as currently downloading."""
        self._state[model_id] = "downloading"

    def set_done(self, model_id: str) -> None:
        r"""Mark *model_id* download as successfully completed."""
        self._state[model_id] = "done"

    def set_error(self, model_id: str, exc: Exception) -> None:
        r"""Record a download failure for *model_id* with the exception message."""
        self._state[model_id] = f"error: {exc}"


_downloads = DownloadTracker()
_quantizations = DownloadTracker()

# Semaphore: only one inference at a time (MLX model is not safe for concurrent use).
# Requests acquire this for the full duration of generation — including the entire
# streaming response — so overlapping requests queue instead of running concurrently.
_inference_semaphore = asyncio.Semaphore(1)


# ── /v1/models ────────────────────────────────────────────────────────────────


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """Return the currently loaded model in OpenAI models-list format.

    Includes a ``capabilities`` field so clients (e.g. OpenCode) can discover
    whether the active model supports tool calling without a separate request.
    """
    model_id = model_manager.model_id or settings.model_id
    capabilities = {"tool_calling": model_manager.supports_tool_calling}
    return ModelList(data=[ModelCard(id=model_id, capabilities=capabilities)])


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

    Two inference modes:
      Single (default): semaphore-guarded, one request at a time via stream_generate.
      Batch (batch_mode=true): concurrent requests via mlx_lm BatchGenerator.
    """
    if settings.batch_mode:
        return await _chat_completions_batch(req)
    return await _chat_completions_single(req)


async def _chat_completions_single(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Single-inference path: semaphore-guarded stream_generate.

    Concurrent requests are serialized (queued) on the inference semaphore rather
    than rejected: MLX/Metal cannot run two generations at once, and clients like
    OpenCode fire overlapping requests (e.g. a chat call plus a title-generation
    call). Queuing lets both succeed in turn instead of failing one of them.
    """

    def per_request_strategy(
        *, max_tokens: int, temperature: float, top_p: float
    ) -> dict[str, Any]:
        return default_strategy(
            max_tokens=req.max_tokens or max_tokens,
            temperature=req.temperature if req.temperature is not None else temperature,
            top_p=req.top_p if req.top_p is not None else top_p,
        )

    messages = OpenAIAdapter.messages_to_dicts(req.messages)

    # Tool-calling turn: enabled when the client sends tools and does not opt out
    # via tool_choice="none". MLX has no native function-calling API, so we let the
    # model's chat template inject the tool signatures, generate the full response,
    # then parse any <tool_call> blocks into OpenAI tool_calls. Generation is
    # buffered (not token-streamed) for these turns so tool calls can be parsed
    # whole; for stream=true we replay the parsed result as SSE chunks.
    tools = OpenAIAdapter.tools_to_dicts(req.tools) if req.tool_choice != "none" else None
    if tools:
        async with _inference_semaphore:
            stream = model_manager.generate(messages, per_request_strategy, tools=tools)
            text = await asyncio.to_thread(stream.collect)
            token_count = stream.token_count
            tps = stream.tokens_per_second
        clean_text, tool_calls = OpenAIAdapter.parse_tool_calls(
            text, model_manager.tool_call_format
        )

        if req.stream:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            return StreamingResponse(
                OpenAIAdapter.tool_call_to_sse(
                    req.model, chunk_id, clean_text, tool_calls, token_count, tps
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return OpenAIAdapter.build_tool_response(
            req.model, clean_text, tool_calls, token_count, tps, req.max_tokens
        )

    if req.stream:
        # Hold the inference semaphore for the ENTIRE streaming response, not just
        # the setup. MLX/Metal generation is not safe to run concurrently: the real
        # work happens later, when FastAPI consumes the SSE generator on a worker
        # thread. If the lock were released here (e.g. via `async with` around the
        # return), two overlapping streaming requests — such as an OpenCode chat
        # plus its title-generation call — would run stream_generate on separate
        # threads at once and abort the Metal command buffer. We acquire up front
        # and release in `finally` so generation is strictly serialized.
        await _inference_semaphore.acquire()
        try:
            stream = model_manager.generate(messages, per_request_strategy)
        except BaseException:
            _inference_semaphore.release()
            raise

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        async def _guarded_sse() -> AsyncGenerator[str, None]:
            try:
                async for chunk in OpenAIAdapter.stream_to_sse(stream, req.model, chunk_id):
                    yield chunk
            finally:
                _inference_semaphore.release()

        return StreamingResponse(
            _guarded_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with _inference_semaphore:
        stream = model_manager.generate(messages, per_request_strategy)
        return await asyncio.to_thread(
            OpenAIAdapter.stream_to_response,
            stream,
            req.model,
            req.max_tokens,
        )


async def _chat_completions_batch(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Batch-inference path: concurrent requests via BatchGenerator."""
    import json as _json
    import time as _time

    from maic.core.batch_manager import batch_manager

    if not batch_manager.is_running:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Batch inference manager is not running. Load a model first.",
                    "type": "server_error",
                }
            },
        )

    messages = OpenAIAdapter.messages_to_dicts(req.messages)
    token_ids: list[int] = model_manager.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )

    max_tokens = req.max_tokens or settings.max_tokens
    gen_kwargs = default_strategy(
        max_tokens=max_tokens,
        temperature=req.temperature if req.temperature is not None else settings.temperature,
        top_p=req.top_p if req.top_p is not None else settings.top_p,
    )
    sampler = gen_kwargs.get("sampler")

    token_gen = batch_manager.submit(token_ids, max_tokens, sampler=sampler)

    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(_time.time())

        async def _batch_sse() -> AsyncGenerator[str, None]:
            from maic.schemas.openai import (
                ChatCompletionChunk,
                ChunkChoice,
                DeltaMessage,
                UsageInfo,
            )

            first = ChatCompletionChunk(
                id=chunk_id, model=req.model, created=created,
                choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
            )
            yield f"data: {first.model_dump_json()}\n\n"

            prefix = (
                f'{{"id":"{chunk_id}","object":"chat.completion.chunk",'
                f'"created":{created},"model":{_json.dumps(req.model)},'
                f'"choices":[{{"index":0,"delta":{{"content":'
            )
            suffix = '},"finish_reason":null}]}'

            count = 0
            t0 = _time.perf_counter()
            async for tok in token_gen:
                count += 1
                yield f"data: {prefix}{_json.dumps(tok)}{suffix}\n\n"
            elapsed = _time.perf_counter() - t0
            tps = round(count / elapsed, 1) if elapsed > 0 else 0.0

            stop = ChatCompletionChunk(
                id=chunk_id, model=req.model, created=created,
                choices=[ChunkChoice(delta=DeltaMessage(), finish_reason="stop")],
                usage=UsageInfo(
                    completion_tokens=count, total_tokens=count,
                    tokens_per_second=tps,
                ),
            )
            yield f"data: {stop.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _batch_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: collect all tokens
    parts: list[str] = []
    async for tok in token_gen:
        parts.append(tok)
    text = "".join(parts)
    token_count = len(parts)
    finish_reason: Literal["stop", "length", "tool_calls"] = (
        "length" if req.max_tokens and token_count >= req.max_tokens else "stop"
    )

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatChoice(
                message=Message(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            completion_tokens=token_count,
            total_tokens=token_count,
            tokens_per_second=0.0,
        ),
    )


# ── Prompt cache management ───────────────────────────────────────────────────


@router.post("/v1/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Reset the KV prompt cache so the next request starts a fresh conversation."""
    model_manager.clear_cache()
    return {"status": "cleared"}


# ── Runtime settings ─────────────────────────────────────────────────────────


class SettingsUpdateRequest(BaseModel):
    r"""Partial update to runtime settings (KV cache + batch mode).

    Changing KV cache values invalidates the prompt cache, which is rebuilt on
    the next inference request with the new parameters.
    """

    max_kv_size: int | None = Field(None, ge=1, le=10000, description="KV cache size: 1-10000 tokens")
    kv_bits: int | None = Field(None, ge=4, le=8, description="KV quantization bits: 4-8")
    kv_group_size: int | None = Field(None, ge=8, le=512, description="KV group size: 8-512")
    batch_mode: bool | None = None


@router.post("/v1/settings")
async def update_settings(req: SettingsUpdateRequest) -> dict[str, str]:
    """Update runtime settings. Clears the prompt cache when KV values change."""
    changed = False
    if req.max_kv_size is not None and req.max_kv_size != settings.max_kv_size:
        settings.max_kv_size = req.max_kv_size if req.max_kv_size > 0 else None
        changed = True
    if req.kv_bits is not None and req.kv_bits != settings.kv_bits:
        settings.kv_bits = req.kv_bits if req.kv_bits > 0 else None
        changed = True
    if req.kv_group_size is not None and req.kv_group_size != settings.kv_group_size:
        settings.kv_group_size = req.kv_group_size
        changed = True
    if changed:
        model_manager.clear_cache()

    if req.batch_mode is not None and req.batch_mode != settings.batch_mode:
        settings.batch_mode = req.batch_mode
        if req.batch_mode and model_manager.is_loaded:
            from maic.core.batch_manager import batch_manager

            batch_manager.start(model_manager._model, model_manager._tokenizer)
        elif not req.batch_mode:
            from maic.core.batch_manager import batch_manager

            batch_manager.stop()

    return {"status": "updated"}


@router.get("/v1/settings")
async def get_settings() -> dict[str, Any]:
    """Return current runtime settings."""
    return {
        "max_kv_size": settings.max_kv_size,
        "kv_bits": settings.kv_bits,
        "kv_group_size": settings.kv_group_size,
        "batch_mode": settings.batch_mode,
    }


# ── Model management endpoints (for the UI) ─────────────────────────────────


class ModelActionRequest(BaseModel):
    r"""Request body for model management endpoints (download, load).

    Args:
        model_id (str): HuggingFace model identifier to act upon.
    """

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
    hub_ids = {item.id for item in hub_models}

    # Include locally downloaded models not in the Hub list
    local_ids = model_manager.local_model_ids()
    local_only = [mid for mid in local_ids if mid not in hub_ids]

    statuses = [
        ModelStatus(
            id=item.id,
            name=item.id.split("/")[-1],
            size_gb=item.size_gb,
            feasible=item.size_gb <= safe_limit,
            downloaded=(dl := model_manager.is_downloaded(item.id)),
            disk_gb=model_manager.disk_size_gb(item.id) if dl else None,
            active=model_manager.model_id == item.id,
            requires_token=item.gated,
            downloading=_downloads.is_downloading(item.id),
            download_error=_downloads.error(item.id),
            downloads=item.downloads,
            quantizing=_quantizations.is_downloading(item.id),
            quantization_info=model_manager.quantization_info(item.id) if dl else None,
            tool_calling=_model_tool_calling(model_manager, item.id, dl),
        )
        for item in sorted(hub_models, key=lambda x: x.size_gb)
    ]

    # Append locally downloaded models missing from Hub
    for mid in local_only:
        disk = model_manager.disk_size_gb(mid) or 0.0
        statuses.append(
            ModelStatus(
                id=mid,
                name=mid.split("/")[-1],
                size_gb=disk,
                feasible=disk <= safe_limit,
                downloaded=True,
                disk_gb=disk,
                active=model_manager.model_id == mid,
                requires_token=False,
                downloading=False,
                download_error=None,
                downloads=0,
                quantizing=_quantizations.is_downloading(mid),
                quantization_info=model_manager.quantization_info(mid),
                tool_calling=_model_tool_calling(model_manager, mid, True),
            )
        )

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


# ── Model quantization ───────────────────────────────────────────────────────


class QuantizeRequest(BaseModel):
    r"""Request body for model quantization.

    Args:
        model_id: Source model to quantize.
        q_bits: Bits per weight (default 4). Must be between 1 and 8.
        q_group_size: Quantization group size (default 64). Must be between 8 and 512.
        quant_predicate: Optional mixed-quant recipe name
            (e.g. ``"mixed_2_6"``, ``"mixed_3_6"``).
    """

    model_id: str
    q_bits: int = Field(4, ge=1, le=8, description="Quantization bits: 1-8")
    q_group_size: int = Field(64, ge=8, le=512, description="Group size: 8-512")
    quant_predicate: str | None = Field(
        None, pattern=r'^[a-zA-Z0-9_-]+$', description="Recipe name: alphanumerics, hyphens, underscores only"
    )


@router.post("/v1/models/quantize")
async def quantize_model(req: QuantizeRequest) -> dict[str, str]:
    """Start background quantization of a downloaded model."""
    mid = req.model_id
    if not model_manager.is_downloaded(mid):
        raise HTTPException(status_code=404, detail="Source model not found on disk.")

    if _quantizations.is_downloading(mid):
        return {"status": "already_quantizing"}

    def _bg_quantize() -> None:
        _quantizations.set_downloading(mid)
        try:
            model_manager.quantize_model(
                mid,
                q_bits=req.q_bits,
                q_group_size=req.q_group_size,
                quant_predicate=req.quant_predicate,
            )
            _quantizations.set_done(mid)
        except Exception as exc:
            _quantizations.set_error(mid, exc)

    threading.Thread(target=_bg_quantize, daemon=True).start()
    return {"status": "started"}
