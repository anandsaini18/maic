r"""Pydantic models for OpenAI-compatible request and response schemas.

Defines the wire format for chat completions (streaming and non-streaming),
model listings, supported-model queries, model status polling, and error
responses. All shapes mirror the OpenAI API so standard client libraries
work out of the box.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

# ── Request ───────────────────────────────────────────────────────────────────


class Message(BaseModel):
    r"""A single message in a chat conversation.

    Args:
        role (str): The speaker — one of ``"system"``, ``"user"``, or ``"assistant"``.
        content (str): The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    r"""Incoming request body for ``POST /v1/chat/completions``.

    Args:
        model (str): Model identifier (e.g. ``"mlx-community/Phi-3.5-mini-instruct-4bit"``).
        messages (list[Message]): The conversation history to continue from.
        max_tokens (int, optional): Maximum tokens to generate. Default: server config.
        temperature (float, optional): Sampling temperature (0.0–2.0). Default: server config.
        top_p (float, optional): Nucleus sampling cutoff (0.0–1.0).
          Default: server config.
        stream (bool): If ``True``, return SSE stream; otherwise return
          full JSON. Default: ``False``.
    """

    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


# ── Non-streaming Response ────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    r"""Token usage statistics included in completion responses.

    Args:
        prompt_tokens (int): Number of tokens in the prompt. Default: ``0``.
        completion_tokens (int): Number of tokens generated. Default: ``0``.
        total_tokens (int): Sum of prompt and completion tokens. Default: ``0``.
        tokens_per_second (float, optional): Generation throughput metric. Default: ``None``.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float | None = None


class ChatChoice(BaseModel):
    r"""A single completion choice in a non-streaming response.

    Args:
        index (int): Choice index (always ``0`` for single-choice responses). Default: ``0``.
        message (Message): The generated assistant message.
        finish_reason (str): Why generation stopped — ``"stop"`` or ``"length"``.
            Default: ``"stop"``.
    """

    index: int = 0
    message: Message
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    r"""Full non-streaming chat completion response (``stream=false``).

    Args:
        id (str): Unique completion identifier (auto-generated).
        object (str): Always ``"chat.completion"``.
        created (int): Unix timestamp of creation (auto-generated).
        model (str): Model identifier that produced this completion.
        choices (list[ChatChoice]): List of generated choices (typically one).
        usage (UsageInfo): Token usage statistics.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Streaming Response (SSE chunks) ──────────────────────────────────────────


class DeltaMessage(BaseModel):
    r"""Incremental message content within an SSE streaming chunk.

    Either ``role`` (first chunk) or ``content`` (subsequent chunks) is set,
    never both simultaneously.

    Args:
        role (str, optional): Set to ``"assistant"`` in the first chunk only.
        content (str, optional): A token fragment of the generated text.
    """

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    r"""A single choice within a streaming SSE chunk.

    Args:
        index (int): Choice index. Default: ``0``.
        delta (DeltaMessage): The incremental message content for this chunk.
        finish_reason (str, optional): Set to ``"stop"`` or ``"length"`` on the
            final chunk; ``None`` for content chunks.
    """

    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    r"""A single SSE chunk in a streaming chat completion response (``stream=true``).

    Args:
        id (str): Shared completion identifier across all chunks (auto-generated).
        object (str): Always ``"chat.completion.chunk"``.
        created (int): Unix timestamp of creation (auto-generated).
        model (str): Model identifier that produced this chunk.
        choices (list[ChunkChoice]): Chunk choices (typically one).
        usage (UsageInfo, optional): Included only in the final chunk with
            cumulative token stats.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]
    usage: UsageInfo | None = None


# ── Models list ───────────────────────────────────────────────────────────────


class ModelCard(BaseModel):
    r"""A single model entry in the ``GET /v1/models`` response.

    Args:
        id (str): HuggingFace model identifier.
        object (str): Always ``"model"``.
        owned_by (str): Model owner. Default: ``"local"``.
    """

    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    r"""Response body for ``GET /v1/models`` — lists the currently loaded model.

    Args:
        object (str): Always ``"list"``.
        data (list[ModelCard]): List of available model cards.
    """

    object: str = "list"
    data: list[ModelCard]


# ── Supported models list (custom endpoint) ───────────────────────────────────


class SupportedModel(BaseModel):
    r"""A model entry with RAM feasibility info for ``GET /v1/models/supported``.

    Args:
        id (str): HuggingFace model identifier.
        size_gb (float): Estimated model size in GiB.
        min_ram_gb (float): Minimum system RAM required (size / 0.8).
        feasible (bool): ``True`` if the model fits within 80% of available RAM.
        requires_token (bool): ``True`` if the model is gated and needs an HF token.
    """

    id: str
    size_gb: float
    min_ram_gb: float
    feasible: bool
    requires_token: bool


class SupportedModelList(BaseModel):
    r"""Response body for ``GET /v1/models/supported``.

    Args:
        available_ram_gb (float): Total system RAM in GiB.
        models (list[SupportedModel]): Curated models sorted by size with feasibility info.
    """

    available_ram_gb: float
    models: list[SupportedModel]


# ── Model status (for /v1/models/status polling endpoint) ─────────────────────


class ModelStatus(BaseModel):
    r"""Live status of a single model for the UI polling endpoint.

    Args:
        id (str): HuggingFace model identifier.
        name (str): Short display name (last segment of the model ID).
        size_gb (float): Estimated model size in GiB.
        feasible (bool): ``True`` if the model fits within 80% of available RAM.
        downloaded (bool): ``True`` if weight files exist on disk.
        disk_gb (float, optional): Actual disk usage in GiB, or ``None`` if not downloaded.
        active (bool): ``True`` if this model is currently loaded for inference.
        requires_token (bool): ``True`` if the model is gated and needs an HF token.
        downloading (bool): ``True`` if a background download is in progress.
        download_error (str, optional): Error message from the last failed download.
        downloads (int): HuggingFace Hub download count.
    """

    id: str
    name: str
    size_gb: float
    feasible: bool
    downloaded: bool
    disk_gb: float | None
    active: bool
    requires_token: bool
    downloading: bool
    download_error: str | None
    downloads: int


class ModelsStatusResponse(BaseModel):
    r"""Response body for ``GET /v1/models/status`` — polled by the UI every 3 seconds.

    Args:
        available_ram_gb (float): Total system RAM in GiB.
        active_model (str, optional): Currently loaded model ID, or ``None``.
        models (list[ModelStatus]): All discovered models with their live status.
    """

    available_ram_gb: float
    active_model: str | None
    models: list[ModelStatus]


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    r"""Inner error object matching the OpenAI error envelope format.

    Args:
        message (str): Human-readable error description.
        type (str): Error category (e.g. ``"server_error"``, ``"model_busy"``).
            Default: ``"server_error"``.
        code (str, optional): Machine-readable error code. Default: ``None``.
    """

    message: str
    type: str = "server_error"
    code: str | None = None


class ErrorResponse(BaseModel):
    r"""Top-level error response envelope returned on failure.

    Args:
        error (ErrorDetail): The error details.
    """

    error: ErrorDetail
