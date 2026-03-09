from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

# ── Request ───────────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


# ── Non-streaming Response ────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Streaming Response (SSE chunks) ──────────────────────────────────────────


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]
    usage: UsageInfo | None = None


# ── Models list ───────────────────────────────────────────────────────────────


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ── Supported models list (custom endpoint) ───────────────────────────────────


class SupportedModel(BaseModel):
    id: str
    size_gb: float
    min_ram_gb: float
    feasible: bool
    requires_token: bool


class SupportedModelList(BaseModel):
    available_ram_gb: float
    models: list[SupportedModel]


# ── Model status (for /v1/models/status polling endpoint) ─────────────────────


class ModelStatus(BaseModel):
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
    available_ram_gb: float
    active_model: str | None
    models: list[ModelStatus]


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    message: str
    type: str = "server_error"
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
