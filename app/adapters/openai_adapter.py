from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal

import psutil

from app.core.model_manager import FEASIBLE_BY_RAM, TOKEN_REQUIRED_MODELS
from app.core.token_stream import TokenStream
from app.schemas.openai import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChunkChoice,
    DeltaMessage,
    Message,
    SupportedModel,
    SupportedModelList,
    UsageInfo,
)

# ── Adapter Pattern ───────────────────────────────────────────────────────────


class OpenAIAdapter:
    """
    Adapter pattern — translates between our internal types and the OpenAI wire format.

    The rest of the app uses TokenStream and plain dicts. Clients expect JSON shaped
    exactly like OpenAI's API. This class handles all that translation so neither
    ModelManager nor the routes need to know about JSON structure.

    All methods are static — this class holds no state, it's just a namespace for
    translation functions.
    """

    @staticmethod
    def messages_to_dicts(messages: list[Message]) -> list[dict[str, str]]:
        """
        Convert Pydantic Message objects into plain dicts that mlx_lm's chat
        template formatter understands (it expects {'role': ..., 'content': ...}).
        """
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def stream_to_response(
        stream: TokenStream,
        model_id: str,
        request_max_tokens: int | None = None,
    ) -> ChatCompletionResponse:
        """
        Collect the entire TokenStream into a single ChatCompletionResponse JSON object.
        Used when the client sends stream=false. Blocks until generation is complete.
        """
        text = stream.collect()
        token_count = stream.token_count
        finish_reason: Literal["stop", "length"] = (
            "length" if request_max_tokens and token_count >= request_max_tokens else "stop"
        )
        return ChatCompletionResponse(
            model=model_id,
            choices=[
                ChatChoice(
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                completion_tokens=token_count,
                total_tokens=token_count,
                tokens_per_second=round(stream.tokens_per_second, 1),
            ),
        )

    @staticmethod
    async def stream_to_sse(
        stream: TokenStream,
        model_id: str,
        chunk_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Convert a TokenStream into Server-Sent Events (SSE) for the client.

        SSE is a simple HTTP streaming protocol where each event is a line starting
        with 'data: ' followed by a JSON string, terminated by two newlines.
        OpenAI clients expect this exact format when stream=true.

        We send three kinds of chunks:
        1. First chunk — carries only the role ('assistant') so the client knows
           who is speaking before the first word arrives.
        2. Content chunks — one per token, carrying the actual text.
        3. Final chunk — carries finish_reason='stop' and no content, then
           the special sentinel 'data: [DONE]' to tell the client we're finished.
        """
        # First chunk — send role
        first_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        for token in stream:
            chunk = ChatCompletionChunk(
                id=chunk_id,
                model=model_id,
                choices=[ChunkChoice(delta=DeltaMessage(content=token))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk — signal stop + usage stats
        stop_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            choices=[ChunkChoice(delta=DeltaMessage(), finish_reason="stop")],
            usage=UsageInfo(
                completion_tokens=stream.token_count,
                total_tokens=stream.token_count,
                tokens_per_second=round(stream.tokens_per_second, 1),
            ),
        )
        yield f"data: {stop_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def build_supported_models() -> SupportedModelList:
        """
        Build the response for GET /v1/models/supported.

        Reads available system RAM, then for each known model computes whether it
        fits within 80% of that RAM. Returns the full list so a UI can show which
        models are usable on this machine right now and which require a token.
        """
        available_gb = psutil.virtual_memory().total / (1024**3)
        safe_limit = available_gb * 0.8

        models = [
            SupportedModel(
                id=mid,
                size_gb=size,
                min_ram_gb=round(size / 0.8, 1),
                feasible=size <= safe_limit,
                requires_token=mid in TOKEN_REQUIRED_MODELS,
            )
            for mid, size in FEASIBLE_BY_RAM
        ]
        return SupportedModelList(available_ram_gb=round(available_gb, 1), models=models)
