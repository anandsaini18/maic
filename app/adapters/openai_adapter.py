r"""Adapter layer translating between internal types and the OpenAI wire format.

Converts ``TokenStream`` output into OpenAI-compatible JSON for both streaming
(SSE) and non-streaming responses. Also builds the supported-models list with
RAM feasibility checks. All adapter methods are stateless and static.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Literal

from app.core.model_sizing import GATED_MODELS, KNOWN_SIZES_BY_RAM
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

# Cache for build_supported_models() — computed once at first call since
# available RAM is constant for the lifetime of the process.
_supported_models_cache: SupportedModelList | None = None


async def _async_token_iter(stream: TokenStream) -> AsyncGenerator[str, None]:
    """
    Offload the synchronous MLX token iterator to a thread pool so it does not
    block the asyncio event loop during inference.

    How it works:
    - A producer function runs `for token in stream` inside a ThreadPoolExecutor
      thread (where blocking CPU/GPU work is acceptable).
    - Each produced token is pushed into an asyncio.Queue via call_soon_threadsafe,
      keeping all queue interactions on the event loop thread.
    - The async generator awaits tokens from the queue, yielding control to the
      event loop between tokens so other requests can be served concurrently.
    - None signals clean end-of-stream; an Exception instance signals an error.

    Exception propagation (ExceptionWrapper pattern):
    - If the producer raises, the exception object itself is pushed into the queue
      instead of being swallowed by the unawaited run_in_executor Future.
    - The async consumer checks each item: strings are yielded, None breaks the
      loop cleanly, and a BaseException instance is re-raised on the event loop
      so the SSE handler can surface a proper HTTP 500 to the client.

    The queue has a bounded size (32) to provide light back-pressure: if the
    consumer (SSE writer) falls behind the producer, the producer pauses rather
    than buffering the entire response in memory.
    """
    loop = asyncio.get_running_loop()
    # Queue carries tokens (str), a clean-end sentinel (None), or a forwarded
    # exception (BaseException) — never both None and an exception for the same error.
    queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue(maxsize=32)

    def _produce() -> None:
        try:
            for token in stream:
                # Use asyncio.run_coroutine_threadsafe with queue.put() (not put_nowait)
                # so the producer thread blocks if the queue is full, applying natural
                # back-pressure instead of raising QueueFull. This is essential for
                # handling slow consumers without buffering the entire response.
                future = asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                future.result()  # blocks until token is enqueued
        except Exception as exc:
            # Push the live exception into the queue so the async side can
            # re-raise it on the event loop thread.  Do NOT raise here — the
            # Future returned by run_in_executor is never awaited, so any
            # exception raised in _produce would be silently discarded.
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        else:
            # Clean completion: send the None sentinel exactly once.
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    loop.run_in_executor(None, _produce)

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


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

        The route handler is responsible for running this in a thread pool
        (via asyncio.to_thread) so the synchronous MLX iteration does not block
        the asyncio event loop.
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

        Performance: The first and last chunks use Pydantic serialization (two
        allocations per request). Content chunks are built with a pre-computed
        string template to avoid creating Pydantic objects on every token.
        """
        created = int(time.time())

        # First chunk — send role (Pydantic used here; only happens once per request)
        first_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            created=created,
            choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Pre-compute the static parts of the content chunk JSON frame.
        # Each content token only varies in the "content" field, so we build
        # everything else once and splice the token in with json.dumps() for
        # correct JSON escaping (handles quotes, backslashes, unicode, etc.).
        # This avoids constructing a full Pydantic object + calling model_dump_json()
        # on every single token — the hot path in a long streaming response.
        chunk_prefix = (
            f'{{"id":"{chunk_id}","object":"chat.completion.chunk",'
            f'"created":{created},"model":{json.dumps(model_id)},'
            f'"choices":[{{"index":0,"delta":{{"content":'
        )
        chunk_suffix = '},"finish_reason":null}]}'

        # Iterate tokens from a thread so MLX inference doesn't block the event loop.
        # No asyncio.sleep(0) needed — each 'await queue.get()' inside
        # _async_token_iter already yields control back to the event loop.
        async for token in _async_token_iter(stream):
            yield f"data: {chunk_prefix}{json.dumps(token)}{chunk_suffix}\n\n"

        # Final chunk — signal stop + usage stats (Pydantic; once per request)
        stop_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            created=created,
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

        Computes which models fit within 80% of available system RAM. Since RAM is
        constant for the process lifetime, the result is cached after the first call.
        """
        global _supported_models_cache
        if _supported_models_cache is not None:
            return _supported_models_cache

        # Use the module-level constant cached at startup instead of re-calling psutil
        from app.core.model_manager import TOTAL_RAM_GB

        available_gb = TOTAL_RAM_GB
        safe_limit = available_gb * 0.8

        models = [
            SupportedModel(
                id=mid,
                size_gb=size,
                min_ram_gb=round(size / 0.8, 1),
                feasible=size <= safe_limit,
                requires_token=mid in GATED_MODELS,
            )
            for mid, size in KNOWN_SIZES_BY_RAM
        ]
        _supported_models_cache = SupportedModelList(
            available_ram_gb=round(available_gb, 1), models=models
        )
        return _supported_models_cache
