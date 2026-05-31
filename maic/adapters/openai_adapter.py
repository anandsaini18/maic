r"""Adapter layer translating between internal types and the OpenAI wire format.

Converts ``TokenStream`` output into OpenAI-compatible JSON for both streaming
(SSE) and non-streaming responses. Also builds the supported-models list with
RAM feasibility checks. All adapter methods are stateless and static.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from maic.core.model_manager import ToolCallFormat
from maic.core.model_sizing import GATED_MODELS, KNOWN_SIZES_BY_RAM
from maic.core.token_stream import TokenStream
from maic.schemas.openai import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChunkChoice,
    DeltaMessage,
    DeltaToolCall,
    DeltaToolCallFunction,
    FunctionCall,
    Message,
    SupportedModel,
    SupportedModelList,
    Tool,
    ToolCall,
    UsageInfo,
)

# ── Adapter Pattern ───────────────────────────────────────────────────────────

# ── Per-format regex patterns ─────────────────────────────────────────────────
#
# Each pattern captures the raw JSON payload (group 1).  Format-specific parsers
# use these instead of trying every pattern on every response.

# Strips Qwen3-style chain-of-thought blocks before any tool-call parsing.
# The block is emitted before the actual response content; leaving it in place
# confuses every downstream parser and is the primary source of parse failures.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Qwen / Hermes: <tool_call>{…}</tool_call>  (multiple calls → multiple blocks)
_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Llama-3: <|python_tag|>[{…}]  — the tag precedes a JSON array
_LLAMA3_TOOL_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*(\[.*?\]|\{.*?\})", re.DOTALL
)

# Mistral / Mixtral: [TOOL_CALLS] [{…}, …]
_MISTRAL_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\]|\{.*?\})", re.DOTALL
)

# DeepSeek-Coder-V2: <|tool▁calls▁begin|>…<|tool▁sep|>{…}<|tool▁calls▁end|>
# Each call is surrounded by <|tool▁sep|>…<|tool▁calls▁end|> or ends at the
# next <|tool▁sep|>.  We capture the whole section and split by the separator.
_DEEPSEEK_SECTION_RE = re.compile(
    r"<\|tool\u2581calls\u2581begin\|>(.*?)<\|tool\u2581calls\u2581end\|>",
    re.DOTALL,
)
_DEEPSEEK_SEP = "<|tool\u2581sep|>"

# Generic XML-tag fallback: any <tag>{…}</tag> whose body contains "name"
_GENERIC_TAG_RE = re.compile(r"<([a-zA-Z_][\w-]*)>\s*(.*?)\s*</\1>", re.DOTALL)

# Fenced-code-block fallback: ```json {…} ``` or ```xml {…} ```
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)\s*```", re.DOTALL)

# Kept for backwards compat: alias to the primary Qwen pattern
_TOOL_CALL_RE = _QWEN_TOOL_CALL_RE


def _loads_tool_json(raw: str) -> Any:
    """Best-effort parse of a model-emitted tool-call payload into JSON.

    Tolerates leading language hints (``xml``/``json``), surrounding prose, and
    the doubled-brace artifact (``{{ ... }}``) some quantized models copy from the
    chat template's example. Returns the parsed object/list, or None if nothing
    JSON-like could be recovered.
    """
    text = raw.strip()
    # Try direct parse first — handles clean objects AND arrays like [{"name":…}]
    # (Llama-3 emits array-format tool calls that brace-slicing would silently drop).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to brace-extraction for wrapped/dirty content (surrounding prose, etc.)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start : end + 1]
    attempts = [candidate]
    if candidate.startswith("{{") and candidate.endswith("}}"):
        # Drop one wrapping brace from each side: {{...}} → {...}
        attempts.append(candidate[1:-1])
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def _tool_calling_caps(model_id: str, mm: Any) -> dict[str, bool] | None:
    """Return capability flags for *model_id*, or ``None`` when unknown.

    Returns a dict with ``tool_calling`` and ``thinking`` boolean flags for
    models that are downloaded on disk.  Returns ``None`` when the model is not
    downloaded and we cannot determine capability from disk.

    Uses the model manager to check the tool-call format — either from the
    loaded model (fast path) or by reading the tokenizer_config.json from disk.
    """
    fmt = mm.tool_call_format_for(model_id)
    if fmt == ToolCallFormat.NONE:
        # NONE can mean "not downloaded" (no tokenizer_config.json) or genuinely no tools.
        # Only return False (no tools) when the model IS downloaded but template lacks markers.
        from maic.core.model_manager import _model_local_path

        local_path = _model_local_path(model_id)
        if (local_path / "tokenizer_config.json").exists():
            thinking = _thinking_cap(model_id, mm)
            return {"tool_calling": False, "thinking": thinking}
        return None  # Not downloaded → capability unknown
    if fmt == ToolCallFormat.UNKNOWN:
        return None  # Template present but format unrecognised → unknown
    thinking = _thinking_cap(model_id, mm)
    return {"tool_calling": True, "thinking": thinking}


def _thinking_cap(model_id: str, mm: Any) -> bool:  # noqa: ANN401
    """Return whether *model_id* supports thinking mode.

    For the currently loaded model, reads the cached ``supports_thinking``
    property.  For other downloaded models, reads ``tokenizer_config.json``
    from disk and calls ``detect_thinking_support``.
    """
    if model_id == mm.model_id:
        return bool(mm.supports_thinking)
    from maic.core.model_manager import _model_local_path, detect_thinking_support

    local_path = _model_local_path(model_id)
    config_path = local_path / "tokenizer_config.json"
    if not config_path.exists():
        return False
    try:
        import json as _json

        cfg = _json.loads(config_path.read_text())
        template = cfg.get("chat_template", "") or ""
        return detect_thinking_support(template)
    except Exception:
        return False


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
    def messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convert Pydantic Message objects into plain dicts that mlx_lm's chat
        template formatter understands.

        Handles the tool-calling shapes too:
        - Structured ``content`` (a list of parts) is flattened to plain text.
        - Assistant ``tool_calls`` are passed through with ``arguments`` decoded from
          OpenAI's JSON-string form back into an object, because the chat template
          serializes arguments itself (``arguments | tojson``).
        - ``tool`` role messages carry their result as ``content``.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            content = m.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
            entry: dict[str, Any] = {"role": m.role, "content": content or ""}
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.name:
                entry["name"] = m.name
            if m.tool_calls:
                calls: list[dict[str, Any]] = []
                for tc in m.tool_calls:
                    raw = tc.function.arguments
                    try:
                        args_obj = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
                    except (json.JSONDecodeError, TypeError):
                        args_obj = {}
                    calls.append(
                        {
                            "id": tc.id,
                            "function": {"name": tc.function.name, "arguments": args_obj},
                        }
                    )
                entry["tool_calls"] = calls
            out.append(entry)
        return out

    @staticmethod
    def tools_to_dicts(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
        """Convert request Tool objects to plain dicts for the chat template.

        The Qwen template dumps each tool verbatim (``tool | tojson``), so the
        OpenAI ``{"type": "function", "function": {...}}`` shape is passed through.
        Returns None when there are no tools (the template then skips the tools block).
        """
        if not tools:
            return None
        return [t.model_dump(exclude_none=True) for t in tools]

    @staticmethod
    def parse_tool_calls(
        text: str,
        fmt: ToolCallFormat = ToolCallFormat.UNKNOWN,
    ) -> tuple[str, list[ToolCall]]:
        """Extract tool calls the model emitted as JSON into OpenAI tool calls.

        Dispatches to a format-specific sub-parser based on ``fmt`` (detected at
        model-load time from the chat template).  When ``fmt`` is ``UNKNOWN`` the
        original try-all fallback chain is used so new/unexpected models still work.

        Returns ``(text_without_tool_calls, tool_calls)``.  If nothing parses as a
        tool call, ``tool_calls`` is empty and the original text is returned unchanged.
        """
        # Strip any chain-of-thought block emitted by thinking-capable models
        # (e.g. Qwen3).  This must happen before format-specific parsing because
        # <think>…</think> content confuses every pattern-match below.
        text = _THINK_RE.sub("", text).strip()

        if fmt == ToolCallFormat.QWEN or fmt == ToolCallFormat.HERMES:
            return OpenAIAdapter._parse_qwen(text)
        if fmt == ToolCallFormat.LLAMA3:
            return OpenAIAdapter._parse_llama3(text)
        if fmt == ToolCallFormat.MISTRAL:
            return OpenAIAdapter._parse_mistral(text)
        if fmt == ToolCallFormat.DEEPSEEK:
            return OpenAIAdapter._parse_deepseek(text)
        # NONE means no tool support — shouldn't reach here, but handle gracefully
        if fmt == ToolCallFormat.NONE:
            return text, []
        # UNKNOWN / default: original try-all fallback chain
        return OpenAIAdapter._parse_fallback(text)

    # ── Format-specific parsers ───────────────────────────────────────────────

    @staticmethod
    def _make_tool_call(item: dict[str, Any]) -> ToolCall | None:
        """Build a ToolCall from a parsed dict, or None if ``name`` is missing."""
        name = item.get("name")
        if not name:
            return None
        # Accept both "arguments" (Qwen/Mistral) and "parameters" (Llama3)
        arguments = item.get("arguments") or item.get("parameters", {})
        args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            function=FunctionCall(name=name, arguments=args_str),
        )

    @staticmethod
    def _collect_regions(
        text: str, regions: list[tuple[str | None, str]]
    ) -> tuple[str, list[ToolCall]]:
        """Parse (full_match, body) pairs into ToolCalls and strip them from text."""
        tool_calls: list[ToolCall] = []
        strip: list[str] = []
        whole_text_used = False

        for full, body in regions:
            obj = _loads_tool_json(body)
            if obj is None:
                continue
            added = False
            for item in obj if isinstance(obj, list) else [obj]:
                if not isinstance(item, dict):
                    continue
                tc = OpenAIAdapter._make_tool_call(item)
                if tc is not None:
                    tool_calls.append(tc)
                    added = True
            if added:
                if full is None:
                    whole_text_used = True
                else:
                    strip.append(full)

        if not tool_calls:
            return text, []
        if whole_text_used:
            return "", tool_calls
        clean = text
        for span in strip:
            clean = clean.replace(span, "")
        return clean.strip(), tool_calls

    @staticmethod
    def _parse_qwen(text: str) -> tuple[str, list[ToolCall]]:
        """Parse Qwen/Hermes <tool_call>…</tool_call> blocks."""
        blocks = list(_QWEN_TOOL_CALL_RE.finditer(text))
        if not blocks:
            # Quantized models sometimes drift to a different tag name; fall through
            return OpenAIAdapter._parse_fallback(text)
        regions: list[tuple[str | None, str]] = [(m.group(0), m.group(1)) for m in blocks]
        return OpenAIAdapter._collect_regions(text, regions)

    @staticmethod
    def _parse_llama3(text: str) -> tuple[str, list[ToolCall]]:
        """Parse Llama-3 <|python_tag|>[{…}] tool calls."""
        m = _LLAMA3_TOOL_CALL_RE.search(text)
        if not m:
            return OpenAIAdapter._parse_fallback(text)
        # Strip everything from the <|python_tag|> onwards — it's all tool call data
        full_match = text[m.start():]
        regions: list[tuple[str | None, str]] = [(full_match, m.group(1))]
        return OpenAIAdapter._collect_regions(text, regions)

    @staticmethod
    def _parse_mistral(text: str) -> tuple[str, list[ToolCall]]:
        """Parse Mistral/Mixtral [TOOL_CALLS] [{…}] sections."""
        m = _MISTRAL_TOOL_CALL_RE.search(text)
        if not m:
            return OpenAIAdapter._parse_fallback(text)
        regions: list[tuple[str | None, str]] = [(m.group(0), m.group(1))]
        return OpenAIAdapter._collect_regions(text, regions)

    @staticmethod
    def _parse_deepseek(text: str) -> tuple[str, list[ToolCall]]:
        """Parse DeepSeek <|tool▁calls▁begin|>…<|tool▁calls▁end|> sections."""
        m = _DEEPSEEK_SECTION_RE.search(text)
        if not m:
            return OpenAIAdapter._parse_fallback(text)
        section = m.group(1)
        # Split by the separator; each part is a separate tool call JSON object
        parts = section.split(_DEEPSEEK_SEP)
        regions: list[tuple[str | None, str]] = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                regions.append((None, stripped))
        clean, calls = OpenAIAdapter._collect_regions(text, regions)
        # Replace the entire DeepSeek section in the original text
        if calls:
            clean = text[: m.start()].strip()
        return clean, calls

    @staticmethod
    def _parse_fallback(text: str) -> tuple[str, list[ToolCall]]:
        """Try-all fallback chain for UNKNOWN or unexpected model output.

        Order: Qwen <tool_call> tags → generic XML tag → fenced block → bare JSON.
        This was the original parse_tool_calls implementation and is preserved as
        the universal fallback for models whose format we haven't classified yet.
        """
        fallback_regions: list[tuple[str | None, str]] = []
        tool_call_blocks = list(_QWEN_TOOL_CALL_RE.finditer(text))
        if tool_call_blocks:
            fallback_regions = [(m.group(0), m.group(1)) for m in tool_call_blocks]
        else:
            generic = [m for m in _GENERIC_TAG_RE.finditer(text) if '"name"' in m.group(2)]
            fenced = [m for m in _FENCE_RE.finditer(text) if '"name"' in m.group(1)]
            if generic:
                fallback_regions = [(m.group(0), m.group(2)) for m in generic]
            elif fenced:
                fallback_regions = [(m.group(0), m.group(1)) for m in fenced]
            elif '"name"' in text and '"arguments"' in text:
                fallback_regions = [(None, text)]

        return OpenAIAdapter._collect_regions(text, fallback_regions)

    @staticmethod
    def build_tool_response(
        model_id: str,
        content: str,
        tool_calls: list[ToolCall],
        token_count: int,
        tokens_per_second: float,
        request_max_tokens: int | None = None,
    ) -> ChatCompletionResponse:
        """Build a non-streaming response that may include tool calls.

        When ``tool_calls`` is non-empty, ``finish_reason`` is ``"tool_calls"`` and
        ``content`` is set to ``None`` if empty (matching OpenAI). Otherwise this
        behaves like a normal text response with stop/length finish reasons.
        """
        finish_reason: Literal["stop", "length", "tool_calls"]
        if tool_calls:
            finish_reason = "tool_calls"
            message = Message(
                role="assistant",
                content=content or None,
                tool_calls=tool_calls,
            )
        else:
            finish_reason = (
                "length" if request_max_tokens and token_count >= request_max_tokens else "stop"
            )
            message = Message(role="assistant", content=content)
        return ChatCompletionResponse(
            model=model_id,
            choices=[ChatChoice(message=message, finish_reason=finish_reason)],
            usage=UsageInfo(
                completion_tokens=token_count,
                total_tokens=token_count,
                tokens_per_second=round(tokens_per_second, 1),
            ),
        )

    @staticmethod
    async def tool_call_to_sse(
        model_id: str,
        chunk_id: str,
        content: str,
        tool_calls: list[ToolCall],
        token_count: int,
        tokens_per_second: float,
    ) -> AsyncGenerator[str, None]:
        """Emit an already-generated tool-call/text result as SSE chunks.

        Generation for tool turns is buffered (not token-streamed), so this sends
        the role chunk, then any leading text, then one delta per tool call (each
        carrying id + name + full arguments), then the terminal chunk with
        ``finish_reason="tool_calls"`` (or ``"stop"`` when there were no tool calls).
        """
        created = int(time.time())

        first = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            created=created,
            choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {first.model_dump_json()}\n\n"

        if content:
            content_chunk = ChatCompletionChunk(
                id=chunk_id,
                model=model_id,
                created=created,
                choices=[ChunkChoice(delta=DeltaMessage(content=content))],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"

        if tool_calls:
            for index, tc in enumerate(tool_calls):
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=index,
                            id=tc.id,
                            type="function",
                            function=DeltaToolCallFunction(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    ]
                )
                tc_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    model=model_id,
                    created=created,
                    choices=[ChunkChoice(delta=delta)],
                )
                yield f"data: {tc_chunk.model_dump_json()}\n\n"
            finish_reason: Literal["stop", "length", "tool_calls"] = "tool_calls"
        else:
            finish_reason = "stop"

        stop_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            created=created,
            choices=[ChunkChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
            usage=UsageInfo(
                completion_tokens=token_count,
                total_tokens=token_count,
                tokens_per_second=round(tokens_per_second, 1),
            ),
        )
        yield f"data: {stop_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


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
        # Strip chain-of-thought blocks from thinking-mode models.
        # The streaming path leaves think blocks intact (they stream as content
        # tokens and the client can render them as reasoning).  For the buffered
        # non-streaming path we strip them so the final content field is clean.
        text = _THINK_RE.sub("", text).strip()
        token_count = stream.token_count
        finish_reason: Literal["stop", "length"] = (
            "length" if request_max_tokens and token_count >= request_max_tokens else "stop"
        )
        prompt_tokens = stream.prompt_tokens
        return ChatCompletionResponse(
            model=model_id,
            choices=[
                ChatChoice(
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=token_count,
                total_tokens=prompt_tokens + token_count,
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
        2. Content chunks — one per token, carrying the actual text. For thinking
           models, tokens inside <think>…</think> are emitted as reasoning_content.
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
        reasoning_prefix = (
            f'{{"id":"{chunk_id}","object":"chat.completion.chunk",'
            f'"created":{created},"model":{json.dumps(model_id)},'
            f'"choices":[{{"index":0,"delta":{{"reasoning_content":'
        )

        # State machine for tracking <think>…</think> spans in the token stream.
        # Tokens inside a think block are emitted as reasoning_content; tokens
        # outside are emitted as content.  A small buffer holds back chars that
        # could still be the beginning of a tag boundary.
        _TAG_OPEN = "<think>"
        _TAG_CLOSE = "</think>"
        in_think = False
        tag_buf = ""

        def _tag_holdback(text: str, tag: str) -> int:
            """Return the number of trailing chars to hold back as a potential tag prefix."""
            for n in range(min(len(tag) - 1, len(text)), 0, -1):
                if text.endswith(tag[:n]):
                    return n
            return 0

        async for raw_token in _async_token_iter(stream):
            tag_buf += raw_token
            # Process the buffer: find complete tag transitions, flush safe content.
            while True:
                if in_think:
                    close_idx = tag_buf.find(_TAG_CLOSE)
                    if close_idx != -1:
                        reasoning = tag_buf[:close_idx]
                        if reasoning:
                            yield f"data: {reasoning_prefix}{json.dumps(reasoning)}{chunk_suffix}\n\n"
                        tag_buf = tag_buf[close_idx + len(_TAG_CLOSE):]
                        in_think = False
                    else:
                        hold = _tag_holdback(tag_buf, _TAG_CLOSE)
                        emit = tag_buf[:-hold] if hold else tag_buf
                        if emit:
                            yield f"data: {reasoning_prefix}{json.dumps(emit)}{chunk_suffix}\n\n"
                        tag_buf = tag_buf[len(emit):]
                        break
                else:
                    open_idx = tag_buf.find(_TAG_OPEN)
                    if open_idx != -1:
                        content = tag_buf[:open_idx]
                        if content:
                            yield f"data: {chunk_prefix}{json.dumps(content)}{chunk_suffix}\n\n"
                        tag_buf = tag_buf[open_idx + len(_TAG_OPEN):]
                        in_think = True
                    else:
                        hold = _tag_holdback(tag_buf, _TAG_OPEN)
                        emit = tag_buf[:-hold] if hold else tag_buf
                        if emit:
                            yield f"data: {chunk_prefix}{json.dumps(emit)}{chunk_suffix}\n\n"
                        tag_buf = tag_buf[len(emit):]
                        break

        # Flush any remaining buffer at end of stream
        if tag_buf:
            if in_think:
                yield f"data: {reasoning_prefix}{json.dumps(tag_buf)}{chunk_suffix}\n\n"
            else:
                yield f"data: {chunk_prefix}{json.dumps(tag_buf)}{chunk_suffix}\n\n"

        # Final chunk — signal stop + usage stats (Pydantic; once per request)
        prompt_tokens = stream.prompt_tokens
        stop_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_id,
            created=created,
            choices=[ChunkChoice(delta=DeltaMessage(), finish_reason="stop")],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=stream.token_count,
                total_tokens=prompt_tokens + stream.token_count,
                tokens_per_second=round(stream.tokens_per_second, 1),
            ),
        )
        yield f"data: {stop_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def build_supported_models() -> SupportedModelList:
        """
        Build the response for GET /v1/models/supported.

        Computes which models fit within 80% of available system RAM. The RAM
        portion is cached (it's constant for the process lifetime). The
        ``capabilities`` field is computed live because it depends on which models
        are downloaded on disk.
        """
        # Use the module-level constant cached at startup instead of re-calling psutil
        from maic.core.model_manager import TOTAL_RAM_GB, model_manager

        available_gb = TOTAL_RAM_GB
        safe_limit = available_gb * 0.8

        models = [
            SupportedModel(
                id=mid,
                size_gb=size,
                min_ram_gb=round(size / 0.8, 1),
                feasible=size <= safe_limit,
                requires_token=mid in GATED_MODELS,
                capabilities=_tool_calling_caps(mid, model_manager),
            )
            for mid, size in KNOWN_SIZES_BY_RAM
        ]
        return SupportedModelList(
            available_ram_gb=round(available_gb, 1), models=models
        )
