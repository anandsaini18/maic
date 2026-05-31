"""Tests for the OpenAI adapter.

stream_to_sse is tested in isolation because its output is an SSE protocol
contract — bugs in the wire format (missing \\n\\n, wrong field names,
out-of-order chunks) are invisible through TestClient's flattened .text.

stream_to_response and build_supported_models are also tested here because
they contain branching logic (finish_reason, RAM feasibility) that's worth
verifying directly rather than only through HTTP integration tests.

messages_to_dicts is trivial enough to skip — it's exercised by the route tests.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from maic.adapters.openai_adapter import OpenAIAdapter, _async_token_iter
from maic.core.model_manager import ToolCallFormat
from maic.core.token_stream import TokenStream

# ── helpers ──────────────────────────────────────────────────────────────────


def _stream(tokens: list[str]) -> TokenStream:
    return TokenStream(iter(tokens))


async def _collect_sse(tokens: list[str], model: str = "m", chunk_id: str = "id-1") -> list[str]:
    """Collect all SSE events from the async generator."""
    return [event async for event in OpenAIAdapter.stream_to_sse(_stream(tokens), model, chunk_id)]


def _parse_sse(event: str) -> dict:
    """Parse a 'data: {...}\\n\\n' SSE event into a dict."""
    return json.loads(event.removeprefix("data: ").strip())


# ── stream_to_sse ────────────────────────────────────────────────────────────


class TestStreamToSSE:
    """SSE wire format — this is a client-facing protocol contract."""

    @pytest.mark.asyncio
    async def test_every_event_has_correct_framing(self):
        events = await _collect_sse(["Hi"])
        for event in events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_event_sequence_role_content_stop_done(self):
        events = await _collect_sse(["Hello", " world"])

        # 1. Role chunk
        role = _parse_sse(events[0])
        assert role["choices"][0]["delta"]["role"] == "assistant"
        assert role["choices"][0]["delta"].get("content") is None
        assert role["choices"][0]["finish_reason"] is None

        # 2. Content chunks
        c1 = _parse_sse(events[1])
        assert c1["choices"][0]["delta"]["content"] == "Hello"
        assert c1["choices"][0]["finish_reason"] is None

        c2 = _parse_sse(events[2])
        assert c2["choices"][0]["delta"]["content"] == " world"

        # 3. Stop chunk
        stop = _parse_sse(events[3])
        assert stop["choices"][0]["finish_reason"] == "stop"
        assert stop["usage"]["completion_tokens"] == 2

        # 4. [DONE] sentinel
        assert events[4] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_chunk_count_matches_tokens(self):
        events = await _collect_sse(["a", "b", "c"])
        # role(1) + content(3) + stop(1) + DONE(1) = 6
        assert len(events) == 6

    @pytest.mark.asyncio
    async def test_all_json_chunks_share_id_and_model(self):
        events = await _collect_sse(["a"], model="org/m", chunk_id="abc")
        for event in events[:-1]:  # skip [DONE]
            parsed = _parse_sse(event)
            assert parsed["id"] == "abc"
            assert parsed["model"] == "org/m"
            assert parsed["object"] == "chat.completion.chunk"

    @pytest.mark.asyncio
    async def test_empty_stream_emits_role_stop_done(self):
        events = await _collect_sse([])
        assert len(events) == 3  # role + stop + DONE
        assert events[-1] == "data: [DONE]\n\n"
        stop = _parse_sse(events[1])
        assert stop["choices"][0]["finish_reason"] == "stop"
        assert stop["usage"]["completion_tokens"] == 0


# ── _async_token_iter exception propagation ──────────────────────────────────


class TestAsyncTokenIter:
    """
    Verifies that exceptions raised by the MLX token generator inside the
    ThreadPoolExecutor thread are propagated to the async consumer rather
    than being silently swallowed into the unawaited run_in_executor Future.

    The old code did:
        except Exception as exc:
            queue.put_nowait(None)   # sends end-of-stream sentinel
            raise exc                # ← went into unawaited Future, silently lost

    The new code pushes the exception object itself into the queue so the
    async side can re-raise it.
    """

    @pytest.mark.asyncio
    async def test_producer_exception_raises_on_consumer_side(self):
        """RuntimeError mid-stream must surface to the caller, not be swallowed."""

        def _failing_gen():
            yield "Hello"
            raise RuntimeError("MLX out of memory")

        stream = TokenStream(_failing_gen())
        with pytest.raises(RuntimeError, match="MLX out of memory"):
            # Collect all items; exception should propagate during iteration.
            [item async for item in _async_token_iter(stream)]

    @pytest.mark.asyncio
    async def test_producer_exception_propagates_through_stream_to_sse(self):
        """Exception must bubble through stream_to_sse to the HTTP layer."""

        def _failing_gen():
            yield "partial"
            raise ValueError("tokenizer exploded")

        stream = TokenStream(_failing_gen())
        with pytest.raises(ValueError, match="tokenizer exploded"):
            [event async for event in OpenAIAdapter.stream_to_sse(stream, "m", "id-1")]

    @pytest.mark.asyncio
    async def test_clean_stream_unaffected_by_error_path(self):
        """Regression: the else-branch sentinel must still work for normal streams."""
        stream = TokenStream(iter(["x", "y"]))
        items = [item async for item in _async_token_iter(stream)]
        assert items == ["x", "y"]

    @pytest.mark.asyncio
    async def test_no_double_sentinel_on_clean_end(self):
        """else-branch sends None exactly once; queue should drain to empty."""
        stream = TokenStream(iter(["a"]))
        items = [item async for item in _async_token_iter(stream)]
        # If a double-None were pushed, a second None would be left in the queue.
        # We verify the generator terminates cleanly with exactly the right items.
        assert items == ["a"]


# ── stream_to_response ───────────────────────────────────────────────────────


class TestStreamToResponse:
    """Non-streaming response — finish_reason branching is the key logic."""

    def test_collects_text_and_usage(self):
        resp = OpenAIAdapter.stream_to_response(_stream(["Hello", " world"]), "m")
        assert resp.choices[0].message.content == "Hello world"
        assert resp.usage.completion_tokens == 2
        assert resp.usage.tokens_per_second is not None

    def test_finish_reason_stop_when_no_limit(self):
        resp = OpenAIAdapter.stream_to_response(_stream(["a", "b"]), "m", request_max_tokens=None)
        assert resp.choices[0].finish_reason == "stop"

    def test_finish_reason_length_at_exact_limit(self):
        resp = OpenAIAdapter.stream_to_response(_stream(["a", "b", "c"]), "m", request_max_tokens=3)
        assert resp.choices[0].finish_reason == "length"

    def test_finish_reason_stop_when_under_limit(self):
        resp = OpenAIAdapter.stream_to_response(_stream(["a"]), "m", request_max_tokens=100)
        assert resp.choices[0].finish_reason == "stop"

    def test_empty_stream_returns_empty_content(self):
        resp = OpenAIAdapter.stream_to_response(_stream([]), "m")
        assert resp.choices[0].message.content == ""
        assert resp.usage.completion_tokens == 0
        assert resp.choices[0].finish_reason == "stop"


# ── tool calling ─────────────────────────────────────────────────────────────


class TestToolCalls:
    """Parsing <tool_call> blocks and shaping OpenAI tool-call responses."""

    def test_parse_extracts_function_and_arguments(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        clean, calls = OpenAIAdapter.parse_tool_calls(text)
        assert clean == ""
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        assert json.loads(calls[0].function.arguments) == {"city": "Paris"}
        assert calls[0].id.startswith("call_")

    def test_parse_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'
            '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>'
        )
        _, calls = OpenAIAdapter.parse_tool_calls(text)
        assert [c.function.name for c in calls] == ["a", "b"]

    def test_parse_keeps_leading_text(self):
        text = 'Let me check.\n<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
        clean, calls = OpenAIAdapter.parse_tool_calls(text)
        assert clean == "Let me check."
        assert calls[0].function.name == "ls"

    def test_parse_handles_response_tag_drift(self):
        # Quantized Qwen-Coder sometimes wraps the call in <response> instead of
        # <tool_call>; the JSON is still correct and must be recovered.
        text = '<response>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</response>'
        clean, calls = OpenAIAdapter.parse_tool_calls(text)
        assert clean == ""
        assert calls[0].function.name == "get_weather"
        assert json.loads(calls[0].function.arguments) == {"city": "Paris"}

    def test_parse_handles_fenced_json(self):
        text = 'Sure.\n```json\n{"name": "ls", "arguments": {"path": "."}}\n```'
        clean, calls = OpenAIAdapter.parse_tool_calls(text)
        assert clean == "Sure."
        assert calls[0].function.name == "ls"

    def test_parse_repairs_doubled_braces(self):
        # The model copies the chat template's buggy example which doubles braces.
        text = '{{"name": "get_weather", "arguments": {"city": "Paris"}}}'
        clean, calls = OpenAIAdapter.parse_tool_calls(text)
        assert clean == ""
        assert calls[0].function.name == "get_weather"
        assert json.loads(calls[0].function.arguments) == {"city": "Paris"}

    def test_parse_fenced_xml_with_doubled_braces(self):
        text = '```xml\n{{"name": "get_weather", "arguments": {"city": "Paris"}}}\n```'
        _, calls = OpenAIAdapter.parse_tool_calls(text)
        assert calls[0].function.name == "get_weather"
        assert json.loads(calls[0].function.arguments) == {"city": "Paris"}

    def test_parse_no_blocks_returns_text_unchanged(self):
        clean, calls = OpenAIAdapter.parse_tool_calls("just a normal answer")
        assert clean == "just a normal answer"
        assert calls == []

    def test_parse_skips_malformed_json(self):
        text = "<tool_call>\nnot json\n</tool_call>"
        clean, calls = OpenAIAdapter.parse_tool_calls(text)
        assert calls == []
        # Nothing valid parsed, so the original text is returned untouched.
        assert clean == text

    def test_tools_to_dicts_passthrough(self):
        from maic.schemas.openai import FunctionDefinition, Tool

        tools = [Tool(function=FunctionDefinition(name="f", description="d"))]
        dicts = OpenAIAdapter.tools_to_dicts(tools)
        assert dicts == [{"type": "function", "function": {"name": "f", "description": "d"}}]

    def test_tools_to_dicts_none(self):
        assert OpenAIAdapter.tools_to_dicts(None) is None
        assert OpenAIAdapter.tools_to_dicts([]) is None

    def test_build_tool_response_with_calls(self):
        _, calls = OpenAIAdapter.parse_tool_calls(
            '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'
        )
        resp = OpenAIAdapter.build_tool_response("m", "", calls, token_count=5, tokens_per_second=10.0)
        assert resp.choices[0].finish_reason == "tool_calls"
        assert resp.choices[0].message.content is None
        assert resp.choices[0].message.tool_calls[0].function.name == "f"

    def test_build_tool_response_plain_text(self):
        resp = OpenAIAdapter.build_tool_response(
            "m", "hello", [], token_count=1, tokens_per_second=10.0
        )
        assert resp.choices[0].finish_reason == "stop"
        assert resp.choices[0].message.content == "hello"
        assert resp.choices[0].message.tool_calls is None

    @pytest.mark.asyncio
    async def test_tool_call_to_sse_emits_tool_call_then_stop(self):
        _, calls = OpenAIAdapter.parse_tool_calls(
            '<tool_call>\n{"name": "ls", "arguments": {"path": "."}}\n</tool_call>'
        )
        events = [
            e
            async for e in OpenAIAdapter.tool_call_to_sse("m", "id-1", "", calls, 3, 12.0)
        ]
        assert events[-1] == "data: [DONE]\n\n"
        parsed = [_parse_sse(e) for e in events if e != "data: [DONE]\n\n"]
        # role chunk first
        assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
        # a tool-call delta carries the function name + arguments
        tc_deltas = [
            tc
            for p in parsed
            for c in p["choices"]
            if c["delta"].get("tool_calls")
            for tc in c["delta"]["tool_calls"]
        ]
        assert tc_deltas[0]["function"]["name"] == "ls"
        assert json.loads(tc_deltas[0]["function"]["arguments"]) == {"path": "."}
        # terminal chunk signals tool_calls
        assert parsed[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ── build_supported_models ───────────────────────────────────────────────────


def _mock_mm_no_tools() -> MagicMock:
    """A minimal model_manager mock where tool_call_format_for always returns NONE."""
    mm = MagicMock()
    mm.tool_call_format_for.return_value = ToolCallFormat.NONE
    return mm


class TestBuildSupportedModels:
    @patch("maic.core.model_manager.TOTAL_RAM_GB", 16.0)
    @patch("maic.adapters.openai_adapter._tool_calling_caps", return_value=None)
    def test_feasibility_uses_80_percent_rule(self, _caps):
        result = OpenAIAdapter.build_supported_models()
        assert result.available_ram_gb == 16.0

        safe = 16.0 * 0.8
        for model in result.models:
            assert model.feasible == (model.size_gb <= safe)
            assert model.min_ram_gb == round(model.size_gb / 0.8, 1)

    @patch("maic.core.model_manager.TOTAL_RAM_GB", 128.0)
    @patch("maic.adapters.openai_adapter._tool_calling_caps", return_value=None)
    def test_requires_token_propagated(self, _caps):
        from maic.core.model_sizing import GATED_MODELS

        result = OpenAIAdapter.build_supported_models()
        for model in result.models:
            assert model.requires_token == (model.id in GATED_MODELS)

    @patch("maic.core.model_manager.TOTAL_RAM_GB", 128.0)
    @patch(
        "maic.adapters.openai_adapter._tool_calling_caps",
        return_value={"tool_calling": True},
    )
    def test_capabilities_propagated_when_present(self, _caps):
        result = OpenAIAdapter.build_supported_models()
        assert all(m.capabilities == {"tool_calling": True} for m in result.models)


# ── parse_tool_calls (format-aware) ──────────────────────────────────────────


class TestParseToolCallsQwen:
    """Qwen / Hermes format: <tool_call>…</tool_call>"""

    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "ls", "arguments": {"path": "."}}\n</tool_call>'
        clean, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert len(calls) == 1
        assert calls[0].function.name == "ls"
        assert json.loads(calls[0].function.arguments) == {"path": "."}
        assert clean == ""

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>'
        )
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert len(calls) == 2
        assert calls[0].function.name == "a"
        assert calls[1].function.name == "b"

    def test_text_before_tool_call_preserved(self):
        text = 'Sure, let me help.\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
        clean, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert "Sure" in clean
        assert len(calls) == 1

    def test_hermes_format_same_as_qwen(self):
        text = '<tool_call>{"name": "g", "arguments": {"k": "v"}}</tool_call>'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.HERMES)
        assert calls[0].function.name == "g"

    def test_no_tool_call_returns_original(self):
        text = "Just a plain response."
        clean, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert calls == []
        assert clean == text


class TestParseToolCallsMistral:
    """Mistral format: [TOOL_CALLS] [{…}]"""

    def test_single_tool_call(self):
        text = '[TOOL_CALLS] [{"name": "search", "arguments": {"q": "mlx"}}]'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.MISTRAL)
        assert len(calls) == 1
        assert calls[0].function.name == "search"

    def test_no_tag_falls_through_to_fallback(self):
        text = "No tool call here."
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.MISTRAL)
        assert calls == []


class TestParseToolCallsLlama3:
    """Llama-3 format: <|python_tag|>[{…}]"""

    def test_single_tool_call(self):
        text = '<|python_tag|>[{"name": "run", "parameters": {"cmd": "ls"}}]'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.LLAMA3)
        assert len(calls) == 1
        assert calls[0].function.name == "run"
        # "parameters" key should be accepted as arguments
        assert json.loads(calls[0].function.arguments) == {"cmd": "ls"}

    def test_no_python_tag_falls_through(self):
        text = "Regular response."
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.LLAMA3)
        assert calls == []


class TestParseToolCallsDeepSeek:
    """DeepSeek format: <|tool▁calls▁begin|>…<|tool▁sep|>…<|tool▁calls▁end|>"""

    def test_single_tool_call(self):
        sep = "<|tool\u2581sep|>"
        text = (
            "<|tool\u2581calls\u2581begin|>"
            f'{sep}{{"name": "calc", "arguments": {{"expr": "2+2"}}}}'
            "<|tool\u2581calls\u2581end|>"
        )
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.DEEPSEEK)
        assert len(calls) == 1
        assert calls[0].function.name == "calc"


class TestParseToolCallsNone:
    def test_none_format_always_returns_empty(self):
        text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        clean, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.NONE)
        assert calls == []
        assert clean == text


class TestParseToolCallsFallback:
    """UNKNOWN format uses the original fallback chain."""

    def test_qwen_tags_still_parsed(self):
        text = '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.UNKNOWN)
        assert calls[0].function.name == "x"

    def test_fenced_block_parsed(self):
        text = '```json\n{"name": "y", "arguments": {"a": 1}}\n```'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.UNKNOWN)
        assert calls[0].function.name == "y"

    def test_bare_json_parsed(self):
        text = '{"name": "z", "arguments": {}}'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.UNKNOWN)
        assert calls[0].function.name == "z"


class TestParseToolCallsThinkBlocks:
    """<think>…</think> blocks must be stripped before any tool-call parsing."""

    def test_think_block_before_qwen_tool_call_stripped(self):
        text = (
            "<think>Let me figure out which tool to call.</think>\n"
            '<tool_call>{"name": "search", "arguments": {"q": "mlx"}}</tool_call>'
        )
        clean, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert len(calls) == 1
        assert calls[0].function.name == "search"
        assert "<think>" not in clean

    def test_think_block_only_no_tool_call(self):
        text = "<think>I should reason about this.</think>\nHere is my answer."
        clean, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert calls == []
        assert "<think>" not in clean
        assert "Here is my answer." in clean

    def test_multiline_think_block_stripped(self):
        text = (
            "<think>\nStep 1: check docs\nStep 2: call tool\n</think>\n"
            '<tool_call>{"name": "docs", "arguments": {}}</tool_call>'
        )
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert len(calls) == 1
        assert calls[0].function.name == "docs"

    def test_no_think_block_unchanged(self):
        text = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        _, calls = OpenAIAdapter.parse_tool_calls(text, ToolCallFormat.QWEN)
        assert len(calls) == 1
        assert calls[0].function.name == "ping"

    def test_think_block_stripped_in_stream_to_response(self):
        tokens = ["<think>reason</think>", " actual answer"]
        response = OpenAIAdapter.stream_to_response(_stream(tokens), "m")
        assert response.choices[0].message.content == "actual answer"
        assert "<think>" not in (response.choices[0].message.content or "")

    def test_stream_to_response_no_think_block_unchanged(self):
        tokens = ["Hello", " world"]
        response = OpenAIAdapter.stream_to_response(_stream(tokens), "m")
        assert response.choices[0].message.content == "Hello world"


class TestMessagesToolCallId:
    def test_messages_to_dicts_tool_role_includes_tool_call_id(self):
        from maic.schemas.openai import Message

        messages = [
            Message(role="tool", content="42", tool_call_id="call_xyz", name="get_answer")
        ]
        result = OpenAIAdapter.messages_to_dicts(messages)
        assert result[0]["tool_call_id"] == "call_xyz"
        assert result[0]["name"] == "get_answer"

    def test_messages_to_dicts_assistant_tool_calls_include_id(self):
        from maic.schemas.openai import FunctionCall, Message, ToolCall

        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_abc",
                        function=FunctionCall(name="search", arguments='{"q":"test"}'),
                    )
                ],
            )
        ]
        result = OpenAIAdapter.messages_to_dicts(messages)
        assert result[0]["tool_calls"][0]["id"] == "call_abc"
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"


class TestPromptTokens:
    def test_prompt_tokens_set_on_token_stream(self):
        stream = _stream(["Hello"])
        stream.prompt_tokens = 5
        assert stream.prompt_tokens == 5

    def test_stream_to_response_uses_prompt_tokens(self):
        stream = _stream(["Hi"])
        stream.prompt_tokens = 10
        response = OpenAIAdapter.stream_to_response(stream, "m")
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 1
        assert response.usage.total_tokens == 11
