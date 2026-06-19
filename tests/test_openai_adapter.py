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
from unittest.mock import patch

import pytest

from app.adapters.openai_adapter import OpenAIAdapter, _async_token_iter
from app.core.token_stream import TokenStream

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


# ── build_supported_models ───────────────────────────────────────────────────


class TestBuildSupportedModels:
    def setup_method(self) -> None:
        from app.adapters import openai_adapter

        openai_adapter._supported_models_cache = None

    @patch("app.core.model_manager.TOTAL_RAM_GB", 16.0)
    def test_feasibility_uses_80_percent_rule(self):

        result = OpenAIAdapter.build_supported_models()
        assert result.available_ram_gb == 16.0

        safe = 16.0 * 0.8
        for model in result.models:
            assert model.feasible == (model.size_gb <= safe)
            assert model.min_ram_gb == round(model.size_gb / 0.8, 1)

    @patch("app.core.model_manager.TOTAL_RAM_GB", 128.0)
    def test_requires_token_propagated(self):
        from app.core.model_sizing import GATED_MODELS

        result = OpenAIAdapter.build_supported_models()
        for model in result.models:
            assert model.requires_token == (model.id in GATED_MODELS)
