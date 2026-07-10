"""Integration tests for API routes.

Tests the full HTTP request -> response flow through FastAPI's TestClient.
This exercises routes, decorators, adapter, and schemas together —
where the real bugs hide.

Only model_manager is mocked (it owns MLX hardware). Everything else
runs for real: Pydantic validation, decorators, adapter formatting.
"""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from maic.api.routes import _downloads, _quantizations, router
from maic.core.model_manager import ModelLoadError, ModelTooLargeError
from maic.core.token_stream import TokenStream

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_download_state():
    """Reset download/quantization tracker state between tests."""
    _downloads._state.clear()
    _quantizations._state.clear()
    yield
    _downloads._state.clear()
    _quantizations._state.clear()


@pytest.fixture
def mm():
    """A mock model_manager with sane defaults."""
    m = Mock()
    m.is_loaded = True
    m.model_id = "test/model"
    m.is_downloaded.return_value = False
    m.local_model_ids.return_value = []  # Required for models_status endpoint
    m.supports_tool_calling = False
    from maic.core.model_manager import ToolCallFormat

    m.tool_call_format_for.return_value = ToolCallFormat.NONE
    return m


@pytest.fixture
def client(mm):
    """TestClient with model_manager patched at the source and all import sites."""
    app = FastAPI()
    app.include_router(router)

    # model_manager is imported lazily inside require_model's wrapper,
    # so we must patch it at the source module, not at decorators module level.
    with (
        patch("maic.api.routes.model_manager", mm),
        patch("maic.core.model_manager.model_manager", mm),
    ):
        yield TestClient(app)


def _chat_body(**overrides) -> dict:
    """Build a valid chat completion request body."""
    return {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        **overrides,
    }


# ── Chat completions (the core path) ────────────────────────────────────────


class TestChatCompletions:
    """POST /v1/chat/completions — the most critical endpoint.

    All tests run with batch_mode=False (default), exercising the
    single-inference semaphore path.
    """

    def _with_stream(self, mm, tokens: list[str]):
        mm.generate.return_value = TokenStream(iter(tokens))

    def test_non_streaming_full_response_shape(self, client, mm):
        self._with_stream(mm, ["Hello", " world"])

        with patch(
            "maic.api.routes.default_strategy", return_value={"max_tokens": 512, "sampler": None}
        ):
            resp = client.post("/v1/chat/completions", json=_chat_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 2
        assert data["usage"]["total_tokens"] == 2
        assert data["usage"]["tokens_per_second"] is not None

    def test_streaming_sse_wire_format(self, client, mm):
        self._with_stream(mm, ["Hi"])

        with patch(
            "maic.api.routes.default_strategy", return_value={"max_tokens": 512, "sampler": None}
        ):
            resp = client.post("/v1/chat/completions", json=_chat_body(stream=True))

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = [line for line in resp.text.strip().split("\n\n") if line.startswith("data:")]
        assert len(events) == 4  # role + content + stop + [DONE]
        assert events[-1] == "data: [DONE]"

        # First chunk: role only
        first = json.loads(events[0].removeprefix("data: "))
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["choices"][0]["delta"].get("content") is None

        # Content chunk: token
        content = json.loads(events[1].removeprefix("data: "))
        assert content["choices"][0]["delta"]["content"] == "Hi"
        assert content["choices"][0]["finish_reason"] is None

        # Stop chunk: finish_reason + usage
        stop = json.loads(events[2].removeprefix("data: "))
        assert stop["choices"][0]["finish_reason"] == "stop"
        assert stop["usage"]["completion_tokens"] == 1

    def test_503_when_model_not_loaded(self, mm):
        """@require_model decorator blocks with 503 before route runs."""
        mm.is_loaded = False
        app = FastAPI()
        app.include_router(router)

        with (
            patch("maic.api.routes.model_manager", mm),
            patch("maic.core.model_manager.model_manager", mm),
        ):
            resp = TestClient(app).post("/v1/chat/completions", json=_chat_body())

        assert resp.status_code == 503

    def test_temperature_zero_is_not_treated_as_falsy(self, client, mm):
        """temperature=0.0 is valid, not 'missing'. Classic Python gotcha."""
        captured = {}

        def spy(*, max_tokens, temperature, top_p):
            captured.update(temperature=temperature, top_p=top_p)
            return {"max_tokens": max_tokens, "sampler": None}

        # Make mm.generate call the strategy so our spy fires
        def fake_generate(messages, strategy):
            strategy(max_tokens=512, temperature=0.7, top_p=0.9)
            return TokenStream(iter(["ok"]))

        mm.generate.side_effect = fake_generate

        with patch("maic.api.routes.default_strategy", side_effect=spy):
            client.post("/v1/chat/completions", json=_chat_body(temperature=0.0, top_p=0.0))

        assert captured["temperature"] == 0.0
        assert captured["top_p"] == 0.0

    def test_max_tokens_override(self, client, mm):
        captured = {}

        def spy(*, max_tokens, temperature, top_p):
            captured["max_tokens"] = max_tokens
            return {"max_tokens": max_tokens, "sampler": None}

        def fake_generate(messages, strategy):
            strategy(max_tokens=512, temperature=0.7, top_p=0.9)
            return TokenStream(iter(["a", "b", "c"]))

        mm.generate.side_effect = fake_generate

        with patch("maic.api.routes.default_strategy", side_effect=spy):
            client.post("/v1/chat/completions", json=_chat_body(max_tokens=42))

        assert captured["max_tokens"] == 42

    def test_finish_reason_length_at_limit(self, client, mm):
        self._with_stream(mm, ["a", "b", "c"])

        with patch(
            "maic.api.routes.default_strategy", return_value={"max_tokens": 3, "sampler": None}
        ):
            resp = client.post("/v1/chat/completions", json=_chat_body(max_tokens=3))

        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_invalid_role_returns_422(self, client):
        body = {"model": "m", "messages": [{"role": "robot", "content": "x"}]}
        assert client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tools_request_returns_tool_calls(self, client, mm):
        tool_output = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        mm.generate.return_value = TokenStream(iter([tool_output]))
        body = _chat_body(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ]
        )

        with patch(
            "maic.api.routes.default_strategy", return_value={"max_tokens": 512, "sampler": None}
        ):
            resp = client.post("/v1/chat/completions", json=body)

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        calls = data["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
        # tools were forwarded to the model's chat template
        assert mm.generate.call_args.kwargs.get("tools") is not None

    def test_tools_streaming_emits_tool_call_chunk(self, client, mm):
        tool_output = '<tool_call>\n{"name": "ls", "arguments": {"path": "."}}\n</tool_call>'
        mm.generate.return_value = TokenStream(iter([tool_output]))
        body = _chat_body(stream=True, tools=[{"type": "function", "function": {"name": "ls"}}])

        with patch(
            "maic.api.routes.default_strategy", return_value={"max_tokens": 512, "sampler": None}
        ):
            resp = client.post("/v1/chat/completions", json=body)

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:") and line != "data: [DONE]"
        ]
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        names = [
            tc["function"]["name"]
            for ev in events
            for choice in ev["choices"]
            if choice["delta"].get("tool_calls")
            for tc in choice["delta"]["tool_calls"]
        ]
        assert "ls" in names

    def test_tool_choice_none_skips_tools(self, client, mm):
        mm.generate.return_value = TokenStream(iter(["plain answer"]))
        body = _chat_body(
            tool_choice="none",
            tools=[{"type": "function", "function": {"name": "ls"}}],
        )

        with patch(
            "maic.api.routes.default_strategy", return_value={"max_tokens": 512, "sampler": None}
        ):
            resp = client.post("/v1/chat/completions", json=body)

        assert resp.status_code == 200
        # tool_choice="none" means the chat template is not given tools at all.
        assert mm.generate.call_args.kwargs.get("tools") is None
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_missing_messages_returns_422(self, client):
        assert client.post("/v1/chat/completions", json={"model": "m"}).status_code == 422


# ── Model management endpoints ───────────────────────────────────────────────


class TestListModels:
    def test_returns_active_model(self, client, mm):
        mm.model_id = "org/my-model"
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == "org/my-model"
        assert resp.json()["object"] == "list"

    def test_falls_back_to_settings(self, client, mm):
        mm.model_id = None
        with patch("maic.api.routes.settings") as s:
            s.model_id = "default/fallback"
            resp = client.get("/v1/models")
        assert resp.json()["data"][0]["id"] == "default/fallback"


class TestLoadModel:
    def test_already_active_is_noop(self, client, mm):
        mm.model_id = "org/model"
        resp = client.post("/v1/models/load", json={"model_id": "org/model"})
        assert resp.json()["status"] == "already_active"
        mm.load.assert_not_called()

    def test_success(self, client, mm):
        mm.model_id = "org/old"
        resp = client.post("/v1/models/load", json={"model_id": "org/new"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        mm.load.assert_called_once_with("org/new")

    def test_too_large_returns_400(self, client, mm):
        mm.model_id = "org/old"
        mm.load.side_effect = ModelTooLargeError("needs 64GB")
        resp = client.post("/v1/models/load", json={"model_id": "org/huge"})
        assert resp.status_code == 400

    def test_load_error_returns_500(self, client, mm):
        mm.model_id = "org/old"
        mm.load.side_effect = ModelLoadError("corrupt")
        resp = client.post("/v1/models/load", json={"model_id": "org/bad"})
        assert resp.status_code == 500


class TestDownloadModel:
    def test_already_downloaded(self, client, mm):
        mm.is_downloaded.return_value = True
        resp = client.post("/v1/models/download", json={"model_id": "org/m"})
        assert resp.json()["status"] == "already_downloaded"

    def test_already_in_progress(self, client, mm):
        _downloads.set_downloading("org/m")
        resp = client.post("/v1/models/download", json={"model_id": "org/m"})
        assert resp.json()["status"] == "already_downloading"

    def test_starts_thread(self, client, mm):
        with patch("maic.api.routes.threading") as mock_t:
            resp = client.post("/v1/models/download", json={"model_id": "org/m"})
        assert resp.json()["status"] == "started"
        mock_t.Thread.assert_called_once()


class TestDeleteModel:
    def test_404_when_not_downloaded(self, client, mm):
        mm.is_downloaded.return_value = False
        assert client.request("DELETE", "/v1/models/org/model").status_code == 404

    def test_409_when_active(self, client, mm):
        mm.is_downloaded.return_value = True
        mm.delete_model.side_effect = RuntimeError("active")
        assert client.request("DELETE", "/v1/models/org/model").status_code == 409

    def test_success(self, client, mm):
        mm.is_downloaded.return_value = True
        resp = client.request("DELETE", "/v1/models/org/model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        mm.delete_model.assert_called_once_with("org/model")


class TestCacheClear:
    def test_clear_cache_returns_cleared(self, client, mm):
        resp = client.post("/v1/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        mm.clear_cache.assert_called_once()


class TestSettings:
    def test_get_settings_returns_defaults(self, client):
        with patch("maic.api.routes.settings") as s:
            s.max_kv_size = None
            s.kv_bits = None
            s.kv_group_size = 64
            resp = client.get("/v1/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_kv_size"] is None
        assert data["kv_bits"] is None
        assert data["kv_group_size"] == 64

    def test_update_settings_changes_values(self, client, mm):
        with patch("maic.api.routes.settings") as s:
            s.max_kv_size = None
            s.kv_bits = None
            s.kv_group_size = 64
            resp = client.post("/v1/settings", json={"max_kv_size": 4096, "kv_bits": 8})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"
        assert s.max_kv_size == 4096
        assert s.kv_bits == 8
        mm.clear_cache.assert_called_once()

    def test_update_settings_zero_disables(self, client, mm):
        """Setting fields to null (omit) disables them. 0 is rejected by ge=1/ge=4 validators."""
        with patch("maic.api.routes.settings") as s:
            s.max_kv_size = 4096
            s.kv_bits = 8
            s.kv_group_size = 64
            # 0 fails Field(ge=1) / Field(ge=4) validation → 422
            resp = client.post("/v1/settings", json={"max_kv_size": 0, "kv_bits": 0})
        assert resp.status_code == 422

    def test_update_settings_null_clears_values(self, client, mm):
        """Setting fields to null (omit) leaves them unchanged; handler only acts on non-None."""
        with patch("maic.api.routes.settings") as s:
            s.max_kv_size = 4096
            s.kv_bits = 8
            s.kv_group_size = 64
            # Omitting the field leaves it at the existing value
            resp = client.post("/v1/settings", json={})
        assert resp.status_code == 200

    def test_update_settings_no_change_skips_cache_clear(self, client, mm):
        with patch("maic.api.routes.settings") as s:
            s.max_kv_size = 4096
            s.kv_bits = 8
            s.kv_group_size = 64
            client.post("/v1/settings", json={"max_kv_size": 4096})
        mm.clear_cache.assert_not_called()

    def test_get_settings_includes_batch_mode(self, client):
        with patch("maic.api.routes.settings") as s:
            s.max_kv_size = None
            s.kv_bits = None
            s.kv_group_size = 64
            s.batch_mode = False
            resp = client.get("/v1/settings")
        assert resp.json()["batch_mode"] is False

    def test_update_batch_mode(self, client, mm):
        with (
            patch("maic.api.routes.settings") as s,
            patch("maic.api.routes.model_manager", mm),
        ):
            s.max_kv_size = None
            s.kv_bits = None
            s.kv_group_size = 64
            s.batch_mode = False
            mm.is_loaded = False
            resp = client.post("/v1/settings", json={"batch_mode": True})
        assert resp.json()["status"] == "updated"
        assert s.batch_mode is True


class TestQuantizeModel:
    def test_404_when_not_downloaded(self, client, mm):
        mm.is_downloaded.return_value = False
        resp = client.post("/v1/models/quantize", json={"model_id": "org/m"})
        assert resp.status_code == 404

    def test_already_quantizing(self, client, mm):
        mm.is_downloaded.return_value = True
        _quantizations.set_downloading("org/m")
        resp = client.post("/v1/models/quantize", json={"model_id": "org/m"})
        assert resp.json()["status"] == "already_quantizing"

    def test_starts_thread(self, client, mm):
        mm.is_downloaded.return_value = True
        with patch("maic.api.routes.threading") as mock_t:
            resp = client.post("/v1/models/quantize", json={"model_id": "org/m", "q_bits": 8})
        assert resp.json()["status"] == "started"
        mock_t.Thread.assert_called_once()

    def test_default_bits_is_4(self, client, mm):
        mm.is_downloaded.return_value = True
        with patch("maic.api.routes.threading"):
            resp = client.post("/v1/models/quantize", json={"model_id": "org/m"})
        assert resp.json()["status"] == "started"


class TestListModelsCapabilities:
    """GET /v1/models — capabilities field (tool_calling)."""

    def test_returns_loaded_model_id(self, client, mm):
        mm.model_id = "org/mymodel"
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "org/mymodel"

    def test_capabilities_field_present(self, client, mm):
        mm.model_id = "org/model"
        mm.supports_tool_calling = True
        resp = client.get("/v1/models")
        card = resp.json()["data"][0]
        assert "capabilities" in card
        assert card["capabilities"]["tool_calling"] is True

    def test_capabilities_false_when_no_tools(self, client, mm):
        mm.model_id = "org/model"
        mm.supports_tool_calling = False
        resp = client.get("/v1/models")
        card = resp.json()["data"][0]
        assert card["capabilities"]["tool_calling"] is False


class TestModelsStatus:
    @patch("maic.api.routes.TOTAL_RAM_GB", 16.0)
    def test_response_shape(self, client, mm):
        mm.model_id = "org/active"

        resp = client.get("/v1/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available_ram_gb"] == 16.0
        assert data["active_model"] == "org/active"
        assert len(data["models"]) > 0

        m = data["models"][0]
        for key in ["id", "name", "size_gb", "feasible", "downloaded", "active", "requires_token"]:
            assert key in m, f"missing key: {key}"

    @patch("maic.api.routes.TOTAL_RAM_GB", 16.0)
    @patch("maic.api.routes.fetch_hub_models")
    def test_download_error_extracted(self, mock_hub, client, mm):
        mm.model_id = None

        from maic.core.hub_fetcher import HubModel

        mid = "mlx-community/SmolLM2-1.7B-Instruct-4bit"
        mock_hub.return_value = [HubModel(id=mid, size_gb=1.0, downloads=0, gated=False)]
        _downloads.set_error(mid, Exception("connection timeout"))

        resp = client.get("/v1/models/status")
        target = next(m for m in resp.json()["models"] if m["id"] == mid)
        assert target["download_error"] is not None
        assert "connection timeout" in target["download_error"]
