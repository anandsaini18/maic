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

from maic.api.routes import _download_state, router
from maic.core.model_manager import ModelLoadError, ModelTooLargeError
from maic.core.token_stream import TokenStream

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_download_state():
    """Reset module-level download state between tests."""
    _download_state.clear()
    yield
    _download_state.clear()


@pytest.fixture
def mm():
    """A mock model_manager with sane defaults."""
    m = Mock()
    m.is_loaded = True
    m.model_id = "test/model"
    m.is_downloaded.return_value = False
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
    """POST /v1/chat/completions — the most critical endpoint."""

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
        body = {"model": "m", "messages": [{"role": "tool", "content": "x"}]}
        assert client.post("/v1/chat/completions", json=body).status_code == 422

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
        _download_state["org/m"] = "downloading"
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


class TestModelsStatus:
    @patch("psutil.virtual_memory")
    def test_response_shape(self, mock_vm, client, mm):
        mem = Mock()
        mem.total = 16 * (1024**3)
        mock_vm.return_value = mem
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

    @patch("psutil.virtual_memory")
    def test_download_error_extracted(self, mock_vm, client, mm):
        mem = Mock()
        mem.total = 16 * (1024**3)
        mock_vm.return_value = mem
        mm.model_id = None

        from maic.core.model_manager import KNOWN_MODEL_SIZES

        mid = next(iter(KNOWN_MODEL_SIZES))
        _download_state[mid] = "error: connection timeout"

        resp = client.get("/v1/models/status")
        target = next(m for m in resp.json()["models"] if m["id"] == mid)
        assert target["download_error"] is not None
        assert "connection timeout" in target["download_error"]
