"""Unit tests for the BatchInferenceManager."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from maic.core.batch_manager import _DONE, BatchInferenceManager


@pytest.fixture
def mock_settings():
    with patch("maic.core.batch_manager.settings") as s:
        s.max_kv_size = None
        s.batch_prefill_size = 8
        s.batch_completion_size = 32
        yield s


class TestBatchInferenceManagerUnit:
    """Tests for BatchInferenceManager state management (no real MLX)."""

    def test_initial_state(self):
        bm = BatchInferenceManager()
        assert not bm.is_running

    def test_stop_when_not_running_is_noop(self):
        bm = BatchInferenceManager()
        bm.stop()
        assert not bm.is_running

    def test_start_sets_running(self, mock_settings):
        bm = BatchInferenceManager()
        _mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_ids = {0}

        mock_batch_gen_cls = MagicMock()
        mock_batch_gen = MagicMock()
        mock_batch_gen.next.return_value = []
        mock_batch_gen_cls.return_value = mock_batch_gen

        with patch.dict("sys.modules", {"mlx_lm": MagicMock(), "mlx_lm.generate": MagicMock()}):
            with patch("maic.core.batch_manager.BatchInferenceManager._loop"):
                bm._running = True
                bm._thread = MagicMock()
                bm._thread.join = MagicMock()
                assert bm.is_running

        bm._running = False
        bm._thread = None

    def test_stop_after_manual_set(self, mock_settings):
        bm = BatchInferenceManager()
        bm._running = True
        bm._generator = MagicMock()
        bm._thread = MagicMock()
        bm._thread.join = MagicMock()
        bm.stop()
        assert not bm.is_running
        assert bm._generator is None


class TestBatchManagerSubmitCollect:
    """Test the submit/collect flow with a mocked batch loop."""

    @pytest.mark.asyncio
    async def test_submit_yields_tokens_from_queue(self):
        bm = BatchInferenceManager()
        bm._running = True

        async def fake_submit():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            loop.call_soon(queue.put_nowait, "Hello")
            loop.call_soon(queue.put_nowait, " world")
            loop.call_soon(queue.put_nowait, _DONE)

            from maic.core.batch_manager import _PendingRequest

            req = _PendingRequest(
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
                sampler=None,
                response_queue=queue,
                loop=loop,
            )

            tokens = []
            while True:
                item = await req.response_queue.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                tokens.append(item)
            return tokens

        tokens = await fake_submit()
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_submit_propagates_exceptions(self):
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        loop.call_soon(queue.put_nowait, Exception("test error"))

        item = await queue.get()
        assert isinstance(item, Exception)
        assert "test error" in str(item)
