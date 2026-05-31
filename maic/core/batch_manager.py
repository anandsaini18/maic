r"""Continuous batching inference manager wrapping mlx_lm's BatchGenerator.

Accepts multiple concurrent generation requests, batches them through a
single model forward pass loop, and streams per-request tokens back through
asyncio queues. Falls back to single-inference (stream_generate) when
batch_mode is disabled in settings.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from maic.core.config import settings

logger = logging.getLogger(__name__)

# Sentinel value pushed into a response queue to signal end-of-stream.
_DONE = object()


@dataclass
class _PendingRequest:
    """A request waiting to be inserted into the batch."""

    prompt_tokens: list[int]
    max_tokens: int
    sampler: Any
    response_queue: asyncio.Queue[str | object | Exception]
    loop: asyncio.AbstractEventLoop
    uid: int | None = field(default=None)


class BatchInferenceManager:
    """Manages a shared mlx_lm BatchGenerator across concurrent requests.

    Lifecycle:
        1. ``start(model, tokenizer)`` — called after model load; creates the
           BatchGenerator and starts the background loop thread.
        2. ``submit(prompt_tokens, max_tokens, sampler)`` — called per request;
           returns an async generator that yields decoded token strings.
        3. ``stop()`` — called on model switch or shutdown; joins the loop thread.
    """

    def __init__(self) -> None:
        self._generator: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._pending: list[_PendingRequest] = []
        self._active: dict[int, _PendingRequest] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, model: Any, tokenizer: Any) -> None:
        """Start the batch inference loop for the given model."""
        self.stop()
        self._model = model
        self._tokenizer = tokenizer
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="batch-loop")
        self._thread.start()
        logger.info("Batch inference manager started")

    def stop(self) -> None:
        """Stop the batch loop and close the generator."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._generator is not None:
            try:
                self._generator.close()
            except Exception:
                pass
            self._generator = None
        self._active.clear()
        self._pending.clear()
        logger.info("Batch inference manager stopped")

    async def submit(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        sampler: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Submit a request and yield decoded tokens as they arrive."""
        queue: asyncio.Queue[str | object | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        req = _PendingRequest(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            response_queue=queue,
            loop=loop,
        )
        with self._lock:
            self._pending.append(req)

        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _loop(self) -> None:
        """Background thread: drain pending requests and step the BatchGenerator."""
        from mlx_lm.generate import BatchGenerator

        self._generator = BatchGenerator(
            self._model,
            stop_tokens=set(self._tokenizer.eos_token_ids),
            max_kv_size=settings.max_kv_size,
            prefill_batch_size=settings.batch_prefill_size,
            completion_batch_size=settings.batch_completion_size,
        )

        while self._running:
            self._drain_pending()

            if not self._active:
                threading.Event().wait(timeout=0.005)
                continue

            try:
                responses = self._generator.next()
            except Exception as exc:
                logger.error("BatchGenerator.next() failed: %s", exc, exc_info=True)
                for req in self._active.values():
                    self._send(req, exc)
                self._active.clear()
                continue

            if not responses:
                threading.Event().wait(timeout=0.001)
                continue

            for resp in responses:
                req = self._active.get(resp.uid)
                if req is None:
                    continue

                if resp.finish_reason is not None:
                    self._send(req, _DONE)
                    self._active.pop(resp.uid, None)
                else:
                    text = self._tokenizer.decode([resp.token])
                    if text:
                        self._send(req, text)

        # Clean up any remaining active requests
        for req in self._active.values():
            self._send(req, _DONE)
        self._active.clear()

    def _drain_pending(self) -> None:
        """Move pending requests into the active BatchGenerator."""
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()

        if not batch:
            return

        prompts = [r.prompt_tokens for r in batch]
        max_tokens_list = [r.max_tokens for r in batch]
        samplers = [r.sampler for r in batch]

        uids = self._generator.insert(
            prompts,
            max_tokens=max_tokens_list,
            samplers=samplers,
        )
        for uid, req in zip(uids, batch, strict=True):
            req.uid = uid
            self._active[uid] = req

    def _send(self, req: _PendingRequest, item: Any) -> None:
        """Thread-safe push into the request's asyncio queue."""
        try:
            req.loop.call_soon_threadsafe(req.response_queue.put_nowait, item)
        except RuntimeError:
            pass


batch_manager = BatchInferenceManager()
