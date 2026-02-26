from __future__ import annotations

import time
from typing import Generator


# ── Iterator Pattern ──────────────────────────────────────────────────────────

class TokenStream:
    """
    Iterator pattern — wraps the raw MLX generator so callers never touch MLX directly.

    The model produces tokens one at a time. This class wraps that process in a
    standard Python iterator, so any consumer (SSE route, non-streaming collector,
    test) can just do:

        for token in stream:
            send_to_client(token)

    It also tracks timing so we can report tokens/sec when generation finishes.
    """

    def __init__(self, generator: Generator[str, None, None]) -> None:
        self._gen = generator
        self._tokens: list[str] = []          # accumulates every yielded token
        self._started_at: float = time.perf_counter()
        self._finished_at: float | None = None

    # ── Iterator protocol ────────────────────────────────────────────────────

    def __iter__(self) -> "TokenStream":
        return self

    def __next__(self) -> str:
        """Pull the next token. Records finish time when the generator is exhausted."""
        try:
            token = next(self._gen)
            self._tokens.append(token)
            return token
        except StopIteration:
            self._finished_at = time.perf_counter()
            raise

    # ── Convenience helpers ──────────────────────────────────────────────────

    def collect(self) -> str:
        """Drain the entire stream and return the complete response as one string."""
        return "".join(self)

    @property
    def token_count(self) -> int:
        """How many tokens have been yielded so far (or total, if stream is done)."""
        return len(self._tokens)

    @property
    def elapsed(self) -> float:
        """Seconds since generation started. Uses current time if not finished yet."""
        end = self._finished_at or time.perf_counter()
        return end - self._started_at

    @property
    def tokens_per_second(self) -> float:
        """Generation speed. Returns 0 if elapsed time is somehow zero."""
        return self.token_count / self.elapsed if self.elapsed > 0 else 0.0
