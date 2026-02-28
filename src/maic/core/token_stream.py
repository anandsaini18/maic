from __future__ import annotations

import time
from collections.abc import Generator

# ─────────────────────────────────────────────────────────────────────────────
# Token stream wrapper — provides iteration + performance metrics
# ─────────────────────────────────────────────────────────────────────────────


class TokenStream:
    """
    Standard Python iterator wrapping the raw MLX token generator.

    Purpose: Hide MLX specifics behind a clean iterator interface so callers
    (API routes, test collectors, non-streaming aggregators) interact with tokens
    the same way regardless of where they come from.

    Usage:
        stream = model_manager.generate(messages, strategy)
        for token in stream:
            send_to_client(token)  # Token by token to user
        # or
        full_response = stream.collect()  # Gather all tokens at once

    Bonus: Automatically tracks generation timing (start/end/duration) so we can
    report tokens/second and other performance metrics to observers.
    """

    def __init__(self, generator: Generator[str, None, None]) -> None:
        """
        Initialize the stream with an MLX token generator.
        Records the start time immediately for accurate elapsed measurement.
        """
        self._gen = generator
        self._tokens: list[str] = []  # Accumulates all tokens for metrics and collect()
        self._started_at: float = time.perf_counter()  # Generation start timestamp
        self._finished_at: float | None = None  # Set when generator is exhausted

    # ─────────────────────────────────────────────────────────────────────────
    # Iterator protocol — allows 'for token in stream:' syntax
    # ─────────────────────────────────────────────────────────────────────────

    def __iter__(self) -> TokenStream:
        """Return self to implement iterator protocol."""
        return self

    def __next__(self) -> str:
        """
        Retrieve the next token from the MLX generator.

        Flow: Pull token → Accumulate in list → Return to caller.
        When generator runs out (StopIteration), mark finish time so elapsed/tps
        metrics become accurate.
        """
        try:
            token = next(self._gen)
            self._tokens.append(token)
            return token
        except StopIteration:
            self._finished_at = time.perf_counter()
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers — metrics and aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def collect(self) -> str:
        """
        Fully consume the stream and return the complete generated text.

        Useful when you need the full response at once (non-streaming routes)
        rather than token-by-token iteration. Internally just joins accumulated
        tokens.
        """
        return "".join(self)

    @property
    def token_count(self) -> int:
        """
        Current count of yielded tokens (live count during generation, final after done).
        Used for progress tracking and performance metrics.
        """
        return len(self._tokens)

    @property
    def elapsed(self) -> float:
        """
        Elapsed time in seconds since generation started.

        Logic: If stream is done, use recorded finish time. While stream is active,
        use current time to give real-time elapsed stats. Accurate either way.
        """
        end = self._finished_at or time.perf_counter()
        return end - self._started_at

    @property
    def tokens_per_second(self) -> float:
        """
        Throughput metric: tokens generated per second.

        Used for performance reporting (e.g., "33.5 tok/s"). Guards against
        division by zero if elapsed somehow is 0 (edge case).
        """
        return self.token_count / self.elapsed if self.elapsed > 0 else 0.0
