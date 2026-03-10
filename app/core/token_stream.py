r"""Token stream wrapper providing iteration and performance metrics.

Wraps the raw MLX token generator behind a standard Python iterator interface
with built-in timing, token counting, and an ``on_complete`` callback for
observer notification. Consumed by both SSE streaming and non-streaming routes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator

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

    def __init__(
        self,
        generator: Generator[str, None, None],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize the stream with an MLX token generator.
        Records the start time immediately for accurate elapsed measurement.

        on_complete: optional zero-arg callback fired exactly once when the
        generator is exhausted. Used by ModelManager to notify observers with
        stats after the full stream is consumed. Replaces the broken
        instance-attribute monkey-patch approach (Python's iterator protocol
        dispatches __next__ via type(obj).__next__, not instance attributes).
        """
        self._gen = generator
        self._count: int = 0  # Count of yielded tokens (avoids growing list)
        self._on_complete = on_complete
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

        Flow: Pull token → Increment counter → Return to caller.
        When generator runs out (StopIteration), mark finish time, fire the
        on_complete callback (if any), then re-raise so callers stop iteration.
        """
        try:
            token = next(self._gen)
            self._count += 1
            return token
        except StopIteration:
            if self._finished_at is None:
                # Only record finish time and fire callback on the first exhaustion
                self._finished_at = time.perf_counter()
                if self._on_complete is not None:
                    self._on_complete()
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers — metrics and aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def collect(self) -> str:
        """
        Fully consume the stream and return the complete generated text.

        Useful when you need the full response at once (non-streaming routes)
        rather than token-by-token iteration. Joins all remaining tokens
        directly without building an intermediate list.
        """
        parts: list[str] = []
        for token in self:
            parts.append(token)
        return "".join(parts)

    @property
    def token_count(self) -> int:
        """
        Current count of yielded tokens (live count during generation, final after done).
        Used for progress tracking and performance metrics.
        """
        return self._count

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
        return self._count / self.elapsed if self.elapsed > 0 else 0.0
