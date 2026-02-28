"""Unit tests for TokenStream iterator and metrics tracking."""

import time
from collections.abc import Generator

import pytest

from app.core.token_stream import TokenStream


class TestTokenStreamIterator:
    """Tests for iterator protocol compliance."""

    def generator_fixture(self) -> Generator[str, None, None]:
        """Simple generator for testing."""
        tokens = ["Hello", " ", "world", "!"]
        yield from tokens

    def test_iter_returns_self(self):
        """__iter__ should return self for iterator protocol."""
        gen = self.generator_fixture()
        stream = TokenStream(gen)
        assert iter(stream) is stream

    def test_next_retrieves_tokens_in_order(self):
        """__next__ should yield tokens one at a time."""
        gen = self.generator_fixture()
        stream = TokenStream(gen)

        assert next(stream) == "Hello"
        assert next(stream) == " "
        assert next(stream) == "world"
        assert next(stream) == "!"

    def test_stopiteration_on_exhaustion(self):
        """__next__ should raise StopIteration when generator exhausted."""
        gen = self.generator_fixture()
        stream = TokenStream(gen)

        # Consume all tokens
        list(stream)

        # Next call should raise StopIteration
        with pytest.raises(StopIteration):
            next(stream)

    def test_for_loop_iteration(self):
        """Stream should work with for loops."""
        gen = self.generator_fixture()
        stream = TokenStream(gen)

        tokens = [token for token in stream]
        assert tokens == ["Hello", " ", "world", "!"]


class TestTokenAccumulation:
    """Tests for token collection and counting."""

    def generator_with_delays(self) -> Generator[str, None, None]:
        """Generator with small delays to test timing."""
        for token in ["tok1", "tok2", "tok3"]:
            yield token
            time.sleep(0.01)  # Small delay for timing tests

    def test_token_count_increments(self):
        """token_count should increase as tokens are yielded."""
        gen = self.generator_with_delays()
        stream = TokenStream(gen)

        assert stream.token_count == 0
        next(stream)
        assert stream.token_count == 1
        next(stream)
        assert stream.token_count == 2

    def test_collect_joins_all_tokens(self):
        """collect() should return concatenated tokens."""

        def simple_gen():
            yield "H"
            yield "i"

        gen = simple_gen()
        stream = TokenStream(gen)
        result = stream.collect()

        assert result == "Hi"
        assert stream.token_count == 2

    def test_collect_with_empty_stream(self):
        """collect() on empty stream should return empty string."""

        def empty_gen():
            return
            yield  # Unreachable, makes it a generator

        gen = empty_gen()
        stream = TokenStream(gen)
        result = stream.collect()

        assert result == ""
        assert stream.token_count == 0

    def test_token_list_accumulates(self):
        """Internal token list should accumulate all tokens."""

        def counting_gen():
            for i in range(5):
                yield f"token{i}"

        gen = counting_gen()
        stream = TokenStream(gen)
        list(stream)  # Consume all

        assert stream.token_count == 5


class TestTimingMetrics:
    """Tests for elapsed time and tokens-per-second metrics."""

    def test_elapsed_increases_over_time(self):
        """elapsed property should increase as time passes."""

        def slow_gen():
            yield "1"
            time.sleep(0.05)
            yield "2"

        gen = slow_gen()
        stream = TokenStream(gen)

        start_elapsed = stream.elapsed
        next(stream)
        next(stream)
        end_elapsed = stream.elapsed

        # End elapsed should be slightly larger (gen took 0.05s)
        assert end_elapsed > start_elapsed
        assert end_elapsed >= 0.04  # Allow small timing variations

    def test_elapsed_after_finish(self):
        """elapsed should be accurate after stream finishes."""

        def timed_gen():
            yield "a"
            time.sleep(0.05)
            yield "b"

        gen = timed_gen()
        stream = TokenStream(gen)

        # Consume all tokens
        list(stream)

        elapsed = stream.elapsed
        assert elapsed >= 0.04  # Should have paused for ~0.05s

    def test_tokens_per_second_calculation(self):
        """tokens_per_second should calculate correctly."""

        def fast_gen():
            for i in range(4):
                yield f"t{i}"

        gen = fast_gen()
        stream = TokenStream(gen)

        start = time.perf_counter()
        list(stream)
        end = time.perf_counter()

        expected_tps = 4 / (end - start)
        actual_tps = stream.tokens_per_second

        # Should be approximately equal (within 10% due to timing variations)
        assert actual_tps * 0.9 <= expected_tps <= actual_tps * 1.1

    def test_tokens_per_second_zero_elapsed(self):
        """tokens_per_second should return 0 if elapsed is 0 (edge case)."""

        def instant_gen():
            yield "instant"

        gen = instant_gen()
        stream = TokenStream(gen)

        # Artificially set finished_at to same as started_at
        next(stream)
        stream._finished_at = stream._started_at

        assert stream.tokens_per_second == 0.0

    def test_elapsed_during_active_stream(self):
        """elapsed should work correctly while stream is still active."""

        def slow_gen():
            yield "1"
            time.sleep(0.05)
            # Don't return yet, stream still active

        gen = slow_gen()
        stream = TokenStream(gen)

        next(stream)
        elapsed_at_one = stream.elapsed

        time.sleep(0.05)
        elapsed_at_two = stream.elapsed

        # Should increase because we slept while monitoring
        assert elapsed_at_two > elapsed_at_one


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_generator(self):
        """Should handle empty generator gracefully."""

        def empty():
            return
            yield

        stream = TokenStream(empty())
        tokens = list(stream)

        assert tokens == []
        assert stream.token_count == 0
        assert stream.elapsed >= 0

    def test_single_token(self):
        """Should handle single-token stream."""

        def single():
            yield "only"

        stream = TokenStream(single())
        assert list(stream) == ["only"]
        assert stream.token_count == 1

    def test_large_tokens(self):
        """Should handle large token strings."""

        def large_gen():
            yield "x" * 10000
            yield "y" * 10000

        stream = TokenStream(large_gen())
        result = stream.collect()

        assert len(result) == 20000
        assert stream.token_count == 2

    def test_whitespace_tokens(self):
        """Should preserve whitespace in tokens."""

        def whitespace_gen():
            yield "   "
            yield "\n"
            yield "\t"

        stream = TokenStream(whitespace_gen())
        result = stream.collect()

        assert result == "   \n\t"

    def test_unicode_tokens(self):
        """Should handle unicode tokens correctly."""

        def unicode_gen():
            yield "Hello"
            yield " 🚀 "
            yield "世界"

        stream = TokenStream(unicode_gen())
        result = stream.collect()

        assert result == "Hello 🚀 世界"
        assert stream.token_count == 3

    def test_finished_at_set_on_stopiteration(self):
        """_finished_at should be set when StopIteration is raised."""

        def tiny_gen():
            yield "x"

        stream = TokenStream(tiny_gen())
        assert stream._finished_at is None

        next(stream)
        assert stream._finished_at is None  # Not finished yet

        try:
            next(stream)
        except StopIteration:
            pass

        assert stream._finished_at is not None


class TestCollectMethod:
    """Tests specific to collect() convenience method."""

    def test_collect_returns_string_type(self):
        """collect() should return str, not list."""

        def text_gen():
            yield "a"
            yield "b"

        stream = TokenStream(text_gen())
        result = stream.collect()

        assert isinstance(result, str)
        assert result == "ab"

    def test_collect_handles_special_chars(self):
        """collect() should preserve special characters."""

        def special_gen():
            yield "["
            yield "{map}"
            yield "]"

        stream = TokenStream(special_gen())
        result = stream.collect()

        assert result == "[{map}]"

    def test_multiple_collect_calls(self):
        """Multiple collect() calls should return same result."""

        def limited_gen():
            yield "a"
            yield "b"

        stream = TokenStream(limited_gen())
        first = stream.collect()

        assert first == "ab"


class TestTimingAccuracy:
    """Tests for precise timing measurement."""

    def test_elapsed_property_uses_finish_time_when_done(self):
        """When finished, elapsed should use recorded finish time."""

        def quick_gen():
            yield "done"

        gen = quick_gen()
        stream = TokenStream(gen)

        list(stream)  # Consume all

        # Get elapsed twice; should be identical since _finished_at is set
        elapsed1 = stream.elapsed
        time.sleep(0.01)
        elapsed2 = stream.elapsed

        # Both should be very close since using same finish time
        assert abs(elapsed1 - elapsed2) < 0.005

    def test_elapsed_uses_current_time_while_active(self):
        """While stream is active, elapsed should track current time."""

        def slow_gen():
            yield "token"
            # Stream left active

        gen = slow_gen()
        stream = TokenStream(gen)

        next(stream)
        elapsed1 = stream.elapsed

        time.sleep(0.02)
        elapsed2 = stream.elapsed

        # Should have increased due to sleep
        assert elapsed2 > elapsed1 + 0.015
