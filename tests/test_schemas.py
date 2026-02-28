"""Tests for Pydantic schemas — only the edge cases that Pydantic
won't catch automatically (role literals, optional vs required fields,
auto-generated defaults).

These tests act as a safety net: if someone changes a field type or
removes Optional, these break before the bug reaches a client.
"""

import pytest
from pydantic import ValidationError

from app.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    DeltaMessage,
    Message,
)


class TestMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            m = Message(role=role, content="hi")
            assert m.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="tool", content="hi")

    def test_invalid_role_human_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="human", content="hi")


class TestChatCompletionRequest:
    def test_defaults(self):
        req = ChatCompletionRequest(
            model="m", messages=[Message(role="user", content="hi")]
        )
        assert req.stream is False
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.top_p is None

    def test_stream_true(self):
        req = ChatCompletionRequest(
            model="m", messages=[Message(role="user", content="hi")], stream=True
        )
        assert req.stream is True


class TestChatCompletionResponse:
    def test_auto_generated_id_and_created(self):
        resp1 = ChatCompletionResponse(
            model="m",
            choices=[],
        )
        resp2 = ChatCompletionResponse(
            model="m",
            choices=[],
        )
        assert resp1.id.startswith("chatcmpl-")
        assert resp1.id != resp2.id  # UUIDs must differ
        assert resp1.object == "chat.completion"
        assert isinstance(resp1.created, int)


class TestChunkChoice:
    def test_finish_reason_none_by_default(self):
        c = ChunkChoice(delta=DeltaMessage(content="hi"))
        assert c.finish_reason is None

    def test_finish_reason_stop(self):
        c = ChunkChoice(delta=DeltaMessage(), finish_reason="stop")
        assert c.finish_reason == "stop"

    def test_invalid_finish_reason_rejected(self):
        with pytest.raises(ValidationError):
            ChunkChoice(delta=DeltaMessage(), finish_reason="timeout")


class TestChatCompletionChunk:
    def test_object_is_chunk_type(self):
        chunk = ChatCompletionChunk(
            model="m",
            choices=[ChunkChoice(delta=DeltaMessage())],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.usage is None  # no usage on content chunks
