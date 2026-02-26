"""Shared pytest fixtures and configuration."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for model tests."""
    tokenizer = Mock()
    tokenizer.eos_token_ids = [128001, 128009]
    tokenizer.decode.return_value = "<|end|>"
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.encode.return_value = [1, 2, 3]
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock MLX model for model tests."""
    model = Mock()
    return model


@pytest.fixture
def mock_mlx_response():
    """Mock streaming response from MLX."""
    response = Mock()
    response.text = "token"
    response.finish_reason = None
    return response


@pytest.fixture
def simple_token_generator():
    """Simple generator of test tokens."""
    def gen():
        tokens = ["Hello", " ", "world", "!"]
        for token in tokens:
            yield token
    return gen()


@pytest.fixture
def temp_models_dir(tmp_path):
    """Temporary directory for model files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir
