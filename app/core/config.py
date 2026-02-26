from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Strategy Pattern ──────────────────────────────────────────────────────────

@runtime_checkable
class GenerationStrategy(Protocol):
    """
    Strategy pattern — a swappable object that controls how the model picks
    the next token at each step.

    Any callable with this signature counts as a strategy. You can pass a
    different one per request without changing ModelManager at all. The strategy
    returns a kwargs dict that gets unpacked straight into mlx_lm.stream_generate().
    """
    def __call__(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict: ...


def default_strategy(*, max_tokens: int, temperature: float, top_p: float) -> dict:
    """
    Standard sampling: pick tokens randomly weighted by probability.

    'temperature' controls creativity — higher = more random, lower = more focused.
    'top_p' (nucleus sampling) cuts off the long tail of unlikely tokens before sampling.
    This is the strategy used for normal chat responses.
    """
    from mlx_lm.sample_utils import make_sampler
    return {
        "max_tokens": max_tokens,
        "sampler": make_sampler(temp=temperature, top_p=top_p),
    }


def greedy_strategy(*, max_tokens: int, temperature: float, top_p: float) -> dict:
    """
    Greedy decoding: always pick the single most probable next token (temp=0).

    Fully deterministic — the same prompt always produces the same output.
    Useful for testing or any task where you want reproducible results.
    The temperature and top_p arguments are ignored here.
    """
    from mlx_lm.sample_utils import make_sampler
    return {
        "max_tokens": max_tokens,
        "sampler": make_sampler(temp=0.0),
    }


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All runtime configuration for the server.

    Values are read from environment variables or a .env file in this priority:
      CLI flag (main.py) > environment variable > .env file > default below.

    models_dir is intentionally separate from the HuggingFace cache — it survives
    .env deletion and can point to an external drive if your internal disk is tight.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_id: str = "mlx-community/Phi-3.5-mini-instruct-4bit"
    # Where downloaded weights live. Model 'org/name' is stored as '{models_dir}/org--name/'
    models_dir: str = "~/models"
    host: str = "0.0.0.0"
    port: int = 8000
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


settings = Settings()
