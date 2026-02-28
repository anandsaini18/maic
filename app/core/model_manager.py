from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import psutil
import mlx_lm

from app.core.config import GenerationStrategy, settings
from app.core.token_stream import TokenStream

logger = logging.getLogger(__name__)


def _model_local_path(model_id: str) -> Path:
    """
    Convert HuggingFace model ID to a local path, replacing '/' with '--'.

    Example: 'mlx-community/Phi-3.5-mini-instruct-4bit' → '~/models/mlx-community--Phi-3.5-mini-instruct-4bit/'

    Using '--' keeps every model in a flat folder (simpler management) rather than
    creating nested subdirectories.
    """
    models_dir = Path(settings.models_dir).expanduser().resolve()
    return models_dir / model_id.replace("/", "--")


def _ensure_model_downloaded(model_id: str, local_path: Path) -> None:
    """
    Download model weights from HuggingFace if not already cached locally.

    First run: Uses hf_transfer (Rust-based, 10-100× faster) to download weights,
    skipping PyTorch (.bin) files since MLX only needs .safetensors.

    Subsequent runs: Detects any cached .safetensors or .npz files and skips
    download, allowing config changes without re-downloading.
    """
    # Check for cached weights (skip download if found)
    weight_files = list(local_path.glob("*.safetensors")) + \
                        list(local_path.glob("*.npz"))
    if weight_files:
        logger.info(
            "Model found locally at '%s' — skipping download.", local_path)
        return

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download

    logger.info(
        "Downloading '%s' → %s  (using hf_transfer for fast download) …",
        model_id,
        local_path,
    )
    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        # Skip PyTorch weights; MLX uses only safetensors
        ignore_patterns=["*.bin", "original/*"],
    )
    logger.info("Download complete: %s", local_path)


# ─────────────────────────────────────────────────────────────────────────────
# Known model sizes (GiB) — used for RAM feasibility checks and UI suggestions
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_MODEL_SIZES: dict[str, float] = {
    # Open models (no token required)
    "mlx-community/SmolLM2-1.7B-Instruct-4bit": 1.0,
    "mlx-community/Phi-3.5-mini-instruct-4bit": 2.3,       # ← default
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": 2.5,
    "mlx-community/gemma-3-4b-it-4bit": 2.6,
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit": 4.0,
    "mlx-community/Qwen3-8B-4bit": 5.0,
    # Gated models (Meta license — accept at hf.co/meta-llama first)
    "mlx-community/Llama-3.2-1B-Instruct-4bit": 0.7,
    "mlx-community/Llama-3.2-3B-Instruct-4bit": 1.8,
    "mlx-community/Llama-3.1-8B-Instruct-4bit": 4.9,
    "mlx-community/Llama-3.3-70B-Instruct-4bit": 40.0,
    # Unfeasible on any MacBook (included for reference)
    "mlx-community/Kimi-K2.5": 658.0,
}

# Models requiring HuggingFace authentication (gated behind license agreements)
TOKEN_REQUIRED_MODELS: frozenset[str] = frozenset({
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.3-70B-Instruct-4bit",
})

# Sorted by size for easy lookup by RAM tier
FEASIBLE_BY_RAM: list[tuple[float, str]] = sorted(
    KNOWN_MODEL_SIZES.items(), key=lambda x: x[1]
)


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class ModelTooLargeError(RuntimeError):
    """Raised when the requested model needs more RAM than this machine safely has."""


class ModelLoadError(RuntimeError):
    """Raised when downloading or loading a model from disk fails."""


# ── Observer Pattern ──────────────────────────────────────────────────────────

@runtime_checkable
class InferenceObserver(Protocol):
    """
    Observer pattern interface — anything that wants to react to generation events.

    You can attach multiple observers to the model manager. Each one gets called
    at three moments: when a token arrives, when generation finishes, and when
    an error occurs. Implement only the hooks you care about.

    Register with: model_manager.register_observer(your_observer)
    """

    def on_token(self, token: str) -> None: ...
    def on_complete(self, stats: dict) -> None: ...
    def on_error(self, error: Exception) -> None: ...


class StatsObserver:
    """
    Built-in observer that logs a performance summary when generation ends.

    Fires on_complete() with a dict containing token_count, elapsed (seconds),
    and tokens_per_second. Nothing is logged per-token to keep logs readable.
    This observer is always active — it's added to every ModelManager by default.
    """

    def on_token(self, token: str) -> None:
        pass  # no-op per token; stats aggregated at on_complete

    def on_complete(self, stats: dict) -> None:
        logger.info(
            "Inference complete — %d tokens in %.2fs (%.1f tok/s)",
            stats["token_count"],
            stats["elapsed"],
            stats["tokens_per_second"],
        )

    def on_error(self, error: Exception) -> None:
        logger.error("Inference error: %s", error)


# ── Facade Pattern ────────────────────────────────────────────────────────────

class ModelManager:
    """
    Facade pattern — the single entry point for everything model-related.

    The rest of the codebase only ever calls two methods:
      - load(model_id)   — download if needed, load weights into memory
      - generate(messages, strategy) — run inference, return a TokenStream

    All the messy MLX details (tokenization, special tokens, stop detection,
    observer notifications) are hidden inside this class.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._observers: list[InferenceObserver] = [StatsObserver()]
        # Populated after load() — text forms of all EOS tokens for stop-string detection
        self._stop_strings: frozenset[str] = frozenset()

    # ── Observer management ──────────────────────────────────────────────────

    def register_observer(self, observer: InferenceObserver) -> None:
        """Attach a new observer. It will receive events for all future generate() calls."""
        self._observers.append(observer)

    def _notify_token(self, token: str) -> None:
        """Broadcast a newly generated token to all observers."""
        for obs in self._observers:
            obs.on_token(token)

    def _notify_complete(self, stats: dict) -> None:
        """Broadcast generation-complete stats (token count, speed) to all observers."""
        for obs in self._observers:
            obs.on_complete(stats)

    def _notify_error(self, error: Exception) -> None:
        """Broadcast an exception to all observers so they can log or react."""
        for obs in self._observers:
            obs.on_error(error)

    # ── RAM feasibility check ────────────────────────────────────────────────

    def _check_ram(self, model_id: str) -> None:
        """
        Fail early if model needs more than 80% of available RAM.

        Looks up model size in KNOWN_MODEL_SIZES and suggests feasible alternatives.
        Unknown models skip this check and let MLX fail later if needed.
        """
        required_gb = KNOWN_MODEL_SIZES.get(model_id)
        if required_gb is None:
            return  # Model size unknown; let MLX attempt to load it

        available_gb = psutil.virtual_memory().total / (1024 ** 3)
        safe_limit = available_gb * 0.8  # Use only 80% to leave headroom for system

        if required_gb > safe_limit:
            feasible = [
                f"  - {mid} (~{size:.1f} GB)"
                for mid, size in FEASIBLE_BY_RAM
                if size <= safe_limit
            ]
            feasible_str = "\n".join(
                feasible) if feasible else "  (none in known list)"
            raise ModelTooLargeError(
                f"\n\n❌  Model '{model_id}' requires ~{required_gb:.0f} GB RAM"
                f" (you have {available_gb:.0f} GB).\n"
                f"Feasible alternatives for your Mac:\n{feasible_str}\n"
            )

    # ── Stop-string derivation ────────────────────────────────────────────────

    def _derive_stop_strings(self) -> frozenset[str]:
        """
        Auto-detect end-of-generation markers specific to the loaded model.

        Problem: Different models signal completion differently (Phi: <|end|>,
        Qwen:
