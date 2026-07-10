r"""MLX model lifecycle management — download, load, generate, and delete.

Implements the Facade pattern: the rest of the app calls ``model_manager.load()``
and ``model_manager.generate()`` without knowing about MLX internals, tokenizer
chat templates, or stop-string detection. Also implements the Observer pattern
for generation event broadcasting and provides TTL-cached disk queries for
the UI polling endpoint.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import psutil

from maic.core.config import GenerationStrategy, settings
from maic.core.model_sizing import KNOWN_SIZES, KNOWN_SIZES_BY_RAM
from maic.core.token_stream import TokenStream

logger = logging.getLogger(__name__)


# ── Tool-call format detection ────────────────────────────────────────────────


class ToolCallFormat(str, Enum):
    """Output format a model uses when emitting tool calls.

    Detected at model-load time by inspecting the chat template Jinja2 source.
    ``NONE`` means the model has no tool-calling support; ``UNKNOWN`` means the
    template was present but didn't match any known pattern (the fallback
    try-all parser is used in that case).
    """

    QWEN = "qwen"       # <tool_call>{"name":…,"arguments":{…}}</tool_call>
    LLAMA3 = "llama3"   # <|python_tag|>[{"name":…,"parameters":{…}}]
    MISTRAL = "mistral" # [TOOL_CALLS] [{"name":…,"arguments":{…}}]
    DEEPSEEK = "deepseek"  # <|tool▁calls▁begin|>…<|tool▁sep|>…<|tool▁calls▁end|>
    HERMES = "hermes"   # <tool_call>{"name":…,"arguments":{…}}</tool_call> (NousHermes variant)
    UNKNOWN = "unknown" # Template present but format not recognised → fallback parser
    NONE = "none"       # No tool-calling support in the chat template


# Ordered list of (marker, format) pairs checked against the raw Jinja2 template.
# Earlier entries take priority; the first match wins.
_FORMAT_MARKERS: list[tuple[str, ToolCallFormat]] = [
    ("<|python_tag|>", ToolCallFormat.LLAMA3),
    ("[TOOL_CALLS]", ToolCallFormat.MISTRAL),
    # DeepSeek uses Unicode word-joiners (U+2581) in special token names.
    # Match both the exact Unicode form and an ASCII fallback.
    ("tool\u2581calls\u2581begin", ToolCallFormat.DEEPSEEK),
    ("tool_calls_begin", ToolCallFormat.DEEPSEEK),  # ASCII version in some configs
    # Qwen / Kimi K2 templates reference '<tool_call>' blocks for model output AND
    # '<tool_response>' for rendering tool results back.  QWEN must be checked before
    # HERMES because Qwen templates contain BOTH markers; the first match wins.
    ("<tool_call>", ToolCallFormat.QWEN),
    # Hermes (NousResearch): uses '<tool_response>' as the result wrapper.
    # Pure Hermes templates contain '<tool_response>' but NOT '<tool_call>', so they
    # still land here after the QWEN check above falls through.
    ("<tool_response>", ToolCallFormat.HERMES),
    # Fallback: template defines tools section but format is unrecognised.
    # Match the Jinja2 conditional that gates the tools block in templates.
    ("{% if tools %}", ToolCallFormat.UNKNOWN),
]


def detect_tool_call_format(chat_template: str) -> ToolCallFormat:
    """Inspect the raw Jinja2 chat template and return the model's tool-call format.

    The template is the raw string stored in ``tokenizer_config.json`` under the
    ``chat_template`` key.  We pattern-match on literal marker strings that appear
    in every known family's template; the first match wins.  Returns ``NONE`` when
    no tool-aware markers are found at all.

    This runs once at model-load time and the result is cached on the manager, so
    it has no impact on hot-path inference performance.
    """
    if not isinstance(chat_template, str) or not chat_template:
        return ToolCallFormat.NONE
    for marker, fmt in _FORMAT_MARKERS:
        if marker in chat_template:
            # Distinguish UNKNOWN from NONE: only return UNKNOWN when the template
            # actually references tools (caught by the "tools" marker above), but
            # didn't match any concrete format.
            return fmt
    return ToolCallFormat.NONE


# Thinking-mode detection markers — any of these in the raw Jinja2 template
# indicates the model supports extended chain-of-thought via enable_thinking=True.
# Qwen3 templates reference ``enable_thinking`` as a template variable; some also
# emit literal ``<think>`` tokens.
_THINKING_MARKERS: tuple[str, ...] = ("enable_thinking", "<think>")


def detect_thinking_support(chat_template: str) -> bool:
    """Return True if the model's chat template supports ``enable_thinking=True``.

    Checks for Qwen3-style thinking markers in the raw Jinja2 template source.
    This runs once at load time; the result is stored on the manager instance.
    """
    if not isinstance(chat_template, str) or not chat_template:
        return False
    return any(marker in chat_template for marker in _THINKING_MARKERS)

# ── Cached total RAM (constant for the lifetime of the process) ───────────────
# Used by hot paths (models_status polling, adapter feasibility checks) to avoid
# repeated psutil.virtual_memory() calls. _check_ram() uses psutil directly so
# tests can still control the value via mocking.
TOTAL_RAM_GB: float = psutil.virtual_memory().total / (1024**3)


def _model_local_path(model_id: str) -> Path:
    """
    Convert HuggingFace model ID to a local path, replacing '/' with '--'.

    Example: 'mlx-community/Phi-3.5-mini-instruct-4bit' → '~/models/mlx-community--Phi-3.5-mini-instruct-4bit/'

    Using '--' keeps every model in a flat folder (simpler management) rather than
    creating nested subdirectories.

    Validates model ID format to prevent path traversal attacks.
    """
    import re

    # Whitelist: org/name format (HuggingFace standard)
    # Allows org names and model names with alphanumerics, hyphens, underscores, dots, and colons
    if not re.match(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.:,-]+$', model_id):
        raise ValueError(
            f"Invalid model ID format: '{model_id}'. "
            f"Expected format: 'organization/model-name' (alphanumerics, hyphens, underscores, dots allowed)"
        )

    models_dir = Path(settings.models_dir).expanduser().resolve()
    safe_name = model_id.replace("/", "--")
    final_path = models_dir / safe_name

    # Verify the resolved path is under models_dir (prevent symlink escape / directory traversal)
    try:
        final_path.resolve().relative_to(models_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Path traversal detected in model_id: {model_id}") from exc

    return final_path


def _ensure_model_downloaded(model_id: str, local_path: Path) -> None:
    """
    Download model weights from HuggingFace if not already cached locally.

    First run: Uses hf_transfer (Rust-based, 10-100× faster) to download weights,
    skipping PyTorch (.bin) files since MLX only needs .safetensors.

    Subsequent runs: Detects any cached .safetensors or .npz files and skips
    download, allowing config changes without re-downloading.
    """
    # Check for cached weights (skip download if found)
    weight_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.npz"))
    if weight_files:
        logger.info("Model found locally at '%s' — skipping download.", local_path)
        return

    # Enable Rust-based downloader for 10-100× faster transfers
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


# Guard against pathological stop strings that can delay visible streaming.
# Real EOS markers are short (< 20 chars), so 64 keeps correctness while
# preventing whole-response buffering if a tokenizer emits a bad marker.
MAX_STOP_LOOKBACK_CHARS = 64

_DOWNLOAD_CACHE_TTL = 30.0  # seconds
_DISK_SIZE_CACHE_TTL = 60.0  # seconds
_LOCAL_IDS_CACHE_TTL = 15.0  # seconds (directory list changes less frequently)


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

    def on_token(self, token: str) -> None:
        r"""Called each time a new token is generated."""
        ...

    def on_complete(self, stats: dict[str, Any]) -> None:
        r"""Called when generation finishes with performance stats."""
        ...

    def on_error(self, error: Exception) -> None:
        r"""Called when generation fails with the raised exception."""
        ...


class StatsObserver:
    """
    Built-in observer that logs a performance summary when generation ends.

    Fires on_complete() with a dict containing token_count, elapsed (seconds),
    and tokens_per_second. Nothing is logged per-token to keep logs readable.
    This observer is always active — it's added to every ModelManager by default.
    """

    def on_token(self, token: str) -> None:
        r"""No-op — stats are aggregated at completion, not per token."""

    def on_complete(self, stats: dict[str, Any]) -> None:
        r"""Log token count, elapsed time, and throughput at INFO level."""
        logger.info(
            "Inference complete — %d tokens in %.2fs (%.1f tok/s)",
            stats["token_count"],
            stats["elapsed"],
            stats["tokens_per_second"],
        )

    def on_error(self, error: Exception) -> None:
        r"""Log the inference error at ERROR level."""
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
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id: str | None = None
        self._observers: list[InferenceObserver] = [StatsObserver()]
        # Populated after load() — text forms of all EOS tokens for stop-string detection
        self._stop_strings: frozenset[str] = frozenset()
        # Tool-call format detected from the chat template at load time.
        self._tool_call_format: ToolCallFormat = ToolCallFormat.NONE
        # Whether the loaded model's chat template supports enable_thinking=True.
        self._supports_thinking: bool = False
        # Cached module references set after load() — avoids re-importing on every generate()
        self._mx: Any = None
        self._mlx_lm: Any = None
        # KV prompt cache — persists across generate() calls within a conversation
        # so previously-seen tokens are not re-processed. Reset on model switch or
        # explicit clear_cache() call. Created lazily in _ensure_prompt_cache().
        self._prompt_cache: Any = None
        # Per-instance TTL caches for is_downloaded / disk_size_gb / local_model_ids.
        # Instance-level (not module-level) so tests that create fresh ModelManager()
        # instances start with an empty cache and don't bleed state across test runs.
        self._download_cache: dict[str, tuple[bool, float]] = {}
        self._disk_size_cache: dict[str, tuple[float | None, float]] = {}
        self._local_ids_cache: tuple[list[str], float] | None = None

    # ── Observer management ──────────────────────────────────────────────────

    def register_observer(self, observer: InferenceObserver) -> None:
        """Attach a new observer. It will receive events for all future generate() calls."""
        self._observers.append(observer)

    def _notify_token(self, token: str) -> None:
        """Broadcast a newly generated token to all observers."""
        for obs in self._observers:
            obs.on_token(token)

    def _notify_complete(self, stats: dict[str, Any]) -> None:
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

        Looks up model size in KNOWN_SIZES (model_sizing) and suggests feasible
        alternatives. Unknown models skip this check and let MLX fail later if needed.

        Reads psutil directly (not the cached constant) so that tests can patch
        psutil.virtual_memory to simulate different RAM configurations. This is
        acceptable because _check_ram is called only at model-load time, not on
        every request, so the psutil call cost is negligible.
        """
        required_gb = KNOWN_SIZES.get(model_id)
        if required_gb is None:
            return  # Model size unknown; let MLX attempt to load it

        available_gb = psutil.virtual_memory().total / (1024**3)
        safe_limit = available_gb * 0.8  # Use only 80% to leave headroom for system

        if required_gb > safe_limit:
            feasible = [
                f"  - {mid} (~{size:.1f} GB)"
                for mid, size in KNOWN_SIZES_BY_RAM
                if size <= safe_limit
            ]
            feasible_str = "\n".join(feasible) if feasible else "  (none in known list)"
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
        Qwen: <|im_end|>, Mistral: </s>). Missing these causes garbage output.

        Solution: Extract stop tokens via two methods:
          1. Decode known eos_token_ids from tokenizer
          2. Probe chat template with sentinel string to find appended tokens

        These are used during generation to truncate output immediately when detected.
        """
        stop: set[str] = set()

        # Method 1: Extract from tokenizer's known EOS token IDs
        for eid in self._tokenizer.eos_token_ids:
            s = self._tokenizer.decode([eid]).strip()
            if s:
                stop.add(s)

        # Method 2: Probe chat template to find tokens appended after content
        # (Catches model-specific markers like <|end|>, <|im_end|>, </s>)
        try:
            SENTINEL = "\x01\x02\x03"  # Unlikely to appear in real templates
            probe_ids: list[int] = self._tokenizer.apply_chat_template(
                [{"role": "assistant", "content": SENTINEL}],
                tokenize=True,
                add_generation_prompt=False,
            )
            sentinel_ids = self._tokenizer.encode(SENTINEL, add_special_tokens=False)
            for i in range(len(probe_ids)):
                if probe_ids[i : i + len(sentinel_ids)] == sentinel_ids:
                    # Tokens after sentinel are the model's end-of-turn markers
                    for eid in probe_ids[i + len(sentinel_ids) :]:
                        s = self._tokenizer.decode([eid]).strip()
                        if s:
                            stop.add(s)
                            # Register with tokenizer for native token-level detection
                            self._tokenizer.add_eos_token(s)
                    break
        except Exception as exc:
            logger.debug("Chat template probe failed: %s", exc)

        return frozenset(stop)

    # ── Model loading ────────────────────────────────────────────────────────

    def load(self, model_id: str | None = None) -> None:
        """
        Download (if needed) and load a model into memory.

        Process: RAM check → Download → Load into MLX → Derive stop markers

        Raises ModelLoadError with troubleshooting hints if any step fails.
        """
        model_id = model_id or settings.model_id
        self._check_ram(model_id)

        local_path = _model_local_path(model_id)

        try:
            _ensure_model_downloaded(model_id, local_path)
        except Exception as exc:
            logger.error("Failed to download model '%s': %s", model_id, exc, exc_info=True)
            raise ModelLoadError(
                f"Failed to download model '{model_id}'. "
                f"Check the model ID, your internet connection, or your HF token for gated models."
            ) from exc

        logger.info("Loading model from '%s' …", local_path)
        try:
            import mlx_lm  # Lazy import; only needed at load time

            # Load from local path (offline after first download)
            result = mlx_lm.load(str(local_path))
            self._model, self._tokenizer = result[0], result[1]
            self._model_id = model_id
            self._stop_strings = self._derive_stop_strings()
            # Detect tool-call format from the chat template (once per load).
            raw_template = getattr(self._tokenizer, "chat_template", "")
            chat_template = raw_template if isinstance(raw_template, str) else ""
            self._tool_call_format = detect_tool_call_format(chat_template)
            self._supports_thinking = detect_thinking_support(chat_template)
            # Cache mlx_lm reference so generate() avoids re-importing on every call.
            # mlx.core is cached lazily on the first generate() call (it's only needed there).
            self._mlx_lm = mlx_lm
            logger.info(
                "Model '%s' loaded. Stop strings: %s. Tool-call format: %s. Thinking: %s",
                model_id,
                self._stop_strings,
                self._tool_call_format.value,
                self._supports_thinking,
            )
        except Exception as exc:
            logger.error("Failed to load model '%s' from disk: %s", model_id, exc, exc_info=True)
            raise ModelLoadError(
                f"Failed to load model '{model_id}'. "
                f"The local files may be corrupted — try re-downloading the model."
            ) from exc

        # Reset prompt cache — the old cache belongs to the previous model
        self._prompt_cache = None
        # Invalidate download/disk-size cache for this model after a successful load
        self._download_cache.pop(model_id, None)
        self._disk_size_cache.pop(model_id, None)

        # Start/restart batch manager if batch mode is enabled
        if settings.batch_mode:
            from maic.core.batch_manager import batch_manager

            batch_manager.start(self._model, self._tokenizer)

    # ── Prompt cache management ─────────────────────────────────────────────

    def _ensure_prompt_cache(self) -> Any:
        """Return the current prompt cache, creating one if needed.

        The cache is created lazily (not in load()) so that the mlx_lm.models.cache
        import only happens when generation is actually requested.
        """
        if self._prompt_cache is None:
            from mlx_lm.models.cache import make_prompt_cache

            cache_kwargs: dict[str, Any] = {}
            if settings.max_kv_size is not None:
                cache_kwargs["max_kv_size"] = settings.max_kv_size
            self._prompt_cache = make_prompt_cache(self._model, **cache_kwargs)
            logger.debug("Created new prompt cache (max_kv_size=%s)", settings.max_kv_size)
        return self._prompt_cache

    def clear_cache(self) -> None:
        """Discard the current prompt cache so the next generate() starts fresh.

        Called when the user starts a new conversation or switches context.
        """
        self._prompt_cache = None
        logger.debug("Prompt cache cleared")

    @property
    def is_loaded(self) -> bool:
        """True once load() has successfully completed."""
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        """The HuggingFace model ID that is currently loaded, or None if not loaded yet."""
        return self._model_id

    @property
    def tokenizer(self) -> Any:
        """The tokenizer for the currently loaded model. None if no model is loaded."""
        return self._tokenizer

    @property
    def tool_call_format(self) -> ToolCallFormat:
        """Tool-call output format detected from the loaded model's chat template.

        Returns ``NONE`` when no model is loaded or the template has no tool support.
        Callers should use this to route to the correct parser rather than trying
        every known format on every response.
        """
        return self._tool_call_format

    @property
    def supports_tool_calling(self) -> bool:
        """True if the currently loaded model's chat template supports tool calls."""
        return self._tool_call_format not in (ToolCallFormat.NONE, ToolCallFormat.UNKNOWN)

    @property
    def supports_thinking(self) -> bool:
        """True if the loaded model supports ``enable_thinking=True`` in apply_chat_template.

        Qwen3 templates expose this variable; passing it to a tokenizer that does not
        support it raises a TypeError/KeyError, so callers must gate on this property.
        """
        return self._supports_thinking

    def tool_call_format_for(self, model_id: str) -> ToolCallFormat:
        """Return the tool-call format for any model — loaded or just downloaded.

        If *model_id* is the currently loaded model, returns the already-detected
        format.  For a downloaded (but not loaded) model, reads the
        ``tokenizer_config.json`` from disk and runs detection.  Falls back to
        ``NONE`` when the model is not on disk or the config is unreadable.
        """
        if model_id == self._model_id:
            return self._tool_call_format
        local_path = _model_local_path(model_id)
        config_path = local_path / "tokenizer_config.json"
        if not config_path.exists():
            return ToolCallFormat.NONE
        try:
            import json as _json

            cfg = _json.loads(config_path.read_text())
            template = cfg.get("chat_template", "") or ""
            return detect_tool_call_format(template)
        except Exception:
            return ToolCallFormat.NONE

    # ── Generation (Facade entry point) ──────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, Any]],
        strategy: GenerationStrategy,
        tools: list[dict[str, Any]] | None = None,
    ) -> TokenStream:
        """
        Run inference on a list of chat messages and return a TokenStream.

        The caller doesn't need to know anything about MLX. Internally this:
        1. Applies the model's chat template to format the conversation correctly
           (each model has its own special tokens for <user>, <assistant>, etc.).
        2. Converts the token IDs to an MLX array and kicks off streaming generation.
        3. Wraps the raw MLX generator in a stop-aware inner generator (_observed)
           that cuts output at the right moment — see the three-level stop logic below.
        4. Wraps everything in a TokenStream so callers can just do `for token in stream`.
        5. Fires observer.on_complete() with speed stats once the stream is exhausted.

        The 'strategy' argument controls sampling (temperature, top_p, max_tokens).
        Pass default_strategy for normal use, greedy_strategy for deterministic output.

        The optional 'tools' argument is a list of OpenAI-style tool definitions.
        When provided, they are passed to the chat template so the model can emit
        tool calls; the caller is responsible for parsing those out of the output.
        """
        # Use cached module references set during load() — avoids sys.modules lookup per call.
        # If _mlx_lm is None (e.g., in tests that set up mm internals without calling load()),
        # fall back to a lazy import so sys.modules patches in tests still work correctly.
        if self._mx is None:
            import mlx.core as mx

            self._mx = mx
        mx = self._mx
        if self._mlx_lm is None:
            import mlx_lm

            self._mlx_lm = mlx_lm
        mlx_lm = self._mlx_lm

        # Apply chat template and tokenize messages in one step.
        # Ensures special tokens are properly encoded for EOS detection.
        # When `tools` are provided, the model's chat template injects the tool
        # signatures and instructs the model to emit <tool_call> blocks; passing
        # None leaves the template's non-tool path unchanged.
        # `enable_thinking=True` is passed only when the template supports it
        # (Qwen3 family); passing it to an unsupported tokenizer raises TypeError.
        template_kwargs: dict[str, Any] = {
            "tools": tools,
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if self._supports_thinking:
            template_kwargs["enable_thinking"] = True
        token_ids: list[int] = self._tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )
        prompt_len = len(token_ids)
        if prompt_len > 4000:
            logger.warning(
                "Large prompt detected: %d tokens (tools=%d). "
                "KV-cache prefill will dominate inference time.",
                prompt_len,
                len(tools) if tools else 0,
            )
        else:
            logger.debug("Prompt length: %d tokens", prompt_len)
        prompt = mx.array(token_ids)

        gen_kwargs = strategy(
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )

        gen_kwargs["prompt_cache"] = self._ensure_prompt_cache()
        if settings.kv_bits is not None:
            gen_kwargs["kv_bits"] = settings.kv_bits
            gen_kwargs["kv_group_size"] = settings.kv_group_size

        raw_gen = mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Stop-aware wrapper with three detection layers (see _observed below)
        # ─────────────────────────────────────────────────────────────────────
        stop_strings = self._stop_strings
        # Rolling buffer to catch stop strings spanning chunk boundaries.
        # We keep only the minimal suffix that could still become a stop marker.
        # This preserves realtime streaming even for short responses.
        max_suffix = min(
            max((len(s) for s in stop_strings), default=0),
            MAX_STOP_LOOKBACK_CHARS,
        )

        def _observed() -> Generator[str, None, None]:
            """
            Three-layer stop detection: native EOS → length limit → text patterns.

            Why three levels? Because different models (and temperatures) fail in
            different ways, and we can't rely on just one mechanism:

            Level 1 — finish_reason == "stop":
                mlx_lm detected an EOS token ID internally and stopped the loop.
                We trim any stop-string text that leaked into the final chunk
                before yielding it, then break.

            Level 2 — finish_reason == "length":
                We hit max_tokens. Yield whatever is buffered and stop cleanly.

            Level 3 — stop string found in rolling text buffer:
                The model emitted its end-of-turn token as plain characters rather
                than as its designated token ID (common with 4-bit quantized models
                at high temperature). We scan the accumulated text and cut off
                when we see a stop string.

            Between stop checks we hold back a small suffix window equal to the
            length of the longest stop string. This prevents us from yielding text
            that might be the start of a stop string that hasn't fully arrived yet.
            """
            suffix = ""  # Accumulates text to detect stop strings
            ended_by_finish_reason = False
            use_token_decode_fallback = False

            def _scan_suffix(text: str) -> tuple[int, int | None]:
                """
                Scan accumulated text for both holdback chars and complete stop matches.

                Returns (holdback_chars, stop_position) where:
                - holdback_chars: trailing chars that might be start of a stop string
                - stop_position: position of rightmost complete stop string match, or None

                This replaces two separate scans (_suffix_holdback_chars + Layer 3 rfind)
                with a single pass over all stop strings.
                """
                if not stop_strings or not text:
                    return 0, None

                holdback = 0
                stop_pos = None

                for stop in stop_strings:
                    # Check for complete stop string match
                    idx = text.rfind(stop)
                    if idx != -1:
                        # Keep the rightmost (maximum) position among all stop strings
                        if stop_pos is None or idx > stop_pos:
                            stop_pos = idx
                    else:
                        # No complete match; check for partial prefix at end (holdback)
                        prefix_cap = min(len(stop) - 1, max_suffix, len(text))
                        for n in range(prefix_cap, 0, -1):
                            if text.endswith(stop[:n]):
                                if n > holdback:
                                    holdback = n
                                break

                return holdback, stop_pos

            def _decode_token_piece(token_value: Any) -> str:
                try:
                    token_id = (
                        int(token_value.item())
                        if hasattr(token_value, "item")
                        else int(token_value)
                    )
                    return self._tokenizer.decode([token_id]) or ""
                except Exception:
                    return ""

            try:
                for token_response in raw_gen:
                    segment = token_response.text or ""

                    if token_response.finish_reason is None:
                        if use_token_decode_fallback:
                            segment = _decode_token_piece(token_response.token)
                        elif not segment:
                            decoded = _decode_token_piece(token_response.token)
                            if decoded:
                                use_token_decode_fallback = True
                                segment = decoded
                    elif use_token_decode_fallback:
                        # Final detokenizer flush may duplicate token-decoded output.
                        segment = ""

                    suffix += segment

                    # Layer 1: MLX detected native EOS token
                    if token_response.finish_reason == "stop":
                        # Trim at the rightmost stop string to preserve any
                        # stop-string-like content legitimately generated earlier.
                        positions = [suffix.rfind(s) for s in stop_strings if s in suffix]
                        clean = suffix[: max(positions)] if positions else suffix
                        if clean:
                            self._notify_token(clean)
                            yield clean
                        ended_by_finish_reason = True
                        break

                    # Layer 2: Hit max_tokens limit
                    if token_response.finish_reason == "length":
                        if suffix:
                            self._notify_token(suffix)
                            yield suffix
                        ended_by_finish_reason = True
                        break

                    # Layer 3 + Normal flow: use single scan for both stop detection and holdback
                    holdback, stop_pos = _scan_suffix(suffix)

                    if stop_pos is not None:
                        # Stop string found in accumulated text
                        clean = suffix[:stop_pos]
                        if clean:
                            self._notify_token(clean)
                            yield clean
                        ended_by_finish_reason = True
                        break

                    # Normal flow: emit everything except the tiny suffix that
                    # could still be the start of a future stop string.
                    safe = suffix[:-holdback] if holdback else suffix
                    if safe:
                        self._notify_token(safe)
                        yield safe
                    suffix = suffix[len(safe) :]

                # Defensive flush: if upstream ends without finish_reason,
                # don't lose buffered text held for stop-string lookback.
                if suffix and not ended_by_finish_reason:
                    self._notify_token(suffix)
                    yield suffix

            except Exception as exc:
                self._notify_error(exc)
                raise

        # Pass on_complete callback directly into TokenStream so it fires reliably
        # on StopIteration regardless of how the stream is consumed (for loop,
        # collect(), next()). This replaces the previous instance-attribute
        # monkey-patch which was silently ignored by Python's iterator protocol.
        def _on_complete() -> None:
            self._notify_complete(
                {
                    "token_count": stream.token_count,
                    "elapsed": stream.elapsed,
                    "tokens_per_second": stream.tokens_per_second,
                }
            )

        stream = TokenStream(_observed(), on_complete=_on_complete)
        stream.prompt_tokens = len(token_ids)
        return stream

    # ── Model management (for UI) ────────────────────────────────────────────

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model's weight files exist on disk. Result is TTL-cached."""
        now = time.monotonic()
        cached = self._download_cache.get(model_id)
        if cached is not None and now < cached[1]:
            return cached[0]

        local_path = _model_local_path(model_id)
        result = bool(list(local_path.glob("*.safetensors")) + list(local_path.glob("*.npz")))
        self._download_cache[model_id] = (result, now + _DOWNLOAD_CACHE_TTL)
        return result

    def delete_model(self, model_id: str) -> None:
        """
        Delete model weights from disk (cannot delete currently loaded model).
        """
        if self._model_id == model_id:
            raise RuntimeError(f"Cannot delete '{model_id}' — it is currently loaded.")
        local_path = _model_local_path(model_id)
        if local_path.exists():
            import shutil

            shutil.rmtree(local_path)
            logger.info("Deleted model weights at '%s'", local_path)
        self._download_cache.pop(model_id, None)
        self._disk_size_cache.pop(model_id, None)
        self._local_ids_cache = None  # Invalidate directory listing cache

    def download_model(self, model_id: str) -> None:
        """Download a model's weights without loading them into memory."""
        self._check_ram(model_id)
        local_path = _model_local_path(model_id)
        _ensure_model_downloaded(model_id, local_path)
        self._download_cache.pop(model_id, None)
        self._disk_size_cache.pop(model_id, None)
        self._local_ids_cache = None  # Invalidate directory listing cache

    def disk_size_gb(self, model_id: str) -> float | None:
        """Return disk size in GB of a downloaded model, or None if not cached."""
        now = time.monotonic()
        cached = self._disk_size_cache.get(model_id)
        if cached is not None and now < cached[1]:
            return cached[0]

        local_path = _model_local_path(model_id)
        if not local_path.exists():
            result = None
        else:
            total = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
            result = round(total / (1024**3), 2)

        self._disk_size_cache[model_id] = (result, now + _DISK_SIZE_CACHE_TTL)
        return result

    def local_model_ids(self) -> list[str]:
        """Scan the models directory and return IDs of all locally downloaded models. Result is TTL-cached."""
        now = time.monotonic()
        if self._local_ids_cache is not None and now < self._local_ids_cache[1]:
            return self._local_ids_cache[0]

        models_dir = Path(settings.models_dir).expanduser().resolve()
        result: list[str] = []
        if not models_dir.exists():
            pass
        else:
            for entry in models_dir.iterdir():
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                # Convert folder name back to HF model ID (org--name -> org/name)
                model_id = entry.name.replace("--", "/", 1)
                # Verify it actually has weight files
                if list(entry.glob("*.safetensors")) or list(entry.glob("*.npz")):
                    result.append(model_id)

        self._local_ids_cache = (result, now + _LOCAL_IDS_CACHE_TTL)
        return result

    # ── Quantization ──────────────────────────────────────────────────────

    def quantization_info(self, model_id: str) -> str | None:
        """Read quantization details from a model's config.json, or None if unquantized."""
        local_path = _model_local_path(model_id)
        config_path = local_path / "config.json"
        if not config_path.exists():
            return None
        try:
            import json as _json

            cfg = _json.loads(config_path.read_text())
            quant = cfg.get("quantization")
            if not quant:
                return None
            bits = quant.get("bits")
            group_size = quant.get("group_size")
            return f"{bits}bit (g={group_size})" if bits else None
        except Exception:
            return None

    def quantize_model(
        self,
        model_id: str,
        *,
        q_bits: int = 4,
        q_group_size: int = 64,
        quant_predicate: str | None = None,
    ) -> str:
        """Quantize a downloaded model and save the result as a new model directory.

        Returns the new model ID (e.g. ``org/model-4bit``).
        """
        source_path = _model_local_path(model_id)
        if not source_path.exists():
            raise ModelLoadError(f"Source model '{model_id}' not found on disk.")

        suffix = f"-{q_bits}bit" if quant_predicate is None else f"-{quant_predicate}"
        new_model_id = f"{model_id}{suffix}"
        output_path = _model_local_path(new_model_id)

        if output_path.exists():
            raise RuntimeError(
                f"Quantized model already exists at '{output_path}'. "
                "Delete it first if you want to re-quantize."
            )

        import mlx_lm

        convert_kwargs: dict[str, Any] = {
            "hf_path": str(source_path),
            "mlx_path": str(output_path),
            "quantize": True,
            "q_bits": q_bits,
            "q_group_size": q_group_size,
        }
        if quant_predicate is not None:
            convert_kwargs["quant_predicate"] = quant_predicate

        logger.info(
            "Quantizing '%s' → '%s' (%d-bit, group_size=%d) …",
            model_id, new_model_id, q_bits, q_group_size,
        )
        mlx_lm.convert(**convert_kwargs)
        logger.info("Quantization complete: %s", output_path)

        self._local_ids_cache = None
        self._download_cache.pop(new_model_id, None)
        self._disk_size_cache.pop(new_model_id, None)
        return new_model_id


# Module-level singleton instance used throughout the app
model_manager = ModelManager()
