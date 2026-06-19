r"""Model size estimation and curated size data for RAM feasibility checks.

Provides three resolution levels for determining a model's memory footprint:
curated ``KNOWN_SIZES`` (human-verified), regex-based ``estimate_size_gb``
(heuristic from model name), and a 7.0 GiB fallback. Also maintains the
``GATED_MODELS`` frozenset for HuggingFace token requirements.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Curated size overrides ────────────────────────────────────────────────────
# Human-verified sizes (GiB) for models whose names don't encode parameter
# count clearly — e.g. "Phi-3.5" is a version number, not "3.5 billion params".
# These take priority over the regex estimator and serve as fallback data when
# the Hub is unreachable.
KNOWN_SIZES: dict[str, float] = {
    "mlx-community/SmolLM2-1.7B-Instruct-4bit": 1.0,
    "mlx-community/Phi-3.5-mini-instruct-4bit": 2.3,
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": 2.5,
    "mlx-community/gemma-3-4b-it-4bit": 2.6,
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit": 4.0,
    "mlx-community/Qwen3-8B-4bit": 5.0,
    "mlx-community/Llama-3.2-1B-Instruct-4bit": 0.7,
    "mlx-community/Llama-3.2-3B-Instruct-4bit": 1.8,
    "mlx-community/Llama-3.1-8B-Instruct-4bit": 4.9,
    "mlx-community/Llama-3.3-70B-Instruct-4bit": 40.0,
}

# Static safety-net for models that require a HuggingFace token regardless of
# what the API reports. The HF "gated" field is the primary signal for dynamic
# discovery; this frozenset ensures known gated models are never served without
# warning even when the API response is inconsistent.
GATED_MODELS: frozenset[str] = frozenset(
    {
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/Llama-3.3-70B-Instruct-4bit",
    }
)

# Pre-sorted (smallest → largest) for the /v1/models/supported endpoint and
# for presenting feasible alternatives in RAM-overflow error messages.
KNOWN_SIZES_BY_RAM: list[tuple[str, float]] = sorted(KNOWN_SIZES.items(), key=lambda x: x[1])


# ── Size estimation ───────────────────────────────────────────────────────────


def estimate_size_gb(model_id: str) -> float | None:
    """Estimate model size in GiB from its name using heuristics.

    Returns None when the name contains no recognisable parameter count so
    that callers can fall back to a curated value rather than silently using a
    wrong default.

    Formula: parameters × bytes-per-parameter + 0.25 GB overhead (config files,
    tokenizer, etc.).

    The parameter regex matches "<N>b" where "b" is followed by a separator or
    end-of-string. This excludes "bit" (b followed by "i") and "bf16" (b
    followed by "f") while correctly matching "7b-instruct", "4b-it", "1.7b".
    The input is lowercased first, so [Bb] is never needed.
    """
    name = model_id.lower()

    param_match = re.search(r"(\d+(?:\.\d+)?)b(?=[-_./\d\s]|$)", name)
    if not param_match:
        return None
    params_b = float(param_match.group(1))

    if re.search(r"2[\-_]?bit", name):
        bits = 2
    elif re.search(r"3[\-_]?bit", name):
        bits = 3
    elif re.search(r"4[\-_]?bit|q4", name):
        bits = 4
    elif re.search(r"6[\-_]?bit", name):
        bits = 6
    elif re.search(r"8[\-_]?bit|q8", name):
        bits = 8
    elif re.search(r"bf16|fp16|float16", name):
        bits = 16
    else:
        bits = 4  # mlx-community models are predominantly 4-bit

    size_gb = (params_b * 1e9 * (bits / 8)) / (1024**3)
    return round(size_gb + 0.25, 1)


def resolve_size_gb(model_id: str) -> float:
    """Best-effort size in GiB for any model ID.

    Resolution order (highest confidence first):
      1. KNOWN_SIZES — human-curated, precise.
      2. estimate_size_gb — regex heuristic, accurate for most mlx-community
         naming conventions.
      3. 7.0 GB fallback — a reasonable mid-range default when both methods
         fail (roughly equivalent to a 4-bit 7B model).

    This function always returns a value so callers never receive None, making
    feasibility comparisons unconditionally safe.
    """
    known = KNOWN_SIZES.get(model_id)
    if known is not None:
        return known

    estimated = estimate_size_gb(model_id)
    if estimated is not None:
        return estimated

    logger.debug("Unknown size for '%s'; defaulting to 7.0 GB.", model_id)
    return 7.0
