from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class CuratedModel(TypedDict):
    alias: str
    hf_id: str
    family: str
    size_gb: float
    min_ram_gb: int
    tool_calling: bool
    description: str


CURATED_MODELS: list[CuratedModel] = [
    {
        "alias": "phi-mini",
        "hf_id": "mlx-community/Phi-3.5-mini-instruct-4bit",
        "family": "none",
        "size_gb": 2.3,
        "min_ram_gb": 6,
        "tool_calling": False,
        "description": "Smallest model. Chat-only. Best for 8GB Macs.",
    },
    {
        "alias": "qwen-4b",
        "hf_id": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "family": "qwen",
        "size_gb": 2.5,
        "min_ram_gb": 6,
        "tool_calling": True,
        "description": "Smallest tool-capable model.",
    },
    {
        "alias": "mistral-7b",
        "hf_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "family": "mistral",
        "size_gb": 4.0,
        "min_ram_gb": 8,
        "tool_calling": True,
        "description": "Stable tool calling. Great perf.",
    },
    {
        "alias": "qwen-coder-7b",
        "hf_id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "family": "qwen",
        "size_gb": 4.3,
        "min_ram_gb": 8,
        "tool_calling": True,
        "description": "Best coding model for 16GB Macs. Recommended for OpenCode.",
    },
    {
        "alias": "deepseek-r1-14b",
        "hf_id": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "family": "qwen",
        "size_gb": 8.5,
        "min_ram_gb": 14,
        "tool_calling": True,
        "description": "DeepSeek R1 reasoning + coding.",
    },
    {
        "alias": "qwen-8b",
        "hf_id": "mlx-community/Qwen3-8B-4bit",
        "family": "qwen",
        "size_gb": 5.0,
        "min_ram_gb": 8,
        "tool_calling": True,
        "description": "Qwen3 8B with thinking mode.",
    },
    {
        "alias": "deepseek-coder",
        "hf_id": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
        "family": "deepseek",
        "size_gb": 9.0,
        "min_ram_gb": 14,
        "tool_calling": True,
        "description": "Best code completion on 16GB Macs.",
    },
    {
        "alias": "qwen-14b",
        "hf_id": "mlx-community/Qwen3-14B-4bit",
        "family": "qwen",
        "size_gb": 8.5,
        "min_ram_gb": 16,
        "tool_calling": True,
        "description": "Qwen3 14B. Strong reasoning for 32GB Macs.",
    },
]


def resolve_hf_id(alias_or_id: str) -> str:
    """Return the HuggingFace repo ID for a curated alias, or pass through as-is."""
    for m in CURATED_MODELS:
        if m["alias"] == alias_or_id:
            return str(m["hf_id"])
    return alias_or_id


def list_installed_models() -> list[str]:
    """Return HF model IDs of all locally downloaded models in ~/models/."""
    models_dir = Path.home() / "models"
    if not models_dir.exists():
        return []
    result = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        has_weights = any(d.glob("*.safetensors")) or (d / "config.json").exists()
        if has_weights:
            hf_id = d.name.replace("--", "/", 1)
            result.append(hf_id)
    return result
