from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


def _models_dir() -> Path:
    """Return ~/models/, creating it if needed."""
    d = Path.home() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _local_dir_for(hf_id: str) -> Path:
    """Return the ~/models/<owner>--<name> path for a given HF repo ID."""
    return _models_dir() / hf_id.replace("/", "--")


# ── Layer 1: hf-transfer (Rust, fastest) ────────────────────────────────────


def _hf_transfer_download(hf_id: str, local_dir: Path) -> Path:
    """Download via snapshot_download with HF_HUB_ENABLE_HF_TRANSFER=1.

    hf-transfer is HuggingFace's Rust-based parallel downloader that integrates
    transparently into snapshot_download via an env var toggle.
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    local_dir.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(hf_id, local_dir=str(local_dir))
    return Path(result)


# ── Layer 2: ThreadPoolExecutor fallback ────────────────────────────────────


def _parallel_download(
    hf_id: str,
    local_dir: Path,
    max_workers: int = 4,
    progress_callback: object = None,
) -> Path:
    """Download all files in a HuggingFace repo concurrently.

    Time complexity: O(max_file_size / bandwidth) — bounded by largest file, not sum.
    Space complexity: O(max_workers × chunk_size) active buffers at any time.
    """
    files = list(list_repo_files(hf_id))
    local_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    completed = 0

    def _download_one(filename: str) -> str:
        nonlocal completed
        hf_hub_download(
            repo_id=hf_id,
            filename=filename,
            local_dir=str(local_dir),
        )
        with lock:
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(files), filename)  # type: ignore[operator]
        return filename

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, f): f for f in files}
        for future in as_completed(futures):
            future.result()  # re-raise any download exception

    return local_dir


# ── Public API ───────────────────────────────────────────────────────────────


def fast_download(
    hf_id: str,
    local_dir: Path | None = None,
    max_workers: int = 4,
    progress_callback: object = None,
) -> Path:
    """Download a HuggingFace repo as fast as possible.

    Strategy (tried in order):
    1. hf-transfer (Rust, ~5-10× speedup) — used when the package is importable.
    2. ThreadPoolExecutor over hf_hub_download (pure-Python parallel fallback).

    Args:
        hf_id: HuggingFace repo ID, e.g. ``"mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"``.
        local_dir: Destination directory.  Defaults to ``~/models/<owner>--<name>``.
        max_workers: Thread count for the fallback path (ignored for hf-transfer).
        progress_callback: ``(completed, total, filename) -> None`` called after each
            file finishes on the fallback path.  Not called for hf-transfer.

    Returns:
        Path to the downloaded model directory.
    """
    if local_dir is None:
        local_dir = _local_dir_for(hf_id)

    try:
        import hf_transfer  # noqa: F401

        return _hf_transfer_download(hf_id, local_dir)
    except ImportError:
        return _parallel_download(
            hf_id,
            local_dir,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )


# ── HF cache symlink ─────────────────────────────────────────────────────────


def ensure_hf_cache_symlink(hf_id: str, local_dir: Path) -> None:
    """Create a symlink in ~/.cache/huggingface/hub/ pointing to local_dir.

    This lets mlx_lm.server find the model by HF ID without re-downloading.
    Skips silently if the link already exists or if local_dir does not exist.
    """
    if not local_dir.exists():
        return

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    link_name = "models--" + hf_id.replace("/", "--")
    link_path = cache_dir / link_name

    if not link_path.exists():
        link_path.symlink_to(local_dir.resolve())
