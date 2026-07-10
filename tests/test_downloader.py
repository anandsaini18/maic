"""Tests for maic.core.downloader — fast parallel model downloader."""
from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

from maic.core.downloader import (
    _local_dir_for,
    _parallel_download,
    ensure_hf_cache_symlink,
    fast_download,
)

HF_ID = "mlx-community/test-model-4bit"

_original_import = builtins.__import__


def _raise_for_hf_transfer(name: str, *args: object, **kwargs: object) -> object:
    """Custom __import__ that raises ImportError only for hf_transfer."""
    if name == "hf_transfer":
        raise ImportError("hf_transfer not installed")
    return _original_import(name, *args, **kwargs)


# ── ensure_hf_cache_symlink ──────────────────────────────────────────────────


def test_ensure_hf_cache_symlink_creates_link(tmp_path: Path) -> None:
    """Symlink is created in <fake_home>/.cache/huggingface/hub/."""
    fake_home = tmp_path / "home"
    local_dir = fake_home / "models" / "mlx-community--test-model-4bit"
    local_dir.mkdir(parents=True)

    with patch("maic.core.downloader.Path.home", return_value=fake_home):
        ensure_hf_cache_symlink(HF_ID, local_dir)

    link = fake_home / ".cache" / "huggingface" / "hub" / "models--mlx-community--test-model-4bit"
    assert link.is_symlink(), "symlink should exist"
    assert link.resolve() == local_dir.resolve()


def test_ensure_hf_cache_symlink_skips_existing(tmp_path: Path) -> None:
    """No error if symlink already points to the correct target."""
    fake_home = tmp_path / "home"
    local_dir = fake_home / "models" / "mlx-community--test-model-4bit"
    local_dir.mkdir(parents=True)

    cache_dir = fake_home / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True)
    link = cache_dir / "models--mlx-community--test-model-4bit"
    link.symlink_to(local_dir.resolve())

    with patch("maic.core.downloader.Path.home", return_value=fake_home):
        # Should not raise
        ensure_hf_cache_symlink(HF_ID, local_dir)

    assert link.is_symlink()


def test_ensure_hf_cache_symlink_skips_missing_local_dir(tmp_path: Path) -> None:
    """No symlink created when local_dir does not exist yet."""
    fake_home = tmp_path / "home"
    local_dir = fake_home / "models" / "does-not-exist"

    with patch("maic.core.downloader.Path.home", return_value=fake_home):
        ensure_hf_cache_symlink(HF_ID, local_dir)

    link = fake_home / ".cache" / "huggingface" / "hub" / "models--mlx-community--test-model-4bit"
    assert not link.exists()


# ── fast_download strategy selection ────────────────────────────────────────


def test_fast_download_uses_hf_transfer_when_available(tmp_path: Path) -> None:
    """When hf_transfer is importable, fast_download sets the env var and calls snapshot_download."""
    fake_hf_transfer = MagicMock()
    dest = tmp_path / "model"
    dest.mkdir()

    with (
        patch.dict("sys.modules", {"hf_transfer": fake_hf_transfer}),
        patch(
            "maic.core.downloader._hf_transfer_download", return_value=dest
        ) as mock_layer1,
    ):
        result = fast_download(HF_ID, local_dir=tmp_path / "model")

    mock_layer1.assert_called_once_with(HF_ID, tmp_path / "model")
    assert result == dest


def test_fast_download_falls_back_to_parallel(tmp_path: Path) -> None:
    """When hf_transfer is not importable, fast_download calls _parallel_download."""
    dest = tmp_path / "model"
    dest.mkdir()

    # Ensure hf_transfer is absent in sys.modules during the test
    import sys

    sys.modules.pop("hf_transfer", None)

    with (
        patch("builtins.__import__", side_effect=_raise_for_hf_transfer),
        patch(
            "maic.core.downloader._parallel_download", return_value=dest
        ) as mock_layer2,
    ):
        result = fast_download(HF_ID, local_dir=tmp_path / "model", max_workers=2)

    mock_layer2.assert_called_once_with(
        HF_ID, tmp_path / "model", max_workers=2, progress_callback=None
    )
    assert result == dest


# ── _parallel_download internals ────────────────────────────────────────────


def test_parallel_download_calls_hf_hub_download_for_each_file(tmp_path: Path) -> None:
    """_parallel_download fetches every file reported by list_repo_files."""
    files = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    calls_made: list[str] = []

    def fake_hf_hub_download(repo_id: str, filename: str, local_dir: str) -> None:
        calls_made.append(filename)

    with (
        patch("maic.core.downloader.list_repo_files", return_value=iter(files)),
        patch("maic.core.downloader.hf_hub_download", side_effect=fake_hf_hub_download),
    ):
        result = _parallel_download(HF_ID, tmp_path, max_workers=2)

    assert sorted(calls_made) == sorted(files)
    assert result == tmp_path


def test_parallel_download_invokes_progress_callback(tmp_path: Path) -> None:
    """progress_callback is called once per downloaded file with correct args."""
    files = ["a.json", "b.safetensors"]
    progress_calls: list[tuple[int, int, str]] = []

    def fake_hf_hub_download(repo_id: str, filename: str, local_dir: str) -> None:
        pass

    def cb(done: int, total: int, filename: str) -> None:
        progress_calls.append((done, total, filename))

    with (
        patch("maic.core.downloader.list_repo_files", return_value=iter(files)),
        patch("maic.core.downloader.hf_hub_download", side_effect=fake_hf_hub_download),
    ):
        _parallel_download(HF_ID, tmp_path, max_workers=1, progress_callback=cb)

    assert len(progress_calls) == 2
    # The total count reported to the callback should always equal len(files)
    for _done, total, _fname in progress_calls:
        assert total == 2


def test_local_dir_for_replaces_slash() -> None:
    assert _local_dir_for("owner/repo").name == "owner--repo"
