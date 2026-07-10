from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from maic.cli import app

runner = CliRunner()


def test_models_list_runs() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0


def test_setup_opencode_creates_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "opencode.json"
    monkeypatch.setattr("maic.cli._opencode_config_path", lambda: config_file)

    result = runner.invoke(app, ["setup", "opencode"])
    assert result.exit_code == 0
    assert config_file.exists()
    data = json.loads(config_file.read_text())
    assert "provider" in data
    assert "maic" in data["provider"]
    assert "baseURL" in data["provider"]["maic"]["options"]


def test_setup_opencode_preserves_existing_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "opencode.json"
    existing = {"provider": {"other": {"npm": "other-pkg", "options": {}}}}
    config_file.write_text(json.dumps(existing))
    monkeypatch.setattr("maic.cli._opencode_config_path", lambda: config_file)

    result = runner.invoke(app, ["setup", "opencode"])
    assert result.exit_code == 0
    data = json.loads(config_file.read_text())
    assert "other" in data["provider"]
    assert "maic" in data["provider"]


def test_setup_opencode_handles_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "opencode.json"
    config_file.write_text("{bad json{{")
    monkeypatch.setattr("maic.cli._opencode_config_path", lambda: config_file)

    result = runner.invoke(app, ["setup", "opencode"])
    assert result.exit_code == 0
    # Backup should have been created
    backup = config_file.with_suffix(".json.bak")
    assert backup.exists()
    # New config should be valid
    data = json.loads(config_file.read_text())
    assert "maic" in data["provider"]


def test_doctor_runs() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_models_list_shows_installed_indicator(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx-community--Qwen2.5-Coder-7B-Instruct-4bit"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    with patch("maic.cli.list_installed_models", return_value=["mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"]):
        result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "↓" in result.output


def test_models_list_no_installed_no_indicator() -> None:
    with patch("maic.cli.list_installed_models", return_value=[]):
        result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    # legend line only appears when something is installed
    assert "↓ = installed" not in result.output


def test_doctor_shows_models_installed_count() -> None:
    with patch("maic.cli.list_installed_models", return_value=["mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"]):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Models installed" in result.output
    assert "1" in result.output


def test_doctor_shows_no_models_hint() -> None:
    with patch("maic.cli.list_installed_models", return_value=[]):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Models installed" in result.output


def test_run_no_models_exits_with_hint() -> None:
    with patch("maic.cli.list_installed_models", return_value=[]):
        result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    assert "No models installed" in result.output
    assert "maic models pull" in result.output


def _patch_run_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all three server-launch strategies so run() returns immediately in tests."""
    import sys
    import types

    monkeypatch.setattr("shutil.which", lambda _: None)
    # Make mlx_lm importable but block exec/subprocess so no real server starts.
    fake_mlx_lm = types.ModuleType("mlx_lm")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr("maic.cli.os.execvp", lambda *_a, **_kw: None)
    monkeypatch.setattr("subprocess.run", lambda *_a, **_kw: None)


def test_run_single_model_auto_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_strategies(monkeypatch)
    with (
        patch("maic.cli.list_installed_models", return_value=["mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"]),
        patch("uvicorn.run"),
    ):
        result = runner.invoke(app, ["run"])
    assert "Auto-selected" in result.output
    assert "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" in result.output


def test_run_with_explicit_model_skips_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_strategies(monkeypatch)
    with (
        patch("maic.cli.list_installed_models") as mock_list,
        patch("uvicorn.run"),
    ):
        result = runner.invoke(app, ["run", "--model", "qwen-coder-7b"])
    mock_list.assert_not_called()
    assert result.exit_code == 0


# ── list_installed_models unit tests ─────────────────────────────────────────


def test_list_installed_models_empty_dir(tmp_path: Path) -> None:
    from maic.core.model_registry import list_installed_models

    with patch("maic.core.model_registry.Path.home", return_value=tmp_path):
        result = list_installed_models()
    assert result == []


def test_list_installed_models_detects_safetensors(tmp_path: Path) -> None:
    from maic.core.model_registry import list_installed_models

    models_dir = tmp_path / "models"
    model_dir = models_dir / "mlx-community--Qwen2.5-Coder-7B-Instruct-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"")

    with patch("maic.core.model_registry.Path.home", return_value=tmp_path):
        result = list_installed_models()
    assert result == ["mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"]


def test_list_installed_models_detects_config_json(tmp_path: Path) -> None:
    from maic.core.model_registry import list_installed_models

    models_dir = tmp_path / "models"
    model_dir = models_dir / "mlx-community--Phi-3.5-mini-instruct-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    with patch("maic.core.model_registry.Path.home", return_value=tmp_path):
        result = list_installed_models()
    assert result == ["mlx-community/Phi-3.5-mini-instruct-4bit"]


def test_list_installed_models_ignores_empty_dirs(tmp_path: Path) -> None:
    from maic.core.model_registry import list_installed_models

    models_dir = tmp_path / "models"
    empty_dir = models_dir / "mlx-community--some-model"
    empty_dir.mkdir(parents=True)

    with patch("maic.core.model_registry.Path.home", return_value=tmp_path):
        result = list_installed_models()
    assert result == []


def test_list_installed_models_missing_dir(tmp_path: Path) -> None:
    from maic.core.model_registry import list_installed_models

    with patch("maic.core.model_registry.Path.home", return_value=tmp_path):
        # tmp_path/models doesn't exist
        result = list_installed_models()
    assert result == []


# ── adapter / schema tests ────────────────────────────────────────────────────


def test_tui_module_importable() -> None:
    """TUI module imports without error."""
    from maic import tui  # noqa: F401

    assert hasattr(tui, "run_tui")


def test_models_to_dicts_passes_tool_call_id() -> None:
    from maic.adapters.openai_adapter import OpenAIAdapter
    from maic.schemas.openai import Message

    messages = [
        Message(
            role="tool",
            content="result",
            tool_call_id="call_abc123",
            name="my_tool",
        )
    ]
    result = OpenAIAdapter.messages_to_dicts(messages)
    assert len(result) == 1
    assert result[0]["tool_call_id"] == "call_abc123"
    assert result[0]["name"] == "my_tool"
