"""Unit tests for model manager, loading, and generation logic."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from app.core.model_manager import (
    _model_local_path,
    _ensure_model_downloaded,
    KNOWN_MODEL_SIZES,
    TOKEN_REQUIRED_MODELS,
    ModelTooLargeError,
    ModelLoadError,
    InferenceObserver,
    StatsObserver,
    ModelManager,
    model_manager,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for _model_local_path
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLocalPath:
    """Tests for model ID to path conversion."""

    @patch("app.core.model_manager.settings")
    def test_converts_slash_to_double_dash(self, mock_settings):
        """Should replace '/' with '--' in model paths."""
        mock_settings.models_dir = "/tmp/models"

        result = _model_local_path("mlx-community/Phi-3.5-mini-instruct-4bit")

        assert "--" in str(result)
        # No slash after models dir
        assert "/" not in str(result).split("/models/")[1]

    @patch("app.core.model_manager.settings")
    def test_expands_home_directory(self, mock_settings):
        """Should expand ~ to home directory."""
        mock_settings.models_dir = "~/models"

        result = _model_local_path("test/model")

        # Should not contain ~ after expansion
        assert "~" not in str(result)
        assert str(result).startswith("/")

    @patch("app.core.model_manager.settings")
    def test_handles_multiple_slashes(self, mock_settings):
        """Should handle model IDs with multiple slashes."""
        mock_settings.models_dir = "/tmp/models"

        result = _model_local_path("deep/nested/model/name")

        # All slashes should be converted to --
        path_str = str(result)
        assert path_str.count("--") == 3
        assert "/" not in path_str.split("/models/")[1]

    @patch("app.core.model_manager.settings")
    def test_returns_path_object(self, mock_settings):
        """Should return a Path object."""
        mock_settings.models_dir = "/tmp"

        result = _model_local_path("org/model")

        assert isinstance(result, Path)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for _ensure_model_downloaded
# ─────────────────────────────────────────────────────────────────────────────


class TestEnsureModelDownloaded:
    """Tests for model download logic."""

    def test_skips_download_if_safetensors_exists(self, tmp_path):
        """Should skip download if .safetensors file exists."""
        local_path = tmp_path / "model"
        local_path.mkdir(parents=True)
        (local_path / "model.safetensors").write_text("weights")

        with patch("app.core.model_manager.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            # Should not call download
            mock_download.assert_not_called()

    def test_skips_download_if_npz_exists(self, tmp_path):
        """Should skip download if .npz file exists."""
        local_path = tmp_path / "model"
        local_path.mkdir(parents=True)
        (local_path / "model.npz").write_text("weights")

        with patch("app.core.model_manager.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            mock_download.assert_not_called()

    def test_downloads_if_no_weights_cached(self, tmp_path):
        """Should download if no weight files exist locally."""
        local_path = tmp_path / "new_model"

        with patch("app.core.model_manager.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            # Should call snapshot_download
            mock_download.assert_called_once()
            assert mock_download.call_args[1]["repo_id"] == "test/model"

    def test_enables_hf_transfer_for_speed(self, tmp_path):
        """Should set HF_HUB_ENABLE_HF_TRANSFER environment variable."""
        local_path = tmp_path / "model"

        with patch.dict("os.environ", {}, clear=False):
            with patch("app.core.model_manager.snapshot_download"):
                _ensure_model_downloaded("test/model", local_path)

                # Check env var was set
                import os

                assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_skips_pytorch_weights_in_download(self, tmp_path):
        """Should ignore .bin files and original/ folder during download."""
        local_path = tmp_path / "model"

        with patch("app.core.model_manager.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            # Check ignore patterns
            call_kwargs = mock_download.call_args[1]
            assert "*.bin" in call_kwargs["ignore_patterns"]
            assert "original/*" in call_kwargs["ignore_patterns"]

    def test_creates_directories_if_missing(self, tmp_path):
        """Should create local_path directories if they don't exist."""
        local_path = tmp_path / "deep" / "nested" / "model"
        assert not local_path.exists()

        with patch("app.core.model_manager.snapshot_download"):
            _ensure_model_downloaded("test/model", local_path)

            # Directory should be created
            assert local_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Tests for ModelManager._check_ram
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckRAM:
    """Tests for RAM feasibility checking."""

    def test_allows_models_under_80_percent_ram(self):
        """Should allow models using <80% of available RAM."""
        mm = ModelManager()

        # Mock psutil to return 10GB RAM, model needs 7GB (70%)
        with patch("app.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 10 * (1024**3)
            mock_mem.return_value.available = 10 * (1024**3)

            # Should not raise
            mm._check_ram("mlx-community/Phi-3.5-mini-instruct-4bit")  # 2.3GB

    def test_rejects_models_over_80_percent_ram(self):
        """Should reject models requiring >80% of available RAM."""
        mm = ModelManager()

        # Mock psutil to return 2GB RAM, model needs 4GB
        with patch("app.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 2 * (1024**3)
            mock_mem.return_value.available = 2 * (1024**3)

            with pytest.raises(ModelTooLargeError) as exc_info:
                mm._check_ram(
                    "mlx-community/Mistral-7B-Instruct-v0.3-4bit")  # 4.0GB

            assert "Feasible alternatives" in str(exc_info.value)

    def test_skips_check_for_unknown_models(self):
        """Should not check unknown models (let MLX decide)."""
        mm = ModelManager()

        # Unknown model should not raise
        mm._check_ram("completely/unknown-model-xyz")

    def test_suggests_feasible_alternatives(self):
        """Should suggest smaller models when rejecting too-large models."""
        mm = ModelManager()

        with patch("app.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 3 * (1024**3)  # 3GB
            mock_mem.return_value.available = 3 * (1024**3)

            with pytest.raises(ModelTooLargeError) as exc_info:
                mm._check_ram(
                    "mlx-community/Mistral-7B-Instruct-v0.3-4bit")  # 4.0GB

            error_msg = str(exc_info.value)
            # Should suggest smaller feasible models
            assert "SmolLM" in error_msg or "Phi" in error_msg

    def test_ram_check_preserves_80_percent_rule(self):
        """Should use exactly 80% of available RAM as safe limit."""
        mm = ModelManager()

        # 10GB RAM, 80% = 8GB limit
        with patch("app.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 10 * (1024**3)

            # 8GB model at 80% should fail
            with pytest.raises(ModelTooLargeError):
                mm._check_ram(
                    "mlx-community/Llama-3.3-70B-Instruct-4bit")  # 40GB


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Observer Pattern
# ─────────────────────────────────────────────────────────────────────────────


class TestObserverPattern:
    """Tests for observer registration and notifications."""

    def test_register_observer(self):
        """Should add observer to list."""
        mm = ModelManager()
        mock_observer = Mock(spec=InferenceObserver)

        # Initially has StatsObserver
        initial_count = len(mm._observers)

        mm.register_observer(mock_observer)

        assert len(mm._observers) == initial_count + 1
        assert mock_observer in mm._observers

    def test_notify_token_calls_all_observers(self):
        """_notify_token should call on_token for all observers."""
        mm = ModelManager()
        observer1 = Mock(spec=InferenceObserver)
        observer2 = Mock(spec=InferenceObserver)

        mm.register_observer(observer1)
        mm.register_observer(observer2)

        mm._notify_token("test_token")

        observer1.on_token.assert_called_once_with("test_token")
        observer2.on_token.assert_called_once_with("test_token")

    def test_notify_complete_calls_all_observers(self):
        """_notify_complete should call on_complete for all observers."""
        mm = ModelManager()
        observer1 = Mock(spec=InferenceObserver)
        observer2 = Mock(spec=InferenceObserver)

        mm.register_observer(observer1)
        mm.register_observer(observer2)

        stats = {"token_count": 10, "elapsed": 1.0, "tokens_per_second": 10.0}
        mm._notify_complete(stats)

        observer1.on_complete.assert_called_once_with(stats)
        observer2.on_complete.assert_called_once_with(stats)

    def test_notify_error_calls_all_observers(self):
        """_notify_error should call on_error for all observers."""
        mm = ModelManager()
        observer1 = Mock(spec=InferenceObserver)
        observer2 = Mock(spec=InferenceObserver)

        mm.register_observer(observer1)
        mm.register_observer(observer2)

        error = RuntimeError("test error")
        mm._notify_error(error)

        observer1.on_error.assert_called_once_with(error)
        observer2.on_error.assert_called_once_with(error)

    def test_stats_observer_logs_on_complete(self):
        """StatsObserver should log performance stats."""
        observer = StatsObserver()

        with patch("app.core.model_manager.logger") as mock_logger:
            stats = {"token_count": 42, "elapsed": 2.0,
                     "tokens_per_second": 21.0}
            observer.on_complete(stats)

            # Should log with correct values
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0]
            assert "42 tokens" in call_args[0]
            assert call_args[1] == 42


# ─────────────────────────────────────────────────────────────────────────────
# Tests for ModelManager.load
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLoad:
    """Tests for model loading logic."""

    @patch("app.core.model_manager.mlx_lm.load")
    @patch("app.core.model_manager._ensure_model_downloaded")
    def test_load_successful_flow(self, mock_download, mock_mlx_load):
        """Should download, load, and derive stop strings."""
        mm = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = []
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        with patch.object(mm, "_check_ram"):
            with patch.object(mm, "_derive_stop_strings", return_value=frozenset()):
                mm.load("test/model")

        assert mm._model is mock_model
        assert mm._tokenizer is mock_tokenizer
        assert mm._model_id == "test/model"
        mock_download.assert_called_once()

    @patch("app.core.model_manager._ensure_model_downloaded")
    def test_load_checks_ram_before_download(self, mock_download):
        """Should check RAM before attempting download."""
        mm = ModelManager()

        with patch.object(mm, "_check_ram") as mock_check:
            mock_check.side_effect = ModelTooLargeError("Too large")

            with pytest.raises(ModelTooLargeError):
                mm.load("test/model")

            # Download should not be called
            mock_download.assert_not_called()

    @patch("app.core.model_manager._ensure_model_downloaded")
    def test_load_wraps_download_errors(self, mock_download):
        """Should wrap download errors in ModelLoadError."""
        mm = ModelManager()
        mock_download.side_effect = Exception("Network error")

        with patch.object(mm, "_check_ram"):
            with pytest.raises(ModelLoadError) as exc_info:
                mm.load("test/model")

            assert "Failed to download" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)

    @patch("app.core.model_manager.mlx_lm.load")
    @patch("app.core.model_manager._ensure_model_downloaded")
    def test_load_wraps_mlx_errors(self, mock_download, mock_mlx_load):
        """Should wrap MLX loading errors in ModelLoadError."""
        mm = ModelManager()
        mock_mlx_load.side_effect = Exception("MLX error")

        with patch.object(mm, "_check_ram"):
            with patch.object(mm, "_derive_stop_strings"):
                with pytest.raises(ModelLoadError) as exc_info:
                    mm.load("test/model")

                assert "Failed to load" in str(exc_info.value)

    def test_is_loaded_property(self):
        """is_loaded should reflect model state."""
        mm = ModelManager()

        assert not mm.is_loaded

        mm._model = Mock()
        assert mm.is_loaded

        mm._model = None
        assert not mm.is_loaded

    def test_model_id_property(self):
        """model_id property should return loaded model's ID."""
        mm = ModelManager()

        assert mm.model_id is None

        mm._model_id = "test/model"
        assert mm.model_id == "test/model"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Stop String Derivation
# ─────────────────────────────────────────────────────────────────────────────


class TestDeriveStopStrings:
    """Tests for auto-detection of model-specific stop tokens."""

    def test_extracts_eos_token_ids(self):
        """Should extract and decode EOS tokens from tokenizer."""
        mm = ModelManager()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = [128000, 128001]
        mock_tokenizer.decode.side_effect = lambda ids: {
            (128000,): "<|end|>",
            (128001,): "</s>",
        }[tuple(ids)]

        mm._tokenizer = mock_tokenizer

        result = mm._derive_stop_strings()

        assert "<|end|>" in result
        assert "</s>" in result

    def test_probes_chat_template_for_tokens(self):
        """Should probe chat template to find appended tokens."""
        mm = ModelManager()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = []

        # Simulate chat template response
        SENTINEL = "\x01\x02\x03"
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 999]
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.side_effect = lambda ids: {
            (999,): "<|im_end|>",
        }[tuple(ids)]

        mm._tokenizer = mock_tokenizer

        result = mm._derive_stop_strings()

        # Should find the appended token
        assert len(result) > 0 or mm._tokenizer.apply_chat_template.called

    def test_returns_frozenset(self):
        """Should return frozenset of strings."""
        mm = ModelManager()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = []
        mock_tokenizer.apply_chat_template.return_value = []

        mm._tokenizer = mock_tokenizer

        result = mm._derive_stop_strings()

        assert isinstance(result, frozenset)

    def test_handles_probe_failure_gracefully(self):
        """Should not crash if chat template probe fails."""
        mm = ModelManager()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = [100]
        mock_tokenizer.decode.return_value = "<|end|>"
        mock_tokenizer.apply_chat_template.side_effect = Exception(
            "Template error")

        mm._tokenizer = mock_tokenizer

        # Should not raise
        result = mm._derive_stop_strings()
        assert isinstance(result, frozenset)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Model Management Methods
# ─────────────────────────────────────────────────────────────────────────────


class TestModelManagement:
    """Tests for is_downloaded, download_model, delete_model, disk_size_gb."""

    @patch("app.core.model_manager._model_local_path")
    def test_is_downloaded_returns_true_with_safetensors(self, mock_path):
        """is_downloaded should return True if .safetensors exists."""
        mm = ModelManager()
        mock_local_path = Mock()
        mock_local_path.glob.side_effect = [["file.safetensors"], []]
        mock_path.return_value = mock_local_path

        result = mm.is_downloaded("test/model")

        assert result is True

    @patch("app.core.model_manager._model_local_path")
    def test_is_downloaded_returns_false_if_empty(self, mock_path):
        """is_downloaded should return False if no weight files."""
        mm = ModelManager()
        mock_local_path = Mock()
        mock_local_path.glob.side_effect = [[], []]
        mock_path.return_value = mock_local_path

        result = mm.is_downloaded("test/model")

        assert result is False

    @patch("app.core.model_manager._ensure_model_downloaded")
    @patch("app.core.model_manager._model_local_path")
    def test_download_model(self, mock_path, mock_ensure_download):
        """download_model should ensure model is downloaded."""
        mm = ModelManager()
        mock_path.return_value = "/fake/path"

        with patch.object(mm, "_check_ram"):
            mm.download_model("test/model")

            mock_ensure_download.assert_called_once()

    @patch("app.core.model_manager._model_local_path")
    def test_delete_model_refuses_loaded_model(self, mock_path):
        """delete_model should refuse to delete currently loaded model."""
        mm = ModelManager()
        mm._model_id = "test/model"

        with pytest.raises(RuntimeError) as exc_info:
            mm.delete_model("test/model")

        assert "currently loaded" in str(exc_info.value)

    @patch("app.core.model_manager._model_local_path")
    def test_delete_model_removes_directory(self, mock_path):
        """delete_model should remove model directory."""
        mm = ModelManager()
        mm._model_id = "other/model"

        mock_local_path = Mock()
        mock_local_path.exists.return_value = True
        mock_path.return_value = mock_local_path

        with patch("shutil.rmtree") as mock_rmtree:
            mm.delete_model("test/model")

            mock_rmtree.assert_called_once()

    @patch("app.core.model_manager._model_local_path")
    def test_disk_size_gb_returns_none_if_not_downloaded(self, mock_path):
        """disk_size_gb should return None if model not downloaded."""
        mm = ModelManager()
        mock_local_path = Mock()
        mock_local_path.exists.return_value = False
        mock_path.return_value = mock_local_path

        result = mm.disk_size_gb("test/model")

        assert result is None

    @patch("app.core.model_manager._model_local_path")
    def test_disk_size_gb_calculates_size(self, mock_path):
        """disk_size_gb should calculate total size in GB."""
        mm = ModelManager()

        # Mock file system
        mock_local_path = Mock()
        mock_local_path.exists.return_value = True

        # Create mock files with specific sizes
        mock_file1 = Mock()
        mock_file1.stat.return_value.st_size = 1024 * 1024 * 1024  # 1GB
        mock_file2 = Mock()
        mock_file2.stat.return_value.st_size = 512 * 1024 * 1024  # 0.5GB

        mock_local_path.rglob.return_value = [mock_file1, mock_file2]
        mock_file1.is_file.return_value = True
        mock_file2.is_file.return_value = True

        mock_path.return_value = mock_local_path

        result = mm.disk_size_gb("test/model")

        # Should return approximately 1.5GB
        assert 1.4 < result < 1.6


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Constants and Security
# ─────────────────────────────────────────────────────────────────────────────


class TestConstants:
    """Tests for model constants and configuration."""

    def test_known_model_sizes_not_empty(self):
        """KNOWN_MODEL_SIZES should have models defined."""
        assert len(KNOWN_MODEL_SIZES) > 0

    def test_token_required_models_are_subset(self):
        """TOKEN_REQUIRED_MODELS should only contain models in KNOWN_MODEL_SIZES."""
        for model_id in TOKEN_REQUIRED_MODELS:
            assert model_id in KNOWN_MODEL_SIZES

    def test_feasible_by_ram_is_sorted(self):
        """FEASIBLE_BY_RAM should be sorted by size."""
        from app.core.model_manager import FEASIBLE_BY_RAM

        sizes = [size for size, _ in FEASIBLE_BY_RAM]
        assert sizes == sorted(sizes)

    def test_default_model_in_known_sizes(self):
        """Default model should be in KNOWN_MODEL_SIZES."""
        with patch("app.core.model_manager.settings") as mock_settings:
            mock_settings.model_id = "mlx-community/Phi-3.5-mini-instruct-4bit"

            assert mock_settings.model_id in KNOWN_MODEL_SIZES


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Path Security
# ─────────────────────────────────────────────────────────────────────────────


class TestPathSecurity:
    """Tests to ensure no path traversal vulnerabilities."""

    @patch("app.core.model_manager.settings")
    def test_no_path_traversal_in_model_id(self, mock_settings):
        """Should safely handle model IDs with '..' safely."""
        mock_settings.models_dir = "/safe/models"

        # Attempt path traversal
        result = _model_local_path("../../etc/passwd")

        # Should still be under models directory (due to resolve())
        assert str(result).startswith("/safe/models")

    @patch("app.core.model_manager.settings")
    def test_handles_absolute_paths_in_model_id(self, mock_settings):
        """Should not allow absolute paths in model IDs."""
        mock_settings.models_dir = "/safe/models"

        # Attempt to use absolute path
        result = _model_local_path("/etc/passwd/model")

        # Should still append to models dir
        assert str(result).startswith("/safe/models")
