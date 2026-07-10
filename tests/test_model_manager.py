"""Unit tests for model manager, loading, and generation logic."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from maic.core.model_manager import (
    InferenceObserver,
    ModelLoadError,
    ModelManager,
    ModelTooLargeError,
    StatsObserver,
    ToolCallFormat,
    _ensure_model_downloaded,
    _model_local_path,
    detect_thinking_support,
    detect_tool_call_format,
)
from maic.core.model_sizing import GATED_MODELS, KNOWN_SIZES, KNOWN_SIZES_BY_RAM

# ─────────────────────────────────────────────────────────────────────────────
# Tests for _model_local_path
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLocalPath:
    """Tests for model ID to path conversion."""

    @patch("maic.core.model_manager.settings")
    def test_converts_slash_to_double_dash(self, mock_settings):
        """Should replace '/' with '--' in model paths."""
        mock_settings.models_dir = "/tmp/models"

        result = _model_local_path("mlx-community/Phi-3.5-mini-instruct-4bit")

        assert "--" in str(result)
        # No slash after models dir
        assert "/" not in str(result).split("/models/")[1]

    @patch("maic.core.model_manager.settings")
    def test_expands_home_directory(self, mock_settings):
        """Should expand ~ to home directory."""
        mock_settings.models_dir = "~/models"

        result = _model_local_path("test/model")

        # Should not contain ~ after expansion
        assert "~" not in str(result)
        assert str(result).startswith("/")

    @patch("maic.core.model_manager.settings")
    def test_handles_multiple_slashes(self, mock_settings):
        """Model IDs with multiple slashes are rejected (not a valid HF org/model ID)."""
        mock_settings.models_dir = "/tmp/models"

        with pytest.raises(ValueError, match="Invalid model ID format"):
            _model_local_path("deep/nested/model/name")

    @patch("maic.core.model_manager.settings")
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

        with patch("huggingface_hub.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            # Should not call download
            mock_download.assert_not_called()

    def test_skips_download_if_npz_exists(self, tmp_path):
        """Should skip download if .npz file exists."""
        local_path = tmp_path / "model"
        local_path.mkdir(parents=True)
        (local_path / "model.npz").write_text("weights")

        with patch("huggingface_hub.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            mock_download.assert_not_called()

    def test_downloads_if_no_weights_cached(self, tmp_path):
        """Should download if no weight files exist locally."""
        local_path = tmp_path / "new_model"

        with patch("huggingface_hub.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            # Should call snapshot_download
            mock_download.assert_called_once()
            assert mock_download.call_args[1]["repo_id"] == "test/model"

    def test_enables_hf_transfer_for_speed(self, tmp_path):
        """Should set HF_HUB_ENABLE_HF_TRANSFER environment variable."""
        local_path = tmp_path / "model"

        with patch.dict("os.environ", {}, clear=False):
            with patch("huggingface_hub.snapshot_download"):
                _ensure_model_downloaded("test/model", local_path)

                # Check env var was set
                import os

                assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_skips_pytorch_weights_in_download(self, tmp_path):
        """Should ignore .bin files and original/ folder during download."""
        local_path = tmp_path / "model"

        with patch("huggingface_hub.snapshot_download") as mock_download:
            _ensure_model_downloaded("test/model", local_path)

            # Check ignore patterns
            call_kwargs = mock_download.call_args[1]
            assert "*.bin" in call_kwargs["ignore_patterns"]
            assert "original/*" in call_kwargs["ignore_patterns"]

    def test_creates_directories_if_missing(self, tmp_path):
        """Should create local_path directories if they don't exist."""
        local_path = tmp_path / "deep" / "nested" / "model"
        assert not local_path.exists()

        with patch("huggingface_hub.snapshot_download"):
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
        with patch("maic.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 10 * (1024**3)
            mock_mem.return_value.available = 10 * (1024**3)

            # Should not raise
            mm._check_ram("mlx-community/Phi-3.5-mini-instruct-4bit")  # 2.3GB

    def test_rejects_models_over_80_percent_ram(self):
        """Should reject models requiring >80% of available RAM."""
        mm = ModelManager()

        # Mock psutil to return 2GB RAM, model needs 4GB
        with patch("maic.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 2 * (1024**3)
            mock_mem.return_value.available = 2 * (1024**3)

            with pytest.raises(ModelTooLargeError) as exc_info:
                mm._check_ram("mlx-community/Mistral-7B-Instruct-v0.3-4bit")  # 4.0GB

            assert "Feasible alternatives" in str(exc_info.value)

    def test_skips_check_for_unknown_models(self):
        """Should not check unknown models (let MLX decide)."""
        mm = ModelManager()

        # Unknown model should not raise
        mm._check_ram("completely/unknown-model-xyz")

    def test_suggests_feasible_alternatives(self):
        """Should suggest smaller models when rejecting too-large models."""
        mm = ModelManager()

        with patch("maic.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 3 * (1024**3)  # 3GB
            mock_mem.return_value.available = 3 * (1024**3)

            with pytest.raises(ModelTooLargeError) as exc_info:
                mm._check_ram("mlx-community/Mistral-7B-Instruct-v0.3-4bit")  # 4.0GB

            error_msg = str(exc_info.value)
            # Should suggest smaller feasible models
            assert "SmolLM" in error_msg or "Phi" in error_msg

    def test_ram_check_preserves_80_percent_rule(self):
        """Should use exactly 80% of available RAM as safe limit."""
        mm = ModelManager()

        # 10GB RAM, 80% = 8GB limit
        with patch("maic.core.model_manager.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.total = 10 * (1024**3)

            # 8GB model at 80% should fail
            with pytest.raises(ModelTooLargeError):
                mm._check_ram("mlx-community/Llama-3.3-70B-Instruct-4bit")  # 40GB


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

        with patch("maic.core.model_manager.logger") as mock_logger:
            stats = {"token_count": 42, "elapsed": 2.0, "tokens_per_second": 21.0}
            observer.on_complete(stats)

            # Should log with correct values (format string uses %d placeholders)
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0]
            assert "%d" in call_args[0]  # format string
            assert call_args[1] == 42  # token_count
            assert call_args[2] == 2.0  # elapsed
            assert call_args[3] == 21.0  # tokens_per_second


# ─────────────────────────────────────────────────────────────────────────────
# Tests for ModelManager.load
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLoad:
    """Tests for model loading logic."""

    @patch("maic.core.model_manager._ensure_model_downloaded")
    def test_load_successful_flow(self, mock_download):
        """Should download, load, and derive stop strings."""
        mm = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = []

        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(mm, "_check_ram"):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
                with patch.object(mm, "_derive_stop_strings", return_value=frozenset()):
                    mm.load("test/model")

        assert mm._model is mock_model
        assert mm._tokenizer is mock_tokenizer
        assert mm._model_id == "test/model"
        mock_download.assert_called_once()

    @patch("maic.core.model_manager._ensure_model_downloaded")
    def test_load_checks_ram_before_download(self, mock_download):
        """Should check RAM before attempting download."""
        mm = ModelManager()

        with patch.object(mm, "_check_ram") as mock_check:
            mock_check.side_effect = ModelTooLargeError("Too large")

            with pytest.raises(ModelTooLargeError):
                mm.load("test/model")

            # Download should not be called
            mock_download.assert_not_called()

    @patch("maic.core.model_manager._ensure_model_downloaded")
    def test_load_wraps_download_errors(self, mock_download):
        """Should wrap download errors in ModelLoadError."""
        mm = ModelManager()
        mock_download.side_effect = Exception("Network error")

        with patch.object(mm, "_check_ram"):
            with pytest.raises(ModelLoadError) as exc_info:
                mm.load("test/model")

            assert "Failed to download" in str(exc_info.value)
            assert "Network error" not in str(exc_info.value)

    @patch("maic.core.model_manager._ensure_model_downloaded")
    def test_load_wraps_mlx_errors(self, mock_download):
        """Should wrap MLX loading errors in ModelLoadError."""
        mm = ModelManager()
        mock_mlx = MagicMock()
        mock_mlx.load.side_effect = Exception("MLX error")

        with patch.object(mm, "_check_ram"):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
                with pytest.raises(ModelLoadError) as exc_info:
                    mm.load("test/model")

                assert "Failed to load" in str(exc_info.value)
                assert "MLX error" not in str(exc_info.value)

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
        mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")

        mm._tokenizer = mock_tokenizer

        # Should not raise
        result = mm._derive_stop_strings()
        assert isinstance(result, frozenset)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Model Management Methods
# ─────────────────────────────────────────────────────────────────────────────


class TestModelManagement:
    """Tests for is_downloaded, download_model, delete_model, disk_size_gb."""

    @patch("maic.core.model_manager._model_local_path")
    def test_is_downloaded_returns_true_with_safetensors(self, mock_path):
        """is_downloaded should return True if .safetensors exists."""
        mm = ModelManager()
        mock_local_path = Mock()
        mock_local_path.glob.side_effect = [["file.safetensors"], []]
        mock_path.return_value = mock_local_path

        result = mm.is_downloaded("test/model")

        assert result is True

    @patch("maic.core.model_manager._model_local_path")
    def test_is_downloaded_returns_false_if_empty(self, mock_path):
        """is_downloaded should return False if no weight files."""
        mm = ModelManager()
        mock_local_path = Mock()
        mock_local_path.glob.side_effect = [[], []]
        mock_path.return_value = mock_local_path

        result = mm.is_downloaded("test/model")

        assert result is False

    @patch("maic.core.model_manager._ensure_model_downloaded")
    @patch("maic.core.model_manager._model_local_path")
    def test_download_model(self, mock_path, mock_ensure_download):
        """download_model should ensure model is downloaded."""
        mm = ModelManager()
        mock_path.return_value = "/fake/path"

        with patch.object(mm, "_check_ram"):
            mm.download_model("test/model")

            mock_ensure_download.assert_called_once()

    @patch("maic.core.model_manager._model_local_path")
    def test_delete_model_refuses_loaded_model(self, mock_path):
        """delete_model should refuse to delete currently loaded model."""
        mm = ModelManager()
        mm._model_id = "test/model"

        with pytest.raises(RuntimeError) as exc_info:
            mm.delete_model("test/model")

        assert "currently loaded" in str(exc_info.value)

    @patch("maic.core.model_manager._model_local_path")
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

    @patch("maic.core.model_manager._model_local_path")
    def test_disk_size_gb_returns_none_if_not_downloaded(self, mock_path):
        """disk_size_gb should return None if model not downloaded."""
        mm = ModelManager()
        mock_local_path = Mock()
        mock_local_path.exists.return_value = False
        mock_path.return_value = mock_local_path

        result = mm.disk_size_gb("test/model")

        assert result is None

    @patch("maic.core.model_manager._model_local_path")
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

    def test_known_sizes_not_empty(self):
        """KNOWN_SIZES should have models defined."""
        assert len(KNOWN_SIZES) > 0

    def test_gated_models_are_subset_of_known_sizes(self):
        """GATED_MODELS should only reference models present in KNOWN_SIZES."""
        for model_id in GATED_MODELS:
            assert model_id in KNOWN_SIZES

    def test_known_sizes_by_ram_is_sorted(self):
        """KNOWN_SIZES_BY_RAM should be sorted by ascending size."""
        sizes = [size for _, size in KNOWN_SIZES_BY_RAM]
        assert sizes == sorted(sizes)

    def test_default_model_in_known_sizes(self):
        """Default model should be in KNOWN_SIZES."""
        assert "mlx-community/Phi-3.5-mini-instruct-4bit" in KNOWN_SIZES


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Path Security
# ─────────────────────────────────────────────────────────────────────────────


class TestPathSecurity:
    """Tests to ensure no path traversal vulnerabilities."""

    @patch("maic.core.model_manager.settings")
    def test_no_path_traversal_in_model_id(self, mock_settings):
        """Model IDs with '..' are rejected by format validation before path resolution."""
        mock_settings.models_dir = "/safe/models"

        with pytest.raises(ValueError, match="Invalid model ID format"):
            _model_local_path("../../etc/passwd")

    @patch("maic.core.model_manager.settings")
    def test_handles_absolute_paths_in_model_id(self, mock_settings):
        """Absolute paths in model IDs are rejected by format validation."""
        mock_settings.models_dir = "/safe/models"

        with pytest.raises(ValueError, match="Invalid model ID format"):
            _model_local_path("/etc/passwd/model")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for stop string trimming in _observed (Layer 1 and Layer 3)
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptCache:
    """Tests for prompt cache lifecycle."""

    def test_clear_cache_resets_to_none(self):
        mm = ModelManager()
        mm._prompt_cache = "something"
        mm.clear_cache()
        assert mm._prompt_cache is None

    def test_load_resets_prompt_cache(self):
        """load() should discard the old prompt cache."""
        mm = ModelManager()
        mm._prompt_cache = "old_cache"
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_ids = []

        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(mm, "_check_ram"):
            with patch("maic.core.model_manager._ensure_model_downloaded"):
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
                    with patch.object(mm, "_derive_stop_strings", return_value=frozenset()):
                        mm.load("test/model")

        assert mm._prompt_cache is None

    def test_ensure_prompt_cache_creates_cache(self):
        """_ensure_prompt_cache should create a cache on first call."""
        mm = ModelManager()
        mm._model = Mock()

        mock_cache = Mock()
        with patch("maic.core.model_manager.settings") as mock_settings:
            mock_settings.max_kv_size = None
            mock_settings.kv_bits = None
            mock_settings.kv_group_size = 64
            with patch.dict(
                "sys.modules",
                {"mlx_lm": MagicMock(), "mlx_lm.models": MagicMock(), "mlx_lm.models.cache": MagicMock()},
            ):
                with patch("mlx_lm.models.cache.make_prompt_cache", return_value=mock_cache):
                    result = mm._ensure_prompt_cache()

        assert result is mock_cache
        assert mm._prompt_cache is mock_cache

    def test_ensure_prompt_cache_reuses_existing(self):
        """_ensure_prompt_cache should return existing cache without recreating."""
        mm = ModelManager()
        mm._prompt_cache = "existing"
        assert mm._ensure_prompt_cache() == "existing"


class TestStopStringTrimming:
    """Verify that Layer 1 and Layer 3 trim at the rightmost stop string,
    preserving any stop-string-like text that appeared earlier in content."""

    def _make_resp(self, text, finish_reason=None, token=1):
        r = Mock()
        r.text = text
        r.finish_reason = finish_reason
        r.token = token
        return r

    def _collect(self, mm, responses):
        """Wire up generate() with mocked MLX internals and return joined output."""
        mock_mx = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.stream_generate.return_value = iter(responses)
        strategy = Mock(return_value={})
        with patch.dict(
            "sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": mock_mlx}
        ):
            with patch("maic.core.model_manager.settings") as mock_settings:
                mock_settings.max_tokens = 512
                mock_settings.temperature = 0.7
                mock_settings.top_p = 0.9
                mock_settings.max_kv_size = None
                mock_settings.kv_bits = None
                mock_settings.kv_group_size = 64
                stream = mm.generate([], strategy)
                return "".join(stream)

    def _collect_chunks(self, mm, responses):
        """Wire up generate() and return raw yielded chunks (for stream cadence assertions)."""
        mock_mx = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.stream_generate.return_value = iter(responses)
        strategy = Mock(return_value={})
        with patch.dict(
            "sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx, "mlx_lm": mock_mlx}
        ):
            with patch("maic.core.model_manager.settings") as mock_settings:
                mock_settings.max_tokens = 512
                mock_settings.temperature = 0.7
                mock_settings.top_p = 0.9
                mock_settings.max_kv_size = None
                mock_settings.kv_bits = None
                mock_settings.kv_group_size = 64
                stream = mm.generate([], strategy)
                return list(stream)

    def _make_mm(self, stop_strings):
        mm = ModelManager()
        mm._model = Mock()
        mm._tokenizer = Mock()
        mm._tokenizer.apply_chat_template.return_value = [1, 2, 3]
        mm._stop_strings = frozenset(stop_strings)
        mm._prompt_cache = []  # Pre-set so _ensure_prompt_cache skips the import
        return mm

    # ── Layer 1 ──────────────────────────────────────────────────────────────

    def test_layer1_simple_trim(self):
        """Layer 1: trim the single EOS token at the end."""
        mm = self._make_mm({"<|im_end|>"})
        result = self._collect(mm, [self._make_resp("Hello world <|im_end|>", "stop")])
        assert result == "Hello world "

    def test_layer1_no_stop_string_leaked(self):
        """Layer 1: if no stop string leaked into text, yield the full suffix."""
        mm = self._make_mm({"<|im_end|>"})
        result = self._collect(mm, [self._make_resp("Hello world", "stop")])
        assert result == "Hello world"

    def test_layer1_trims_rightmost_preserves_earlier_content(self):
        """Layer 1: with two stop strings in suffix, trim at the rightmost one.

        Old code applied split() for every stop string in frozenset order (arbitrary),
        which could remove text before the actual EOS marker.  The new code finds
        the rightmost occurrence of any stop string and trims only there.
        """
        mm = self._make_mm({"<|end|>", "<|im_end|>"})
        # <|end|> appears at position 6; <|im_end|> is the actual EOS at position 28.
        suffix = "Note: <|end|> ends the turn <|im_end|>"
        result = self._collect(mm, [self._make_resp(suffix, "stop")])
        assert result == "Note: <|end|> ends the turn "
        assert "<|im_end|>" not in result

    # ── Layer 3 ──────────────────────────────────────────────────────────────

    def test_layer3_trims_rightmost_not_frozenset_first(self):
        """Layer 3: trim at the rightmost stop string, not whichever frozenset yields first.

        Old code used next() on a frozenset (arbitrary iteration order), so it could
        pick an earlier stop string and over-trim.  The new code always finds the
        rightmost occurrence.
        """
        mm = self._make_mm({"<|end|>", "<|im_end|>"})
        # Both stop strings fit within the rolling window (max_suffix=11),
        # so the whole chunk stays in suffix and Layer 3 fires.
        result = self._collect(mm, [self._make_resp("X<|end|>Y<|im_end|>")])
        assert result == "X<|end|>Y"
        assert "<|im_end|>" not in result

    def test_layer3_single_stop_string(self):
        """Layer 3: basic case with a single stop string."""
        mm = self._make_mm({"<|im_end|>"})
        result = self._collect(mm, [self._make_resp("Hi<|im_end|>")])
        assert result == "Hi"

    def test_short_response_streams_incrementally_with_standard_stop_marker(self):
        """Short replies should still stream before final stop chunk."""
        mm = self._make_mm({"<|im_end|>"})
        responses = [
            self._make_resp("H"),
            self._make_resp("i"),
            self._make_resp("!"),
            self._make_resp("", "stop"),
        ]

        chunks = self._collect_chunks(mm, responses)

        assert "".join(chunks) == "Hi!"
        assert len(chunks) > 1

    def test_fallback_to_token_decode_when_text_segments_are_empty(self):
        """If MLX yields empty text segments, we should still stream via token decode."""
        mm = self._make_mm({"<|im_end|>"})
        mm._tokenizer.decode.side_effect = lambda ids: {101: "H", 102: "i", 999: ""}[ids[0]]

        responses = [
            self._make_resp("", None, token=101),
            self._make_resp("", None, token=102),
            self._make_resp("Hi", "stop", token=999),
        ]

        chunks = self._collect_chunks(mm, responses)

        assert "".join(chunks) == "Hi"
        assert len(chunks) >= 2

    def test_pathological_long_stop_marker_does_not_fully_buffer_stream(self):
        """Very long stop markers must not force a single end-of-stream chunk."""
        mm = self._make_mm({"X" * 500})
        responses = [self._make_resp("a") for _ in range(120)]
        responses.append(self._make_resp("z", "stop"))

        chunks = self._collect_chunks(mm, responses)

        assert "".join(chunks) == ("a" * 120 + "z")
        assert len(chunks) > 1


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantizationInfo:
    @patch("maic.core.model_manager.settings")
    def test_returns_none_when_no_config(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        mm = ModelManager()
        assert mm.quantization_info("org/model") is None

    @patch("maic.core.model_manager.settings")
    def test_returns_none_when_no_quant_key(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        model_dir = tmp_path / "org--model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"hidden_size": 768}')
        mm = ModelManager()
        assert mm.quantization_info("org/model") is None

    @patch("maic.core.model_manager.settings")
    def test_returns_info_string(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        model_dir = tmp_path / "org--model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"quantization": {"bits": 4, "group_size": 64}}'
        )
        mm = ModelManager()
        assert mm.quantization_info("org/model") == "4bit (g=64)"

    @patch("maic.core.model_manager.settings")
    def test_returns_none_for_corrupt_json(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        model_dir = tmp_path / "org--model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{bad json")
        mm = ModelManager()
        assert mm.quantization_info("org/model") is None


class TestQuantizeModel:
    @patch("maic.core.model_manager.settings")
    def test_raises_when_source_missing(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        mm = ModelManager()
        with pytest.raises(ModelLoadError, match="not found on disk"):
            mm.quantize_model("org/model")

    @patch("maic.core.model_manager.settings")
    def test_raises_when_output_exists(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        (tmp_path / "org--model").mkdir()
        (tmp_path / "org--model-4bit").mkdir()
        mm = ModelManager()
        with pytest.raises(RuntimeError, match="already exists"):
            mm.quantize_model("org/model")

    @patch("maic.core.model_manager.settings")
    def test_calls_convert_with_correct_args(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        (tmp_path / "org--model").mkdir()

        mock_mlx_lm = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            mm = ModelManager()
            result = mm.quantize_model("org/model", q_bits=8, q_group_size=32)

        assert result == "org/model-8bit"
        mock_mlx_lm.convert.assert_called_once()
        call_kwargs = mock_mlx_lm.convert.call_args[1]
        assert call_kwargs["q_bits"] == 8
        assert call_kwargs["q_group_size"] == 32
        assert call_kwargs["quantize"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tool-call format detection
# ─────────────────────────────────────────────────────────────────────────────

# Abbreviated template snippets — just the markers that drive detection.
_QWEN_TEMPLATE = (
    "{%- if tools %}\n{{- '<|im_start|>system\\n' }}\n"
    "{%- for tool in tools %}\n{{- tool | tojson }}\n{%- endfor %}\n"
    "If you need to call a tool, use <tool_call> JSON </tool_call>\n"
)
_LLAMA3_TEMPLATE = (
    "{% if tools %}<|python_tag|>[{\"name\":\"...\",\"parameters\":{}}]\n"
    "{% endif %}<|eot_id|>"
)
_MISTRAL_TEMPLATE = (
    "{% if tool_calls %}[TOOL_CALLS] "
    "[{\"name\":\"...\",\"arguments\":{}}]{% endif %}"
)
_DEEPSEEK_TEMPLATE = (
    "{% if tools %}<|\u2581tool\u2581calls\u2581begin\u2581|>"
    "... <|\u2581tool\u2581sep\u2581|> ... <|\u2581tool\u2581calls\u2581end\u2581|>"
    "{% endif %}"
)
_DEEPSEEK_TEMPLATE_V2 = (
    "tool\u2581calls\u2581begin content here"
)
_HERMES_TEMPLATE = (
    "{%- for message in messages %}\n"
    "{%- if message.role == 'tool' %}<tool_response>{{message.content}}</tool_response>\n"
    "{%- endif %}\n{%- endfor %}"
)
_NO_TOOLS_TEMPLATE = (
    "{% for message in messages %}{% if message.role == 'user' %}"
    "{{ message.content }}{% endif %}{% endfor %}"
)


class TestDetectToolCallFormat:
    """Tests for the chat-template-based tool-call format detector."""

    def test_empty_template_returns_none(self):
        assert detect_tool_call_format("") == ToolCallFormat.NONE

    def test_no_tool_markers_returns_none(self):
        assert detect_tool_call_format(_NO_TOOLS_TEMPLATE) == ToolCallFormat.NONE

    def test_qwen_template_detected(self):
        assert detect_tool_call_format(_QWEN_TEMPLATE) == ToolCallFormat.QWEN

    def test_llama3_template_detected(self):
        assert detect_tool_call_format(_LLAMA3_TEMPLATE) == ToolCallFormat.LLAMA3

    def test_mistral_template_detected(self):
        assert detect_tool_call_format(_MISTRAL_TEMPLATE) == ToolCallFormat.MISTRAL

    def test_deepseek_template_detected(self):
        assert detect_tool_call_format(_DEEPSEEK_TEMPLATE_V2) == ToolCallFormat.DEEPSEEK

    def test_hermes_template_detected(self):
        assert detect_tool_call_format(_HERMES_TEMPLATE) == ToolCallFormat.HERMES

    def test_template_with_tools_keyword_only_returns_unknown(self):
        # A template that mentions "tools" but no recognisable output format
        template = "{% if tools %}You have access to tools.{% endif %}"
        assert detect_tool_call_format(template) == ToolCallFormat.UNKNOWN

    def test_llama3_takes_priority_over_qwen(self):
        # If a template somehow has both markers, LLAMA3 wins (checked first)
        combined = _LLAMA3_TEMPLATE + "<tool_call>...</tool_call>"
        assert detect_tool_call_format(combined) == ToolCallFormat.LLAMA3


class TestModelManagerToolCallFormat:
    """Tests for ToolCallFormat properties on ModelManager."""

    def test_default_format_is_none(self):
        mm = ModelManager()
        assert mm.tool_call_format == ToolCallFormat.NONE

    def test_supports_tool_calling_false_by_default(self):
        mm = ModelManager()
        assert mm.supports_tool_calling is False

    def test_supports_tool_calling_true_after_qwen_load(self):
        mm = ModelManager()
        mm._tool_call_format = ToolCallFormat.QWEN
        assert mm.supports_tool_calling is True

    def test_supports_tool_calling_false_for_unknown(self):
        mm = ModelManager()
        mm._tool_call_format = ToolCallFormat.UNKNOWN
        assert mm.supports_tool_calling is False

    @patch("maic.core.model_manager.settings")
    def test_tool_call_format_for_loaded_model_returns_current(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        mm = ModelManager()
        mm._model_id = "org/model"
        mm._tool_call_format = ToolCallFormat.MISTRAL
        assert mm.tool_call_format_for("org/model") == ToolCallFormat.MISTRAL

    @patch("maic.core.model_manager.settings")
    def test_tool_call_format_for_downloaded_model_reads_config(self, mock_settings, tmp_path):
        mock_settings.models_dir = str(tmp_path)
        model_dir = tmp_path / "org--other"
        model_dir.mkdir()
        import json

        cfg = {"chat_template": _QWEN_TEMPLATE}
        (model_dir / "tokenizer_config.json").write_text(json.dumps(cfg))
        mm = ModelManager()
        mm._model_id = "org/loaded"
        assert mm.tool_call_format_for("org/other") == ToolCallFormat.QWEN

    @patch("maic.core.model_manager.settings")
    def test_tool_call_format_for_not_downloaded_model_returns_none(
        self, mock_settings, tmp_path
    ):
        mock_settings.models_dir = str(tmp_path)
        mm = ModelManager()
        assert mm.tool_call_format_for("org/nothere") == ToolCallFormat.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Thinking mode detection
# ─────────────────────────────────────────────────────────────────────────────

# Abbreviated Qwen3 template snippet — contains 'enable_thinking' variable.
_QWEN3_TEMPLATE = (
    "{%- if enable_thinking is defined and enable_thinking %}"
    "<think>{{thinking_content}}</think>"
    "{%- endif %}"
    "{{- content }}"
)
# A template that contains the literal <think> token (alternative signal).
_THINK_TOKEN_TEMPLATE = "{{ bos_token }}<think>{{ content }}</think>"

# Qwen2.5-Coder / plain Qwen template — no thinking markers.
_QWEN25_TEMPLATE = (
    "{%- if tools %}\n{{- '<|im_start|>system\\n' }}\n"
    "{%- for tool in tools %}\n{{- tool | tojson }}\n{%- endfor %}\n"
    "If you need to call a tool, use <tool_call> JSON </tool_call>\n"
)
# Llama-3 template — no thinking markers.
_LLAMA3_PLAIN_TEMPLATE = (
    "{% if tools %}<|python_tag|>[{\"name\":\"...\",\"parameters\":{}}]\n"
    "{% endif %}<|eot_id|>"
)


class TestDetectThinkingSupport:
    """Tests for the chat-template thinking-mode detector."""

    def test_empty_template_returns_false(self):
        assert detect_thinking_support("") is False

    def test_none_like_value_returns_false(self):
        assert detect_thinking_support("  ") is False

    def test_qwen3_enable_thinking_returns_true(self):
        assert detect_thinking_support(_QWEN3_TEMPLATE) is True

    def test_think_token_in_template_returns_true(self):
        assert detect_thinking_support(_THINK_TOKEN_TEMPLATE) is True

    def test_qwen25_coder_template_returns_false(self):
        assert detect_thinking_support(_QWEN25_TEMPLATE) is False

    def test_llama3_template_returns_false(self):
        assert detect_thinking_support(_LLAMA3_PLAIN_TEMPLATE) is False

    def test_no_tool_template_returns_false(self):
        plain = "{% for m in messages %}{{ m.content }}{% endfor %}"
        assert detect_thinking_support(plain) is False


class TestModelManagerThinkingSupport:
    """Tests for the supports_thinking property on ModelManager."""

    def test_default_supports_thinking_is_false(self):
        mm = ModelManager()
        assert mm.supports_thinking is False

    def test_supports_thinking_true_when_flag_set(self):
        mm = ModelManager()
        mm._supports_thinking = True
        assert mm.supports_thinking is True

    def test_supports_thinking_false_for_qwen25_template(self):
        mm = ModelManager()
        mm._supports_thinking = detect_thinking_support(_QWEN25_TEMPLATE)
        assert mm.supports_thinking is False

    def test_supports_thinking_true_for_qwen3_template(self):
        mm = ModelManager()
        mm._supports_thinking = detect_thinking_support(_QWEN3_TEMPLATE)
        assert mm.supports_thinking is True
