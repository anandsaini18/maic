"""
Unit Test Suite for LLM Model Manager and Token Streaming

This directory contains comprehensive unit tests ensuring the model manager
and token streaming logic are reliable, secure, and performant.

## Test Files

### test_token_stream.py
Tests for the TokenStream iterator wrapper that provides a clean interface
over raw MLX token generation.

**Coverage:**
- Iterator protocol (__iter__, __next__)
- Token accumulation and counting
- Timing metrics (elapsed, tokens_per_second)
- collect() aggregation method
- Edge cases (empty streams, unicode, whitespace)
- Timing accuracy for active vs finished streams

**Key Tests:**
- `test_next_retrieves_tokens_in_order` — Ensures tokens come out in order
- `test_tokens_per_second_calculation` — Validates throughput metrics
- `test_collect_joins_all_tokens` — Confirms full text reconstruction
- `test_unicode_tokens` — Ensures international text support

### test_model_manager.py
Tests for model loading, generation, RAM checking, and observer notifications.
Uses mocking to avoid downloading real models or requiring MLX GPU code.

**Coverage:**
- Path conversion (_model_local_path) with security checks
- Model downloading with caching logic
- RAM feasibility checking (80% safe threshold)
- Stop string derivation (multiple detection layers)
- Observer pattern registration and notification
- Model lifecycle (load, download, delete, check status)
- Path traversal security

**Key Tests:**
- `test_allows_models_under_80_percent_ram` — RAM safety validation
- `test_rejects_models_over_80_percent_ram` — Prevents OOM conditions
- `test_skips_download_if_safetensors_exists` — Download caching
- `test_register_observer` — Observer pattern functionality
- `test_no_path_traversal_in_model_id` — Security against path attacks

### conftest.py
Shared pytest fixtures used across all tests:
- `mock_tokenizer` — Pre-configured mock tokenizer
- `mock_model` — Pre-configured mock MLX model
- `simple_token_generator` — Test token generator
- `temp_models_dir` — Temporary directory for file operations

## Running Tests

### Install test dependencies:
```bash
pip install pytest pytest-cov pytest-timeout
```

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_token_stream.py
```

### Run specific test class:
```bash
pytest tests/test_model_manager.py::TestCheckRAM
```

### Run specific test:
```bash
pytest tests/test_model_manager.py::TestCheckRAM::test_allows_models_under_80_percent_ram
```

### Run with coverage report:
```bash
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

### Run only security tests:
```bash
pytest -m security
```

### Run with verbose output:
```bash
pytest -v
```

### Run tests matching a pattern:
```bash
pytest -k "ram" # Runs all tests with "ram" in name
```

## Test Organization

Tests are organized into logical classes for clarity:

**TokenStream Tests:**
- `TestTokenStreamIterator` — Python iterator protocol
- `TestTokenAccumulation` — Token collection
- `TestTimingMetrics` — Performance metrics
- `TestEdgeCases` — Boundary conditions
- `TestCollectMethod` — Aggregation method
- `TestTimingAccuracy` — Precise timing measurement

**ModelManager Tests:**
- `TestModelLocalPath` — Path conversion security
- `TestEnsureModelDownloaded` — Download caching
- `TestCheckRAM` — Memory safety
- `TestObserverPattern` — Observer notifications
- `TestModelLoad` — Complete load workflow
- `TestDeriveStopStrings` — Stop token detection
- `TestModelManagement` — Status/cleanup methods
- `TestConstants` — Configuration validation
- `TestPathSecurity` — Path traversal prevention

## Coverage Goals

Current test coverage targets:
- **tokenstream.py**: ~95% (all public methods + edge cases)
- **model_manager.py**: ~90% (all critical paths, mocked external deps)

Run `pytest --cov=app --cov-report=term-missing` to see coverage details.

## Mocking Strategy

The test suite uses `unittest.mock` to avoid:
- Downloading real models from HuggingFace
- Loading actual MLX models (requires GPU)
- Filesystem side effects
- Network operations

Mocks are configured to return realistic data, catching most integration issues
while keeping tests fast and isolated.

## Security Tests

Path security tests ensure no path traversal vulnerabilities:
- `test_no_path_traversal_in_model_id` — Prevents `../../etc/passwd` style attacks
- `test_handles_absolute_paths_in_model_id` — Normalizes absolute paths
- RAM checking prevents denial-of-service via OOM

## Performance Notes

- Full test suite runs in <10 seconds (mocked, no network)
- Individual tests typically <100ms
- Timing tests use 50-100ms sleeps for precision

## Continous Integration

These tests are designed for CI/CD pipelines:
- No external dependencies (network, GPU)
- Deterministic (no flaky timing issues)
- Fast feedback (all tests < 30 seconds)
- Clear failure messages

Example GitHub Actions / GitLab CI configuration:
```yaml
test:
  script:
    - pip install -q pytest pytest-cov pytest-timeout
    - pytest --cov=app --cov-report=xml
    - coverage report --fail-under=85
```

## Extending Tests

To add new tests:

1. Create test class in appropriate file
2. Follow naming convention: `test_<what_is_being_tested>`
3. Use descriptive docstrings
4. Add relevant pytest markers (`@pytest.mark.security`, etc.)
5. Use fixtures from conftest.py when possible
6. Mock external dependencies

Example:
```python
class TestNewFeature:
    \"\"\"Tests for new feature.\"\"\"
    
    @pytest.mark.security
    def test_validates_input(self):
        \"\"\"Should reject invalid input.\"\"\"
        # test code
```

## Troubleshooting

### Import errors:
Make sure you're running pytest from project root:
```bash
cd /Users/mukul/Codes/llm
pytest
```

### Mock errors:
Check that mocked objects match expected interface by looking at the actual
class being mocked (e.g., `InferenceObserver`, `GenerationStrategy`)

### Timing test failures:
Timing tests may occasionally fail on very slow systems. CI/CD agents
should have sufficient CPU. Local failures usually indicate system load.
"""
