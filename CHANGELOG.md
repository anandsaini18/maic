# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-28

### Added
- **GitHub Actions CI pipeline** (`ci.yml`) — runs lint + test + build on every push/PR
  - Backend lint job: ruff, black, mypy across Python 3.10/3.11/3.12 matrix
  - Backend test job: pytest with coverage, uploads HTML + XML reports
  - Frontend job: ESLint + TypeScript + Vite production build
  - Coverage comment on PRs via `MishaKav/pytest-coverage-comment`
  - CI Gate job — single required status check for branch protection
- **PR auto-labeling** (`pr-labeler.yml`) — labels by file path and PR size
- **Dependabot** (`dependabot.yml`) — weekly dependency updates for pip, npm, and GitHub Actions
- **`pyproject.toml`** — single source of truth for all project config (replaces `pytest.ini`, `mypy.ini`, `requirements.txt`, `requirements-dev.txt`)
- **`justfile`** — standard task runner replacing `run.sh` and `run_tests.sh`
- **Integration tests** (`test_routes.py`) — 22 tests covering all HTTP endpoints through FastAPI `TestClient`, exercising routes + decorators + adapter + schemas together
- **Adapter unit tests** (`test_openai_adapter.py`) — 12 tests for SSE wire format, finish_reason branching, RAM feasibility
- **Schema edge-case tests** (`test_schemas.py`) — 10 tests for Pydantic validation (role literals, optional defaults, auto-generated IDs)
- 15 GitHub labels for area, type, and PR size categorization
- `pytest-asyncio` and `httpx` as dev dependencies

### Changed
- `mlx-lm` moved to optional `[mlx]` dependency group — CI runs on Linux without it (tests mock MLX)
- `just setup` installs `.[dev,mlx]` locally; CI installs `.[dev]` only
- ruff config now includes isort, bugbear, and pyupgrade rules
- black + ruff share `line-length = 100`

### Removed
- `run.sh` — replaced by `just setup` + `just dev`
- `run_tests.sh` — replaced by `just test` and variants
- `pytest.ini` — config moved to `pyproject.toml`
- `mypy.ini` — config moved to `pyproject.toml`
- `requirements.txt` — dependencies moved to `pyproject.toml`
- `requirements-dev.txt` — dependencies moved to `pyproject.toml [dev]`

---

## [0.1.0] - 2026-02-26

### Added
- **Initial Release** of Maic
- MLX-optimized inference engine for Apple Silicon (M1/M2/M3)
- OpenAI-compatible REST API (`/v1/chat/completions`)
- Modern React-based web chat interface with Tailwind CSS styling
- Model management system with:
  - One-click HuggingFace model downloads
  - Automatic RAM requirement validation
  - Model activation/deactivation
  - Real-time download progress tracking
- Real-time monitoring dashboard with:
  - Token-per-minute (TPM) charts
  - Memory usage tracking
  - Model status visualization
- Advanced generation parameters:
  - Temperature control
  - Top-p (nucleus sampling)
  - Max tokens configuration
- CLI argument support for model selection and server configuration
- Strategy pattern for pluggable generation strategies (default, greedy)

### Tech Stack
- **Backend:** FastAPI, MLX, Uvicorn, Pydantic
- **Frontend:** React 19, Vite, TypeScript, Tailwind CSS
- **Build Tools:** ESLint, PostCSS, esbuild

---

## Planned Features (Future Releases)

### [0.2.0] - Streaming Improvements
- [ ] WebSocket support for faster streaming responses
- [ ] Server-sent events (SSE) optimization
- [ ] Response buffering and backpressure handling

### [0.3.0] - Extended Model Support
- [ ] GGUF format support
- [ ] Custom quantization profiles
- [ ] LoRA adapter support

### [0.4.0] - Advanced Features
- [ ] Conversation history and management
- [ ] Export chat sessions (PDF, JSON)
- [ ] User preferences persistence
- [ ] Dark/Light theme toggle

### [0.5.0] - Multi-User & Deployment
- [ ] User authentication (optional)
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, GCP, Azure)

---

## [0.6.0] - 2026-03-10

### Added
- **HuggingFace Hub dynamic model discovery** (`hub_fetcher.py`) — polls the configured Hub org for text-generation models, parses gated status, and caches results with a 5-minute TTL (30-second retry TTL on failure). Falls back to three offline defaults when the Hub is unreachable
- **Model size estimation** (`model_sizing.py`) — curated `KNOWN_SIZES` dict + regex estimator that parses parameter count and bit-width from model names; `resolve_size_gb()` cascades curated → regex → 7 GB default and always returns a value
- **Observer pattern** (`InferenceObserver` protocol + `StatsObserver`) — any code can attach to `ModelManager` and receive `on_token`, `on_complete`, and `on_error` events without touching the generation loop; `StatsObserver` is registered by default and logs tokens-per-second on completion
- **Concurrency guard** — `asyncio.Semaphore(1)` in routes serializes inference; a second concurrent request receives `HTTP 503 model_busy` immediately rather than queuing unboundedly
- **`DownloadTracker` class** (`routes.py`) — encapsulates background download state behind a typed interface (`is_downloading`, `error`, `set_downloading`, `set_done`, `set_error`), replacing a bare module-level `dict[str, str]`

### Changed
- **MLX event-loop blocking eliminated** — `_async_token_iter()` offloads the synchronous `TokenStream` iterator to a `ThreadPoolExecutor` thread via an `asyncio.Queue(maxsize=32)`; the async SSE generator awaits tokens from the queue, yielding control to the event loop between tokens
- **`load()` offloaded to thread pool** — `POST /v1/models/load` now uses `await asyncio.to_thread(model_manager.load, mid)` so disk I/O and MLX weight-mapping do not freeze the event loop
- **`stream_to_response()` offloaded to thread pool** — non-streaming completions run `await asyncio.to_thread(OpenAIAdapter.stream_to_response, ...)` instead of blocking the event loop during token collection
- **Fixed broken `on_complete` callback** — replaced the silent `stream.__next__ = fn` monkey-patch (ignored by Python's iterator protocol) with a proper `on_complete: Callable[[], None] | None` parameter on `TokenStream`, called on `StopIteration`
- **Per-token Pydantic allocation eliminated** — SSE content chunks are now built with a pre-computed string prefix/suffix; only `json.dumps(token)` varies per token. Pydantic serialization used only for the first and last chunks (2 per request, not N)
- **`is_downloaded()` and `disk_size_gb()` TTL-cached** — per-instance 30 s / 60 s caches avoid filesystem `glob` and `stat` calls on every 3-second UI poll; caches are explicitly invalidated on load, download, and delete
- **`TOTAL_RAM_GB` module-level constant** — `psutil.virtual_memory()` called once at startup; used in hot paths (`models_status`, adapter feasibility). `_check_ram()` still reads `psutil` directly for test patchability
- **`mlx_lm` cached after `load()`** — `self._mlx_lm` set once; `generate()` skips `sys.modules` lookup on every call
- **`make_sampler` cached** — `@lru_cache(maxsize=16)` on `_cached_sampler(temp, top_p)` in `config.py`; repeated requests with the same generation parameters reuse the same sampler object
- **`TokenStream._tokens` list replaced** — replaced with `self._count: int`; avoids growing list allocation for the common case where only token count matters
- **SSE `created` timestamp pre-computed** — `int(time.time())` called once before the SSE loop instead of per-chunk

### Fixed
- Stop-string detection now uses three layers: native EOS token detection (finish_reason == "stop"), max-token length limit (finish_reason == "length"), and rolling text-buffer scan for stop strings that leak as plain characters in quantized models
- Three-layer stop logic includes a holdback window equal to the longest stop string to prevent yielding text that could be the start of an arriving stop marker

