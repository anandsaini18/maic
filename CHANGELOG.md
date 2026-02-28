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

## [Unreleased]

- Fix pre-existing test failures in `test_model_manager.py` (stale mock paths)
- Frontend test setup (Vitest)
