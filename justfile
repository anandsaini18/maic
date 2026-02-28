# Project commands — run `just` or `just --list` to see all available recipes
# All commands assume .venv/ exists (created by `just setup`).

set dotenv-load

VENV := ".venv/bin"

# Default: show available commands
default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────────────

# Create venv and install all dependencies (Python + MLX + frontend)
setup:
    python3 -m venv .venv
    {{VENV}}/pip install -e ".[dev,mlx]"
    cd frontend && npm ci

# Install Python dependencies only (with MLX for macOS)
install:
    {{VENV}}/pip install -e ".[dev,mlx]"

# Install frontend dependencies only
install-frontend:
    cd frontend && npm ci

# ── Development ──────────────────────────────────────────────────────────────

# Start the backend server (pass args like: just dev --model mlx-community/Llama-3.2-1B-Instruct-4bit)
dev *ARGS:
    {{VENV}}/python main.py {{ARGS}}

# Start the frontend dev server (Vite, port 5173, proxies to backend)
dev-frontend:
    cd frontend && npm run dev

# Start both backend and frontend (requires backend .env configured)
dev-all:
    just dev &
    just dev-frontend

# ── Build ────────────────────────────────────────────────────────────────────

# Build the frontend for production (outputs to static/)
build:
    cd frontend && npm run build

# ── Testing ──────────────────────────────────────────────────────────────────

# Run all tests with coverage
test *ARGS:
    {{VENV}}/python -m pytest {{ARGS}}

# Quick test run — no coverage, minimal output
test-quick:
    {{VENV}}/python -m pytest -q --no-cov

# Run only security-related tests
test-security:
    {{VENV}}/python -m pytest -m security -v

# Run only TokenStream tests
test-token:
    {{VENV}}/python -m pytest tests/test_token_stream.py -v

# Run only ModelManager tests
test-manager:
    {{VENV}}/python -m pytest tests/test_model_manager.py -v

# Rerun only previously failed tests
test-failed:
    {{VENV}}/python -m pytest --lf -v

# ── Linting & Formatting ────────────────────────────────────────────────────

# Run all linters (Python + frontend)
lint: lint-python lint-frontend

# Lint Python code (ruff + black check + mypy)
lint-python:
    {{VENV}}/ruff check app/ tests/
    {{VENV}}/black --check app/ tests/
    {{VENV}}/mypy app/

# Lint frontend (ESLint + TypeScript)
lint-frontend:
    cd frontend && npm run lint

# Auto-format Python code
fmt:
    {{VENV}}/ruff check --fix app/ tests/
    {{VENV}}/black app/ tests/

# ── CI (mirrors GitHub Actions locally) ──────────────────────────────────────

# Run the full CI pipeline locally
ci: lint test build
    @echo "All CI checks passed."

# ── Cleanup ──────────────────────────────────────────────────────────────────

# Remove build artifacts and caches
clean:
    rm -rf .venv htmlcov .pytest_cache .mypy_cache .ruff_cache
    rm -rf static/assets static/index.html
    rm -rf frontend/node_modules
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
