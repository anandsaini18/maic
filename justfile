# Project commands — run `just` or `just --list` to see all available recipes
# All commands assume .venv/ exists (created by `just setup`).

set dotenv-load

VENV := ".venv/bin"

# Default: show available commands in logical order
default:
    @echo ""
    @echo "  Maic — Available Commands"
    @echo "  ──────────────────────────────────────────────────────────"
    @echo ""
    @echo "  Getting Started (run in this order)"
    @echo "    just setup           Create venv + install all dependencies"
    @echo "    just build           Build frontend for production"
    @echo "    just start           setup → build → launch both (default: Llama-3.2-1B)"
    @echo "    just start --model ID  same with a custom model"
    @echo ""
    @echo "  Development"
    @echo "    just dev             Start backend server"
    @echo "    just dev-frontend    Start frontend dev server (port 5173)"
    @echo "    just dev-all         Start both in background + foreground"
    @echo "    just install         Reinstall Python deps only"
    @echo "    just install-frontend  Reinstall frontend deps only"
    @echo ""
    @echo "  Testing"
    @echo "    just test            Run all tests with coverage"
    @echo "    just test-quick      Quick run — no coverage"
    @echo "    just test-token      TokenStream tests only"
    @echo "    just test-manager    ModelManager tests only"
    @echo "    just test-failed     Rerun previously failed tests"
    @echo ""
    @echo "  Code Quality"
    @echo "    just lint            Run all linters (Python + frontend)"
    @echo "    just lint-python     ruff + black + mypy"
    @echo "    just lint-frontend   ESLint + TypeScript"
    @echo "    just fmt             Auto-format Python code"
    @echo ""
    @echo "  CI & Cleanup"
    @echo "    just ci              Full CI pipeline (lint + test + build)"
    @echo "    just clean           Remove build artifacts and caches"
    @echo ""

# Setup → build → launch backend + frontend in separate terminals
# Default model: mlx-community/Llama-3.2-1B-Instruct-4bit
# Override:      just start --model mlx-community/Phi-3.5-mini-instruct-4bit
start *ARGS:
    #!/usr/bin/env bash
    set -e
    DIR="$(pwd)"
    ARGS="{{ARGS}}"
    DEFAULT_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
    MODEL_ARGS="${ARGS:---model $DEFAULT_MODEL}"
    BACKEND_URL="http://localhost:8001/v1/models"
    echo ""
    echo "  Maic — Launcher"
    echo "  ─────────────────────────────────"
    echo ""

    echo "  [1/3] Setting up environment..."
    if [[ ! -d ".venv" ]]; then
        just setup
    else
        echo "        venv already exists, skipping"
    fi
    echo ""

    echo "  [2/3] Building frontend..."
    just build
    echo ""

    echo "  [3/3] Launching backend ($MODEL_ARGS)..."
    BACKEND=$(mktemp /tmp/maic-backend-XXXX)
    FRONTEND=$(mktemp /tmp/maic-frontend-XXXX)
    printf '#!/bin/bash\ncd "%s" && just dev %s\n' "$DIR" "$MODEL_ARGS" > "$BACKEND"
    printf '#!/bin/bash\ncd "%s" && just dev-frontend\n' "$DIR" > "$FRONTEND"
    chmod +x "$BACKEND" "$FRONTEND"
    open -a Terminal "$BACKEND"

    echo "        Waiting for backend on port 8001..."
    ELAPSED=0
    SPIN='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    until nc -z 127.0.0.1 8001 2>/dev/null; do
        if [[ $ELAPSED -ge 180 ]]; then
            echo ""
            echo "  Backend did not start after 180s — check the backend terminal."
            exit 1
        fi
        IDX=$((ELAPSED % 10))
        printf "\r        %s  %ds elapsed (model loading...)" "${SPIN:$IDX:1}" "$ELAPSED"
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done
    echo ""
    echo "        Backend ready!"
    echo ""

    echo "  [4/4] Launching frontend..."
    open -a Terminal "$FRONTEND"
    echo "        Frontend → http://localhost:5173"
    echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

# Create venv and install all dependencies (Python + MLX + frontend)
setup:
    python3.12 -m venv .venv
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
    cd frontend && ([ -d node_modules ] || npm ci) && npm run build

# ── Testing ──────────────────────────────────────────────────────────────────

# Run all tests with coverage
test *ARGS:
    {{VENV}}/python -m pytest {{ARGS}}

# Quick test run — no coverage, minimal output
test-quick:
    {{VENV}}/python -m pytest -q --no-cov

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
