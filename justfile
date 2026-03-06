set dotenv-load

VENV := ".venv/bin"

[private]
default:
    @just --list

# ── Setup ─────────────────────────────────────────────────────────────────────

# Create .venv and install Python + frontend deps
[group: 'setup']
setup:
    @echo "→ Creating virtual environment..."
    python3.12 -m venv .venv
    @echo "→ Installing Python dependencies..."
    {{VENV}}/pip install -e ".[dev,mlx]" -q
    @echo "→ Installing frontend dependencies..."
    cd frontend && npm ci --silent
    @echo "✓ Setup complete — run 'just dev' to start"

# ── Development ───────────────────────────────────────────────────────────────

# Start backend API server (pass args e.g. just dev --model mlx-community/Llama-3.2-1B-Instruct-4bit)
[group: 'dev']
dev *ARGS:
    @[ -d .venv ] || (echo "✗ No venv found — run 'just setup' first" && exit 1)
    @echo "→ Starting backend on http://localhost:8001"
    {{VENV}}/python main.py {{ARGS}}

# Start frontend Vite dev server (proxies API to port 8001)
[group: 'dev']
dev-frontend:
    @echo "→ Starting frontend dev server on http://localhost:5173"
    cd frontend && npm run dev

# Build frontend then start backend (production mode, serves UI from /static)
[group: 'dev']
run *ARGS:
    @[ -d .venv ] || (echo "✗ No venv found — run 'just setup' first" && exit 1)
    @echo "→ Building frontend..."
    @cd frontend && ([ -d node_modules ] || npm ci --silent) && npm run build --silent
    @echo "→ Starting backend on http://localhost:8001"
    {{VENV}}/python main.py {{ARGS}}

# ── Build ─────────────────────────────────────────────────────────────────────

# Build frontend for production (outputs to static/)
[group: 'build']
build:
    @echo "→ Building frontend..."
    @cd frontend && ([ -d node_modules ] || (echo "  Installing deps first..." && npm ci --silent)) && npm run build
    @echo "✓ Frontend built → static/"

# ── Testing ───────────────────────────────────────────────────────────────────

# Run tests — pass extra args e.g. just test -k token  or  just test --lf -q
[group: 'test']
test *ARGS:
    @echo "→ Running tests..."
    {{VENV}}/python -m pytest {{ARGS}}

# ── Code Quality ──────────────────────────────────────────────────────────────

# Run all linters: ruff + mypy (Python) and ESLint + tsc (frontend)
[group: 'quality']
lint:
    @echo "→ Linting Python (ruff)..."
    {{VENV}}/ruff check src/ tests/
    @echo "→ Type-checking Python (mypy)..."
    {{VENV}}/mypy src/
    @echo "→ Linting frontend (ESLint + tsc)..."
    cd frontend && npm run lint
    @echo "✓ All lint checks passed"

# Auto-format Python code with ruff + black
[group: 'quality']
fmt:
    @echo "→ Formatting Python..."
    {{VENV}}/ruff check --fix src/ tests/
    {{VENV}}/black src/ tests/
    @echo "✓ Done"

# ── CI ────────────────────────────────────────────────────────────────────────

# Run full CI pipeline locally: lint → test → build
[group: 'ci']
ci: lint test build
    @echo "✓ All CI checks passed"

# ── Cleanup ───────────────────────────────────────────────────────────────────

# Remove venv, build artifacts, and caches
[group: 'ci']
clean:
    @echo "→ Cleaning build artifacts and caches..."
    rm -rf .venv htmlcov .pytest_cache .mypy_cache .ruff_cache
    rm -rf static/assets static/index.html
    rm -rf frontend/node_modules
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    @echo "✓ Clean"
