r"""Application entry point — CLI parsing, FastAPI app factory, and server startup.

Wires together the FastAPI application with model lifecycle management,
frontend static file serving, and uvicorn. Run directly or via ``just dev``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.model_manager import ModelLoadError, ModelTooLargeError, model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    r"""Parse CLI arguments for model selection, host, and port overrides.

    Returns:
        argparse.Namespace: Parsed arguments with ``model``, ``host``, and ``port`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Local LLM REST server (MLX + OpenAI-compatible API)",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help=(
            "HuggingFace model ID to load (e.g. mlx-community/Llama-3.2-1B-Instruct-4bit). "
            "Overrides MODEL_ID env var and .env file."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind (default from config, usually 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default from config, usually 8000)",
    )
    return parser.parse_args()


# ── FastAPI lifespan — model load on startup ──────────────────────────────────

def make_lifespan(model_id: str):
    r"""Create a FastAPI lifespan context manager that loads the model on startup.

    Args:
        model_id (str): HuggingFace model identifier to load when the server starts.

    Returns:
        Callable: An async context manager suitable for FastAPI's ``lifespan`` parameter.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            model_manager.load(model_id)
        except ModelLoadError as exc:
            # Print friendly message; re-raise as RuntimeError so uvicorn exits
            # cleanly without double-printing a traceback.
            print(str(exc), file=sys.stderr)
            raise RuntimeError("Model load failed — see error above.") from None
        yield
        # Shutdown: nothing to clean up for MLX

    return lifespan


# ── App factory ───────────────────────────────────────────────────────────────

def configure_frontend(app: FastAPI, static_dir: Path) -> None:
    """Mount built frontend assets and root route, with clear failure mode."""
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(static_dir)
    if not static_dir.exists():
        return

    index_file = static_dir / "index.html"
    assets_dir = static_dir / "assets"

    if index_file.exists() and assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/")
        async def serve_ui():
            return FileResponse(str(index_file))

        return

    logger.warning(
        "Frontend build artifacts are missing (expected %s and %s). Run `just build`.",
        index_file,
        assets_dir,
    )

    @app.get("/")
    async def serve_ui_unavailable():
        return HTMLResponse(
            (
                "<h1>Frontend build missing</h1>"
                "<p>Run <code>just build</code> and restart the server.</p>"
            ),
            status_code=503,
        )


def create_app(model_id: str) -> FastAPI:
    r"""Build and configure the FastAPI application instance.

    Sets up the OpenAI-compatible API router, frontend static file serving,
    and model lifespan management.

    Args:
        model_id (str): HuggingFace model identifier to load at startup.

    Returns:
        FastAPI: Fully configured application ready for ``uvicorn.run()``.
    """
    app = FastAPI(
        title="Local LLM Server",
        description=(
            "OpenAI-compatible REST API for local MLX models on Apple Silicon.\n\n"
            "**Patterns**: Facade · Iterator · Strategy · Decorator · Observer · Adapter"
        ),
        version="1.0.0",
        lifespan=make_lifespan(model_id),
    )
    app.include_router(router)
    configure_frontend(app, Path(__file__).parent / "static")

    return app


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    r"""Server entrypoint — resolve config, check RAM feasibility, and start uvicorn.

    Priority for settings: CLI flag > env var / ``.env`` > built-in default.
    Exits with code 1 if the requested model exceeds available RAM.
    """
    args = parse_args()

    # CLI flag > env var / .env > built-in default (model selection priority)
    model_id = args.model or settings.model_id
    host = args.host or settings.host
    port = args.port or settings.port

    # ── RAM feasibility check (before uvicorn starts) ─────────────────────────
    # ModelTooLargeError is caught here so we exit cleanly with a friendly
    # message, never entering uvicorn's lifespan machinery.
    try:
        from app.core.model_manager import ModelManager
        ModelManager()._check_ram(model_id)
    except ModelTooLargeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    logger.info("Starting server — model: %s  host: %s  port: %d", model_id, host, port)
    logger.info("Swagger UI: http://%s:%d/docs", host if host != "0.0.0.0" else "localhost", port)

    app = create_app(model_id)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
