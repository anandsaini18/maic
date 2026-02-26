from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager

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

def create_app(model_id: str) -> FastAPI:
    app = FastAPI(
        title="Local LLM Server",
        description=(
            "OpenAI-compatible REST API for local MLX models on Apple Silicon.\n\n"
            "**Patterns**: Facade · Iterator · Strategy · Decorator · Observer · Adapter"
        ),
        version="0.1.0",
        lifespan=make_lifespan(model_id),
    )
    app.include_router(router)

    # Serve the Vite-built frontend at the root
    from pathlib import Path
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/")
        async def serve_ui():
            return FileResponse(str(static_dir / "index.html"))

    return app


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
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
