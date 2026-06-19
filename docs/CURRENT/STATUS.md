# Status

Last updated: 2026-03-10

## Current State

- Backend serves a local OpenAI-compatible chat endpoint at `/v1/chat/completions`.
- Frontend is a React + Vite app built into `static/` and served by FastAPI.
- Model lifecycle operations (status, download, load, delete) are available from backend routes used by the UI.
- Test coverage exists for routes, schemas, adapters, token streaming, model manager, and frontend static serving.

## In Progress

- Adopted Notdefined documentation framework (`docs/CURRENT`, `docs/DECISIONS`, `docs/SPECS`) for repository management.

## Known Constraints

- Primary target platform is Apple Silicon macOS with MLX runtime.
- Offline/private behavior depends on model assets being available locally.

## Recently Shipped

- Initial repo documentation migration to Notdefined framework structure.
- Streaming reliability hardening:
  - backend now limits pathological stop-marker lookback to avoid end-only buffering
  - backend now falls back to token-level decode when MLX emits empty text segments
  - frontend SSE consumer now parses full SSE events (`\n\n` framed) for robust incremental rendering
  - SSE transport now uses explicit no-transform/keep-alive headers and async flush points per chunk
  - frontend requests now send `Accept: text/event-stream` for streaming calls
- Added `just dev-built` to build frontend and start backend in one command
- Release workflow now triggers on pushes to `main`, derives the package version from `pyproject.toml`, creates the corresponding git tag, and publishes artifacts to GitHub Releases + PyPI (skipping if that version tag already exists)
