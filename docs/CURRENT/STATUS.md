# Status

Last updated: 2026-03-05

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
