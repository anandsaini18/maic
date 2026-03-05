# Architecture

Last updated: 2026-03-05

## System Shape Today

Maic is a two-part local-first system:
- Backend: FastAPI service for chat completion + model lifecycle APIs.
- Frontend: React SPA for chat and model controls.

The production frontend build is served from `static/` by the backend.

## Backend

- Entry point: `main.py`
- API layer: `app/api/`
- Core runtime and generation: `app/core/`
- OpenAI wire-format adapters: `app/adapters/`
- Pydantic contracts: `app/schemas/`

Boundary rules:
- Route handlers map transport concerns only.
- Business logic stays in `app/core`.
- OpenAI response shaping stays in `app/adapters`.

## Frontend

- Application code: `frontend/src/`
- API client and types: `frontend/src/api/`
- Stateful orchestration: `frontend/src/hooks/`
- Presentational components: `frontend/src/components/`

## Verification Topology

- Backend and contract tests: `tests/`
- Quick backend checks: `just test-quick`
- Full stack checks: `just ci`
- Frontend checks: `cd frontend && npm run lint && npm run build`
