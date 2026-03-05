# AGENTS.md (Backend)

Scope: `app/`.

This file overrides root guidance for backend-specific decisions.

## Canonical Read Order

When working under `app/`, use this order:
1) `app/AGENTS.md`
2) root `AGENTS.md`
3) `docs/CURRENT/STATUS.md`
4) `docs/CURRENT/ARCHITECTURE.md`
5) `docs/CURRENT/ROADMAP.md`

## Backend Overview

Stack:
- FastAPI routes in `api/`
- Pydantic schemas in `schemas/`
- Core model/runtime logic in `core/`
- OpenAI response translation in `adapters/`

Primary responsibilities:
- Serve OpenAI-compatible chat endpoint
- Manage local model download/load/delete lifecycle
- Stream token output safely and consistently

## Backend Architecture Boundaries

- `api/routes.py`: transport + HTTP mapping only
- `api/decorators.py`: reusable route guards/timing wrappers
- `core/model_manager.py`: model lifecycle + generation behavior
- `core/token_stream.py`: iterator and generation metrics
- `adapters/openai_adapter.py`: wire-format translation only
- `schemas/openai.py`: request/response contracts

Do not move heavy generation/business logic into routes or adapters.

## API And Contract Rules

- All request/response shapes must be explicit in Pydantic schemas.
- Maintain compatibility for current OpenAI-style fields unless intentionally versioning a change.
- Map internal exceptions to stable HTTP codes/messages.
- For streaming changes, preserve SSE contract (`data: ...` chunks + `[DONE]`).

## Safety And Validation

- Treat request payloads and model identifiers as untrusted.
- Keep path handling normalized and constrained to configured model directory.
- Preserve RAM feasibility guardrails before expensive model operations.
- Avoid exposing raw internal errors directly when a safer user-facing message is available.

## Testing Rules

For backend behavior changes:
- Add/update targeted tests in `tests/`.
- Prefer route integration coverage for endpoint behavior (`tests/test_routes.py`).
- Keep unit tests for core logic (`test_model_manager.py`, `test_token_stream.py`, `test_openai_adapter.py`).

## Backend Verification

Run the narrowest relevant checks:
- `just test-quick`
- Targeted pytest file(s) for changed module(s)
- `just lint-python` when touching typed or shared backend code
