# AGENTS.md (Root)

Scope: Entire repository. More specific `AGENTS.md` files in subdirectories override this file.

## Notdefined Framework (Repo Contract)

This repository follows the Notdefined documentation model: keep current reality separate from historical plans.

Canonical truth read order:
1) Most specific `AGENTS.md` in current path
2) `docs/CURRENT/STATUS.md`
3) `docs/CURRENT/ARCHITECTURE.md`
4) `docs/CURRENT/ROADMAP.md`
5) `docs/DECISIONS/*` (ADRs)
6) `docs/SPECS/active/*`

Historical context only (never authoritative):
- `docs/SPECS/done/*`
- `docs/SPECS/archive/*`
- old notes/plans

If docs conflict with running code/config, code/config wins and docs must be updated in the same change.

## Project Overview

Maic is a local-first LLM stack for Apple Silicon:
- Backend: FastAPI + MLX model runtime in `app/` with entrypoint `main.py`
- Frontend: React + TypeScript + Vite in `frontend/`
- Build output: frontend production bundle emitted to `static/` and served by backend

Primary product goals:
- OpenAI-compatible chat API (`/v1/chat/completions`)
- Local model lifecycle management (download, load, delete)
- Private/offline-first execution after model download

## Repo-Wide Conventions

- Keep diffs minimal and localized.
- Preserve OpenAI compatibility for existing endpoints unless explicitly changing API behavior.
- Treat all external inputs as untrusted (request body, model IDs, filesystem paths, streamed payloads).
- Keep route handlers thin; business logic belongs in `app/core` and format translation in `app/adapters`.
- Keep frontend components presentational where possible; stateful behavior belongs in hooks.
- Do not add dependencies, change lockfiles, or change toolchains without approval.

## Documentation Rules (Non-Negotiable)

After any change that affects behavior, architecture, build/dev workflow, dependencies, or platform quirks:
- Update relevant docs in the same change.
- Keep `AGENTS.md` short; put implementation detail in `docs/CURRENT/*` or ADRs.
- Ensure docs describe reality as it runs today (no aspirational text).

Docs taxonomy:
- `docs/CURRENT/`: current truth (`STATUS.md`, `ARCHITECTURE.md`, `ROADMAP.md`)
- `docs/DECISIONS/`: ADRs (Context / Decision / Consequences)
- `docs/SPECS/active/`: active implementation specs
- `docs/SPECS/done/`: shipped specs still accurate at ship time
- `docs/SPECS/archive/`: superseded or stale specs with archive reason

## Verification Matrix

Run the narrowest relevant checks:
- Backend-only changes: `just test-quick`
- Route/schema/backend contract changes: `just test` (or targeted `pytest` file)
- Frontend-only changes: `cd frontend && npm run lint && npm run build`
- Cross-stack changes: `just ci`

## Delivery Standard

Before marking work complete, report:
- What changed
- Which checks were run
- Any known gaps or follow-up work
