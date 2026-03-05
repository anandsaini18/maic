# AGENTS.md (Frontend)

Scope: `frontend/`.

This file overrides root guidance for frontend-specific decisions.

## Canonical Read Order

When working under `frontend/`, use this order:
1) `frontend/AGENTS.md`
2) root `AGENTS.md`
3) `docs/CURRENT/STATUS.md`
4) `docs/CURRENT/ARCHITECTURE.md`
5) `docs/CURRENT/ROADMAP.md`

## Frontend Overview

Stack:
- React 19 + TypeScript
- Vite build/dev server
- Tailwind + CSS variables for theming

Core flow:
- `hooks/useChat.ts`: chat state, streaming consumption, message lifecycle
- `hooks/useModels.ts`: model polling and lifecycle actions
- `api/client.ts`: backend HTTP calls
- `components/`: UI composition and presentation

## Frontend Architecture Rules

- Keep network logic in `src/api/client.ts`; do not call `fetch` directly from components.
- Keep shared API/domain types in `src/api/types.ts`.
- Keep business/state logic in hooks; components should primarily render props.
- Keep markdown rendering through `src/lib/markdown.ts` only.
- Any use of `dangerouslySetInnerHTML` must use sanitized HTML output.

## UI And Styling Conventions

- Reuse existing design tokens and CSS variables before adding new raw colors.
- Prefer existing shared components (`SliderGroup`, `ToggleRow`, etc.) over one-off variants.
- Keep layout responsive for desktop + mobile; avoid fixed-width-only assumptions.
- Keep animations purposeful and lightweight; avoid gratuitous motion.

## Contract-Sync Rules

When adding/changing a setting in UI:
- Update `GenerationSettings` type in `src/api/types.ts`
- Update payload mapping in `hooks/useChat.ts`
- Confirm backend schema supports it; if not, add backend support in same change

When changing model management UI behavior:
- Keep parity with `/v1/models/status`, `/v1/models/download`, `/v1/models/load`, `/v1/models/{id}`

## Frontend Verification

Run before completion:
- `cd frontend && npm run lint`
- `cd frontend && npm run build`

If API interactions changed, also run relevant backend route tests.
