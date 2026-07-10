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
- Package: `maic/` (installable via `pip install maic`; entry point `maic` CLI)
- Backend: FastAPI + MLX model runtime; fallback entrypoint `main.py` for raw uvicorn
- Frontend: React + TypeScript + Vite in `frontend/`; production bundle served from `static/`
- Preferred server: `maic run` delegates to `mlx-openai-server` subprocess when available, falls back to `mlx_lm.server`, then to the built-in FastAPI server

Primary product goals:
- OpenAI-compatible chat API (`/v1/chat/completions`) with tool/function calling and thinking/reasoning mode
- Local model lifecycle management (download, load, delete, quantize)
- OpenCode-ready: `maic setup opencode` writes the provider config automatically
- Private/offline-first execution after initial model download

## CLI Commands

All commands are under the `maic` entry point (defined in `pyproject.toml` → `maic.cli:app`):

| Command | Purpose |
|---|---|
| `maic run [--model ALIAS] [--port PORT]` | Start inference server; interactive model picker if >1 installed |
| `maic models list [--tool-calling] [--max-ram N]` | Browse curated registry with RAM/tool-call indicators |
| `maic models pull ALIAS_OR_HF_ID` | Download a model from HuggingFace Hub |
| `maic setup opencode [--port PORT]` | Write Maic provider into `~/.config/opencode/opencode.json` |
| `maic doctor [--port PORT]` | Check Python, MLX, mlx-openai-server, server liveness, OpenCode config, installed models, RAM |

## API Endpoints

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | Returns active model + `capabilities: {tool_calling: bool}` |
| `GET /v1/models/supported` | Curated list with RAM requirements and feasibility |
| `GET /v1/models/status` | Full status for all Hub + local models (downloading, active, tool_calling) |
| `POST /v1/chat/completions` | Streaming + non-streaming; tool calls; `reasoning_content` delta in thinking mode |
| `POST /v1/models/download` | Background download; returns immediately |
| `POST /v1/models/load` | Switch active model |
| `DELETE /v1/models/{model_id}` | Remove weights from disk |
| `POST /v1/models/quantize` | Background quantization |
| `POST /v1/cache/clear` | Reset KV prompt cache |
| `GET /v1/settings` / `POST /v1/settings` | Runtime KV cache + batch mode settings |

## Model Capability Detection

`ToolCallFormat` enum in `maic/core/model_manager.py`, detected at model-load time by inspecting the tokenizer chat template:

| Format | Detection marker |
|---|---|
| `QWEN` | `<tool_call>` blocks (Qwen, Kimi K2) |
| `LLAMA3` | `<\|python_tag\|>` |
| `MISTRAL` | `[TOOL_CALLS]` |
| `DEEPSEEK` | `tool▁calls▁begin` (U+2581 word-joiner tokens) |
| `HERMES` | `<tool_response>` (NousHermes variant) |
| `UNKNOWN` | `{% if tools %}` present but no known marker → fallback parser |
| `NONE` | No tool-calling markers in template |

`model_manager.supports_tool_calling` is `True` for any format except `NONE` and `UNKNOWN`.
`model_manager.supports_thinking` is `True` when `enable_thinking` appears in the chat template (Qwen3 family).

## Curated Model Registry

8 models in `maic/core/model_registry.py` (aliases, HF IDs, RAM tiers, tool-calling flag).
Recommended for OpenCode: `qwen-coder-7b` (`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`).

## Repo-Wide Conventions

- Keep diffs minimal and localized.
- Preserve OpenAI compatibility for existing endpoints unless explicitly changing API behavior.
- Treat all external inputs as untrusted (request body, model IDs, filesystem paths, streamed payloads).
- Keep route handlers thin; business logic belongs in `maic/core` and format translation in `maic/adapters`.
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
- System health: `maic doctor`
- Backend-only changes: `just test-quick`
- Route/schema/backend contract changes: `just test` (or targeted `pytest` file)
- Frontend-only changes: `cd frontend && npm run lint && npm run build`
- Cross-stack changes: `just ci`

## Delivery Standard

Before marking work complete, report:
- What changed
- Which checks were run
- Any known gaps or follow-up work
