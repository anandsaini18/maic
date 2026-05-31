# Status

Last updated: 2026-05-31

## Current State

- Backend serves a local OpenAI-compatible chat endpoint at `/v1/chat/completions`.
- **Multi-model tool-calling support**: Any `mlx-community` model whose chat template supports
  function calling (Qwen2.5, Qwen3, Kimi K2, Llama-3.x, Mistral/Mixtral, DeepSeek-Coder-V2,
  Hermes/NousHermes) works with OpenCode's agentic tool-calling protocol out of the box.
- Tool-call format is auto-detected at model-load time from the chat template; format-specific
  parsers replace the previous try-all fallback for known model families.
- `GET /v1/models` advertises `capabilities.tool_calling` so clients (OpenCode, frontend) can
  discover tool support without trial inference.
- Frontend model selector shows a "tools" badge for models known to support function calling.
- Frontend is a React + Vite app built into `static/` and served by FastAPI.
- Model lifecycle operations (status, download, load, delete) are available from backend routes used by the UI.
- Test coverage exists for routes, schemas, adapters, token streaming, model manager, and frontend static serving.

## In Progress

- Adopted Notdefined documentation framework (`docs/CURRENT`, `docs/DECISIONS`, `docs/SPECS`) for repository management.

## Known Constraints

- Primary target platform is Apple Silicon macOS with MLX runtime.
- Offline/private behavior depends on model assets being available locally.
- `tool_calling` capability is `null` for models not yet downloaded (template unreadable).

## Recently Shipped

- **Multi-model tool-calling with format detection** (feature/support-opencode):
  - `maic/core/model_manager.py`: `ToolCallFormat` enum (`QWEN`, `LLAMA3`, `MISTRAL`,
    `DEEPSEEK`, `HERMES`, `UNKNOWN`, `NONE`); `detect_tool_call_format(chat_template)`
    pattern-matches the Jinja2 template source; `ModelManager.tool_call_format` property
    stores the result at load time; `supports_tool_calling` bool property; `tool_call_format_for(model_id)`
    reads disk for non-loaded downloaded models.
  - `maic/adapters/openai_adapter.py`: `parse_tool_calls(text, fmt)` now dispatches to
    format-specific sub-parsers (`_parse_qwen`, `_parse_llama3`, `_parse_mistral`,
    `_parse_deepseek`, `_parse_fallback`); fallback chain preserved as `UNKNOWN` path.
  - `maic/schemas/openai.py`: `ModelCard.capabilities`, `SupportedModel.capabilities`
    (`dict[str, bool] | None`); `ModelStatus.tool_calling` (`bool | None`).
  - `maic/api/routes.py`: `GET /v1/models` includes `capabilities`; `GET /v1/models/status`
    includes `tool_calling`; tool-call parse call passes `model_manager.tool_call_format`.
  - `frontend/src/api/types.ts`: `Model.tool_calling?: boolean | null`.
  - `frontend/src/components/inspector/ModelOption.tsx`: "tools" badge when `tool_calling === true`.

- **OpenAI tool-calling support** (initial, enabling OpenCode TUI):
  - `maic/schemas/openai.py`: added `Tool`, `ToolCall`, `FunctionCall`, `FunctionDefinition`,
    `DeltaToolCall`, `DeltaToolCallFunction`; `Message` now accepts `tool` role, optional
    `tool_calls`, and structured/list `content`; request carries `tools`/`tool_choice`;
    finish-reason literals include `"tool_calls"`.
  - `maic/adapters/openai_adapter.py`: `messages_to_dicts` handles tool-call history and
    tool-role messages; `tools_to_dicts` converts tool defs for the chat template.
  - `maic/core/model_manager.py`: `generate()` accepts `tools=` and forwards it to
    `apply_chat_template`, so tool definitions are injected into the prompt.
  - `maic/api/routes.py`: tool turns are buffered, parsed, then returned (or replayed as SSE)
    with `finish_reason="tool_calls"`.

## Historically Shipped

- Initial repo documentation migration to Notdefined framework structure.
- Streaming reliability hardening:
  - backend now limits pathological stop-marker lookback to avoid end-only buffering
  - backend now falls back to token-level decode when MLX emits empty text segments
  - frontend SSE consumer now parses full SSE events (`\n\n` framed) for robust incremental rendering
  - SSE transport now uses explicit no-transform/keep-alive headers and async flush points per chunk
  - frontend requests now send `Accept: text/event-stream` for streaming calls
- Added `just dev-built` to build frontend and start backend in one command
- Release workflow now triggers on pushes to `main`, derives the package version from `pyproject.toml`, creates the corresponding git tag, and publishes artifacts to GitHub Releases + PyPI (skipping if that version tag already exists)
