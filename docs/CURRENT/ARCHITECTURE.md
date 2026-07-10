# Architecture

Last updated: 2026-05-31

## System Shape Today

Maic is a two-part local-first system:
- Backend: FastAPI service for chat completion + model lifecycle APIs.
- Frontend: React SPA for chat and model controls.

The production frontend build is served from `static/` by the backend.

## Backend

- Entry point: `main.py`
- API layer: `maic/api/`
- Core runtime and generation: `maic/core/`
- OpenAI wire-format adapters: `maic/adapters/`
- Pydantic contracts: `maic/schemas/`

Boundary rules:
- Route handlers map transport concerns only.
- Business logic stays in `maic/core`.
- OpenAI response shaping stays in `maic/adapters`.

### Tool-calling flow

```
Model load
  → detect_tool_call_format(tokenizer.chat_template) → ToolCallFormat enum
  → stored as model_manager.tool_call_format
  → model_manager.supports_tool_calling (bool) exposed for capability advertising

Client request (with tools=[...])
  → routes.py: tools_to_dicts() → model_manager.generate(tools=...)
  → chat template injects tool signatures into prompt
  → model generates tool-call text (format depends on model family)
  → routes.py: OpenAIAdapter.parse_tool_calls(text, model_manager.tool_call_format)
      dispatches to format-specific parser:
        QWEN / HERMES  → <tool_call>…</tool_call> blocks
        LLAMA3         → <|python_tag|>[{…}] JSON array
        MISTRAL        → [TOOL_CALLS] [{…}] JSON array
        DEEPSEEK       → <|tool▁calls▁begin|>…<|tool▁sep|>…<|tool▁calls▁end|>
        UNKNOWN        → fallback: XML tag → fenced block → bare JSON
      all paths tolerate doubled-brace artifact {{...}} and "parameters" vs "arguments"
  → build_tool_response / tool_call_to_sse → client sees finish_reason="tool_calls"
```

Tool results (`role:"tool"`) flow back through `messages_to_dicts` as
`<tool_response>` entries in the next prompt.

### Supported tool-calling model families

| Family | Format enum | Template marker |
|--------|-------------|-----------------|
| Qwen2.5 / Qwen3 / Kimi K2 | `QWEN` | `<tool_call>` |
| NousHermes / Hermes-2 | `HERMES` | `<tool_response>` |
| Llama-3.x Instruct | `LLAMA3` | `<\|python_tag\|>` |
| Mistral / Mixtral Instruct | `MISTRAL` | `[TOOL_CALLS]` |
| DeepSeek-Coder-V2 | `DEEPSEEK` | `tool▁calls▁begin` |

Any model whose template contains a `tools` variable but doesn't match the above
gets `UNKNOWN`, which triggers the fallback chain so it may still work.

### Capability advertisement

- `GET /v1/models` → `ModelCard.capabilities = {"tool_calling": bool}`
- `GET /v1/models/status` → `ModelStatus.tool_calling = bool | null`
- `GET /v1/models/supported` → `SupportedModel.capabilities = {"tool_calling": bool} | null`

For loaded models the value comes from the in-memory `model_manager.tool_call_format`.
For downloaded (not loaded) models the `tokenizer_config.json` is read from disk.
For not-yet-downloaded models the field is `null` / omitted.

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
