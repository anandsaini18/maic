# Maic Developer Guide

**Status:** Current  
**Last updated:** 2026-05-31  
**Audience:** Developers installing and integrating Maic

---

## Table of Contents

1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Model Management](#5-model-management)
6. [Running the Server](#6-running-the-server)
7. [OpenCode Integration](#7-opencode-integration)
8. [System Health Check](#8-system-health-check)
9. [API Reference](#9-api-reference)
10. [Supported Model Families](#10-supported-model-families)
11. [Development Setup](#11-development-setup)
12. [Known Limitations](#12-known-limitations)

---

## 1. Overview

Maic is a local LLM inference server for Apple Silicon (MLX backend). It exposes an OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`) and a `maic` CLI for model management, server startup, and integration setup.

Primary use case: run large language models entirely on-device and connect tools such as OpenCode, Cursor, or any OpenAI-SDK-compatible client to them without external API calls or internet access after the initial model download.

Key properties:

- OpenAI-compatible API — drop-in replacement for `https://api.openai.com` in any SDK
- MLX-powered — runs on Apple Silicon unified memory; no GPU required
- Tool calling — QWEN, LLAMA3, MISTRAL, and DEEPSEEK model families support function/tool calling
- Thinking/reasoning — Qwen3 family supports `reasoning_content` deltas in streaming mode
- OpenCode-ready — `maic setup opencode` writes the provider config automatically

---

## 2. Requirements

| Requirement | Minimum |
|---|---|
| Hardware | Apple Silicon (M1, M2, M3, or M4) |
| macOS | 13 Ventura or later |
| Python | 3.10+ |
| RAM | 8 GB (16 GB recommended for coding workloads) |
| Disk | 5–10 GB free per model |

Intel Macs and Linux are not supported. MLX only runs on Apple Silicon.

---

## 3. Installation

Install from PyPI with the MLX optional dependency group:

```bash
pip install "maic[mlx]"
```

The `[mlx]` group installs `mlx`, `mlx-lm`, and `mlx-openai-server`. Omitting it installs only the CLI and FastAPI server without the inference backend — useful for CI or non-Apple machines.

Verify the installation:

```bash
maic --version
maic doctor
```

---

## 4. Quick Start

Five commands from a clean install to a working inference server:

```bash
pip install "maic[mlx]"
maic doctor
maic models list
maic models pull qwen-coder-7b
maic run
```

Step-by-step:

1. `pip install "maic[mlx]"` — installs Maic and the MLX inference backend
2. `maic doctor` — confirms Python version, MLX availability, and system RAM
3. `maic models list` — shows all curated models with size and RAM requirements
4. `maic models pull qwen-coder-7b` — downloads `qwen-coder-7b` to `~/models/`
5. `maic run` — starts the inference server on `http://127.0.0.1:8001`

---

## 5. Model Management

### Listing models

```bash
maic models list
```

Displays all curated models. Columns: installation status, alias, HuggingFace ID, disk size, minimum RAM, tool calling support, description. Models requiring more RAM than the system has are dimmed but can still be attempted on unified-memory Macs.

Filter to tool-capable models only:

```bash
maic models list --tool-calling
```

Filter by maximum RAM budget:

```bash
maic models list --max-ram 16
```

### Downloading a model

```bash
maic models pull <alias-or-hf-id>
```

Pass a curated alias (recommended) or any raw HuggingFace repo ID:

```bash
maic models pull qwen-coder-7b
maic models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

Both forms resolve to the same download. After download completes, `pull` also creates a HuggingFace cache symlink so that `mlx_lm` and other tools that read from the HF cache find the model automatically.

### Model storage location

Models are stored at `~/models/<org>--<repo>/`. Example:

```
~/models/mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/
```

The HuggingFace ID is reconstructed from the directory name by replacing the first `--` with `/`.

### Curated model registry

| Alias | HuggingFace ID | Size | Min RAM | Tool Calling | Notes |
|---|---|---|---|---|---|
| `phi-mini` | `mlx-community/Phi-3.5-mini-instruct-4bit` | 2.3 GB | 6 GB | No | Chat-only. Best for 8 GB Macs. |
| `qwen-4b` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 2.5 GB | 6 GB | Yes | Smallest tool-capable model. |
| `mistral-7b` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4.0 GB | 8 GB | Yes | Stable tool calling. |
| `qwen-coder-7b` | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB | 8 GB | Yes | Recommended for OpenCode on 16 GB Macs. |
| `qwen-8b` | `mlx-community/Qwen3-8B-4bit` | 5.0 GB | 8 GB | Yes | Qwen3 with thinking mode. |
| `deepseek-r1-14b` | `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | 8.5 GB | 14 GB | Yes | DeepSeek R1 reasoning + coding. |
| `deepseek-coder` | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit` | 9.0 GB | 14 GB | Yes | Best code completion on 16 GB Macs. |
| `qwen-14b` | `mlx-community/Qwen3-14B-4bit` | 8.5 GB | 16 GB | Yes | Strong reasoning for 32 GB Macs. |

---

## 6. Running the Server

### Start with interactive model picker

```bash
maic run
```

If exactly one model is installed, it is selected automatically. If multiple models are installed, an arrow-key selection prompt is displayed.

### Start with a specific model

```bash
maic run --model qwen-coder-7b
maic run --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

### Custom port and host

```bash
maic run --port 8001 --host 127.0.0.1
```

Default: `--host 127.0.0.1 --port 8001`.

To bind to all interfaces (for LAN access):

```bash
maic run --host 0.0.0.0 --port 8001
```

### Server startup strategy

`maic run` selects a server backend in priority order:

| Priority | Backend | Condition | Tool Calling |
|---|---|---|---|
| 1 | `mlx-openai-server` subprocess | Binary found outside current venv | Yes (via Maic adapter) |
| 2 | `mlx_lm.server` module | `mlx_lm` importable | No (native) |
| 3 | Maic built-in FastAPI | Fallback | Yes |

Strategy 3 (Maic FastAPI) is always available and is the only backend that supports tool calling natively. `pip install "maic[mlx]"` installs `mlx-openai-server` which triggers Strategy 1 in most setups.

### Verify the server is running

```bash
curl http://localhost:8001/v1/models
```

A successful response returns a JSON object listing the currently loaded model:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
      "object": "model",
      "owned_by": "maic",
      "capabilities": {
        "tool_calling": true,
        "thinking": false
      }
    }
  ]
}
```

---

## 7. OpenCode Integration

OpenCode is a terminal-based AI coding assistant. This section covers configuring OpenCode to use Maic as its inference provider.

### Step 1: Install OpenCode

```bash
npm install -g opencode-ai
```

Verify:

```bash
opencode --version
```

### Step 2: Start Maic

```bash
maic run --model qwen-coder-7b
```

Leave this running in a terminal. Confirm the server is up:

```bash
curl http://localhost:8001/v1/models
```

### Step 3: Auto-configure the provider

```bash
maic setup opencode
```

This command writes (or merges into) `~/.config/opencode/opencode.json` with:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "maic": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Maic (local MLX)",
      "options": {
        "baseURL": "http://localhost:8001/v1",
        "apiKey": "maic-local"
      }
    }
  }
}
```

The command merges into any existing config — it does not overwrite unrelated providers.

If the existing config has invalid JSON, the original file is backed up to `opencode.json.bak` before writing.

### Step 4: Register models for the OpenCode TUI

**This step is required.** OpenCode does not auto-discover models from `/v1/models` for custom providers. Each model you want to appear in the OpenCode model picker must be explicitly listed in the config under the `models` key.

Open `~/.config/opencode/opencode.json` and add a `models` block inside the `maic` provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "maic": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Maic (local MLX)",
      "options": {
        "baseURL": "http://localhost:8001/v1",
        "apiKey": "maic-local"
      },
      "models": {
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": {
          "name": "Qwen2.5 Coder 7B (local)"
        },
        "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit": {
          "name": "DeepSeek R1 14B (local)"
        }
      }
    }
  }
}
```

Rules for the `models` block:

- The key must be the exact HuggingFace model ID (same value returned by `/v1/models`).
- The `name` string is what appears in the OpenCode model picker.
- Add one entry per downloaded model.
- After editing, restart OpenCode for the changes to take effect.

### Step 5: Verify in OpenCode

Launch OpenCode:

```bash
opencode
```

Open the model selector (default keybind: `/` or `m` depending on version). The `Maic (local MLX)` provider and each registered model should appear.

### Using a non-default port

If Maic runs on a port other than `8001`:

```bash
maic setup opencode --port 8002
```

Then update the `baseURL` in the `models` block entries to match.

---

## 8. System Health Check

```bash
maic doctor
```

Runs a set of checks and prints a status table. Each check and its fix:

| Check | Pass condition | Fix when failing |
|---|---|---|
| Python >= 3.10 | Python 3.10 or newer in PATH | Install Python 3.10+ |
| MLX available | `import mlx.core` succeeds | `pip install "maic[mlx]"` |
| mlx-openai-server | Binary found in PATH | `pip install mlx-openai-server` |
| Server running | `/v1/models` responds on the given port | `maic run --model <alias>` |
| OpenCode config | `~/.config/opencode/opencode.json` exists | `maic setup opencode` |
| OpenCode maic provider | `provider.maic` key present in config | `maic setup opencode` |
| Models installed | At least one model in `~/models/` | `maic models pull <alias>` |
| RAM detected | `psutil` reports total RAM | `pip install psutil` |

Check a non-default port:

```bash
maic doctor --port 8002
```

---

## 9. API Reference

The Maic server exposes an OpenAI-compatible REST API. All endpoints are prefixed with `/v1`.

### GET /v1/models

Returns the currently loaded model with its capabilities.

```bash
curl http://localhost:8001/v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
      "object": "model",
      "owned_by": "maic",
      "capabilities": {
        "tool_calling": true,
        "thinking": false
      }
    }
  ]
}
```

### POST /v1/chat/completions

Standard chat completions endpoint. Supports both streaming and non-streaming responses.

**Supported request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Model ID (ignored if a model is already loaded) |
| `messages` | array | Yes | Conversation history |
| `stream` | boolean | No | `true` for SSE streaming, `false` for single response |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `temperature` | float | No | Sampling temperature (0.0–2.0) |
| `top_p` | float | No | Nucleus sampling threshold |
| `tools` | array | No | Tool/function definitions (requires tool-calling model) |
| `tool_choice` | string/object | No | Tool selection mode (`auto`, `none`, or specific tool) |

**Non-streaming request:**

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Write a Python hello world."}],
    "stream": false
  }'
```

**Streaming request:**

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Write a Python hello world."}],
    "stream": true
  }'
```

**Usage object in non-streaming response:**

```json
{
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170,
    "tokens_per_second": 34.7
  }
}
```

`tokens_per_second` is a Maic extension; standard OpenAI clients ignore unknown fields.

**Tool calling example:**

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }]
  }'
```

**Thinking/reasoning mode:**

For Qwen3 models with thinking support, a `reasoning_content` field appears in streaming deltas:

```json
{"delta": {"reasoning_content": "Let me think about this..."}}
{"delta": {"content": "The answer is..."}}
```

### Additional endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/models/supported` | GET | Curated model list with RAM requirements and feasibility |
| `/v1/models/status` | GET | Status for all Hub and local models (downloading, active, tool_calling) |
| `/v1/models/download` | POST | Start a background model download; returns immediately |
| `/v1/models/load` | POST | Switch the active model |
| `/v1/models/{model_id}` | DELETE | Remove model weights from disk |
| `/v1/models/quantize` | POST | Start background quantization |
| `/v1/cache/clear` | POST | Reset KV prompt cache |
| `/v1/settings` | GET | Read runtime settings (KV cache, batch mode) |
| `/v1/settings` | POST | Update runtime settings |

---

## 10. Supported Model Families

Maic detects tool-calling capability at model-load time by inspecting the tokenizer chat template. The detection markers and resulting behavior:

| Family | Detection marker in chat template | Tool Calling | Thinking | Example models |
|---|---|---|---|---|
| QWEN | `<tool_call>` | Yes | Qwen3 only | Qwen2.5-Coder-7B, Qwen3-8B, Qwen3-14B |
| LLAMA3 | `<\|python_tag\|>` | Yes | No | Llama-3.1-8B-Instruct, Llama-3.3-70B |
| MISTRAL | `[TOOL_CALLS]` | Yes | No | Mistral-7B-Instruct-v0.3, Mistral-Nemo |
| DEEPSEEK | `tool_calls_begin` (U+2581 tokens) | Yes | No | DeepSeek-Coder-V2-Lite |
| HERMES | `<tool_response>` | Yes | No | NousHermes variants |
| UNKNOWN | `{% if tools %}` present, no known marker | No | No | Fallback; tool calls not attempted |
| NONE | No tool-calling markers | No | No | Phi-3.5-mini, Gemma |

`model_manager.supports_tool_calling` is `True` for any format except `NONE` and `UNKNOWN`.  
`model_manager.supports_thinking` is `True` when `enable_thinking` appears in the chat template (Qwen3 family).

The DeepSeek R1 14B distillation is based on Qwen weights and is detected as `QWEN` family.

---

## 11. Development Setup

This section applies to contributors building from source. End-users install via `pip install "maic[mlx]"` and do not need these steps.

### Prerequisites

- `just` task runner: `brew install just`
- Python 3.10+

### Clone and install

```bash
git clone <repo-url>
cd llm
just setup
```

`just setup` creates a virtualenv, installs all dependencies including `.[dev,mlx]`, and installs frontend npm packages.

### Common tasks

| Command | Action |
|---|---|
| `just dev` | Start the FastAPI backend with uvicorn (auto-reload) |
| `just test` | Run the full pytest suite with coverage |
| `just test-quick` | Run pytest without coverage (faster) |
| `just lint` | Run ruff and mypy |
| `just fmt` | Auto-format with ruff and black |
| `just ci` | Full lint + test pipeline (mirrors CI) |

### Project structure

```
maic/               Installable Python package
  cli.py            CLI entry point (maic command)
  api/routes.py     FastAPI endpoints
  api/decorators.py require_model, timing decorators
  adapters/openai_adapter.py  SSE streaming, response shaping
  core/model_manager.py       MLX model lifecycle (singleton)
  core/token_stream.py        Generator wrapper with timing stats
  core/config.py              Pydantic settings + generation strategies
  core/model_registry.py      Curated model list
  schemas/openai.py           Request/response Pydantic models
frontend/           React + Vite chat UI
tests/              pytest test suite
docs/               Project documentation
```

### Dependency conventions

- All config lives in `pyproject.toml` (pytest, ruff, black, mypy — no separate config files).
- `mlx-lm` is an optional dependency (`[mlx]` group). It only works on macOS/Apple Silicon.
- CI installs `.[dev]` (no mlx). Local dev installs `.[dev,mlx]`.
- Do not add dependencies, change lockfiles, or change toolchains without approval.

### Coding conventions

- Target Python 3.10+. Use `from __future__ import annotations` at the top of every file.
- Import from `collections.abc`, not `typing`, for `AsyncGenerator`, `Generator`, `Callable`.
- `mlx_lm` and `mlx.core` must be lazy-imported inside methods. Importing at module level breaks Linux CI.
- Use `raise ... from exc` in except clauses (ruff B904).
- Ruff config: `select = ["E", "W", "F", "I", "B", "UP"]`, line-length 100.

### Testing conventions

- Prefer integration tests through `TestClient` over isolated unit tests.
- Test behavior (HTTP request to response), not implementation details.
- Patch `maic.core.model_manager.model_manager` (the source), not at the import site.
- Patch `psutil.virtual_memory` directly, not `maic.api.routes.psutil`.
- `temperature=0.0` is valid (not missing). Use `is not None` checks, not truthiness.

---

## 12. Known Limitations

| Limitation | Detail |
|---|---|
| Single model at a time | Only one model can be loaded in memory. Switching models requires restarting the server or calling `POST /v1/models/load`. |
| RAM constraint for long contexts | On 16 GB Macs, models above ~8–9 GB may OOM when processing long context windows. OpenCode sends the full workspace, which can be several thousand tokens. |
| Tool calling and mlx_lm.server | `mlx_lm.server` (Strategy 2) does not support tool calling natively. Tool calling requires Maic's built-in FastAPI server (Strategy 3). `pip install "maic[mlx]"` installs `mlx-openai-server` which uses Strategy 1 and routes tool calls through the Maic adapter. |
| OpenCode model registration | OpenCode does not auto-discover models from `/v1/models` for custom providers. Each model must be manually added to the `models` block in `~/.config/opencode/opencode.json`. See [Section 7, Step 4](#step-4-register-models-for-the-opencode-tui). |
| Apple Silicon only | MLX does not run on Intel Macs or Linux. The CLI and FastAPI server install on any platform, but inference requires Apple Silicon. |
