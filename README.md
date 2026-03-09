# Maic 🪄  
## Run LLM Models Locally on MacBook (No Subscription • No API Key • Fully Offline)

**Maic** is a high-performance local LLM server optimized for **Apple Silicon (M1, M2, M3, M4)**.

It lets you run large language models directly on your MacBook without:

- ❌ OpenAI API keys  
- ❌ Monthly subscription  
- ❌ Internet dependency  
- ❌ Sending data to external servers  

If you're searching for:

> **How to run LLM models on MacBook locally without any subscription or API key**

Maic is built exactly for that.

---

## 🚀 What Makes Maic Different?

Most local LLM tools are:

- CLI-only  
- Not OpenAI-compatible  
- Hard to integrate  
- Not optimized for Apple Silicon  

Maic provides:

- ✅ MLX-optimized inference engine  
- ✅ OpenAI-compatible API (`/v1/chat/completions`)  
- ✅ Modern web-based chat interface  
- ✅ Real-time monitoring  
- ✅ One-click model downloads  

It is both **developer-friendly and privacy-first**.

---

## 🧠 What Is Maic?

Maic is a local AI inference server built on:

- **MLX (Apple’s machine learning framework)**
- FastAPI backend
- React frontend
- Fully optimized for ARM64 macOS

It turns your MacBook into a **self-hosted AI server**.

---

## 🍏 Why Apple Silicon Optimization Matters

Maic is built specifically for:

- MacBook Pro M1 / M2 / M3 / M4
- MacBook Air M-series  
- ARM64 macOS systems  

Benefits:

- Lower memory usage  
- Higher token throughput  
- Efficient Metal acceleration via MLX  
- Stable performance for local inference  

Ideal for running:

- Llama 3  
- Mistral  
- Phi-3  
- Other MLX-compatible models  

---

## ✨ Features

### 🚀 MLX-Optimized Inference
Blazing-fast inference tuned for Apple Silicon.

### 🔌 OpenAI-Compatible API
Drop-in replacement for OpenAI’s API.

Compatible with:
- LangChain  
- Open WebUI  
- VS Code extensions  
- Custom AI tools  
- Any OpenAI-compatible client  

### 🎨 Modern Web UI
Chat interface built with:
- React  
- Tailwind CSS  
- Framer Motion  

### 📦 Smart Model Management
- One-click HuggingFace downloads  
- Automatic RAM requirement validation  
- Live progress tracking  

### 📊 Real-Time Monitoring
- Tokens per minute (TPM)  
- Memory usage  
- Live performance charts  

### 🛡️ Privacy First
Everything runs locally.

No telemetry.  
No cloud dependency.  
No API billing.

---

## ⚡ How to Run LLM Models Locally on MacBook

### Prerequisites

- Python 3.10+
- Node.js 22+
- [just](https://github.com/casey/just) command runner (`brew install just`)

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/anandsaini18/maic.git
cd maic
just setup
```

This automatically:
- Creates a Python virtual environment (`.venv/`)
- Installs backend dependencies + MLX
- Installs frontend dependencies and builds the UI

### 2️⃣ Start the Server

```bash
just dev
```

Open your browser: **http://localhost:8000**

You now have a fully local LLM running on your Mac.

If `/` shows **Frontend build missing**, generate static assets and restart:

```bash
just build
just dev
```

### Available Commands

Run `just` to see all commands:

| Command | Description |
|---------|-------------|
| `just setup` | Create venv + install everything |
| `just dev` | Start the backend server |
| `just dev-built` | Build frontend, then start backend server |
| `just dev --model mlx-community/Llama-3.2-1B-Instruct-4bit` | Start with a specific model |
| `just dev-frontend` | Start Vite dev server (port 5173) |
| `just build` | Build frontend for production |
| `just test` | Run all tests with coverage |
| `just test-quick` | Fast test run, no coverage |
| `just lint` | Run all linters (Python + frontend) |
| `just fmt` | Auto-format Python code |
| `just ci` | Run the full CI pipeline locally |
| `just clean` | Remove build artifacts and caches |

---

## 🔌 API Example (OpenAI Compatible)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
    "stream": true
  }'
```

Maic works as a local OpenAI replacement.


---

## 🎯 Use Cases

- Run LLM offline on Mac
- Private AI assistant
- Develop AI apps without OpenAI billing
- Local-first AI experimentation
- Secure enterprise prototyping

---

## ⚡ Performance

Benchmarked on **Apple M1 Pro 16GB** — Mistral 7B Instruct v0.3 (4-bit quantized, 4.0 GB), 5 runs per prompt, greedy decoding.

| Metric | Maic (M1 Pro 16GB) | LM Studio (M1 Pro 32GB)¹ |
|--------|-------------------|--------------------------|
| Decode speed — mean | **38.4 tok/s** | 37.1 tok/s |
| Decode speed — median | 39.3 tok/s | — |
| Decode speed — p95 | 40.0 tok/s | — |
| Time to first token — mean | 297 ms | — |

> ¹ LM Studio 0.3.10 reference numbers from [lmstudio-ai/mlx-engine #103](https://github.com/lmstudio-ai/mlx-engine/issues/103), M1 Pro 32GB, MLX backend, no speculative decoding. RAM difference (16 GB vs 32 GB) does not affect decode throughput for models that fit in unified memory.

Maic matches or exceeds LM Studio on identical hardware — **+3.6% faster** on half the RAM.

See [`benchmarks/BENCHMARK_REPORT.md`](benchmarks/BENCHMARK_REPORT.md) for the full breakdown.

---

## 📊 Comparison

| Feature | Maic | Cloud APIs | CLI-only Tools |
|---------|------|------------|----------------|
| No API Key | ✅ | ❌ | ✅ |
| Fully Offline | ✅ | ❌ | ✅ |
| Web UI | ✅ | ❌ | ❌ |
| OpenAI Compatible | ✅ | ✅ | ❌ |
| Apple Silicon Optimized | ✅ | ❌ | ⚠️ |



---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | FastAPI, MLX, Uvicorn, Pydantic |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Testing | pytest, pytest-cov, pytest-asyncio, httpx |
| Linting | ruff, black, mypy, ESLint |
| CI/CD | GitHub Actions (lint, test, build matrix) |
| Config | `pyproject.toml` (single source of truth) |
| Task Runner | [just](https://github.com/casey/just) |

---

## 📖 Configuration

Configure via `.env` file or environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `mlx-community/Phi-3.5-mini-instruct-4bit` | HuggingFace model ID |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `MAX_TOKENS` | `512` | Max tokens per response |
| `TEMPERATURE` | `0.7` | Sampling temperature |

Priority: CLI flag > env var > `.env` file > built-in default.



---

## 🧪 Testing

```bash
just test           # All tests with coverage
just test-quick     # Fast run, no coverage
just test-security  # Security tests only
just lint           # All linters (Python + frontend)
just ci             # Full CI pipeline locally
```

Tests are organized by layer:

| Test file | Covers | Approach |
|-----------|--------|----------|
| `test_routes.py` | HTTP endpoints, decorators, adapter, schemas | Integration via `TestClient` |
| `test_openai_adapter.py` | SSE wire format, finish_reason logic, RAM feasibility | Unit tests |
| `test_schemas.py` | Pydantic validation edge cases (roles, literals, defaults) | Unit tests |
| `test_model_manager.py` | Model loading, RAM checks, observer pattern, security | Unit tests |
| `test_token_stream.py` | Iterator protocol, timing, token accumulation | Unit tests |

---

## 📈 Search Queries This Project Helps With

- how to run llm models on macbook locally without any subscription or api key
- run llm locally mac m1 m2 m3 m4
- apple silicon local llm
- openai alternative mac offline
- local openai server mac
- chatgpt alternative mac

---

## 🛣 Roadmap

- ~~Benchmark suite vs llama.cpp~~ — done (see [benchmarks/](benchmarks/))
- Improved Metal acceleration
- Multi-model support
- Built-in local RAG support
- Native macOS app packaging

---

## 🤝 Contributing

Pull requests and issues are welcome.

---

## 📄 License

MIT License.

---

Built with ❤️ for the Apple Silicon and Mac AI community.
