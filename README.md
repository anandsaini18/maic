# Maic 🪄  
## Run LLM Models Locally on MacBook (No Subscription • No API Key • Fully Offline)

**Maic** is a high-performance local LLM server optimized for **Apple Silicon (M1, M2, M3)**.

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

- MacBook Pro M1 / M2 / M3  
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

### Available Commands

Run `just` to see all commands:

| Command | Description |
|---------|-------------|
| `just setup` | Create venv + install everything |
| `just dev` | Start the backend server |
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
| Testing | pytest, pytest-cov |
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

## 📈 Search Queries This Project Helps With

- how to run llm models on macbook locally without any subscription or api key
- run llm locally mac m1
- apple silicon local llm
- openai alternative mac offline
- local openai server mac
- chatgpt alternative mac

---

## 🛣 Roadmap

- Benchmark suite vs llama.cpp
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