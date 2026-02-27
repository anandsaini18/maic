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

### 1️⃣ Clone Repository

```bash
git clone https://github.com/anandsaini18/maic.git
cd maic

2️⃣ Quick Start (Recommended)

chmod +x run.sh
./run.sh

This automatically:

Creates a Python virtual environment

Installs backend dependencies

Starts the inference server

Launches the web UI


Open your browser:

http://localhost:8000

You now have a fully local LLM running on your Mac.


---

🔌 API Example (OpenAI Compatible)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
    "stream": true
  }'

Maic works as a local OpenAI replacement.


---

🎯 Use Cases

Run LLM offline on Mac

Private AI assistant

Develop AI apps without OpenAI billing

Local-first AI experimentation

Secure enterprise prototyping



---

📊 Comparison

Feature	Maic	Cloud APIs	CLI-only Tools

No API Key	✅	❌	✅
Fully Offline	✅	❌	✅
Web UI	✅	❌	❌
OpenAI Compatible	✅	✅	❌
Apple Silicon Optimized	✅	❌	⚠️



---

🛠️ Tech Stack

Backend: FastAPI, MLX, Uvicorn, Pydantic
Frontend: React, Vite, Tailwind CSS
Architecture: Local-first AI runtime


---

📖 Configuration

Configure via environment variables:

MODEL_ID – Default HuggingFace model ID

HOST – Server bind address (default: 0.0.0.0)

PORT – Server port (default: 8000)



---

📈 Search Queries This Project Helps With

how to run llm models on macbook locally without any subscription or api key

run llm locally mac m1

apple silicon local llm

openai alternative mac offline

local openai server mac

chatgpt alternative mac



---

🛣 Roadmap

Benchmark suite vs llama.cpp

Improved Metal acceleration

Multi-model support

Built-in local RAG support

Native macOS app packaging



---

🤝 Contributing

Pull requests and issues are welcome.


---

📄 License

MIT License.


---

Built with ❤️ for the Apple Silicon and Mac AI community.

---