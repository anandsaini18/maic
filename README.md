# Maic 🪄

**Maic** is a high-performance local language model server specifically optimized for Apple Silicon (M1/M2/M3). It provides an **OpenAI-compatible API** and a sleek, modern web-based chat interface, allowing you to run powerful models like Llama 3, Mistral, and Phi-3 entirely on your own hardware with maximum efficiency.

Maic is for users who want privacy, zero cost, and local control — tinkerers, developers who want an OpenAI-compatible local API, privacy-conscious users, and anyone experimenting with open-source models on Apple Silicon. Its unique value is the MLX-optimized inference server with an API other tools can consume.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![MLX](https://img.shields.io/badge/optimization-MLX-orange.svg)

## ✨ Features

- **🚀 MLX-Optimized Inference:** Built on Apple's MLX framework for blazing-fast performance on macOS.
- **🔌 OpenAI-Compatible API:** A drop-in replacement for `v1/chat/completions`. Use Maic with your favorite AI tools and extensions.
- **🎨 Modern Web UI:** A beautiful, responsive chat interface built with React, Tailwind CSS, and Framer Motion.
- **📦 Model Management:** 
  - One-click downloads directly from HuggingFace.
  - Automatic memory (RAM) requirement checks before loading.
  - Real-time download progress and status tracking.
- **📊 Real-time Monitoring:** Track inference speed (TPM - Tokens Per Minute) and memory usage with integrated live charts.
- **🛡️ Privacy First:** Everything runs locally on your machine. No data ever leaves your device.
- **⚙️ Advanced Controls:** Fine-tune your experience with adjustable temperature, top-p, and max tokens.

## 🛠️ Tech Stack

- **Backend:** FastAPI, MLX, Uvicorn, Pydantic
- **Frontend:** React, Vite, Tailwind CSS, Lucide Icons
- **Deployment:** Simple shell scripts for easy setup on macOS

## 🚀 Getting Started

### Prerequisites

- **Apple Silicon Mac** (M1, M2, M3 series)
- **macOS** 13.5 or later
- **Python 3.9+**
- **Node.js & npm** (for frontend development)

### Quick Start (Automated)

The easiest way to get started is by using the provided `run.sh` script, which handles virtual environment setup and dependency installation:

```bash
chmod +x run.sh
./run.sh
```

### Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/maic.git
   cd maic
   ```

2. **Setup Backend:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```

3. **Setup Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Open your browser and navigate to `http://localhost:8000` (or the port specified by Vite).

## 🔌 API Usage

Maic exposes an OpenAI-compatible API. You can interact with it using standard tools like `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Explain quantum entanglement like I am five."}],
    "stream": true
  }'
```

## 📖 Configuration

You can configure Maic using environment variables or a `.env` file:

- `MODEL_ID`: Default HuggingFace model ID to load.
- `HOST`: Server bind address (default: `0.0.0.0`).
- `PORT`: Server port (default: `8000`).

## 🤝 Contributing

Contributions are welcome! Whether it's reporting bugs, suggesting features, or submitting pull requests, your help is appreciated.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ for the Apple Silicon community.
