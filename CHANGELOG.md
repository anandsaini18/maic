# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-26

### Added
- **Initial Release** of Maic
- MLX-optimized inference engine for Apple Silicon (M1/M2/M3)
- OpenAI-compatible REST API (`/v1/chat/completions`)
- Modern React-based web chat interface with Tailwind CSS styling
- Model management system with:
  - One-click HuggingFace model downloads
  - Automatic RAM requirement validation
  - Model activation/deactivation
  - Real-time download progress tracking
- Real-time monitoring dashboard with:
  - Token-per-minute (TPM) charts
  - Memory usage tracking
  - Model status visualization
- Advanced generation parameters:
  - Temperature control
  - Top-p (nucleus sampling)
  - Max tokens configuration
- CLI argument support for model selection and server configuration
- Automated setup script (`run.sh`) for easy macOS deployment
- Comprehensive testing checklist (`TEST_UI_FLOW.md`)
- User guide documentation (`USER_GUIDE.md`)
- Strategy pattern for pluggable generation strategies (default, greedy)

### Tech Stack
- **Backend:** FastAPI, MLX, Uvicorn, Pydantic
- **Frontend:** React 19, Vite, TypeScript, Tailwind CSS
- **Build Tools:** ESLint, PostCSS, esbuild

---

## Planned Features (Future Releases)

### [0.2.0] - Streaming Improvements
- [ ] WebSocket support for faster streaming responses
- [ ] Server-sent events (SSE) optimization
- [ ] Response buffering and backpressure handling

### [0.3.0] - Extended Model Support
- [ ] GGUF format support
- [ ] Custom quantization profiles
- [ ] LoRA adapter support

### [0.4.0] - Advanced Features
- [ ] Conversation history and management
- [ ] Export chat sessions (PDF, JSON)
- [ ] User preferences persistence
- [ ] Dark/Light theme toggle

### [0.5.0] - Multi-User & Deployment
- [ ] User authentication (optional)
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, GCP, Azure)

---

## [Unreleased]

### Under Development
- Integration with Apple's new Neural Engine
- Multi-GPU support (when available)
- Analytics dashboard for inference metrics
