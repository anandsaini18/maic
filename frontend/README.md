# Maic Frontend

A modern React + TypeScript web UI for interacting with the local LLM server. Built with Vite, Tailwind CSS, and optimized for desktop and mobile.

## Overview

The frontend provides a full-featured chat interface with:
- **Real-time streaming** of LLM responses via Server-Sent Events (SSE)
- **Model management** — download, activate, and delete models
- **Generation settings** — temperature, top-P, top-K, and more
- **Live performance monitoring** — tokens-per-second chart and inference timing
- **Dark/light theme** — automatic system preference detection
- **Responsive design** — desktop + mobile support
- **Markdown rendering** — with syntax highlighting via Highlight.js

## Tech Stack

- **React 19** — UI framework with modern hooks
- **TypeScript** — type safety
- **Vite 7** — lightning-fast build tool with HMR
- **Tailwind CSS** — utility-first styling
- **Marked** — markdown parser
- **Highlight.js** — code syntax highlighting

## Project Structure

```
src/
├── App.tsx                    # Root component with two-panel layout
├── api/
│   ├── client.ts             # Fetch wrappers for backend endpoints
│   └── types.ts              # TypeScript interfaces
├── hooks/
│   ├── useChat.ts            # Chat state, SSE consumer, history
│   ├── useModels.ts          # Model list polling & management
│   ├── useTheme.ts           # Dark/light theme toggle
│   └── useToast.ts           # Toast notification state
├── lib/
│   └── markdown.ts           # Marked + Highlight.js renderer
└── components/
    ├── chat/
    │   ├── ChatPanel.tsx      # Left panel (messages + input)
    │   ├── ChatHeader.tsx     # Model badge, theme toggle
    │   ├── ChatInput.tsx      # Textarea with token counter
    │   ├── Message.tsx        # User/assistant bubble
    │   ├── MessageList.tsx    # Scrollable history
    │   ├── TypingIndicator.tsx # Loading animation
    │   └── WelcomeScreen.tsx  # Empty state suggestions
    └── inspector/
        ├── InspectorPanel.tsx   # Right sidebar
        ├── GenerationSettings.tsx # Settings sliders
        ├── ModelSelector.tsx    # Model management panel
        ├── ModelOption.tsx      # Individual model card
        ├── ModelTag.tsx         # Status badges
        ├── SliderGroup.tsx      # Reusable slider
        ├── StatusBar.tsx        # RAM, online/offline, timing
        ├── ToggleRow.tsx        # Reusable toggle
        └── TpmChart.tsx         # Live tokens/sec chart
```

## Development

### Prerequisites
- Node.js 18+ (or use the included setup script)

### Setup

```bash
npm install
npm run dev
```

The dev server starts at `http://localhost:5173` with HMR enabled.

### Build

```bash
npm run build
```

Outputs optimized bundles to `dist/`. The `run.sh` script in the project root automatically builds and serves from `static/`.

### Type Checking

```bash
npm run type-check
```

### Linting

```bash
npm run lint
```

## Key Features

### Chat with Streaming
- Messages sent via `/v1/chat/completions` with `stream: true`
- SSE stream parsed token-by-token, updating the UI in real time
- Multi-turn context preserved across conversation history
- System prompt support (optional)

### Model Management
- `/v1/models/status` polled every 3 seconds for live download state
- Download, activate, or delete models via HTTP endpoints
- Models tagged with status: `active`, `ready`, `downloading`, `requires-token`, `too-large`
- RAM feasibility check prevents loading models that exceed 80% unified memory

### Generation Settings
Real-time sliders for:
- **Temperature** (0.0–2.0) — randomness control
- **Top-P** (0.0–1.0) — nucleus sampling
- **Top-K** (1–100) — top-K filtering
- **Min-P** (0.0–1.0) — minimum probability threshold
- **Repetition Penalty** (0.5–2.0) — reduce hallucination
- **Max Tokens** (1–4096) — response length
- **Stream toggle** — toggle between streaming and blocking responses

### Performance Monitoring
- **Tokens-per-second chart** — live 20-point rolling window
- **Inference timing** — displays wall-clock time of last response
- **Status bar** — RAM usage, online/offline indicator, last response time

## API Contract

The frontend expects these endpoints from the backend:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/models` | OpenAI-format model list |
| `GET` | `/v1/models/supported` | All models with RAM feasibility |
| `GET` | `/v1/models/status` | Full status (download state, active, feasible) |
| `POST` | `/v1/chat/completions` | Streaming SSE or blocking JSON |
| `POST` | `/v1/models/download` | Background download |
| `POST` | `/v1/models/load` | Switch active model |
| `DELETE` | `/v1/models/{model_id}` | Delete weights from disk |

## Styling

Uses Tailwind CSS with a custom color palette:
- **Light mode:** Gray backgrounds, dark text
- **Dark mode:** Dark backgrounds, light text
- Responsive breakpoints (`sm`, `md`, `lg`, `xl`)
- Custom components for buttons, inputs, badges

## Accessibility

- Semantic HTML (`<button>`, `<nav>`, `<main>`)
- ARIA labels on interactive elements
- Keyboard navigation support
- High contrast for light/dark modes

## Performance

- **Lazy code splitting** — components split by route/section
- **Tree-shaking** — unused code removed in production build
- **Minification** — all bundles gzipped
- **Preload critical assets** — main bundle prioritized

## Troubleshooting

**Blank page on load?**
- Check browser console for errors
- Verify backend is running (`python main.py`)
- Clear browser cache and reload

**Models not showing?**
- Ensure backend `/v1/models` endpoint is working
- Check network tab for failed requests
- Verify CORS is enabled on backend

**Streaming not working?**
- Backend must support SSE on `/v1/chat/completions`
- Check browser supports `ReadableStream` API (all modern browsers)
- Verify no proxy/firewall blocking streaming responses

## License

MIT
