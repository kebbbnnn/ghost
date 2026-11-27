# Ghost

An **embeddable multi-provider LLM library and OpenAI/Anthropic-compatible proxy**
that routes requests to ChatGPT, Gemini, and Spectre — no API key or separate server required.

> ⚠️ **Disclaimer:** This is a reverse-engineering research project.
> Use responsibly and in accordance with each platform's Terms of Service.

---

## Plug Directly Into Another Python Project (No Server)

Install Ghost directly into any Python project from a local checkout or Git URL — no FastAPI, Uvicorn, or open port needed:

```bash
# Core library only (curl-cffi, httpx, pydantic, json-repair)
pip install -e /path/to/ghost

# Or with optional extras: [openai], [anthropic], [vision], [video], [server], [gui], [all]
pip install -e "/path/to/ghost[openai,anthropic]"
```

### 1. Direct Python Client (Sync & Async)

```python
from ghost import Ghost, AsyncGhost

# Synchronous (scripts, Django, Flask, Jupyter)
with Ghost() as client:
    # One-liner
    print(client.ask("What is 2+2?", model="auto"))

    # OpenAI-shaped API (streaming & non-streaming, tool emulation, failover)
    resp = client.chat.completions.create(
        model="gemini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp["choices"][0]["message"]["content"])

    for chunk in client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Count to 3"}],
        stream=True,
    ):
        print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)

    # Anthropic Messages API
    msg = client.messages.create(
        model="gemini",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello Claude-style!"}],
    )

# Native Async (FastAPI, asyncio apps) — opt into warm_up=True for long-lived apps
async with AsyncGhost(warm_up=True) as client:
    reply = await client.ask("Explain Raft in 2 sentences", model="spectre")
```

### 2. Zero-Server Drop-In for Official OpenAI & Anthropic SDKs

Ghost ships pure-`httpx` in-process transports (`GhostTransport` / `AsyncGhostTransport`) so real `openai.OpenAI` / `anthropic.Anthropic` clients (and frameworks like LangChain, Instructor, or PydanticAI) talk to Ghost directly in memory with zero network sockets:

```python
import ghost

# Real openai.OpenAI / openai.AsyncOpenAI — no server running
oa = ghost.openai_client()
resp = oa.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
oa.close()

# Real anthropic.Anthropic / anthropic.AsyncAnthropic — no server running
an = ghost.anthropic_client()
msg = an.messages.create(
    model="gemini",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello!"}],
)
an.close()

# Or inject into any httpx-compatible framework (e.g. LangChain)
http = ghost.httpx_client()  # or ghost.async_httpx_client()
```

> **Full documentation:** See the [Usage Guide](./docs/usage.md) for installation options, multi-turn chat, tool calling, vision/video, Anthropic Messages, LangChain/`httpx` integration, lifecycle best practices, error handling, and offline testing.

> **Tip on Embedded vs. Server Mode:** In-process mode is ideal for single-process apps and scripts. If you run many separate worker processes (e.g., Gunicorn/Celery), running `ghost-server` (or `python -m ghost.server`) lets all workers share one warm guest session and rate-limit state.

---

## Supported Providers

| Provider | Model Prefix | Auth | Multi-Turn |
|---|---|---|---|
| **ChatGPT** | `auto`, `gpt-*` | None (anonymous guest) | ✅ Server-side |
| **Gemini** | `gemini*`, `gemini-tree-of-thoughts` | None (anonymous) | ✅ Server-side |
| **Spectre** | `spectre`, `spectre-ultra` | Multi-agent orchestrator | ✅ Delegated |

---

## Features

- **Zero-server Python library** — embed directly via `Ghost` / `AsyncGhost` with zero FastAPI dependency or open ports
- **Drop-in OpenAI & Anthropic SDK compatibility** — pure-`httpx` in-process transports (`ghost.openai_client()`, `ghost.anthropic_client()`) plus optional HTTP proxy endpoints (`/v1/chat/completions`, `/v1/messages`)
- **Tool calling & parameter emulation** — automatic `tools`, `tool_choice`, `stop`, `max_tokens`, and `temperature` emulation
- **Streaming & non-streaming** — both `stream=True` and `stream=False` supported across sync, async, and HTTP modes
- **Multi-provider routing & failover** — routes by model prefix and fails over cleanly while preserving 429 rate-limit signals
- **Automatic session management** — device IDs, PoW sentinels, and token refresh handled transparently

---

## Requirements

**Python 3.11+**

```bash
# Minimal library install (no server/GUI/OpenCV)
pip install -e .

# Full development environment (server + GUI + vision + video + SDK extras)
pip install -e ".[all]"
# or: pip install -r requirements.txt
```

| Extra | Packages | Purpose |
|---|---|---|
| *(core)* | `curl-cffi`, `httpx[http2]`, `pydantic`, `json-repair` | Embedded library & in-process transports |
| `[openai]` / `[anthropic]` | `openai`, `anthropic` | Official SDK drop-in factories |
| `[server]` | `fastapi`, `uvicorn[standard]`, `python-multipart` | Optional HTTP proxy server (`ghost-server`) |
| `[vision]` / `[video]` | `Pillow`, `opencv-python-headless`, `numpy` | Image dimensions & video keyframe chat |
| `[gui]` | `pywebview` | Desktop GUI dashboard |

---

## Quick Start (HTTP Proxy Mode)

### 1. Start the proxy

```bash
ghost-server
# or: python3 -m ghost.server   (python3 -m src.server also works in this repo)
```

The server will be available at `http://127.0.0.1:8100`.

### 2. Test with the interactive chat client

```bash
# ChatGPT (default)
python3 chat.py

# Gemini
python3 chat.py gemini
```

### 3. Use with curl

```bash
# Streaming
curl -s -N -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true
  }'
```

### 4. Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8100/v1",
    api_key="not-needed",  # required by SDK but ignored by proxy
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 5. Use as a backend for coding agents

Works with any tool that supports an OpenAI-compatible endpoint:

```bash
# Cursor — set the OpenAI Base URL in settings to:
# http://127.0.0.1:8100/v1

export OPENAI_BASE_URL=http://127.0.0.1:8100/v1
```

---

## Desktop GUI (All-in-One Mode)

Instead of running the proxy server and chat client separately, you can launch
Ghost as a **desktop application** with a built-in web dashboard:

```bash
# Activate the virtual environment first
source venv/bin/activate

# Launch the desktop GUI
python3 examples/gui/gui.py
```

This will:

1. Auto-start the FastAPI proxy on `http://127.0.0.1:8100`
2. Open a native desktop window (via [pywebview](https://pywebview.flowrl.com))
   pointing at the built-in dashboard

> **Extra dependency:** `pywebview>=5.1` is already listed in `requirements.txt`.
> If you haven't installed it yet, run `pip install pywebview`.

When you close the window, the proxy server shuts down automatically.

---

## API Reference

### `POST /v1/chat/completions`

Standard OpenAI Chat Completions format.

**Request body:**

```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

### `POST /v1/messages`

Anthropic Messages API format (streaming & non-streaming), plus `POST /v1/messages/count_tokens`.

### `GET /v1/models`

Returns the aggregated list of models from all providers.

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

---

## Available Models

| Provider | Model ID | Notes |
|---|---|---|
| ChatGPT | `auto`, `gpt-5-5` | Default anonymous guest session |
| Gemini | `gemini`, `gemini-tree-of-thoughts` | Anonymous Gemini web session & Tree of Thoughts |
| Spectre | `spectre`, `spectre-ultra` | Single-model router & multi-step DAG orchestrator |

---

## Project Structure

```text
.
├── pyproject.toml                       # Packaging metadata & optional extras (ships ghost* only)
├── ghost/
│   ├── __init__.py                      # Public library API exports
│   ├── engine.py                        # Server-free GhostEngine orchestration core
│   ├── client.py                        # Direct sync (Ghost) & async (AsyncGhost) clients
│   ├── adapters/
│   │   └── httpx_transport.py           # Pure-httpx transports & OpenAI/Anthropic SDK factories
│   ├── server.py                        # Optional FastAPI HTTP proxy delegating to GhostEngine
│   ├── converters/
│   │   └── anthropic.py                 # Anthropic <-> OpenAI request/response conversion
│   ├── core/
│   │   ├── provider.py                  # BaseProvider interface
│   │   ├── exceptions.py                # Shared exception types
│   │   └── emulation/                   # Tool calling & parameter emulation pipeline
│   └── providers/
│       ├── __init__.py                  # ProviderRegistry & model routing
│       ├── chatgpt/                     # ChatGPT anonymous guest provider
│       ├── gemini/                      # Gemini & Tree of Thoughts provider
│       └── spectre/                     # Spectre multi-agent orchestrator
├── src/                                 # Dev-only backwards-compat alias (excluded from wheels)
├── docs/
│   ├── usage.md                         # Full library & SDK usage guide
│   └── designs/                         # Design specifications
├── chat.py                              # Interactive multi-turn test client
├── tests/                               # Pytest test suite
├── ARCHITECTURE.md                      # Deep-dive: design, bugs, decisions
└── README.md
```

---

## How It Works

### ChatGPT (Anonymous)

ChatGPT's web frontend uses an anonymous guest session backed by undocumented
`/backend-anon` endpoints. The proxy replicates that session:

1. **Device ID** — loads `chatgpt.com` to receive an `oai-did` cookie
2. **PoW Sentinel** — solves a SHA-3 proof-of-work challenge
3. **Chat** — sends messages to `/backend-anon/conversation` with sentinel headers
4. **Multi-turn** — conversation/message IDs are tracked for context continuity

### Gemini (Anonymous)

Uses Gemini's batch-RPC protocol with anonymous tokens, mapping responses
to the OpenAI completion schema.

For the full technical deep-dive, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Limitations

- **Guest rate limits** — anonymous ChatGPT/Gemini sessions have tighter rate
  limits. Heavy use may trigger 429 responses.
- **No persistence** — conversation state is in-memory. Restarting the proxy
  invalidates all active conversation IDs.
- **No image generation** — only text chat is supported.
- **Model selection** — guests cannot access premium models directly; the
  backend may silently reroute to a lower tier.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[Proxy Error: HTTP Error 403]` | PoW token expired | Session auto-refreshes; retry after 1–2s |
| `Address already in use` on port 8100 | Another server instance | `lsof -ti :8100 \| xargs kill -9` |

---

## Development

```bash
# Run HTTP proxy with auto-reload
python3 -m uvicorn ghost.server:app --port 8100 --reload

# Run tests
pytest
```

---

## License

ISC — see `package.json`. Research and educational use only.
