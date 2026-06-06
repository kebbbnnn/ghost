# GhostTunnel

A **multi-provider, OpenAI-compatible HTTP proxy** that routes requests to
various AI platforms — no API key required.

> ⚠️ **Disclaimer:** This is a reverse-engineering research project.
> Use responsibly and in accordance with each platform's Terms of Service.

---

## Supported Providers

| Provider | Model Prefix | Auth | Multi-Turn |
|---|---|---|---|
| **ChatGPT** | `auto`, `gpt-*` | None (anonymous guest) | ✅ Server-side |
| **Gemini** | `gemini*` | None (anonymous) | ✅ Server-side |
| **Claude** | `claude*` | Session cookie (via CLI tool) | ✅ Server-side (conversation reuse) |

---

## Features

- **Drop-in OpenAI replacement** — any client that uses `openai` SDK or raw
  `POST /v1/chat/completions` works without modification
- **Streaming & non-streaming** — both `stream: true` and `stream: false` supported
- **Multi-provider routing** — automatically routes based on model name prefix
- **Multi-turn conversations** — full context continuity across turns
- **Automatic session management** — device IDs, PoW sentinels, token refresh,
  and cookie rotation are handled transparently
- **Claude account rotation** — rotate across multiple Claude accounts to
  bypass per-account rate limits, with hot-reload support
- **Automatic retry** — 403 PoW expiry and 429 throttling are recovered
  silently with exponential backoff

---

## Requirements

**Python 3.11+**

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `fastapi` | HTTP server framework |
| `uvicorn[standard]` | ASGI server |
| `curl-cffi` | HTTP client with TLS fingerprint impersonation |
| `pydantic` | Request/response validation |
| `playwright` | Browser automation for Claude login (CLI tool only) |

---

## Quick Start

### 1. Start the proxy

```bash
python3 -m src.server
```

The server will be available at `http://127.0.0.1:8100`.

### 2. Test with the interactive chat client

```bash
# ChatGPT (default)
python3 chat.py

# Gemini
python3 chat.py gemini

# Claude
python3 chat.py claude-3-5-sonnet-20241022
```

### 3. Use with curl

```bash
# Streaming
curl -s -N -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
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
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 5. Use as a backend for coding agents

Works with any tool that supports an OpenAI-compatible endpoint:

```bash
# Claude Code
export ANTHROPIC_BASE_URL=http://127.0.0.1:8100/v1

# Cursor — set the OpenAI Base URL in settings to:
# http://127.0.0.1:8100/v1
```

---

## Desktop GUI (All-in-One Mode)

Instead of running the proxy server and chat client separately, you can launch
GhostTunnel as a **desktop application** with a built-in web dashboard:

```bash
# Activate the virtual environment first
source venv/bin/activate

# Launch the desktop GUI
python3 examples/gui.py
```

This will:

1. Auto-start the FastAPI proxy on `http://127.0.0.1:8100`
2. Open a native desktop window (via [pywebview](https://pywebview.flowrl.com))
   pointing at the built-in dashboard
3. Provide a **Claude Login Helper** — log in to Claude directly from the GUI
   (no CLI needed)

> **Extra dependency:** `pywebview>=5.1` is already listed in `requirements.txt`.
> If you haven't installed it yet, run `pip install pywebview`.

When you close the window, the proxy server shuts down automatically.

---

## Claude Setup

Claude requires authenticated session cookies. Use the CLI tool to log in:

```bash
# Add an account (opens a browser window for login)
python3 src/providers/claude/manage_claude.py --add

# List saved accounts
python3 src/providers/claude/manage_claude.py --list
```

> **Note:** Use Claude's **email/password** login, not Google OAuth — Google
> blocks automated browsers.

You can add multiple accounts. The proxy rotates between them automatically
and handles rate limits per account. Accounts are stored in
`src/providers/claude/accounts.json` and can be hot-reloaded without
restarting the server.

---

## API Reference

### `POST /v1/chat/completions`

Standard OpenAI Chat Completions format.

**Request body:**

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

### `GET /v1/models`

Returns the aggregated list of models from all providers.

### `GET /health`

Liveness probe. Returns `{"status": "ok", "session_ready": true}`.

---

## Available Models

| Provider | Model ID | Notes |
|---|---|---|
| ChatGPT | `auto` | Default; currently resolves to GPT-5-mini for guests |
| Gemini | `gemini` | Anonymous Gemini web interface |
| Claude | `claude-sonnet-4-6` | Claude Sonnet 4.6 (latest) |
| Claude | `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet |
| Claude | `claude-3-opus-20240229` | Claude 3 Opus |
| Claude | `claude-3-5-haiku-20241022` | Claude 3.5 Haiku |

---

## Project Structure

```
.
├── src/
│   ├── server.py                        # FastAPI proxy — OpenAI-compatible endpoints
│   ├── core/
│   │   ├── provider.py                  # BaseProvider interface
│   │   └── exceptions.py               # Shared exception types
│   ├── providers/
│   │   ├── __init__.py                  # Provider registry & model routing
│   │   ├── chatgpt/
│   │   │   └── session.py              # ChatGPT anonymous guest provider
│   │   ├── gemini/
│   │   │   └── session.py              # Gemini anonymous provider
│   │   └── claude/
│   │       ├── session.py              # Claude web API provider
│   │       ├── rotator.py              # Multi-account rotator with cooldowns
│   │       ├── manage_claude.py        # CLI tool for account login & management
│   │       └── accounts.json           # Stored session credentials
├── chat.py                              # Interactive multi-turn test client
├── tests/                               # Test & probe scripts
├── ARCHITECTURE.md                      # Deep-dive: design, bugs, decisions
├── requirements.txt
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

### Claude (Authenticated)

Uses Claude's undocumented web API with session cookies:

1. **Authentication** — `sessionKey` cookies extracted via Playwright CLI tool
2. **Organization** — fetches the user's org ID from `/api/organizations`
3. **Conversation Reuse** — creates a conversation once, reuses it for subsequent
   messages to minimize API calls and avoid frequency throttling
4. **Account Rotation** — round-robin across multiple accounts with per-account
   cooldown tracking and file-watched hot-reloading
5. **Rate Limit Handling** — distinguishes frequency throttling (short backoff + retry)
   from quota exhaustion (cooldown based on `resets_at` timestamp)

For the full technical deep-dive, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Limitations

- **Guest rate limits** — anonymous ChatGPT/Gemini sessions have tighter rate
  limits. Heavy use may trigger 429 responses.
- **Claude requires login** — session cookies must be extracted via the CLI tool.
  Cookies may expire and need re-extraction.
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
| `All Claude accounts rate limited` | Frequency throttling or quota | Wait for cooldown or add more accounts |
| `Address already in use` on port 8100 | Another server instance | `lsof -ti :8100 \| xargs kill -9` |
| Claude login blocked by Google | Playwright Chromium rejected | Use email/password login, not Google OAuth |

---

## Development

```bash
# Run with auto-reload
python3 -m uvicorn src.server:app --port 8100 --reload

# Run tests
python3 tests/test_pow.py
python3 tests/probe_claude.py
```

---

## License

ISC — see `package.json`. Research and educational use only.
