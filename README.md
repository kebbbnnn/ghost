# ChatGPT Anonymous Proxy

An **OpenAI-compatible HTTP proxy** that routes requests to ChatGPT's
undocumented anonymous guest API — no account, no API key required.

> ⚠️ **Disclaimer:** This is a reverse-engineering research project.
> Use responsibly and in accordance with ChatGPT's Terms of Service.

---

## Features

- **Drop-in OpenAI replacement** — any client that uses `openai` SDK or raw
  `POST /v1/chat/completions` works without modification
- **Streaming & non-streaming** — both `stream: true` and `stream: false` supported
- **Multi-turn conversations** — full context continuity across multiple turns
- **Automatic session management** — device ID, PoW sentinel, and token refresh
  are handled transparently
- **Automatic retry** — 403 PoW expiry is recovered silently without breaking
  the active conversation

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

**Node.js** (for PoW VM, optional — only needed if `vm.py` can't run standalone)

```bash
npm install
```

---

## Quick Start

### 1. Start the proxy

```bash
python3 -m uvicorn src.server:app --port 8100
```

The server will be available at `http://127.0.0.1:8100`.

### 2. Test with the interactive chat client

```bash
python3 chat.py
```

```
==================================================
ChatGPT Proxy Interactive Test
Type your message and press Enter. Type 'exit' to quit.
==================================================

You: hello

ChatGPT: Hello! How can I help you today?

You: what did i just say?

ChatGPT: You just said "hello"!
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

# Non-streaming
curl -s -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false
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

---

## API Reference

### `POST /v1/chat/completions`

Standard OpenAI Chat Completions format with two optional extra fields for
multi-turn conversation continuity.

**Request body:**

```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": true,

  // Optional — for multi-turn continuity (returned in previous response)
  "conversation_id": "6a005...",
  "parent_message_id": "7f4e8..."
}
```

**Streaming response** (`stream: true`):

Each SSE chunk includes a non-standard `conversation_id` field alongside the
standard OpenAI fields. Use this to maintain conversation context:

```
data: {"id": "<msg-uuid>", "object": "chat.completion.chunk", "model": "auto",
       "conversation_id": "6a005...",
       "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}

data: [DONE]
```

**For multi-turn:** Pass `conversation_id` and the last `id` as `parent_message_id`
on subsequent requests.

### `GET /v1/models`

Returns the list of available model slugs.

### `GET /health`

Liveness probe. Returns `{"status": "ok", "session_ready": true}`.

---

## Available Models

| Model ID | Notes |
|---|---|
| `auto` | Let ChatGPT choose (default; currently resolves to GPT-5-mini for guests) |
| `gpt-5-5` | Request GPT-5 (may fall back) |
| `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`, etc. | All mapped to `auto` for guest sessions |

---

## Project Structure

```
.
├── src/
│   ├── server.py              # FastAPI proxy — OpenAI-compatible endpoints
│   ├── converter.py           # SSE parser: ChatGPT format → OpenAI format
│   ├── session/
│   │   └── guest_session.py   # Session lifecycle, PoW, streaming
│   └── sentinel/
│       ├── token_generator.py # PoW solver pipeline
│       ├── pow_solver.py      # SHA-3 proof-of-work implementation
│       ├── browser_env.py     # Browser environment simulation
│       └── vm.py              # JS VM for Turnstile/PoW execution
├── chat.py                    # Interactive multi-turn test client
├── research/                  # Reverse engineering scripts (exploratory)
├── tests/                     # Automated tests
├── ARCHITECTURE.md            # Deep-dive: design, bugs, decisions, regression checklist
├── requirements.txt
└── README.md
```

---

## How It Works

ChatGPT's web frontend uses an anonymous guest session backed by a set of
undocumented `/backend-anon` endpoints. This proxy replicates that session:

1. **Device ID** — loads `chatgpt.com` to receive an `oai-did` cookie (stable
   UUID identifying the browser device)
2. **PoW Sentinel** — calls `/backend-anon/sentinel/chat-requirements`, solves
   a SHA-3 proof-of-work challenge, and generates three `Openai-Sentinel-*`
   request headers
3. **Chat** — sends messages to `/backend-anon/conversation` with the sentinel
   headers and streams back SSE responses
4. **Multi-turn** — conversation ID and parent message ID are extracted from
   responses and re-injected into subsequent requests

For the full technical deep-dive, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Multi-Turn Example (manual)

```bash
# Turn 1
TURN1=$(curl -s -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"My name is Alice."}],"stream":false}')

CONV_ID=$(echo $TURN1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('conversation_id',''))")
MSG_ID=$(echo $TURN1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('id',''))")

# Turn 2 — model remembers "Alice"
curl -s -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"auto\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"My name is Alice.\"},
      {\"role\": \"assistant\", \"content\": \"Nice to meet you, Alice!\"},
      {\"role\": \"user\", \"content\": \"What is my name?\"}
    ],
    \"conversation_id\": \"$CONV_ID\",
    \"parent_message_id\": \"$MSG_ID\",
    \"stream\": false
  }"
```

---

## Limitations

- **Guest rate limits** — anonymous sessions have tighter rate limits than
  logged-in accounts. Heavy use may trigger 429 responses.
- **No persistence** — conversation state is in-memory. Restarting the proxy
  invalidates all active conversation IDs.
- **Single session** — all concurrent requests share one `GuestSession` and
  one device identity.
- **No image generation** — only text chat is supported.
- **Model selection** — guests cannot access GPT-4o or o-series models
  directly; the backend may silently reroute to a lower tier.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[Proxy Error: HTTP Error 403]` | PoW token expired | Session auto-refreshes; retry after 1–2s |
| Empty response (no content) | Stale `parent_message_id` | Don't reuse old IDs across server restarts |
| `Address already in use` on port 8100 | Another server instance running | `pkill -f "uvicorn src.server"` |
| `NameError: HTTPError` | Old code before the import fix | Pull latest and restart |

---

## Development

```bash
# Run with auto-reload during development
python3 -m uvicorn src.server:app --port 8100 --reload

# Run tests
python3 tests/test_pow.py
python3 test_proxy.py
```

---

## License

ISC — see `package.json`. Research and educational use only.
