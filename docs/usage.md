# Ghost Usage Guide

This guide covers every way to embed Ghost directly into another Python project without running a server, plus configuration, lifecycle best practices, error handling, and server mode.

---

## 1. Installation

Ghost requires **Python 3.11+**.

### Install directly into another project

```bash
# Editable local install (best when developing alongside another repo)
pip install -e /path/to/ghost

# Standard local install
pip install /path/to/ghost

# Install directly from Git
pip install "git+https://github.com/<your-org>/ghost.git"
```

In another project's `pyproject.toml`:

```toml
dependencies = [
    "ghost @ file:///path/to/ghost",
    # or: "ghost[openai] @ git+https://github.com/<your-org>/ghost.git",
]
```

### Optional Extras

Core installation pulls only four lightweight runtime packages (`curl-cffi`, `httpx[http2]`, `pydantic`, `json-repair`) — no FastAPI, Uvicorn, OpenCV, or PyWebView.

| Extra | Installs | When you need it |
|---|---|---|
| *(core)* | `curl-cffi`, `httpx[http2]`, `pydantic`, `json-repair` | `Ghost`, `AsyncGhost`, `GhostTransport`, `AsyncGhostTransport`, `httpx_client` |
| `ghost[openai]` | `openai>=1.0.0` | `ghost.openai_client()`, `ghost.async_openai_client()` |
| `ghost[anthropic]` | `anthropic>=0.25.0` | `ghost.anthropic_client()`, `ghost.async_anthropic_client()` |
| `ghost[vision]` | `Pillow>=10.0.0` | Exact image dimensions on ChatGPT multimodal uploads |
| `ghost[video]` | `opencv-python-headless`, `numpy` | `client.video_chat(...)` keyframe extraction |
| `ghost[server]` | `fastapi`, `uvicorn[standard]`, `python-multipart`, `python-dotenv` | Running `ghost-server` / `python -m ghost.server` |
| `ghost[gui]` | `pywebview` (+ `pyobjc-*` on macOS) | Desktop GUI (`examples/gui/gui.py`) |
| `ghost[all]` | All of the above | Full local dev environment |

```bash
pip install -e "/path/to/ghost[openai,anthropic]"
```

---

## 2. Choosing an Integration Mode

| Use Case | Recommended API | Needs Server? | Needs `openai`/`anthropic` pkg? |
|---|---|---|---|
| New Python script or app (sync) | `Ghost` | No | No |
| Async Python app (FastAPI, `asyncio`) | `AsyncGhost` | No | No |
| Existing code using `openai.OpenAI` / `AsyncOpenAI` | `ghost.openai_client()` / `ghost.async_openai_client()` | No | Yes (`ghost[openai]`) |
| Existing code using `anthropic.Anthropic` / `AsyncAnthropic` | `ghost.anthropic_client()` / `ghost.async_anthropic_client()` | No | Yes (`ghost[anthropic]`) |
| LangChain, Instructor, PydanticAI, custom `httpx` code | `ghost.httpx_client()` / `GhostTransport` | No | Depends on framework |
| Multi-process app (Gunicorn/Celery) or non-Python clients | `ghost-server` (`python -m ghost.server`) | Yes | Optional |

---

## 3. Direct Python Client (`Ghost` & `AsyncGhost`)

### Quick One-Liner (`ask`)

```python
from ghost import Ghost

with Ghost() as client:
    reply = client.ask("Explain Raft consensus in two sentences.", model="auto")
    print(reply)

    # With a system prompt
    sql = client.ask(
        "Top 5 customers by spend",
        model="gemini",
        system="Return only valid PostgreSQL SQL with no markdown fences.",
    )
```

### OpenAI-Shaped Chat Completions (Sync)

`client.chat.completions.create(...)` accepts standard OpenAI Chat Completions parameters and returns standard OpenAI-formatted dicts (or yields chunk dicts when `stream=True`).

```python
from ghost import Ghost

with Ghost() as client:
    # Non-streaming
    resp = client.chat.completions.create(
        model="auto",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    print(resp["choices"][0]["message"]["content"])

    # Streaming
    stream = client.chat.completions.create(
        model="gemini",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )
    with stream:
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            print(delta.get("content", ""), end="", flush=True)
    print()
```

> **Tip:** Wrapping sync streams in `with stream:` ensures that if you `break` out of the loop early, the upstream HTTP stream is cancelled and closed immediately.

### Multi-Turn Conversations

Pass the growing `messages` list on each turn (standard stateless OpenAI style). Optionally pass `conversation_id` and `parent_message_id` when you want provider-side thread continuity:

```python
from ghost import Ghost

messages = []
with Ghost() as client:
    messages.append({"role": "user", "content": "My favorite language is Rust."})
    resp = client.chat.completions.create(model="auto", messages=messages)
    reply = resp["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})

    messages.append({"role": "user", "content": "What is my favorite language?"})
    resp2 = client.chat.completions.create(model="auto", messages=messages)
    print(resp2["choices"][0]["message"]["content"])
```

### Tool / Function Calling & Parameter Emulation

Ghost runs `EmulationPipeline` automatically inside `GhostEngine` — `tools`, `tool_choice`, `stop`, `max_tokens`, and `temperature` work in-process just like in server mode:

```python
import json
from ghost import Ghost

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]

with Ghost() as client:
    # Turn 1: model emits tool_calls
    resp = client.chat.completions.create(
        model="auto",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    assistant_msg = resp["choices"][0]["message"]
    messages.append(assistant_msg)

    # Execute requested tools and append role="tool" results
    for tc in assistant_msg.get("tool_calls") or []:
        args = json.loads(tc["function"]["arguments"])
        tool_output = json.dumps({"city": args["city"], "temp_c": 22, "condition": "sunny"})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["function"]["name"],
                "content": tool_output,
            }
        )

    # Turn 2: model synthesizes final answer
    final = client.chat.completions.create(model="auto", messages=messages, tools=tools)
    print(final["choices"][0]["message"]["content"])
```

### Multimodal Vision & Video

Attach images via OpenAI `image_url` content parts (HTTPS URL or base64 `data:image/...` URI):

```python
import base64
from ghost import Ghost

with open("diagram.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

with Ghost() as client:
    resp = client.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this architecture diagram."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    )
    print(resp["choices"][0]["message"]["content"])
```

For video understanding (requires `pip install "ghost[video]"`):

```python
from ghost import Ghost

with open("demo.mp4", "rb") as f:
    video_bytes = f.read()

with Ghost() as client:
    with client.video_chat(video_bytes, "Describe what happens in this clip.", model="auto") as stream:
        for chunk in stream:
            print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

### Anthropic Messages API (Direct Client)

```python
from ghost import Ghost

with Ghost() as client:
    # Non-streaming
    msg = client.messages.create(
        model="gemini",
        max_tokens=1024,
        system="You are a careful code reviewer.",
        messages=[{"role": "user", "content": "Review: eval(user_input)"}],
    )
    print(msg["content"][0]["text"])

    # Streaming Anthropic events
    with client.messages.create(
        model="gemini",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a haiku about Python."}],
        stream=True,
    ) as stream:
        for event in stream:
            if event.get("type") == "content_block_delta":
                print(event["delta"].get("text", ""), end="", flush=True)

    # Token estimation
    tokens = client.messages.count_tokens(
        model="gemini",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print("\nEstimated input tokens:", tokens["input_tokens"])
```

### Listing & Retrieving Models

```python
from ghost import Ghost

with Ghost() as client:
    models = client.models.list()
    print([m["id"] for m in models["data"]])

    gemini_info = client.models.retrieve("gemini")
```

### Native Async (`AsyncGhost`)

Use `AsyncGhost` inside `asyncio` applications (FastAPI, Discord/Telegram bots, async workers) for zero thread overhead:

```python
import asyncio
from ghost import AsyncGhost

async def main():
    async with AsyncGhost(warm_up=True) as client:
        # One-liner
        print(await client.ask("Hello!", model="auto"))

        # Streaming
        stream = await client.chat.completions.create(
            model="gemini",
            messages=[{"role": "user", "content": "List 3 facts about Mars."}],
            stream=True,
        )
        async for chunk in stream:
            print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)

asyncio.run(main())
```

Integrating into a host FastAPI app's lifespan (keeping one warm client shared across requests):

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ghost import AsyncGhost

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncGhost(warm_up=True) as client:
        app.state.ghost = client
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/summarize")
async def summarize(text: str):
    reply = await app.state.ghost.ask(f"Summarize:\n{text}", model="auto")
    return {"summary": reply}
```

---

## 4. Drop-In for Official OpenAI & Anthropic SDKs (Zero Server)

Ghost implements pure-`httpx` in-process transports (`GhostTransport` and `AsyncGhostTransport`) that dispatch directly to `GhostEngine` in memory — no FastAPI, no ASGI, no open port.

### Official OpenAI Python SDK

Requires `pip install "ghost[openai]"` (or `pip install openai`).

```python
import ghost

# Synchronous openai.OpenAI
client = ghost.openai_client()
try:
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Hello from OpenAI SDK!"}],
    )
    print(resp.choices[0].message.content)

    # Streaming works identically
    for chunk in client.chat.completions.create(
        model="gemini",
        messages=[{"role": "user", "content": "Count to 3"}],
        stream=True,
    ):
        print(chunk.choices[0].delta.content or "", end="", flush=True)
finally:
    client.close()
```

Async variant:

```python
import asyncio
import ghost

async def main():
    async with ghost.async_openai_client() as client:
        resp = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello async!"}],
        )
        print(resp.choices[0].message.content)

asyncio.run(main())
```

### Official Anthropic Python SDK

Requires `pip install "ghost[anthropic]"` (or `pip install anthropic`).

```python
import ghost

# Synchronous anthropic.Anthropic
client = ghost.anthropic_client()
try:
    msg = client.messages.create(
        model="gemini",
        max_tokens=512,
        messages=[{"role": "user", "content": "Hello from Anthropic SDK!"}],
    )
    print(msg.content[0].text)
finally:
    client.close()
```

Async variant:

```python
import asyncio
import ghost

async def main():
    async with ghost.async_anthropic_client() as client:
        async with client.messages.stream(
            model="gemini",
            max_tokens=512,
            messages=[{"role": "user", "content": "Write 2 lines about space."}],
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)

asyncio.run(main())
```

### Frameworks Accepting `httpx` (LangChain, Instructor, PydanticAI, etc.)

Inject `ghost.httpx_client()` / `ghost.async_httpx_client()` directly into any library that accepts a custom `httpx` client:

```python
from langchain_openai import ChatOpenAI
import ghost

http = ghost.httpx_client()
llm = ChatOpenAI(
    model="auto",
    api_key="ghost",
    base_url="http://ghost.local/v1",
    http_client=http,
)
print(llm.invoke("What is 2+2?").content)
http.close()
```

### Sharing One `GhostEngine` Across Multiple Clients

By default, each client/transport creates its own isolated `GhostEngine`. To share one warm session across both an OpenAI SDK client and an Anthropic SDK client in the same process:

```python
import ghost

engine = ghost.GhostEngine()
oa = ghost.openai_client(engine=engine)
an = ghost.anthropic_client(engine=engine)
# Note: transports with an externally passed `engine=` do NOT close it automatically;
# close the engine yourself when shutting down.
```

---

## 5. Models & Routing

Model names route automatically based on prefix:

| Model ID / Prefix | Provider | Notes |
|---|---|---|
| `auto`, `gpt-*`, *(other)* | ChatGPT | Default guest session (overridable via `default_provider="gemini"`) |
| `gemini`, `gemini*` | Gemini | Anonymous Gemini web session |
| `gemini-tree-of-thoughts` | Gemini ToT | Multi-step Tree of Thoughts reasoning with `reasoning_content` |
| `claude*`, `anthropic*` | Gemini | Routed to Gemini for large context windows |
| `spectre` | Spectre (Fast) | Coordinator picks the best single worker model |
| `spectre-ultra` | Spectre (Ultra) | Coordinator generates and executes a parallel multi-step DAG |

---

## 6. Configuration Options & Environment Variables

### Client / Factory Options

All direct clients (`Ghost`, `AsyncGhost`) and factories (`openai_client`, `anthropic_client`, `httpx_client`, and async variants) accept:

| Parameter | Default | Description |
|---|---|---|
| `default_provider` | `None` | Override fallback default for `auto`/unknown models (`"chatgpt"` or `"gemini"`) |
| `enable_fallback` | `True` | Automatically fail over to an alternate provider on non-429 upstream errors |
| `warm_up` | `False` | Eagerly pre-fetch tokens and run background refresh loops on entry/init (sync & `AsyncGhost`) |
| `raw_logging` | `False` | Write detailed request/upstream/response traces to disk (disabled by default in library mode) |
| `registry` | `None` | Inject a custom `ProviderRegistry` (useful for custom providers or unit testing) |
| `engine` | `None` | Share an existing `GhostEngine` instance |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GHOST_DEFAULT_PROVIDER` | *(empty → ChatGPT)* | Set to `gemini` to default `auto` requests to Gemini |
| `SPECTRE_COORDINATOR_MODEL` | `auto` | Model used by Spectre's coordinator |
| `SPECTRE_WORKERS` | `auto,gemini` | Comma-separated worker model IDs for Spectre |
| `GHOST_RAW_LOGGING` | `true` *(server)* / `false` *(library)* | Enable raw debug trace files |
| `GHOST_RAW_LOG_FILE` | `<root>/debug_raw.log` | Path for appended raw traces |
| `GHOST_RAW_LOG_LATEST_FILE` | `<root>/debug_raw_latest.log` | Path for latest trace snapshot |
| `GHOST_DATA_DIR` | `<root>/data` | Directory used by server mode's thread persistence API |
| `PORT` | `8100` | Port used by `ghost-server` / `python -m ghost.server` |

---

## 7. Error Handling

Ghost preserves rate-limit semantics rather than silently hiding throttling behind a different provider:

- **`RateLimitError`** (HTTP `429` in transports): Raised directly without failover so your application or SDK retry policy can back off properly.
- **`ModelNotFoundError`** (subclass of both `ProviderError` and `ValueError`; HTTP `400`/`404` in transports): Raised when no provider supports the requested model.
- **`TokenExpiredError`** / **`ProviderError`**: Upstream session or provider errors (HTTP `500` in transports).
- **`ImportError`**: Raised with an explicit install command if an optional extra (`[openai]`, `[anthropic]`, `[video]`) is used without being installed.

```python
import time
from ghost import Ghost, ModelNotFoundError, ProviderError, RateLimitError

with Ghost() as client:
    try:
        reply = client.ask("Hello!", model="auto")
    except RateLimitError:
        time.sleep(2)
        reply = client.ask("Hello!", model="auto")
    except ModelNotFoundError as exc:
        print("Bad model:", exc)
    except ProviderError as exc:
        print("Upstream failure:", exc)
```

---

## 8. Lifecycle & Performance Best Practices

1. **Reuse one client instance**: Create a single `Ghost` / `AsyncGhost` (or SDK client) per application/worker rather than instantiating one per request. Reusing a client preserves device cookies, solved PoW tokens, and HTTP keep-alive pools, avoiding a 1–3s cold-start handshake on every call.
2. **Enable `warm_up=True` for long-lived apps**: Keeps ChatGPT sentinel tokens and Gemini CSRF tokens refreshed in the background. Leave `warm_up=False` (the default) for short-lived CLI scripts.
3. **Always close clients**: Use `with Ghost() as client:` / `async with AsyncGhost() as client:` (or call `.close()` / `await .aclose()`) so background refresh tasks and sockets shut down cleanly.
4. **Embedded vs. Server Mode**:
   - Use **embedded library mode** for single-process Python apps, CLIs, notebooks, background scripts, and test suites.
   - Use **server mode** (`ghost-server`) when running many separate OS worker processes (e.g., Gunicorn/Celery) so all workers share one guest session and avoid compounding anonymous rate limits.

---

## 9. Testing Your App Without Network Calls

Inject mock `BaseProvider` instances through `ProviderRegistry` to unit-test code that uses Ghost deterministically offline:

```python
from ghost import BaseProvider, Ghost, ProviderRegistry

class StubProvider(BaseProvider):
    async def get_models(self):
        return [{"id": "auto", "object": "model", "created": 0, "owned_by": "stub"}]

    async def non_streaming_chat(self, messages, model, conversation_id=None, parent_message_id=None):
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "stubbed"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def stream_chat(self, messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id":"chatcmpl-stub","object":"chat.completion.chunk","created":0,"model":"auto","choices":[{"index":0,"delta":{"role":"assistant","content":"stubbed"},"finish_reason":null}]}\n\n'
        yield 'data: {"id":"chatcmpl-stub","object":"chat.completion.chunk","created":0,"model":"auto","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

stub = StubProvider()
registry = ProviderRegistry(chatgpt=stub, gemini=stub, spectre=stub)

with Ghost(registry=registry) as client:
    assert client.ask("anything") == "stubbed"
```

---

## 10. Optional Server & GUI Modes

If you do want an HTTP proxy (for non-Python tools like `curl`, Cursor, Claude Code, or multi-worker deployments):

```bash
# Install with server extras
pip install -e "/path/to/ghost[server]"

# Start HTTP proxy on http://127.0.0.1:8100
ghost-server
# or:
python3 -m ghost.server
# (legacy command `python3 -m src.server` also still works in this repo)
```

For the desktop GUI:

```bash
pip install -e "/path/to/ghost[server,gui]"
python3 examples/gui/gui.py
```
