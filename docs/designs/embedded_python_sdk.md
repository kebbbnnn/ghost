# Embedded Python SDK & In-Process Drop-In Design

> **Status:** Approved  
> **Date:** 2026-11-27

---

## 1. Understanding Summary

- **What is being built**: An installable Python package (`ghost`, via `pyproject.toml`) that lets any Python project embed Ghost in-process with zero server/port required, providing:
  1. A **direct Python client** (`Ghost` for sync, `AsyncGhost` for async) covering all providers (`auto`/ChatGPT, `gemini`, `gemini-tree-of-thoughts`, `spectre`, `spectre-ultra`), streaming & non-streaming, multi-turn IDs, parameter/tool emulation (`EmulationPipeline`), automatic provider failover, and Anthropic Messages support.
  2. **True OpenAI & Anthropic SDK drop-in compatibility** — pure-`httpx` in-process transports and factory helpers so real `openai.OpenAI` / `openai.AsyncOpenAI` and `anthropic.Anthropic` / `anthropic.AsyncAnthropic` clients (and frameworks built on them) talk directly to Ghost in memory with no HTTP server or FastAPI dependency.
- **Why it exists**: Eliminate the operational friction of running and managing a separate `uvicorn`/`FastAPI` proxy process and port when integrating Ghost into another Python codebase.
- **Who it is for**: Python host projects installing Ghost via local/editable path (`pip install -e ../ghost`), git URL (`pip install git+...`), or wheel, plus existing users of this repo's server/GUI/Docker setup.
- **Key constraints**:
  - Core install stays lean (`curl-cffi`, `httpx[http2]`, `pydantic`, `json-repair`); `fastapi`, `uvicorn`, `pywebview`, `opencv-python-headless`, `numpy`, `Pillow`, `openai`, and `anthropic` live in optional extras.
  - Zero unexpected import-time side effects (no creating `./data`, no writing `debug_raw.log` into the host project, no starting background loops or opening HTTP connections on `import ghost`).
  - Backwards compatibility in this repo via a dev-only `src/` `sys.modules` alias (excluded from built wheels) so `python -m src.server`, Docker, GUI, and existing tests continue working unchanged.
  - Fix the pre-existing duplicate `def _build_synthesis_prompt` `IndentationError` in `providers/gemini/tot.py`.
- **Explicit non-goals**:
  - Non-Python language bindings (Node/TS/Go/CLI stdio bridge).
  - Publishing to public PyPI.
  - Sharing guest sessions across separate OS processes without a server (multi-process apps needing shared session pooling should still run server mode).
  - Bundling the dashboard/threads persistence API into the library client.

---

## 2. Non-Functional Requirements & Assumptions

- **Performance & cold-start**: Lazy session initialization by default (zero startup cost on `import` or client construction). Long-lived host apps can opt into `warm_up=True` (via context manager or `start()`/`close()`) to keep PoW/CSRF tokens warm just like the server does.
- **Concurrency & isolation**: Each `Ghost` / `AsyncGhost` client owns isolated provider instances via `ProviderRegistry`, preventing cross-event-loop `asyncio.Lock` and HTTP client conflicts.
- **Filesystem & logging hygiene**: Raw trace file logging defaults to **off** in library mode (`raw_logging=False`, opt-in via client option or `GHOST_RAW_LOGGING=true`); `logging.basicConfig()` is never invoked on library import; `DATA_DIR` is never created on import.
- **Reliability & error semantics**: Automatic failover to a fallback provider on initialization/upstream error, while **propagating `RateLimitError` (429)** without silent failover so callers can apply backoff.
- **Optional media dependencies**: OpenCV/NumPy video extraction stays behind `ghost[video]`; Pillow dimension detection is lazy-imported behind `ghost[vision]`.
- **Maintenance**: `ghost/server.py` delegates chat, model, and Anthropic execution to the shared `GhostEngine`, keeping a single source of truth.

---

## 3. Decision Log

| Decision | Alternatives Considered | Why Chosen |
|---|---|---|
| Installable `ghost` package + dev-only `src/` `sys.modules` alias | Keep `src/` (collides in host projects); drop `src/` (breaks Docker/tests/scripts) | Avoids namespace collision in consumer environments while preserving 100% repo and `monkeypatch` compatibility |
| Server-free `GhostEngine` + pure-`httpx` in-process transport | `httpx.ASGITransport` over FastAPI; duck-typed client only | Zero FastAPI dependency for library consumers + real `openai`/`anthropic` SDK and framework compatibility |
| Per-client `ProviderRegistry` + persistent event-loop thread for sync `Ghost` | Module-global singletons; `asyncio.run()` per call | Prevents cross-loop lock errors, keeps guest sessions warm across sync calls, avoids deadlocks inside active loops (e.g. Jupyter) |
| Lazy init & `raw_logging=False` by default in library mode | Eager warm-up & auto-logging on import | Zero import-time network/disk side effects; opt-in `warm_up=True` / `raw_logging=True` when desired |

---

## 4. Architecture & Components

### Package Layout

```text
pyproject.toml               # packages: include = ["ghost*"] only (excludes src/)
ghost/
├── __init__.py              # Public API exports
├── engine.py                # Server-free GhostEngine orchestration core
├── client.py                # Direct sync (Ghost) & async (AsyncGhost) clients + _AsyncLoopRunner
├── adapters/
│   ├── __init__.py
│   └── httpx_transport.py   # GhostTransport, AsyncGhostTransport, SDK client factories
├── core/                    # BaseProvider (+ close()), exceptions, config, emulation, debug_logger, video_processor
├── converters/              # Anthropic <-> OpenAI converters
├── providers/
│   ├── __init__.py          # ProviderRegistry + backwards-compatible default registry
│   ├── chatgpt/
│   ├── gemini/
│   └── spectre/             # Dependency-injected provider_resolver
└── server.py                # Thin FastAPI server delegating to GhostEngine
src/
└── __init__.py              # Dev-only MetaPathFinder aliasing src.* -> ghost.* in sys.modules
```

### Core Components

1. **`ProviderRegistry` (`ghost/providers/__init__.py`)**
   - Instantiates isolated `ChatGPTProvider`, `GeminiProvider`, and `SpectreProvider(provider_resolver=...)` instances per client.
   - Handles model routing (`spectre*`, `gemini*`/`claude*`/`anthropic*`, `default_provider`, default ChatGPT), failover resolution, `get_all_models()`, `start_background_tasks()`, and `aclose()` (cancels background tasks and closes provider HTTP sessions).
   - Retains module-level default registry symbols (`PROVIDERS`, `get_all_models`, `get_provider_for_model`, `get_fallback_provider`) for server and test compatibility.

2. **`GhostEngine` (`ghost/engine.py`)**
   - Server-free core executing:
     - `list_models()` / `get_model(model_id)`
     - `chat_completion(req)` / `chat_completion_with_metadata(req)`
     - `stream_chat_completion_sse(req)` / `stream_chat_completion(req)`
     - `anthropic_message(body, ...)` / `anthropic_message_with_metadata(body, ...)`
     - `stream_anthropic_message_sse(body, ...)` / `stream_anthropic_message(body, ...)`
     - `count_tokens(body)`
     - `video_chat(...)` (lazy-imports `extract_keyframes` from `ghost.core.video_processor`)
   - Enforces `EmulationPipeline`, provider failover (preserving `RateLimitError`), and opt-in `RequestTrace` logging (`raw_logging=False` by default).

3. **Direct Clients (`ghost/client.py`)**
   - `AsyncGhost`: Native `asyncio` client exposing `.ask(...)`, `.chat.completions.create(...)`, `.messages.create(...)`, `.messages.count_tokens(...)`, `.models.list()`, `.models.retrieve(...)`, `.video_chat(...)`, `.start()`, `.aclose()`, and `async with`.
   - `Ghost`: Synchronous mirror backed by `_AsyncLoopRunner` (a dedicated daemon thread running a persistent `asyncio` event loop), bridging async generators over a bounded `queue.Queue` and cleanly closing upstream async generators on early caller exit.

4. **Pure-`httpx` In-Process Transports & SDK Factories (`ghost/adapters/httpx_transport.py`)**
   - `AsyncGhostTransport(httpx.AsyncBaseTransport)` and `GhostTransport(httpx.BaseTransport)` route OpenAI (`/v1/chat/completions`, `/v1/models`, `/v1/models/{id}`), Anthropic (`/v1/messages`, `/v1/v1/messages`, `/messages`, `/v1/messages/count_tokens`, `/messages/count_tokens`), and utility (`/health`, `/api/hello`, `/api/event_logging/batch`) requests directly to `GhostEngine` in memory, returning standard `httpx.Response` objects including incremental `AsyncByteStream`/`SyncByteStream` SSE streams and standard headers/error schemas.
   - Factory helpers: `httpx_client()`, `async_httpx_client()`, `openai_client()`, `async_openai_client()`, `anthropic_client()`, `async_anthropic_client()`.

---

## 5. Error Handling & Verification Plan

- **Errors**: `RateLimitError` propagates directly in Python clients and maps to HTTP 429 in transports; unknown models raise `ValueError` (HTTP 400/404 in transports); missing optional dependencies (`openai`, `anthropic`, `cv2`) raise clear `ImportError` installation instructions.
- **Verification**:
  1. All existing `tests/` pass via `src/` compatibility shims without regression.
  2. New `tests/test_sdk.py` verifies sync/async direct clients, streaming & early-break cancellation, tool emulation, provider failover, 429 propagation, Anthropic messages, per-client isolation, `warm_up`/close lifecycle, and `GhostTransport` / `AsyncGhostTransport` over real `httpx`.
  3. Packaging verification confirms only `ghost*` is packaged, `import ghost` succeeds with FastAPI/OpenCV/PyWebView unavailable and zero disk writes, and `python -m src.server` resolves cleanly.
