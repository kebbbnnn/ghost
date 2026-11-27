"""
ghost/server.py — Thin FastAPI proxy server delegating to GhostEngine.

Exposes OpenAI-compatible and Anthropic-compatible HTTP endpoints plus the
built-in dashboard, thread persistence, and SSE telemetry stream.

Endpoints:
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions
  POST /v1/messages (plus /v1/v1/messages, /messages)
  POST /v1/messages/count_tokens
  POST /api/video/chat
  GET  /health, /api/status, /api/threads, /api/events
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ghost.core.config import THREADS_FILE, ensure_data_dir
from ghost.core.exceptions import ModelNotFoundError, RateLimitError
from ghost.engine import (
    ChatCompletionRequest,
    ChatMessage,
    GhostEngine,
    is_security_monitor_prompt as _is_security_monitor_prompt,
    make_resilient_stream,
    make_security_monitor_fallback_response as _make_security_monitor_fallback_response,
)
from ghost.providers import (
    PROVIDERS,
    _default_registry,
    get_all_models,
    get_fallback_provider,
    get_provider_for_model,
)

log = logging.getLogger(__name__)

SERVER_START_TIME = int(time.time())
listeners: list[asyncio.Queue] = []


async def broadcast_event(event_type: str, data: dict) -> None:
    if not listeners:
        return
    dead_listeners = []
    for queue in listeners:
        try:
            queue.put_nowait({"event": event_type, "data": data})
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except Exception:
                pass
            try:
                queue.put_nowait({"event": event_type, "data": data})
            except Exception:
                dead_listeners.append(queue)
        except Exception:
            dead_listeners.append(queue)
    for q in dead_listeners:
        if q in listeners:
            listeners.remove(q)


# Shared engine bound dynamically to module-level provider helpers so
# monkeypatching ``ghost.server.get_provider_for_model`` or ``get_all_models``
# continues to work seamlessly in tests.
engine = GhostEngine(
    registry=_default_registry,
    raw_logging=None,  # Respect global RAW_LOGGING_ENABLED in server mode
    on_event=broadcast_event,
    provider_resolver=lambda model: get_provider_for_model(model),
    fallback_resolver=lambda prov: get_fallback_provider(prov),
    models_resolver=lambda: get_all_models(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dir()
    for p in PROVIDERS:
        await p.start_background_tasks()
    yield
    for p in PROVIDERS:
        await p.stop_background_tasks()


app = FastAPI(
    title="Ghost Multi-Platform Proxy",
    description="OpenAI-compatible API proxy using isolated guest sessions",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response header & error builders
# ---------------------------------------------------------------------------

def make_openai_headers(
    request_id: str,
    elapsed_ms: int = 0,
    conversation_id: str | None = None,
    parent_message_id: str | None = None,
    is_stream: bool = False,
) -> dict[str, str]:
    headers = {
        "X-Request-ID": f"req_{request_id.replace('.', '')}",
        "OpenAI-Organization": "org-ghost",
        "OpenAI-Processing-Ms": str(max(1, elapsed_ms)),
        "OpenAI-Version": "2020-10-01",
    }
    if is_stream:
        headers.update(
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        headers["Content-Type"] = "application/json"
    if conversation_id:
        headers["X-Ghost-Conversation-Id"] = conversation_id
    if parent_message_id:
        headers["X-Ghost-Parent-Message-Id"] = parent_message_id
    return headers


def make_anthropic_headers(
    request_id: str,
    conversation_id: str | None = None,
    parent_message_id: str | None = None,
    is_stream: bool = False,
    anthropic_beta: str | None = None,
) -> dict[str, str]:
    headers = {
        "Request-ID": f"req_{request_id.replace('.', '')}",
        "Anthropic-Version": "2023-06-01",
        "Anthropic-Ratelimit-Requests-Limit": "1000",
        "Anthropic-Ratelimit-Requests-Remaining": "999",
        "Anthropic-Ratelimit-Tokens-Limit": "100000",
        "Anthropic-Ratelimit-Tokens-Remaining": "99000",
    }
    if anthropic_beta:
        headers["anthropic-beta"] = anthropic_beta
    if is_stream:
        headers.update(
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        headers["Content-Type"] = "application/json"
    if conversation_id:
        headers["X-Ghost-Conversation-Id"] = conversation_id
    if parent_message_id:
        headers["X-Ghost-Parent-Message-Id"] = parent_message_id
    return headers


def make_openai_error_response(
    message: str,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    headers = make_openai_headers(request_id) if request_id else {}
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
        headers=headers,
    )


def make_anthropic_error_response(
    message: str,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    request_id: str | None = None,
) -> JSONResponse:
    headers = make_anthropic_headers(request_id) if request_id else {}
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    path = request.url.path
    if "messages" in path:
        return make_anthropic_error_response(str(exc), status_code=400)
    if path.startswith("/v1/chat/completions"):
        return make_openai_error_response(str(exc), status_code=400)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ---------------------------------------------------------------------------
# Dashboard static files
# ---------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gui_dashboard = os.path.join(project_root, "examples", "gui", "dashboard", "public")
examples_dashboard = os.path.join(project_root, "examples", "dashboard", "public")
src_dashboard = os.path.join(os.path.dirname(__file__), "dashboard", "public")

dashboard_path = None
for candidate in (gui_dashboard, examples_dashboard, src_dashboard):
    if os.path.exists(candidate):
        dashboard_path = candidate
        break

if dashboard_path:
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    if dashboard_path:
        return RedirectResponse(url="/dashboard/")
    return {"name": "Ghost API", "status": "running"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    models = await engine.list_models()
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    model = await engine.get_model(model_id)
    if model is not None:
        return model
    return make_openai_error_response(
        f"Model '{model_id}' not found",
        status_code=404,
        error_type="invalid_request_error",
        code="model_not_found",
    )


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        try:
            plan = engine.prepare_chat_stream(req, emit_error_chunks=True)
        except ModelNotFoundError as exc:
            return make_openai_error_response(
                str(exc),
                status_code=400,
                error_type="invalid_request_error",
                request_id=str(time.time()),
            )
        stream_headers = make_openai_headers(
            request_id=plan.request_id,
            conversation_id=plan.conversation_id,
            parent_message_id=plan.parent_message_id,
            is_stream=True,
        )
        return StreamingResponse(
            plan.stream,
            media_type="text/event-stream",
            headers=stream_headers,
        )

    request_id = str(time.time())
    try:
        result = await engine.chat_completion_with_metadata(req)
        resp_headers = make_openai_headers(
            request_id=result.request_id,
            elapsed_ms=result.elapsed_ms,
            conversation_id=result.conversation_id,
            parent_message_id=result.parent_message_id,
            is_stream=False,
        )
        return JSONResponse(result.data, headers=resp_headers)
    except ModelNotFoundError as exc:
        return make_openai_error_response(
            str(exc), status_code=400, error_type="invalid_request_error", request_id=request_id
        )
    except RateLimitError as exc:
        return make_openai_error_response(
            str(exc),
            status_code=429,
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            request_id=request_id,
        )
    except Exception as exc:
        log.exception("Non-streaming error")
        return make_openai_error_response(
            str(exc), status_code=500, error_type="server_error", request_id=request_id
        )


# ---------------------------------------------------------------------------
# Video Chat
# ---------------------------------------------------------------------------

@app.post("/api/video/chat")
async def video_chat(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form("auto"),
    conversation_id: str = Form(None),
    parent_message_id: str = Form(None),
    history: str = Form("[]"),
):
    content_type = video.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Expected a video file, got '{content_type}'")

    video_bytes = await video.read()
    if len(video_bytes) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video file too large (max 100MB)")

    log.info(
        "Video chat: received %s (%.1f MB), model=%s",
        video.filename,
        len(video_bytes) / 1e6,
        model,
    )

    try:
        prior_messages = json.loads(history)
    except (json.JSONDecodeError, TypeError):
        prior_messages = []

    try:
        plan = await engine.prepare_video_chat_stream(
            video_bytes,
            prompt,
            model=model,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            history=prior_messages,
            filename=video.filename or "video",
            emit_error_chunks=True,
        )
    except (ValueError, ModelNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        log.exception("Video frame extraction failed")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {exc}")

    return StreamingResponse(
        plan.stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
@app.post("/v1/v1/messages")
@app.post("/messages")
async def anthropic_messages(req: Request):
    request_id = str(time.time())
    try:
        body = await req.json()
    except Exception:
        return make_anthropic_error_response("Invalid JSON", status_code=400, request_id=request_id)

    beta_header = req.headers.get("anthropic-beta", "")
    stream = bool(body.get("stream", False))

    if stream:
        try:
            plan = engine.prepare_anthropic_stream(
                body, beta_header=beta_header, emit_error_chunks=True
            )
        except (ValueError, ModelNotFoundError) as exc:
            return make_anthropic_error_response(str(exc), status_code=400, request_id=request_id)

        stream_headers = make_anthropic_headers(
            request_id=plan.request_id,
            conversation_id=plan.conversation_id,
            parent_message_id=plan.parent_message_id,
            is_stream=True,
            anthropic_beta=plan.effective_beta,
        )
        return StreamingResponse(
            plan.stream,
            media_type="text/event-stream",
            headers=stream_headers,
        )

    try:
        result = await engine.anthropic_message_with_metadata(body, beta_header=beta_header)
        resp_headers = make_anthropic_headers(
            request_id=result.request_id,
            conversation_id=result.conversation_id,
            parent_message_id=result.parent_message_id,
            is_stream=False,
            anthropic_beta=result.effective_beta,
        )
        return JSONResponse(result.data, headers=resp_headers)
    except (ValueError, ModelNotFoundError) as exc:
        return make_anthropic_error_response(str(exc), status_code=400, request_id=request_id)
    except RateLimitError as exc:
        return make_anthropic_error_response(
            str(exc), status_code=429, error_type="rate_limit_error", request_id=request_id
        )
    except Exception as exc:
        log.exception("Non-streaming error")
        return make_anthropic_error_response(
            str(exc), status_code=500, error_type="api_error", request_id=request_id
        )


@app.api_route("/api/hello", methods=["GET", "HEAD"])
async def anthropic_hello():
    return {"status": "ok"}


@app.post("/v1/messages/count_tokens")
@app.post("/messages/count_tokens")
async def anthropic_count_tokens(req: Request):
    try:
        body = await req.json()
    except Exception:
        return {"input_tokens": 100}
    return engine.count_tokens(body)


@app.post("/api/event_logging/batch")
async def anthropic_event_logging_batch():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Health & Status check
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def get_status():
    models = await engine.list_models()
    return {
        "status": "ok",
        "start_time": SERVER_START_TIME,
        "providers_count": len(PROVIDERS),
        "models_count": len(models),
    }


# ---------------------------------------------------------------------------
# Threads Persistence API
# ---------------------------------------------------------------------------

@app.get("/api/threads")
async def get_threads_api():
    if not os.path.exists(THREADS_FILE):
        return []
    try:
        with open(THREADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error loading threads: {e}")
        return []


@app.post("/api/threads")
async def save_threads_api(req: Request):
    try:
        threads = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(threads, (list, dict)):
        raise HTTPException(status_code=400, detail="Payload must be a list or dict")

    temp_file = f"{THREADS_FILE}.tmp.{os.getpid()}"
    try:
        ensure_data_dir()
        os.makedirs(os.path.dirname(THREADS_FILE), exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(threads, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, THREADS_FILE)
        return {"status": "success"}
    except Exception as e:
        log.error(f"Error saving threads: {e}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to save threads: {e}")


@app.get("/api/events")
async def sse_events(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    listeners.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            if queue in listeners:
                listeners.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _free_port(port: int) -> None:
    """Kill any process currently listening on *port* so we can bind cleanly."""
    import signal
    import subprocess

    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                log.info("Freed port %d by killing PID %s", port, pid)
            except ProcessLookupError:
                pass
    except FileNotFoundError:
        pass


def main() -> None:
    """CLI entrypoint for ``ghost-server`` and ``python -m ghost.server``."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 8100))
    _free_port(port)
    uvicorn.run("ghost.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
