"""
server.py — FastAPI proxy server exposing an OpenAI-compatible API
backed by isolated platform providers (ChatGPT, Gemini, etc.).

Endpoints:
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions
  POST /api/video/chat
"""
import logging
import json
import asyncio
import os


from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

SERVER_START_TIME = int(time.time())

from src.core.config import THREADS_FILE
listeners = []

async def broadcast_event(event_type: str, data: dict):
    if not listeners:
        return
    dead_listeners = []
    for queue in listeners:
        try:
            await queue.put({"event": event_type, "data": data})
        except Exception:
            dead_listeners.append(queue)
    for q in dead_listeners:
        if q in listeners:
            listeners.remove(q)


from contextlib import asynccontextmanager
from src.providers import get_all_models, get_provider_for_model, PROVIDERS
from src.core.exceptions import RateLimitError
from src.core.video_processor import extract_keyframes
from src.converters.anthropic import (
    anthropic_to_openai_request,
    openai_to_anthropic_stream,
    openai_to_anthropic_json,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks for all providers
    for p in PROVIDERS:
        await p.start_background_tasks()
    yield
    # Stop background tasks on shutdown
    for p in PROVIDERS:
        await p.stop_background_tasks()

app = FastAPI(
    title="GhostTunnel Multi-Platform Proxy",
    description="OpenAI-compatible API proxy using isolated guest sessions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount dashboard static files
import os
dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard/public")
app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    models = await get_all_models()
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    models = await get_all_models()
    for m in models:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict | None = None
    conversation_id: str | None = None
    parent_message_id: str | None = None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Route to the correct provider based on requested model
    provider = get_provider_for_model(req.model)
    if not provider:
        raise HTTPException(status_code=400, detail=f"No provider supports model '{req.model}'")

    messages = [m.model_dump() for m in req.messages]

    request_id = str(time.time())

    if req.stream:
        async def telemetry_stream_generator():
            await broadcast_event("request_start", {
                "id": request_id,
                "model": req.model,
                "stream": True,
                "messages_count": len(messages),
            })
            token_count = 0
            try:
                async for chunk in provider.stream_chat(messages, req.model, req.conversation_id, req.parent_message_id):
                    token_count += 1
                    yield chunk
            except Exception as e:
                await broadcast_event("request_error", {
                    "id": request_id,
                    "error": str(e),
                })
                raise e
            finally:
                await broadcast_event("request_end", {
                    "id": request_id,
                    "tokens": token_count,
                })

        return StreamingResponse(
            telemetry_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        await broadcast_event("request_start", {
            "id": request_id,
            "model": req.model,
            "stream": False,
            "messages_count": len(messages),
        })
        try:
            resp_dict = await provider.non_streaming_chat(messages, req.model, req.conversation_id, req.parent_message_id)
            await broadcast_event("request_end", {
                "id": request_id,
            })
            return JSONResponse(resp_dict)
        except RateLimitError as e:
            await broadcast_event("request_error", {
                "id": request_id,
                "error": str(e),
            })
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            log.exception("Non-streaming error")
            await broadcast_event("request_error", {
                "id": request_id,
                "error": str(e),
            })
            raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------------
# Video Chat (All-in-one: upload + extract + complete)
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
    """All-in-one video understanding endpoint.

    Accepts a video file + prompt via multipart form data.
    Extracts keyframes server-side using OpenCV + pHash deduplication,
    constructs a timestamped multi-image message, and streams the
    model's response back as SSE.
    """
    # Validate content type
    content_type = video.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Expected a video file, got '{content_type}'")

    # Read video bytes
    video_bytes = await video.read()
    if len(video_bytes) > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(status_code=413, detail="Video file too large (max 100MB)")

    log.info("Video chat: received %s (%.1f MB), model=%s", video.filename, len(video_bytes) / 1e6, model)

    # Extract keyframes in a thread pool (CPU-bound OpenCV work)
    try:
        keyframes = await asyncio.to_thread(extract_keyframes, video_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Video frame extraction failed")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {e}")

    if not keyframes:
        raise HTTPException(status_code=400, detail="No keyframes could be extracted from the video.")

    log.info("Extracted %d keyframes from video", len(keyframes))

    # Build the user message content array with timestamped frames
    content_parts = []
    for kf in keyframes:
        content_parts.append({"type": "text", "text": f"[Frame at {kf['timestamp']}]"})
        content_parts.append({"type": "image_url", "image_url": {"url": kf["base64"]}})
    content_parts.append({"type": "text", "text": prompt})

    user_message = {"role": "user", "content": content_parts}

    # Parse prior conversation history
    try:
        prior_messages = json.loads(history)
    except (json.JSONDecodeError, TypeError):
        prior_messages = []

    messages = prior_messages + [user_message]

    # Route to provider
    provider = get_provider_for_model(model)
    if not provider:
        raise HTTPException(status_code=400, detail=f"No provider supports model '{model}'")

    request_id = str(time.time())

    # Stream the response (same SSE format as /v1/chat/completions)
    async def video_stream_generator():
        await broadcast_event("request_start", {
            "id": request_id,
            "model": model,
            "stream": True,
            "messages_count": len(messages),
            "video": True,
            "keyframes": len(keyframes),
        })
        token_count = 0
        try:
            async for chunk in provider.stream_chat(messages, model, conversation_id, parent_message_id):
                token_count += 1
                yield chunk
        except Exception as e:
            await broadcast_event("request_error", {
                "id": request_id,
                "error": str(e),
            })
            raise e
        finally:
            await broadcast_event("request_end", {
                "id": request_id,
                "tokens": token_count,
            })

    return StreamingResponse(
        video_stream_generator(),
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
async def anthropic_messages(req: Request):
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
        
    try:
        openai_req = anthropic_to_openai_request(body)
    except Exception as e:
        log.exception("Error converting Anthropic request")
        raise HTTPException(status_code=400, detail=str(e))
        
    model = openai_req.get("model", "auto")
    provider = get_provider_for_model(model)
    if not provider:
        raise HTTPException(status_code=400, detail=f"No provider supports model '{model}'")
        
    messages = openai_req.get("messages", [])
    stream = openai_req.get("stream", False)
    
    if stream:
        return StreamingResponse(
            openai_to_anthropic_stream(provider.stream_chat(messages, model), model=model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            resp_dict = await provider.non_streaming_chat(messages, model)
            anthropic_resp = await openai_to_anthropic_json(resp_dict)
            return JSONResponse(anthropic_resp)
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            log.exception("Non-streaming error")
            raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------------
# Health & Status check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/status")
async def get_status():
    models = await get_all_models()
            
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
        with open(THREADS_FILE, "r") as f:
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
    
    try:
        with open(THREADS_FILE, "w") as f:
            json.dump(threads, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        log.error(f"Error saving threads: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save threads: {e}")


@app.get("/api/events")
async def sse_events(request: Request):
    queue = asyncio.Queue()
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
        }
    )



# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _free_port(port: int) -> None:
    """Kill any process currently listening on *port* so we can bind cleanly."""
    import subprocess, signal
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                log.info("Freed port %d by killing PID %s", port, pid)
            except ProcessLookupError:
                pass
    except FileNotFoundError:
        pass  # lsof not available on this platform


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8100))
    _free_port(port)
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=False)
