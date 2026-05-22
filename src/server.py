"""
server.py — FastAPI proxy server exposing an OpenAI-compatible API
backed by isolated platform providers (ChatGPT, Claude, Gemini, etc.).

Endpoints:
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions
"""
import logging
import json
import asyncio
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

SERVER_START_TIME = int(time.time())
ACCOUNTS_FILE = os.path.join(os.path.dirname(__file__), "providers/claude/accounts.json")
THREADS_FILE = os.path.join(os.path.dirname(__file__), "threads.json")
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
            try:
                async for chunk in provider.stream_chat(messages, req.model, req.conversation_id, req.parent_message_id):
                    await broadcast_event("request_token", {
                        "id": request_id,
                        "model": req.model,
                    })
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
            await broadcast_event("request_token", {
                "id": request_id,
                "model": req.model,
            })
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
    from src.providers.claude.session import ClaudeProvider
    
    models = await get_all_models()
    
    claude_data = {}
    for p in PROVIDERS:
        if isinstance(p, ClaudeProvider):
            # Access rotator state
            await p.rotator._load_accounts_if_changed()
            claude_data = {
                "accounts": p.rotator.active_accounts,
                "cooldowns": p.rotator.cooldowns,
                "current_index": p.rotator.current_index
            }
            break

    raw_accounts = []
    if os.path.exists(ACCOUNTS_FILE):
        try:
            with open(ACCOUNTS_FILE, "r") as f:
                raw_accounts = json.load(f).get("accounts", [])
        except Exception:
            pass
            
    return {
        "status": "ok",
        "start_time": SERVER_START_TIME,
        "providers_count": len(PROVIDERS),
        "models_count": len(models),
        "claude": claude_data,
        "raw_accounts": raw_accounts
    }


class ClaudeAccountAddRequest(BaseModel):
    email: str
    sessionKey: str


@app.post("/api/accounts/add")
async def add_claude_account(req: ClaudeAccountAddRequest):
    # Load existing accounts
    accounts_data = {"accounts": []}
    if os.path.exists(ACCOUNTS_FILE):
        try:
            with open(ACCOUNTS_FILE, "r") as f:
                accounts_data = json.load(f)
        except Exception:
            pass

    # Check if email already exists and update, or append new
    updated = False
    for account in accounts_data.get("accounts", []):
        if account.get("email") == req.email:
            account["sessionKey"] = req.sessionKey
            account["added_at"] = datetime.now(tz=timezone.utc).isoformat()
            account["status"] = "active"
            updated = True
            break

    if not updated:
        accounts_data.setdefault("accounts", []).append({
            "email": req.email,
            "sessionKey": req.sessionKey,
            "added_at": datetime.now(tz=timezone.utc).isoformat(),
            "status": "active"
        })

    try:
        os.makedirs(os.path.dirname(ACCOUNTS_FILE), exist_ok=True)
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump(accounts_data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save account: {e}")

    # Force rotator reloading
    from src.providers.claude.session import ClaudeProvider
    for p in PROVIDERS:
        if isinstance(p, ClaudeProvider):
            await p.rotator._load_accounts_if_changed()

    return {"status": "ok"}


@app.delete("/api/accounts/{index}")
async def delete_claude_account(index: int):
    # Load existing accounts
    if not os.path.exists(ACCOUNTS_FILE):
        raise HTTPException(status_code=404, detail="Accounts file not found")

    try:
        with open(ACCOUNTS_FILE, "r") as f:
            accounts_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read accounts: {e}")

    accounts = accounts_data.get("accounts", [])
    if index < 0 or index >= len(accounts):
        raise HTTPException(status_code=400, detail="Invalid account index")

    deleted = accounts.pop(index)

    try:
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump(accounts_data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save accounts: {e}")

    # Force rotator reloading
    from src.providers.claude.session import ClaudeProvider
    for p in PROVIDERS:
        if isinstance(p, ClaudeProvider):
            await p.rotator._load_accounts_if_changed()

    return {"status": "ok", "deleted": deleted.get("email")}


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8100, reload=False)
