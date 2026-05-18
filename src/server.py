"""
server.py — FastAPI proxy server exposing an OpenAI-compatible API
backed by isolated platform providers (ChatGPT, Claude, Gemini, etc.).

Endpoints:
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions
"""
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

SERVER_START_TIME = int(time.time())

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

    if req.stream:
        return StreamingResponse(
            provider.stream_chat(messages, req.model, req.conversation_id, req.parent_message_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            resp_dict = await provider.non_streaming_chat(messages, req.model, req.conversation_id, req.parent_message_id)
            return JSONResponse(resp_dict)
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            log.exception("Non-streaming error")
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
            
    return {
        "status": "ok",
        "start_time": SERVER_START_TIME,
        "providers_count": len(PROVIDERS),
        "models_count": len(models),
        "claude": claude_data
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8100, reload=False)
