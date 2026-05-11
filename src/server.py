"""
server.py — FastAPI proxy server exposing an OpenAI-compatible API
backed by an anonymous ChatGPT guest session.

Endpoints:
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions
"""
import asyncio
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .session.guest_session import GuestSession, RateLimitError, TokenExpiredError
from .converter import (
    map_model,
    parse_chatgpt_sse,
    chatgpt_event_to_openai_chunk,
    format_openai_chunk,
    build_non_streaming_response,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="ChatGPT Anonymous Proxy",
    description="OpenAI-compatible API proxy using the ChatGPT guest session",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared session (refreshed automatically)
_session = GuestSession()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    {
        "id": "gpt-5-5",
        "object": "model",
        "created": 1746057600,
        "owned_by": "openai",
    },
    {
        "id": "auto",
        "object": "model",
        "created": 1746057600,
        "owned_by": "openai",
    },
]


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    for m in AVAILABLE_MODELS:
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
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict | None = None
    conversation_id: str | None = None
    parent_message_id: str | None = None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    target_model = map_model(req.model)
    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream_response(messages, target_model, req.conversation_id, req.parent_message_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _non_streaming_response(messages, target_model, req.conversation_id, req.parent_message_id)


async def _stream_response(messages: list, model: str, conversation_id: str | None = None, parent_message_id: str | None = None):
    """Generator that streams OpenAI-compatible SSE chunks."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    content_tracker = []

    try:
        # Retry once on token expiry
        for attempt in range(2):
            try:
                async for raw_line in _session.stream_chat(messages, model, conversation_id, parent_message_id):
                    event = parse_chatgpt_sse(raw_line)
                    if event is None:
                        continue

                    chunk = chatgpt_event_to_openai_chunk(event, model, chunk_id)
                    if chunk is None:
                        continue

                    # Track content for accurate delta streaming
                    choices = chunk.get("choices", [])
                    if choices:
                        delta_content = choices[0].get("delta", {}).get("content", "")
                        if delta_content:
                            new_text = _get_incremental_text(content_tracker, delta_content)
                            if new_text:
                                chunk["choices"][0]["delta"]["content"] = new_text
                            else:
                                chunk["choices"][0]["delta"].pop("content", None)

                    yield format_openai_chunk(chunk)

                    if choices and choices[0].get("finish_reason"):
                        break

                break  # success, exit retry loop

            except TokenExpiredError:
                if attempt == 0:
                    log.warning("Token expired, refreshing and retrying...")
                    await _session.ensure_ready()
                    content_tracker.clear()
                else:
                    raise

    except RateLimitError:
        error_chunk = {
            "error": {"message": "Rate limited by ChatGPT", "type": "rate_limit_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except Exception as exc:
        log.exception("Stream error")
        error_chunk = {"error": {"message": str(exc), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"


def _get_incremental_text(tracker: list, new_full_text: str) -> str:
    """Return only the new characters since last emission."""
    prev = tracker[0] if tracker else ""
    if new_full_text.startswith(prev):
        delta = new_full_text[len(prev):]
    else:
        delta = new_full_text
    if tracker:
        tracker[0] = new_full_text
    else:
        tracker.append(new_full_text)
    return delta


async def _non_streaming_response(messages: list, model: str, conversation_id: str | None = None, parent_message_id: str | None = None) -> JSONResponse:
    """Collect full streaming response and return as a single completion."""
    full_content = ""
    finish_reason = "stop"

    try:
        for attempt in range(2):
            try:
                async for raw_line in _session.stream_chat(messages, model, conversation_id, parent_message_id):
                    event = parse_chatgpt_sse(raw_line)
                    if not event or event.get("done"):
                        continue

                    msg = event.get("message", {})
                    parts = msg.get("content", {}).get("parts", [])
                    for part in parts:
                        if isinstance(part, str):
                            full_content = part  # ChatGPT sends full text each time

                    status = msg.get("status", "")
                    if status == "max_tokens":
                        finish_reason = "length"
                break
            except TokenExpiredError:
                if attempt == 0:
                    await _session.ensure_ready()
                else:
                    raise

    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by ChatGPT")
    except Exception as exc:
        log.exception("Non-streaming error")
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(build_non_streaming_response(full_content, model, finish_reason))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "session_ready": bool(_session._oai_did)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8100, reload=False)
