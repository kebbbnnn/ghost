"""
converter.py — SurfSense SSE (Vercel AI SDK protocol) → OpenAI format converter.

SurfSense streams events in the Vercel AI SDK wire format:
    data: {"type": "text-delta", "id": "...", "delta": "Hello"}
    data: {"type": "finish", "usage": {...}}
    data: [DONE]

This module parses those events and converts them into OpenAI-compatible
streaming chunks.
"""
import json
import time
import uuid
import logging

log = logging.getLogger(__name__)


def parse_surfsense_sse_line(line: str) -> dict | None:
    """Parse a single SSE data line from SurfSense's stream.

    Expects lines in the form ``data: <json>`` or ``data: [DONE]``.
    Returns a parsed dict, or None for blank / unparseable lines.
    """
    line = line.strip()
    if not line:
        return None

    if not line.startswith("data: "):
        return None

    payload = line[6:]

    if payload == "[DONE]":
        return {"type": "__done__"}

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        log.debug("Unparseable SSE payload: %r", payload[:200])
        return None


def surfsense_event_to_delta(event: dict) -> str | None:
    """Extract the text delta from a Vercel AI SDK event.

    Returns the incremental text string, or None if the event is not
    a text-producing event.

    Wire format observed:
        {"type": "text-delta", "id": "...", "delta": "Hello"}
    """
    event_type = event.get("type", "")

    if event_type == "text-delta":
        # The live wire uses "delta", not "textDelta"
        return event.get("delta") or event.get("textDelta", "")

    # Reasoning tokens — fold into output for OpenAI compat
    if event_type == "reasoning-delta":
        return event.get("delta") or event.get("textDelta", "")

    return None


def is_done_event(event: dict) -> bool:
    """Return True if this event signals end-of-stream."""
    if event.get("type") == "__done__":
        return True
    if event.get("type") == "finish":
        return True
    return False


def is_error_event(event: dict) -> dict | None:
    """Return error dict if this event is an error, else None."""
    if event.get("type") == "error":
        return {
            "message": event.get("message", "Unknown error from SurfSense"),
            "type": "server_error",
        }
    if "error" in event:
        err = event["error"]
        return {
            "message": err.get("message", str(err)),
            "type": err.get("code", "server_error"),
        }
    return None


def build_openai_streaming_chunk(
    delta: str,
    model: str,
    chunk_id: str,
    finish_reason: str | None = None,
) -> dict:
    """Build a single OpenAI-compatible streaming completion chunk."""
    choice = {
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason,
    }
    if delta:
        choice["delta"] = {"content": delta}

    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }


def format_openai_chunk(chunk: dict) -> str:
    """Format a chunk dict as an SSE data line."""
    return f"data: {json.dumps(chunk)}\n\n"


def build_non_streaming_response(
    content: str,
    model: str,
    finish_reason: str = "stop",
) -> dict:
    """Build a complete (non-streaming) OpenAI chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
