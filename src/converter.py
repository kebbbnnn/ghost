"""
converter.py — SSE stream parser: converts ChatGPT SSE → OpenAI format.
"""
import json
import time
import uuid


MODEL_MAP = {
    # Map common OpenAI names to guest-available ChatGPT slugs
    "gpt-4o":               "auto",
    "gpt-4o-mini":          "auto",
    "gpt-4":                "auto",
    "gpt-4-turbo":          "auto",
    "gpt-3.5-turbo":        "auto",
    "gpt-3.5-turbo-16k":    "auto",
    "o1":                   "auto",
    "o1-mini":              "auto",
    "o3":                   "auto",
    "gpt-5-5":              "gpt-5-5",
    "auto":                 "auto",
}
DEFAULT_MODEL = "auto"


def map_model(requested: str) -> str:
    """Map an OpenAI model name to a ChatGPT guest model slug."""
    return MODEL_MAP.get(requested, DEFAULT_MODEL)


def parse_chatgpt_sse(line: str) -> dict | None:
    """
    Parse a single SSE line from the ChatGPT backend.
    Returns the parsed JSON data dict or None if the line should be skipped.
    """
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def chatgpt_event_to_openai_chunk(event: dict, model: str, chunk_id: str) -> dict | None:
    """
    Convert a parsed ChatGPT SSE event into an OpenAI-compatible chunk dict.
    Returns None if the event carries no useful content.
    """
    if event.get("done"):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    msg = event.get("message")
    if not msg:
        return None

    if msg.get("author", {}).get("role") != "assistant":
        return None

    content_obj = msg.get("content", {})
    parts = content_obj.get("parts", [])
    text_delta = ""
    for part in parts:
        if isinstance(part, str):
            text_delta = part
            break

    status = msg.get("status", "")
    finish_reason = None
    if status in ("finished_successfully", "stopped"):
        finish_reason = "stop"
    elif status == "max_tokens":
        finish_reason = "length"

    if not text_delta and not finish_reason:
        return None

    delta = {}
    if text_delta:
        delta["content"] = text_delta
    if msg.get("author", {}).get("role") == "assistant" and text_delta:
        delta["role"] = "assistant"

    chunk_id_to_use = msg.get("id", chunk_id)
    return {
        "id": chunk_id_to_use,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "conversation_id": event.get("conversation_id"),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


def format_openai_chunk(chunk: dict) -> str:
    """Format a chunk dict as an SSE data line."""
    return f"data: {json.dumps(chunk)}\n\n"


def build_non_streaming_response(content: str, model: str, finish_reason: str = "stop") -> dict:
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
