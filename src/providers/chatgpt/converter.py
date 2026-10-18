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
        parsed = json.loads(payload)
        # Guard: the new delta-encoding format sends `data: "v1"` (a JSON string, not an object).
        # Return None for anything that isn't a dict so callers can safely call .get().
        if not isinstance(parsed, dict):
            return None
        return parsed
    except json.JSONDecodeError:
        return None



def chatgpt_event_to_openai_chunk(event: dict, model: str, chunk_id: str) -> dict | None:
    """
    Convert a parsed ChatGPT SSE event into an OpenAI-compatible chunk dict.
    Returns None if the event carries no useful content.

    Handles two formats:
    1. Legacy format: {"message": {"author": {"role": "assistant"}, "content": {...}}}
    2. New delta format: {"p": "/message/content/parts/0", "o": "append", "v": "text"}
       or continuation: {"v": "text"} (shorthand delta without path)
    """
    if not isinstance(event, dict):
        return None

    # --- [DONE] marker ---
    if event.get("done"):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    # --- message_stream_complete signals end of stream ---
    if event.get("type") == "message_stream_complete":
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    # --- New delta-encoding format ---
    event_v = event.get("v")
    event_o = event.get("o")

    # Case 1: event carries a list of patch operations:
    # {"p": "", "o": "patch", "v": [...]} or shorthand {"v": [...]}
    if isinstance(event_v, list):
        text_parts = []
        is_finished = False
        for op in event_v:
            if isinstance(op, dict):
                p_path = op.get("p", "")
                if op.get("o") == "append" and isinstance(op.get("v"), str):
                    # Strictly capture only content parts, ignoring metadata/fallbackMarkdown
                    if not p_path or p_path.startswith("/message/content/parts"):
                        text_parts.append(op["v"])
                elif p_path == "/message/end_turn" and op.get("v") is True:
                    is_finished = True

        if text_parts:
            combined = "".join(text_parts).replace("<escape>", "").replace("</escape>", "")
            if combined:
                return {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": combined}, "finish_reason": "stop" if is_finished else None}],
                }
        if is_finished:
            return {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        return None

    # Case 2: String delta:
    # Either {"p": "/message/content/parts/0", "o": "append", "v": "text"}
    # or shorthand continuation {"v": "text"}
    if event_v is not None and isinstance(event_v, str) and (event_o in ("append", None)):
        cleaned_v = event_v.replace("<escape>", "").replace("</escape>", "")
        if cleaned_v:
            role = "assistant" if ("o" in event or event.get("p") == "/message/content/parts/0") else None
            delta_dict = {"content": cleaned_v}
            if role:
                delta_dict["role"] = role
            return {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta_dict, "finish_reason": None}],
            }
        return None

    # --- Legacy format or message initiation: {"message": {...}} or {"v": {"message": {...}}} ---
    msg = event.get("message")
    if not msg and isinstance(event_v, dict):
        msg = event_v.get("message")

    if not msg:
        return None

    if msg.get("author", {}).get("role") != "assistant":
        return None

    content_obj = msg.get("content", {})
    parts = content_obj.get("parts", [])
    text_delta = ""
    for part in parts:
        if isinstance(part, str):
            text_delta = part.replace("<escape>", "").replace("</escape>", "")
            break

    # Only assign finish_reason if this message is addressed to the user (not an internal tool recipient)
    recipient = msg.get("recipient", "all")
    status = msg.get("status", "")
    finish_reason = None
    if recipient in ("all", None, ""):
        if status in ("finished_successfully", "stopped"):
            finish_reason = "stop"
        elif status == "max_tokens":
            finish_reason = "length"

    if not text_delta:
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
