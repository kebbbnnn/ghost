"""
converter.py — Gemini batch-RPC response parser and OpenAI format converter.

Gemini's StreamGenerate endpoint returns data in Google's proprietary batch-RPC
format. The response contains MULTIPLE progressive JSON chunks — each chunk
has the full content generated so far (not incremental deltas).

This module handles:
  1. Parsing ALL chunks and extracting the final/longest content
  2. Parsing individual chunks for true streaming (incremental deltas)
  3. Converting parsed data into OpenAI-compatible response dicts
  4. Generating SSE streaming chunks
"""
import json
import logging
import re
import time
import uuid

from src.core.emulation.stream_parser import (
    ALL_TOOL_TAGS,
    strip_gemini_html_artifacts,
    strip_googleusercontent_artifacts,
)

log = logging.getLogger(__name__)

AUTOLINK_URL_PATTERN = re.compile(
    r"\[(https?://[^\s\[\]\(\)\"']+?)\]\(https?://[^\s\[\]\(\)\"']+?\)"
)

GEMINI_CANNED_REFUSALS = (
    "i'm having a hard time fulfilling your request",
    "i am having a hard time fulfilling your request",
    "i can't use google drive because required gmail settings are off",
    "i cannot use google drive because required gmail settings are off",
    "i'm still learning how to answer this",
    "i can't help with that right now",
    "cannot fulfill this request",
    "can't fulfill this request",
    "unable to process prompts that use",
    "automated developer roleplay",
    "system instruction overrides",
    "bypass safety policies",
    "execute unverified shell commands",
    "unable to process this request",
    "unable to fulfill this request",
    "cannot process this request",
    "i seem to be encountering an error",
    "encountering an error",
    "encountered an error",
    "could you try again",
    "can i try something else for you",
    "try something else for you",
    "something went wrong",
    "temporary issue. please try again",
    "automated tool calls are currently unavailable",
    "encountered restrictions",
    "restrictions in the workspace",
    "i am working on your request and will continue shortly",
    "will continue shortly",
)


def is_gemini_canned_refusal(text: str) -> bool:
    """Detect Gemini platform post-generation filter refusals or Workspace extension errors."""
    if not text:
        return False
    stripped = text.strip().lower()
    if any(re.search(rf"<{tag}\b", stripped) for tag in ALL_TOOL_TAGS):
        return False
    if len(stripped) > 800:
        return False
    return any(pat in stripped for pat in GEMINI_CANNED_REFUSALS)


def transpile_gemini_code_reference(content: str, prompt: str = "") -> str:
    """
    If Gemini invoked its internal server-side `data_analysis_tool` (emitting
    ```python?code_reference&code_event_index=N blocks instead of <tool_call>/<invoke>),
    transpile the generated Python script(s) into local Bash <tool_call> invocations
    so the client coding harness executes them on the user's local filesystem.
    """
    if not content or "```python?code_reference" not in content:
        return content

    outside_fences = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    if "<tool_call>" in outside_fences or "<invoke" in outside_fences:
        return content

    bash_tool_name = "Bash"
    if prompt:
        if '<tool name="Bash">' in prompt or '"name": "Bash"' in prompt:
            bash_tool_name = "Bash"
        elif '<tool name="bash">' in prompt or '"name": "bash"' in prompt:
            bash_tool_name = "bash"
        elif '<tool name="run_command">' in prompt or '"name": "run_command"' in prompt:
            bash_tool_name = "run_command"
        elif "CRITICAL MODE: TOOL_CALL_REQUIRED" not in prompt:
            return content

    blocks = re.findall(r"```python\?code_reference[^\n]*\n(.*?)```", content, flags=re.DOTALL)
    if not blocks:
        return content

    file_writing_blocks = [
        b for b in blocks
        if any(w in b for w in ("open(", ".write(", ".write_text(", ".write_bytes("))
    ]
    selected = file_writing_blocks if file_writing_blocks else [blocks[0]]

    tool_calls = []
    for b in selected:
        script = b.strip()
        if not script:
            continue
        cmd = f"python3 - << 'GHOST_PYEOF'\n{script}\nGHOST_PYEOF"
        payload = {
            "name": bash_tool_name,
            "arguments": {
                "command": cmd,
                "description": "Execute generated Python script locally",
            },
        }
        tool_calls.append(f"<tool_call>{json.dumps(payload)}</tool_call>")

    return "\n".join(tool_calls) if tool_calls else content


# ---------------------------------------------------------------------------
# Gemini batch-RPC → internal dict
# ---------------------------------------------------------------------------

def parse_gemini_response(raw_text: str) -> dict:
    """
    Parse Gemini's batch-RPC response into a structured dict.

    IMPORTANT: Gemini returns multiple progressive JSON chunks in a single
    response body. Each chunk contains the full content generated so far
    (growing with each chunk). We iterate ALL chunks and return the
    last/longest valid content, ignoring post-generation canned refusal
    overwrites when valid content has already been generated.

    Returns:
        {
            "content": str,
            "conversation_id": str,
            "response_id": str,
            "choice_id": str,
        }

    Raises:
        ValueError: If the response cannot be parsed.
    """
    try:
        lines = raw_text.split("\n")

        best_content = ""
        fallback_refusal = ""
        conversation_id = ""
        response_id = ""
        choice_id = ""

        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("["):
                continue

            try:
                outer = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            # Each outer payload is a list; search for items with a JSON
            # string at position [2] (the inner data envelope)
            items = outer if isinstance(outer[0], list) else [outer]
            for item in items:
                if not isinstance(item, list) or len(item) < 3:
                    continue
                if not isinstance(item[2], str):
                    continue

                try:
                    inner = json.loads(item[2])
                except json.JSONDecodeError:
                    continue

                # Extract content at inner[4][0][1][0]
                content = _extract_content(inner)
                if content:
                    # Clean trailing rogue closing braces/fences after complete tool tag
                    content = re.sub(r"(</(?:tool_call|invoke|[a-z]+)>)\s*[\}\]]+\s*$", r"\1", content)
                    if is_gemini_canned_refusal(content):
                        fallback_refusal = content
                    elif len(content) >= len(best_content):
                        best_content = content

                # Extract conversation tracking IDs (take latest)
                try:
                    cid = inner[1][0]
                    rid = inner[1][1]
                    if cid:
                        conversation_id = cid
                    if rid:
                        response_id = rid
                except (IndexError, TypeError):
                    pass

                # Extract choice ID
                cid = _find_choice_id(inner)
                if cid:
                    choice_id = cid

        final_content = best_content or fallback_refusal
        if not final_content and not conversation_id:
            raise ValueError("No valid content found in any response chunk")

        return {
            "content": final_content,
            "conversation_id": conversation_id,
            "response_id": response_id,
            "choice_id": choice_id,
        }

    except ValueError:
        raise
    except (json.JSONDecodeError, IndexError, TypeError, KeyError) as e:
        raise ValueError(f"Failed to parse Gemini response: {e}")


def parse_streaming_chunks(raw_text: str):
    """
    Parse the progressive chunks from a Gemini response and yield
    incremental text deltas.

    Each chunk from Gemini contains the FULL text so far. We compute
    the delta by subtracting the previous chunk's text.

    Yields:
        dict with keys:
            "delta": str (incremental new text)
            "content": str (full text so far)
            "conversation_id": str
            "response_id": str
            "choice_id": str
            "is_final": bool
    """
    lines = raw_text.split("\n")
    prev_content = ""
    pending_result = None

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("["):
            continue

        try:
            outer = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        items = outer if isinstance(outer[0], list) else [outer]
        for item in items:
            if not isinstance(item, list) or len(item) < 3:
                continue
            if not isinstance(item[2], str):
                continue

            try:
                inner = json.loads(item[2])
            except json.JSONDecodeError:
                continue

            content = _extract_content(inner)
            if not content:
                continue

            # Clean trailing rogue closing braces/fences after complete tool tag
            content = re.sub(r"(</(?:tool_call|invoke|[a-z]+)>)\s*[\}\]]+\s*$", r"\1", content)

            # Ignore post-generation canned refusal overwrite if valid content was already yielded
            if prev_content and is_gemini_canned_refusal(content):
                continue

            # Compute incremental delta
            delta = content[len(prev_content):] if content.startswith(prev_content) else content

            result = {
                "delta": delta,
                "content": content,
                "conversation_id": "",
                "response_id": "",
                "choice_id": "",
                "is_final": False,
            }

            try:
                result["conversation_id"] = inner[1][0] or ""
                result["response_id"] = inner[1][1] or ""
            except (IndexError, TypeError):
                pass

            cid = _find_choice_id(inner)
            if cid:
                result["choice_id"] = cid

            if pending_result is not None:
                yield pending_result

            prev_content = content
            pending_result = result

    if pending_result is not None:
        pending_result["is_final"] = True
        yield pending_result


def _extract_content(inner) -> str:
    """Extract text content from a parsed inner JSON structure."""
    try:
        candidates = inner[4]
        if candidates and len(candidates) > 0:
            first = candidates[0]
            parts = first[1]
            if isinstance(parts, list) and len(parts) > 0:
                text_segments = []
                for p in parts:
                    if isinstance(p, str):
                        text_segments.append(p)
                    elif isinstance(p, list) and len(p) > 0 and isinstance(p[0], str):
                        text_segments.append(p[0])
                if text_segments:
                    text = "".join(text_segments)
                    text = strip_googleusercontent_artifacts(text)
                    text = strip_gemini_html_artifacts(text)
                    if "[http" in text:
                        text = AUTOLINK_URL_PATTERN.sub(r"\1", text)
                    return text
    except (IndexError, TypeError):
        pass
    return ""


def _find_choice_id(data) -> str:
    """
    Recursively search the parsed response for a choice ID (rc_-prefixed string).

    The choice ID location varies depending on whether there are multiple
    response candidates. This performs a depth-limited search.
    """
    # First try the standard location: inner[4] candidates
    try:
        candidates = data[4]
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, list) and len(candidate) > 0:
                    if isinstance(candidate[0], str) and candidate[0].startswith("rc_"):
                        return candidate[0]
    except (IndexError, TypeError):
        pass

    # Fallback: breadth-first search for rc_ string
    queue = [data]
    depth = 0
    max_depth = 5
    while queue and depth < max_depth:
        next_queue = []
        for item in queue:
            if isinstance(item, str) and item.startswith("rc_"):
                return item
            elif isinstance(item, list):
                next_queue.extend(item)
        queue = next_queue
        depth += 1

    return ""


# ---------------------------------------------------------------------------
# Internal → OpenAI format
# ---------------------------------------------------------------------------

def build_non_streaming_response(
    content: str,
    model: str,
    finish_reason: str = "stop",
    conversation_id: str = "",
    response_id: str = "",
) -> dict:
    """Build a complete (non-streaming) OpenAI chat completion response."""
    resp = {
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
    # Include conversation tracking for multi-turn
    if conversation_id:
        resp["conversation_id"] = conversation_id
    if response_id:
        resp["parent_message_id"] = response_id
    return resp


def build_streaming_chunks(
    content: str,
    model: str,
    chunk_id: str,
    conversation_id: str = "",
    response_id: str = "",
):
    """
    Generator that yields SSE-formatted strings simulating token-by-token streaming.

    Splits the content into word-sized pieces and wraps each in an OpenAI
    streaming chunk format.
    """
    # First chunk: role announcement
    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    }
    if conversation_id:
        first_chunk["conversation_id"] = conversation_id
    if response_id:
        first_chunk["parent_message_id"] = response_id
    yield format_openai_chunk(first_chunk)

    # Content chunks: split by words, preserving whitespace
    words = _split_into_tokens(content)
    for word in words:
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": word},
                "finish_reason": None,
            }],
        }
        yield format_openai_chunk(chunk)

    # Final chunk: finish
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield format_openai_chunk(final_chunk)


def _split_into_tokens(text: str) -> list[str]:
    """
    Split text into word-like tokens for simulated streaming.
    Preserves leading whitespace on each token so reassembly is lossless.
    """
    if not text:
        return []

    tokens = []
    current = ""
    for char in text:
        if char in (" ", "\n", "\t") and current:
            tokens.append(current)
            current = char
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


def format_openai_chunk(chunk: dict) -> str:
    """Format a chunk dict as an SSE data line."""
    return f"data: {json.dumps(chunk)}\n\n"
