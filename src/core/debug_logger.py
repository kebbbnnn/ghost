"""
src/core/debug_logger.py — Comprehensive raw request and response file logger.

Captures and records:
1. Incoming client requests (OpenAI / Anthropic / Video)
2. Transpiled upstream payloads & human-readable prompt text sent to AI backends (ChatGPT, Gemini, etc.)
3. Raw upstream response lines / SSE events received over the wire
4. Accumulated model output text
5. Final client response stream chunks or JSON responses
"""
from __future__ import annotations

import contextvars
import datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any

from src.core.config import RAW_LOG_FILE, RAW_LOG_LATEST_FILE, RAW_LOGGING_ENABLED

log = logging.getLogger(__name__)

# File write lock to prevent interleaved writes from concurrent requests
_FILE_LOCK = threading.Lock()


def _sanitize_for_logging(obj: Any, max_str_len: int = 1000) -> Any:
    """
    Recursively sanitize objects for logging.
    Omit long base64 strings and truncate overly large non-text blobs
    to keep log files human-readable and avoid bloating the editor.
    """
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if isinstance(v, str) and (v.startswith("data:image/") or v.startswith("data:video/")):
                comma_idx = v.find(",")
                prefix = v[:comma_idx + 1] if comma_idx != -1 else "data:..."
                sanitized[k] = f"{prefix}<base64 data omitted: {len(v)} chars>"
            elif isinstance(v, str) and len(v) > max_str_len and re.match(r"^[A-Za-z0-9+/=]{100,}$", v.strip()):
                sanitized[k] = f"<base64 data omitted: {len(v)} chars>"
            else:
                sanitized[k] = _sanitize_for_logging(v, max_str_len)
        return sanitized
    elif isinstance(obj, list):
        return [_sanitize_for_logging(item, max_str_len) for item in obj]
    elif isinstance(obj, str):
        if obj.startswith("data:image/") or obj.startswith("data:video/"):
            comma_idx = obj.find(",")
            prefix = obj[:comma_idx + 1] if comma_idx != -1 else "data:..."
            return f"{prefix}<base64 data omitted: {len(obj)} chars>"
        if len(obj) > max_str_len and re.match(r"^[A-Za-z0-9+/=]{100,}$", obj.strip()):
            return f"<base64 data omitted: {len(obj)} chars>"
        return obj
    return obj


def _format_json(data: Any) -> str:
    """Format an object as indented JSON, falling back to repr on non-serializable objects."""
    try:
        sanitized = _sanitize_for_logging(data)
        return json.dumps(sanitized, indent=2, ensure_ascii=False)
    except Exception:
        try:
            return repr(data)
        except Exception:
            return "<unserializable object>"


def _extract_readable_prompt(payload: Any) -> str:
    """
    Extract readable text from incoming requests or upstream payloads.
    Presents human-readable turns so developers don't have to parse escaped JSON strings.
    """
    if not payload:
        return "(no prompt payload captured)"

    lines = []

    # Case 1: Plain string prompt (Gemini or raw completions)
    if isinstance(payload, str):
        return payload.strip()

    # Case 2: Dict payload
    if isinstance(payload, dict):
        # Check for system prompt
        if "system" in payload and payload["system"]:
            lines.append("=== System Instructions ===")
            lines.append(str(payload["system"]).strip())
            lines.append("")

        # Check for tools
        if "tools" in payload and payload["tools"]:
            lines.append("=== Tools Defined ===")
            for t in payload["tools"]:
                t_name = t.get("name") or t.get("function", {}).get("name", "unknown")
                t_desc = t.get("description") or t.get("function", {}).get("description", "")
                lines.append(f"• Tool: {t_name}")
                if t_desc:
                    lines.append(f"  Description: {t_desc.strip()}")
            lines.append("")

        # Check for messages (OpenAI, ChatGPT, or Anthropic format)
        messages = payload.get("messages", [])
        if isinstance(messages, list):
            for idx, msg in enumerate(messages, start=1):
                role = msg.get("role") or msg.get("author", {}).get("role", "unknown")
                lines.append(f"--- Turn {idx} ({role}) ---")

                # OpenAI / Anthropic format
                content = msg.get("content")
                if isinstance(content, str):
                    lines.append(content.strip())
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, str):
                            lines.append(part.strip())
                        elif isinstance(part, dict):
                            p_type = part.get("type", "")
                            if p_type == "text":
                                lines.append(part.get("text", "").strip())
                            elif p_type == "tool_use":
                                lines.append(f"[Tool Call: {part.get('name')}({part.get('input')})]")
                            elif p_type == "tool_result":
                                lines.append(f"[Tool Result for {part.get('tool_use_id')}: {part.get('content')}]")
                            elif p_type == "image_url" or p_type == "image":
                                lines.append("[Attached Image]")
                            else:
                                lines.append(f"[{p_type}]: {_format_json(part)}")
                elif isinstance(content, dict):
                    # ChatGPT parts format: {"content_type": "text", "parts": [...]}
                    parts = content.get("parts", [])
                    for part in parts:
                        if isinstance(part, str):
                            lines.append(part.strip())
                        else:
                            lines.append(_format_json(part))

                # Check for assistant tool_calls in OpenAI format
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        lines.append(f"[Assistant Tool Call: {fn.get('name')}({fn.get('arguments')})]")

                lines.append("")

    return "\n".join(lines).strip() if lines else _format_json(payload)


class RequestTrace:
    """Captures the full lifecycle of a single request from client to upstream and back."""

    def __init__(self, request_id: str, endpoint: str, model: str = "", stream: bool = False):
        self.request_id = request_id
        self.endpoint = endpoint
        self.model = model
        self.stream = stream
        self.start_time = time.time()
        self.end_time: float | None = None

        # 1. Incoming Client Request
        self.client_request: Any = None

        # 2. Upstream Request Payload (Sent to ChatGPT / Gemini / etc.)
        self.provider_name: str = ""
        self.upstream_url: str = ""
        self.upstream_payload: Any = None
        self.upstream_attempt: int = 1

        # 3. Raw Upstream Response (from ChatGPT / Gemini / etc.)
        self.upstream_status: int | None = 200
        self.upstream_raw_lines: list[str] = []
        self.upstream_accumulated_text: list[str] = []

        # 4. Final Client Response
        self.client_chunks: list[str] = []
        self.client_response: Any = None

        # Completion info
        self.error: str | None = None
        self.finished: bool = False

    def log_client_request(self, payload: Any):
        """Record incoming client request body."""
        self.client_request = payload

    def log_upstream_request(self, provider_name: str, url: str, payload: Any, attempt: int = 1):
        """Record upstream request payload and provider details."""
        self.provider_name = provider_name
        self.upstream_url = url
        self.upstream_payload = payload
        self.upstream_attempt = attempt

    def log_upstream_chunk(self, line: str):
        """Record raw line/chunk received from upstream provider."""
        self.upstream_raw_lines.append(line)

        # Attempt incremental text extraction for quick readability
        trimmed = line.strip()
        if trimmed.startswith("data: "):
            json_str = trimmed[6:].strip()
            if json_str and json_str != "[DONE]":
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        # ChatGPT delta format: {"v": "text"}
                        v = parsed.get("v")
                        if isinstance(v, str):
                            self.upstream_accumulated_text.append(v)
                        # ChatGPT legacy format: {"message": {"content": {"parts": [...]}}}
                        msg = parsed.get("message")
                        if isinstance(msg, dict):
                            parts = msg.get("content", {}).get("parts", [])
                            if parts and isinstance(parts[0], str):
                                # Replace with full text if reported as full message
                                self.upstream_accumulated_text = [parts[0]]
                except Exception:
                    pass

    def log_upstream_response(self, status: int, body_or_text: Any):
        """Record non-streaming or error response received from upstream provider."""
        self.upstream_status = status
        if isinstance(body_or_text, str):
            self.upstream_raw_lines.append(body_or_text)
            self.upstream_accumulated_text.append(body_or_text)
        elif isinstance(body_or_text, (dict, list)):
            self.upstream_raw_lines.append(_format_json(body_or_text))

    def log_client_chunk(self, chunk: str):
        """Record chunk yielded back to the client."""
        self.client_chunks.append(chunk)

    def log_client_response(self, response: Any):
        """Record final non-streaming response returned to the client."""
        self.client_response = response

    def log_error(self, err: Exception | str):
        """Record any exception encountered during request execution."""
        self.error = str(err)

    def build_trace_text(self) -> str:
        """Construct the formatted, human-readable trace string."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        duration = (self.end_time or time.time()) - self.start_time

        sections = []

        # ==================== Header ====================
        sections.append("=" * 80)
        sections.append(f"GHOSTTUNNEL RAW TRACE | {now_str} | ID: {self.request_id}")
        sections.append("=" * 80)
        sections.append(f"Client Endpoint : {self.endpoint}")
        sections.append(f"Requested Model : {self.model or 'auto'}")
        sections.append(f"Streaming Mode  : {self.stream}")
        sections.append(f"Provider Target : {self.provider_name or 'Auto / Not specified'}")
        if self.upstream_attempt > 1:
            sections.append(f"Retry Attempt   : {self.upstream_attempt}")
        sections.append(f"Duration        : {duration:.3f}s")
        if self.error:
            sections.append(f"Status          : ERROR ({self.error})")
        else:
            sections.append("Status          : SUCCESS")
        sections.append("")

        # ==================== 1. Client Request ====================
        sections.append("-" * 80)
        sections.append("1. INCOMING CLIENT REQUEST (Raw Body)")
        sections.append("-" * 80)
        if self.client_request is not None:
            sections.append(_format_json(self.client_request))
        else:
            sections.append("(no client request captured)")
        sections.append("")

        # ==================== 2. Upstream Request Payload ====================
        sections.append("-" * 80)
        sections.append(f"2. UPSTREAM REQUEST PAYLOAD (Sent to: {self.upstream_url or self.provider_name or 'Provider'})")
        sections.append("-" * 80)
        if self.upstream_payload is not None:
            sections.append(_format_json(self.upstream_payload))
        else:
            sections.append("(no upstream payload captured)")
        sections.append("")

        # ==================== 2b. Readable Upstream Prompt Text ====================
        sections.append("-" * 80)
        sections.append("2b. UPSTREAM PROMPT TEXT (Readable turns passed to AI backend)")
        sections.append("-" * 80)
        prompt_source = self.upstream_payload if self.upstream_payload is not None else self.client_request
        readable_text = _extract_readable_prompt(prompt_source)
        sections.append(readable_text)
        sections.append("")

        # ==================== 3. Raw Upstream Response ====================
        sections.append("-" * 80)
        sections.append(f"3. RAW UPSTREAM RESPONSE (From {self.provider_name or 'Provider'} | Status: {self.upstream_status})")
        sections.append("-" * 80)
        if self.upstream_raw_lines:
            sections.append("\n".join(self.upstream_raw_lines))
        else:
            sections.append("(no upstream response chunks captured)")
        sections.append("")

        # ==================== 3b. Accumulated Raw Model Text ====================
        if self.upstream_accumulated_text:
            sections.append("-" * 80)
            sections.append("3b. ACCUMULATED RAW MODEL TEXT (Decoded from upstream)")
            sections.append("-" * 80)
            sections.append("".join(self.upstream_accumulated_text).strip())
            sections.append("")

        # ==================== 4. Final Client Response ====================
        sections.append("-" * 80)
        sections.append(f"4. FINAL CLIENT RESPONSE (Returned to {self.endpoint})")
        sections.append("-" * 80)
        if self.client_chunks:
            sections.append("".join(self.client_chunks))
        elif self.client_response is not None:
            sections.append(_format_json(self.client_response))
        else:
            sections.append("(no client response captured)")
        sections.append("")

        # ==================== Footer ====================
        sections.append("=" * 80)
        sections.append(f"END OF TRACE [{self.request_id}] | Elapsed: {duration:.3f}s | "
                        f"Upstream Chunks: {len(self.upstream_raw_lines)} | Client Chunks: {len(self.client_chunks)}")
        sections.append("=" * 80)
        sections.append("\n\n")

        return "\n".join(sections)

    def finish(self):
        """Finalize the trace and write to log files atomically."""
        if self.finished:
            return
        self.finished = True
        self.end_time = time.time()

        if not RAW_LOGGING_ENABLED:
            return

        trace_text = self.build_trace_text()

        try:
            with _FILE_LOCK:
                # 1. Append to primary raw log file
                os.makedirs(os.path.dirname(os.path.abspath(RAW_LOG_FILE)), exist_ok=True)
                with open(RAW_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(trace_text)

                # 2. Overwrite latest raw log file for zero-scroll instant inspection
                os.makedirs(os.path.dirname(os.path.abspath(RAW_LOG_LATEST_FILE)), exist_ok=True)
                with open(RAW_LOG_LATEST_FILE, "w", encoding="utf-8") as f:
                    f.write(trace_text)

            log.info("Logged raw request & response to %s (and %s)", RAW_LOG_FILE, RAW_LOG_LATEST_FILE)
        except Exception as e:
            log.warning("Failed to write raw debug log to %s: %e", RAW_LOG_FILE, e)


# ContextVar for tracing active requests across async call trees
current_trace: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar("current_trace", default=None)


def start_trace(request_id: str, endpoint: str, model: str = "", stream: bool = False) -> RequestTrace:
    """Create a new RequestTrace and bind it to the current async context."""
    trace = RequestTrace(request_id, endpoint, model=model, stream=stream)
    current_trace.set(trace)
    return trace


def get_current_trace() -> RequestTrace | None:
    """Retrieve the active RequestTrace for the current async task context."""
    return current_trace.get(None)
