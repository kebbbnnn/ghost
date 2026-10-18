"""
src/core/emulation/stream_parser.py — Sliding-window lookahead parser for streaming SSE chunks
and non-streaming response transformation.
Handles:
  1. Real-time interception and suppression of <tool_call> tags
  2. Conversion into standard OpenAI delta.tool_calls SSE chunks
  3. Stop sequence truncation and max_tokens enforcement
  4. Non-streaming response parsing with malformed JSON recovery
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator

log = logging.getLogger(__name__)

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


def _sanitize_json_string(raw: str) -> str:
    """Tolerant cleaner for LLM JSON outputs (fixes unescaped newlines, trailing commas)."""
    clean = raw.strip()
    # Remove markdown code fences if wrapped
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    # Remove trailing commas before closing braces
    clean = re.sub(r",\s*(\}|\])", r"\1", clean)
    return clean


def parse_tool_payload(raw_payload: str) -> dict | None:
    """Safely parse a string payload into a tool call dict with 'name' and 'arguments'."""
    clean = _sanitize_json_string(raw_payload)
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            return {"name": data["name"], "arguments": args}
    except Exception as e:
        log.debug("Direct JSON parse failed on tool payload (%s), trying regex: %s", e, clean[:80])

    # Fallback regex extraction: "name": "...", "arguments": ...
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', clean)
    if name_match:
        func_name = name_match.group(1)
        # Find start of arguments object
        args_match = re.search(r'"arguments"\s*:\s*(\{.*\}|\[.*\]|"[^"]*")', clean, re.DOTALL)
        if args_match:
            try:
                args = json.loads(args_match.group(1))
                return {"name": func_name, "arguments": args}
            except Exception:
                return {"name": func_name, "arguments": {}}
        return {"name": func_name, "arguments": {}}

    return None


class StreamEmulationProcessor:
    """
    Sliding-window stream processor that buffers potential tag boundaries
    and transforms emulated tool calls into standard OpenAI SSE chunks.
    """

    def __init__(self, context: dict):
        self.context = context
        self.model = context.get("model", "auto")
        self.stop_sequences: list[str] = context.get("stop", [])
        self.max_tokens: int | None = context.get("max_tokens")
        self.is_synthesis_phase: bool = context.get("is_synthesis_phase", False)
        self.executed_tools: set[str] = set(context.get("executed_tool_names", []))

        self.chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())

        self.buffer = ""
        self.in_tool_tag = False
        self.tool_accumulator = ""
        self.tool_calls_emitted: list[dict] = []
        self.tool_call_index = 0
        self.conversation_id: str | None = None
        self.parent_message_id: str | None = None

        self.total_tokens_estimated = 0
        self.stop_triggered = False
        self.length_triggered = False
        self.first_chunk_sent = False

    async def process_stream(
        self, raw_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Consumes provider SSE lines and yields emulated OpenAI SSE strings."""
        async for line in raw_stream:
            if self.stop_triggered or self.length_triggered:
                break

            line_str = line.strip()
            if not line_str.startswith("data: "):
                continue

            payload = line_str[6:].strip()
            if payload == "[DONE]":
                break

            try:
                chunk_dict = json.loads(payload)
            except Exception:
                continue

            if not isinstance(chunk_dict, dict):
                continue

            # Capture chunk_id and model from upstream if available
            if "id" in chunk_dict and chunk_dict["id"]:
                self.chunk_id = chunk_dict["id"]
            if "model" in chunk_dict and chunk_dict["model"]:
                self.model = chunk_dict["model"]
            if "conversation_id" in chunk_dict and chunk_dict["conversation_id"]:
                self.conversation_id = chunk_dict["conversation_id"]
            if "parent_message_id" in chunk_dict and chunk_dict["parent_message_id"]:
                self.parent_message_id = chunk_dict["parent_message_id"]

            choices = chunk_dict.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            text_delta = delta.get("content", "")

            # If there's content text, process it through the state machine
            if text_delta:
                async for sse_chunk in self._feed_text(text_delta):
                    yield sse_chunk
                    if self.stop_triggered or self.length_triggered:
                        break

        # Stream ended or truncated: flush buffers and emit final chunks
        async for sse_chunk in self._finalize():
            yield sse_chunk

        yield "data: [DONE]\n\n"

    async def _feed_text(self, text: str) -> AsyncGenerator[str, None]:
        """Processes incoming text fragment with lookahead sliding window."""
        cleaned = text.replace("<escape>", "").replace("</escape>", "")
        if not cleaned:
            return
        self.buffer += cleaned

        while self.buffer:
            if not self.in_tool_tag:
                # Look for OPEN_TAG
                open_idx = self.buffer.find(OPEN_TAG)
                if open_idx != -1:
                    # Emit everything before OPEN_TAG as normal content
                    pre_text = self.buffer[:open_idx]
                    pre_text = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", pre_text)
                    if pre_text and pre_text.strip():
                        async for c in self._emit_content(pre_text):
                            yield c
                            if self.stop_triggered or self.length_triggered:
                                return

                    # Transition to in_tool_tag
                    self.in_tool_tag = True
                    self.tool_accumulator = ""
                    self.buffer = self.buffer[open_idx + len(OPEN_TAG):]
                    continue

                # Not found: check if buffer ends with a potential prefix of OPEN_TAG
                match_prefix_len = 0
                for length in range(min(len(OPEN_TAG) - 1, len(self.buffer)), 0, -1):
                    if OPEN_TAG.startswith(self.buffer[-length:]):
                        match_prefix_len = length
                        break

                if match_prefix_len > 0:
                    # Safe text to emit is before the prefix
                    safe_text = self.buffer[:-match_prefix_len]
                    if safe_text:
                        async for c in self._emit_content(safe_text):
                            yield c
                            if self.stop_triggered or self.length_triggered:
                                return
                    self.buffer = self.buffer[-match_prefix_len:]
                    break  # Wait for more input to resolve tag prefix
                else:
                    # Entire buffer is safe content
                    safe_text = self.buffer
                    self.buffer = ""
                    async for c in self._emit_content(safe_text):
                        yield c
                        if self.stop_triggered or self.length_triggered:
                            return
                    break

            else:
                # Inside tool tag: accumulate until CLOSE_TAG
                close_idx = self.buffer.find(CLOSE_TAG)
                if close_idx != -1:
                    self.tool_accumulator += self.buffer[:close_idx]
                    self.buffer = self.buffer[close_idx + len(CLOSE_TAG):]
                    self.in_tool_tag = False

                    # Emit complete tool call chunk
                    async for c in self._emit_tool_call(self.tool_accumulator):
                        yield c
                    self.tool_accumulator = ""
                    continue

                # Not closed yet: check for potential prefix of CLOSE_TAG
                match_prefix_len = 0
                for length in range(min(len(CLOSE_TAG) - 1, len(self.buffer)), 0, -1):
                    if CLOSE_TAG.startswith(self.buffer[-length:]):
                        match_prefix_len = length
                        break

                if match_prefix_len > 0:
                    safe_acc = self.buffer[:-match_prefix_len]
                    self.tool_accumulator += safe_acc
                    self.buffer = self.buffer[-match_prefix_len:]
                    break
                else:
                    self.tool_accumulator += self.buffer
                    self.buffer = ""
                    break

    async def _emit_content(self, text: str) -> AsyncGenerator[str, None]:
        """Emits standard OpenAI delta.content chunk after stop and max_tokens checks."""
        # Strip internal google card tokens if any leak through
        text = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", text)
        if not text:
            return

        # Stop sequence checking
        if self.stop_sequences:
            for s in self.stop_sequences:
                if s in text:
                    idx = text.find(s)
                    truncated_text = text[:idx]
                    if truncated_text:
                        yield self._build_chunk(delta={"content": truncated_text}, finish_reason=None)
                    self.stop_triggered = True
                    yield self._build_chunk(delta={}, finish_reason="stop")
                    return

        # Max tokens checking
        if self.max_tokens is not None:
            # Approximate 4 characters per token
            est_tokens = max(1, len(text) // 4)
            if self.total_tokens_estimated + est_tokens >= self.max_tokens:
                remaining_tokens = max(0, self.max_tokens - self.total_tokens_estimated)
                safe_chars = remaining_tokens * 4
                truncated_text = text[:safe_chars]
                if truncated_text:
                    yield self._build_chunk(delta={"content": truncated_text}, finish_reason=None)
                self.length_triggered = True
                yield self._build_chunk(delta={}, finish_reason="length")
                return
            self.total_tokens_estimated += est_tokens

        role_attr = {"role": "assistant"} if not self.first_chunk_sent else {}
        self.first_chunk_sent = True
        yield self._build_chunk(
            delta={"content": text, **role_attr},
            finish_reason=None,
        )

    async def _emit_tool_call(self, raw_tool_str: str) -> AsyncGenerator[str, None]:
        """Emits delta.tool_calls SSE chunk."""
        parsed = parse_tool_payload(raw_tool_str)
        if not parsed:
            # If payload couldn't be parsed, fallback to flushing raw text
            fallback_text = f"<{OPEN_TAG}>{raw_tool_str}</{CLOSE_TAG}>"
            async for c in self._emit_content(fallback_text):
                yield c
            return

        # Suppress re-invocation of tools whose results were already provided in synthesis phase
        if self.is_synthesis_phase and parsed["name"] in self.executed_tools:
            return

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        func_name = parsed["name"]
        arguments_str = (
            json.dumps(parsed["arguments"])
            if not isinstance(parsed["arguments"], str)
            else parsed["arguments"]
        )

        tool_call_delta = {
            "index": self.tool_call_index,
            "id": call_id,
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments_str,
            },
        }
        self.tool_calls_emitted.append(tool_call_delta)
        self.tool_call_index += 1

        role_attr = {"role": "assistant"} if not self.first_chunk_sent else {}
        self.first_chunk_sent = True

        yield self._build_chunk(
            delta={"tool_calls": [tool_call_delta], **role_attr},
            finish_reason=None,
        )

    async def _finalize(self) -> AsyncGenerator[str, None]:
        """Flushes remaining buffers and emits the final finish_reason chunk."""
        if self.stop_triggered or self.length_triggered:
            return

        # If stream closed while inside tool tag, attempt to parse what we got
        if self.in_tool_tag:
            full_tool_payload = self.tool_accumulator + self.buffer
            self.buffer = ""
            self.in_tool_tag = False
            if full_tool_payload.strip():
                async for c in self._emit_tool_call(full_tool_payload):
                    yield c
        elif self.buffer:
            # Remaining buffered text (like unmatched prefix '<')
            if not self.tool_calls_emitted or self.buffer.strip():
                async for c in self._emit_content(self.buffer):
                    yield c
            self.buffer = ""

        # Determine finish reason
        if self.tool_calls_emitted:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        yield self._build_chunk(delta={}, finish_reason=finish_reason)

    def _build_chunk(self, delta: dict, finish_reason: str | None) -> str:
        """Serializes dictionary into SSE line."""
        chunk = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if self.conversation_id:
            chunk["conversation_id"] = self.conversation_id
        if self.parent_message_id:
            chunk["parent_message_id"] = self.parent_message_id
        return f"data: {json.dumps(chunk)}\n\n"


def process_non_streaming_response(raw_resp: dict, context: dict) -> dict:
    """
    Parses full response dictionary for non-streaming completions:
    extracts tool calls, enforces stop/max_tokens, and formats standard message.tool_calls.
    """
    resp = dict(raw_resp)
    choices = resp.get("choices", [])
    if not choices:
        return resp

    choice = dict(choices[0])
    message = dict(choice.get("message", {}))
    raw_content = message.get("content") or ""

    stop_sequences: list[str] = context.get("stop", [])
    max_tokens: int | None = context.get("max_tokens")
    is_synthesis_phase: bool = context.get("is_synthesis_phase", False)
    executed_tools: set[str] = set(context.get("executed_tool_names", []))

    raw_content = raw_content.replace("<escape>", "").replace("</escape>", "")
    raw_content = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", raw_content)

    # 1. Enforce stop sequences
    if stop_sequences:
        for s in stop_sequences:
            if s in raw_content:
                raw_content = raw_content[: raw_content.find(s)]
                choice["finish_reason"] = "stop"

    # 2. Extract <tool_call> blocks
    pattern = re.compile(rf"{re.escape(OPEN_TAG)}(.*?){re.escape(CLOSE_TAG)}", re.DOTALL)
    matches = pattern.findall(raw_content)

    if matches:
        tool_calls = []
        for i, match in enumerate(matches):
            parsed = parse_tool_payload(match)
            if parsed:
                if is_synthesis_phase and parsed["name"] in executed_tools:
                    continue

                call_id = f"call_{uuid.uuid4().hex[:12]}"
                args_str = (
                    json.dumps(parsed["arguments"])
                    if not isinstance(parsed["arguments"], str)
                    else parsed["arguments"]
                )
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args_str,
                    },
                })

        clean_content = pattern.sub("", raw_content).strip()

        if tool_calls:
            message["content"] = clean_content if clean_content else None
            message["tool_calls"] = tool_calls
            choice["message"] = message
            choice["finish_reason"] = "tool_calls"
            resp["choices"] = [choice]
            return resp
        elif clean_content:
            # All tool calls were suppressed echoes; emit synthesized content
            message["content"] = clean_content
            message.pop("tool_calls", None)
            choice["message"] = message
            choice["finish_reason"] = "stop"
            resp["choices"] = [choice]
            return resp

    # 3. Enforce max_tokens if no tool calls
    if max_tokens is not None:
        est_tokens = max(1, len(raw_content) // 4)
        if est_tokens > max_tokens:
            raw_content = raw_content[: max_tokens * 4]
            choice["finish_reason"] = "length"

    message["content"] = raw_content
    choice["message"] = message
    resp["choices"] = [choice]
    return resp
