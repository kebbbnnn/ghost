"""
session.py — Anonymous Gemini guest session lifecycle manager.

Handles:
  1. Fetching gemini.google.com/app to extract session tokens (thykhd, cfb2h, FdrFJe)
  2. Managing NID + COMPASS cookies for session identity
  3. Sending chat messages via the StreamGenerate batch-RPC endpoint
  4. Multi-turn conversation state tracking
  5. Token refresh and session reset on auth failures

Protocol Summary:
  - GET /app → extract thykhd (CSRF "at" token), cfb2h (bot server version)
  - POST /_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate
    - Query: bl={bot_server}, _reqid={counter}, rt=c
    - Body: form-encoded f.req (nested JSON) + at={thykhd}
  - Response: Google batch-RPC format, parsed via converter.py
"""
import asyncio
import typing
import json
import re
import time
import uuid
import logging
import urllib.parse

import httpx

from src.core.provider import BaseProvider
from src.core.exceptions import RateLimitError, TokenExpiredError, ProviderError
from .converter import (
    parse_gemini_response,
    build_non_streaming_response,
    build_streaming_chunks,
    format_openai_chunk,
)

log = logging.getLogger(__name__)

BASE_URL = "https://gemini.google.com"
APP_URL = f"{BASE_URL}/app"
STREAM_GENERATE_URL = (
    f"{BASE_URL}/_/BardChatUi/data/"
    "assistant.lamda.BardFrontendService/StreamGenerate"
)

# Refresh tokens proactively before they expire.
# The actual TTL is unknown for anonymous sessions; 5 minutes is conservative.
TOKEN_TTL_SECONDS = 300


class GeminiProvider(BaseProvider):
    """
    Anonymous Gemini guest session provider.
    Manages token extraction, conversation state, and request serialization.
    """

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._at_token: str = ""
        self._bot_server: str = ""
        self._sid: str = ""
        self._conversation_id: str = ""
        self._response_id: str = ""
        self._choice_id: str = ""
        self._reqid: int = 100000
        self._token_fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def get_models(self) -> list[dict]:
        return [
            {
                "id": "gemini",
                "object": "model",
                "created": 1746057600,
                "owned_by": "google",
            },
        ]

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> typing.AsyncGenerator[str, None]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        try:
            for attempt in range(2):
                try:
                    prompt = self._build_prompt(messages, conversation_id)
                    raw = await self._send_message(
                        prompt, conversation_id, parent_message_id
                    )
                    parsed = parse_gemini_response(raw)
                    self._update_conversation_state(parsed)

                    for sse_chunk in build_streaming_chunks(
                        parsed["content"],
                        model,
                        chunk_id,
                        parsed.get("conversation_id", ""),
                        parsed.get("response_id", ""),
                    ):
                        yield sse_chunk

                    break  # success

                except TokenExpiredError:
                    if attempt == 0:
                        log.warning("Token expired, refreshing and retrying...")
                        await self._reset_session(hard=False)
                        continue
                    raise

        except RateLimitError:
            error_chunk = {
                "error": {
                    "message": "Rate limited by Gemini",
                    "type": "rate_limit_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        except Exception as exc:
            log.exception("Stream error")
            error_chunk = {
                "error": {"message": str(exc), "type": "server_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        for attempt in range(2):
            try:
                prompt = self._build_prompt(messages, conversation_id)
                raw = await self._send_message(
                    prompt, conversation_id, parent_message_id
                )
                parsed = parse_gemini_response(raw)
                self._update_conversation_state(parsed)

                return build_non_streaming_response(
                    parsed["content"],
                    model,
                    "stop",
                    parsed.get("conversation_id", ""),
                    parsed.get("response_id", ""),
                )
            except TokenExpiredError:
                if attempt == 0:
                    log.warning("Token expired, refreshing and retrying...")
                    await self._reset_session(hard=False)
                    continue
                raise

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                http2=True,
            )
        return self._http

    def _is_token_expired(self) -> bool:
        return (time.time() - self._token_fetched_at) > TOKEN_TTL_SECONDS

    async def ensure_ready(self):
        """Ensure we have valid tokens; refresh if expired."""
        async with self._lock:
            if not self._at_token or self._is_token_expired():
                await self._refresh_tokens()

    async def start_background_tasks(self):
        self._bg_task = asyncio.create_task(self._maintain_session_loop())

    async def stop_background_tasks(self):
        if hasattr(self, "_bg_task") and self._bg_task:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass

    async def _maintain_session_loop(self):
        """Periodically checks and refreshes the token in the background before it expires."""
        while True:
            try:
                async with self._lock:
                    if not self._at_token or self._is_token_expired():
                        await self._refresh_tokens()
            except Exception as e:
                log.error(f"Gemini background token refresh failed: {e}")
            await asyncio.sleep(10)

    async def _refresh_tokens(self):
        """
        GET the Gemini app page and extract session tokens from the HTML.

        Extracts:
          - thykhd → CSRF "at" token
          - cfb2h  → bot server version string
          - FdrFJe → session ID (captured for future use)
        """
        http = await self._get_http()

        resp = await http.get(APP_URL, headers=self._base_headers())
        resp.raise_for_status()
        body = resp.text

        if "CAPTCHA" in body:
            raise RateLimitError(
                "Gemini detected automated access (CAPTCHA). Try again later."
            )

        # Extract thykhd (CSRF/at token)
        thykhd_match = re.search(r'"thykhd":"(.*?)"', body)
        if not thykhd_match:
            raise TokenExpiredError(
                "Failed to extract thykhd token from Gemini page. "
                "Anonymous access may be blocked or the page structure changed."
            )
        self._at_token = thykhd_match.group(1)

        # Extract cfb2h (bot server version)
        cfb2h_match = re.search(r'"cfb2h":"(.*?)"', body)
        if cfb2h_match:
            self._bot_server = cfb2h_match.group(1)
        else:
            # Fallback — will likely fail but lets the request attempt proceed
            log.warning("Could not extract cfb2h; using fallback bot server string")
            self._bot_server = "boq_assistant-bard-web-server_20260510.18_p0"

        # Extract FdrFJe (session ID)
        fdrfje_match = re.search(r'"FdrFJe":"([\d-]+)"', body)
        if fdrfje_match:
            self._sid = fdrfje_match.group(1)

        # Store cookies from the response
        # httpx.AsyncClient automatically handles cookies across requests

        self._token_fetched_at = time.time()
        log.info(
            "Gemini tokens refreshed: bot_server=%s, at_token=%s...",
            self._bot_server,
            self._at_token[:30],
        )

    async def _reset_session(self, hard: bool = True):
        """Reset session state. On hard reset, also clear the HTTP client."""
        self._at_token = ""
        self._bot_server = ""
        self._token_fetched_at = 0.0

        if hard:
            self._conversation_id = ""
            self._response_id = ""
            self._choice_id = ""
            if self._http and not self._http.is_closed:
                await self._http.aclose()
            self._http = None
            log.warning("Gemini session hard reset: all state cleared")
        else:
            log.warning("Gemini session soft reset: tokens cleared, conversation preserved")

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def _send_message(
        self,
        prompt: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> str:
        """
        Send a message to the Gemini StreamGenerate endpoint.
        Returns the raw response text for parsing.
        """
        await self.ensure_ready()
        http = await self._get_http()

        # Use tracked conversation state, falling back to caller-provided IDs
        conv_id = conversation_id or self._conversation_id
        resp_id = parent_message_id or self._response_id
        choice_id = self._choice_id

        # Build the nested JSON payload
        message_struct = json.dumps([
            [prompt],
            None,
            [conv_id, resp_id, choice_id],
        ])
        f_req = json.dumps([None, message_struct])

        # Query parameters
        params = {
            "bl": self._bot_server,
            "_reqid": str(self._reqid),
            "rt": "c",
        }

        # Form-encoded body
        body = urllib.parse.urlencode({
            "f.req": f_req,
            "at": self._at_token,
        })

        headers = {
            **self._base_headers(),
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "X-Same-Domain": "1",
        }

        try:
            resp = await http.post(
                STREAM_GENERATE_URL,
                params=params,
                content=body,
                headers=headers,
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"HTTP request to Gemini failed: {e}")

        # Increment request ID for next call
        self._reqid += 100000

        # Handle error responses
        if resp.status_code == 429:
            raise RateLimitError("Gemini rate limit hit (429)")
        if resp.status_code in (401, 403):
            await self._reset_session(hard=True)
            raise TokenExpiredError(
                f"Gemini rejected request ({resp.status_code}). Session reset."
            )
        if resp.status_code != 200:
            raise ProviderError(
                f"Gemini returned unexpected status {resp.status_code}: "
                f"{resp.text[:200]}"
            )

        return resp.text

    def _update_conversation_state(self, parsed: dict):
        """Update tracked conversation IDs from a parsed response."""
        if parsed.get("conversation_id"):
            self._conversation_id = parsed["conversation_id"]
        if parsed.get("response_id"):
            self._response_id = parsed["response_id"]
        if parsed.get("choice_id"):
            self._choice_id = parsed["choice_id"]

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        messages: list[dict],
        conversation_id: str | None = None,
    ) -> str:
        """
        Convert OpenAI-format messages into a single text prompt for Gemini.

        If we have an active conversation (conversation_id is set), we only
        send the latest user message — Gemini tracks context server-side.

        If this is a fresh conversation, we concatenate all messages into
        a single prompt so the model gets full context.
        """
        # Check if we're continuing an existing conversation
        has_active_conv = bool(conversation_id or self._conversation_id)

        if has_active_conv and messages:
            # Only send the last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return self._extract_text(msg)
            # Fallback: last message regardless of role
            return self._extract_text(messages[-1])

        # Fresh conversation: concatenate all messages
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            text = self._extract_text(msg)
            if role == "system":
                parts.append(f"[System]: {text}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {text}")
            else:
                parts.append(text)

        return "\n\n".join(parts)

    @staticmethod
    def _extract_text(msg: dict) -> str:
        """Extract plain text from an OpenAI message (handles string or array content)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        return str(content)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_headers() -> dict:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://gemini.google.com",
            "Referer": "https://gemini.google.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }
