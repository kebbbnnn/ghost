"""
session.py — Perplexity guest session provider.

Manages anonymous Perplexity search sessions, token warming, and query execution.
Uses curl-cffi for Cloudflare clearance and handles SSE parsing via converter.py.
"""
import asyncio
import typing
import json
import uuid
import time
import logging

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import HTTPError

from src.core.provider import BaseProvider
from src.core.exceptions import RateLimitError, TokenExpiredError, ProviderError
from .converter import (
    parse_perplexity_event,
    build_openai_streaming_chunk,
    format_openai_chunk,
    build_non_streaming_response,
)

log = logging.getLogger(__name__)

BASE_URL = "https://www.perplexity.ai"
SESSION_URL = f"{BASE_URL}/api/auth/session"
ASK_URL = f"{BASE_URL}/rest/sse/perplexity_ask"

# Refresh session proactively before cookies expire (e.g. every 15 mins)
SESSION_TTL_SECONDS = 900

class PerplexityProvider(BaseProvider):
    """
    Anonymous Perplexity search provider.
    Maintains guest cookies and parses the custom SSE events from the ask endpoint.
    """

    def __init__(self):
        self._session: AsyncSession | None = None
        self._session_fetched_at: float = 0.0
        self._lock = asyncio.Lock()
        self._bg_task: asyncio.Task | None = None

    async def get_models(self) -> list[dict]:
        return [
            {
                "id": "perplexity",
                "object": "model",
                "created": 1746057600,
                "owned_by": "perplexity",
            },
        ]

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> AsyncSession:
        if self._session is None:
            self._session = AsyncSession(impersonate="chrome131")
        return self._session

    def _is_session_expired(self) -> bool:
        return (time.time() - self._session_fetched_at) > SESSION_TTL_SECONDS

    async def ensure_ready(self):
        """Ensure we have fresh Cloudflare cookies and active guest session."""
        async with self._lock:
            if not self._session or self._is_session_expired():
                await self._refresh_session()

    async def _refresh_session(self):
        """Perform a request to the session URL to obtain required cookies."""
        session = await self._get_session()
        log.info("Refreshing Perplexity guest session...")
        try:
            resp = await session.get(SESSION_URL, headers=self._base_headers())
            resp.raise_for_status()
            self._session_fetched_at = time.time()
            log.info("Perplexity guest session warmed up. Cookies: %s", 
                     list(session.cookies.get_dict().keys()))
        except Exception as e:
            log.error("Failed to warm up Perplexity guest session: %s", e)
            # Do not raise immediately; let first query attempt retry / fail naturally

    async def _reset_session(self):
        """Tear down session and force new cookies on next call."""
        async with self._lock:
            if self._session:
                try:
                    await self._session.close()
                except Exception:
                    pass
                self._session = None
            self._session_fetched_at = 0.0
            log.warning("Perplexity session reset complete")

    async def start_background_tasks(self):
        # Eagerly pre-warm connection
        try:
            await self.ensure_ready()
        except Exception as e:
            log.warning("Eager Perplexity pre-warm failed: %s", e)
            
        self._bg_task = asyncio.create_task(self._maintain_session_loop())

    async def stop_background_tasks(self):
        if self._bg_task:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None
            
        if self._session:
            await self._session.close()
            self._session = None

    async def _maintain_session_loop(self):
        """Background loop to periodically refresh session before expiry."""
        while True:
            try:
                if not self._session or self._is_session_expired():
                    await self.ensure_ready()
            except Exception as e:
                log.error("Perplexity background session refresh failed: %s", e)
            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Chat execution
    # ------------------------------------------------------------------

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
                    await self.ensure_ready()
                    session = await self._get_session()
                    
                    # Convert messages into Perplexity payload
                    prompt = self._build_prompt(messages, conversation_id)
                    
                    # Track multi-turn state
                    frontend_context_uuid = conversation_id or str(uuid.uuid4())
                    frontend_uuid = str(uuid.uuid4())
                    last_backend_uuid = parent_message_id
                    
                    # Build request payload matching Perplexity REST SSE format
                    payload = {
                        "query_str": prompt,
                        "params": {
                            "attachments": [],
                            "frontend_context_uuid": frontend_context_uuid,
                            "frontend_uuid": frontend_uuid,
                            "is_incognito": False,
                            "language": "en-US",
                            "last_backend_uuid": last_backend_uuid,
                            "mode": "concise",
                            "model_preference": "turbo",
                            "source": "default",
                            "sources": ["web"],
                            "version": "2.18",
                        }
                    }
                    
                    headers = {
                        "Accept": "text/event-stream",
                        "Content-Type": "application/json",
                        "Origin": BASE_URL,
                        "Referer": f"{BASE_URL}/",
                    }
                    
                    log.info("Sending Perplexity SSE query: %r (context=%s, last_backend=%s)", 
                             prompt[:100], frontend_context_uuid, last_backend_uuid)
                             
                    # Track incremental text deltas
                    tracker = {"last_yielded_idx": -1, "full_text": ""}
                    
                    # First chunk: role announcement
                    yield format_openai_chunk(
                        build_openai_streaming_chunk(
                            delta="",
                            model=model,
                            chunk_id=chunk_id,
                            conversation_id=frontend_context_uuid,
                            message_id=last_backend_uuid or ""
                        )
                    )
                    
                    # Make post request and stream lines
                    resp = await session.post(ASK_URL, json=payload, headers=headers, stream=True)
                    
                    if resp.status_code == 429:
                        raise RateLimitError("Rate limited by Perplexity")
                    if resp.status_code in (401, 403):
                        await self._reset_session()
                        raise TokenExpiredError("Perplexity cookie / Turnstile expired")
                    if resp.status_code != 200:
                        raise ProviderError(f"Perplexity returned status {resp.status_code}")
                        
                    async for line_bytes in resp.aiter_lines(delimiter=b"\r\n\r\n"):
                        if not line_bytes:
                            continue
                        line_str = line_bytes.decode("utf-8")
                        event = parse_perplexity_event(line_str, tracker)
                        if not event:
                            continue
                            
                        # If we have context or backend UUID in progress, capture them
                        if event.get("context_uuid"):
                            frontend_context_uuid = event["context_uuid"]
                        if event.get("backend_uuid"):
                            last_backend_uuid = event["backend_uuid"]
                            
                        if event.get("delta"):
                            yield format_openai_chunk(
                                build_openai_streaming_chunk(
                                    delta=event["delta"],
                                    model=model,
                                    chunk_id=chunk_id,
                                    conversation_id=frontend_context_uuid,
                                    message_id=last_backend_uuid
                                )
                            )
                            
                        if event.get("done"):
                            break
                            
                    # Final stop chunk
                    yield format_openai_chunk(
                        build_openai_streaming_chunk(
                            delta="",
                            model=model,
                            chunk_id=chunk_id,
                            conversation_id=frontend_context_uuid,
                            message_id=last_backend_uuid,
                            finish_reason="stop"
                        )
                    )
                    break  # Success
                    
                except (TokenExpiredError, HTTPError) as e:
                    if attempt == 0:
                        log.warning("Perplexity session expired or blocked; resetting and retrying: %s", e)
                        await self._reset_session()
                        continue
                    raise
                    
        except RateLimitError as exc:
            yield format_openai_chunk({"error": {"message": str(exc), "type": "rate_limit_error"}})
        except Exception as exc:
            log.exception("Perplexity stream error")
            yield format_openai_chunk({"error": {"message": str(exc), "type": "server_error"}})
            
        yield "data: [DONE]\n\n"

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        full_content = ""
        frontend_context_uuid = conversation_id or str(uuid.uuid4())
        last_backend_uuid = parent_message_id
        
        for attempt in range(2):
            try:
                await self.ensure_ready()
                session = await self._get_session()
                
                prompt = self._build_prompt(messages, conversation_id)
                frontend_uuid = str(uuid.uuid4())
                
                payload = {
                    "query_str": prompt,
                    "params": {
                        "attachments": [],
                        "frontend_context_uuid": frontend_context_uuid,
                        "frontend_uuid": frontend_uuid,
                        "is_incognito": False,
                        "language": "en-US",
                        "last_backend_uuid": last_backend_uuid,
                        "mode": "concise",
                        "model_preference": "turbo",
                        "source": "default",
                        "sources": ["web"],
                        "version": "2.18",
                    }
                }
                
                headers = {
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/",
                }
                
                resp = await session.post(ASK_URL, json=payload, headers=headers, stream=True)
                
                if resp.status_code == 429:
                    raise RateLimitError("Rate limited by Perplexity")
                if resp.status_code in (401, 403):
                    await self._reset_session()
                    raise TokenExpiredError("Perplexity session expired")
                if resp.status_code != 200:
                    raise ProviderError(f"Perplexity returned status {resp.status_code}")
                    
                tracker = {"last_yielded_idx": -1, "full_text": ""}
                async for line_bytes in resp.aiter_lines(delimiter=b"\r\n\r\n"):
                    if not line_bytes:
                        continue
                    line_str = line_bytes.decode("utf-8")
                    event = parse_perplexity_event(line_str, tracker)
                    if not event:
                        continue
                        
                    if event.get("context_uuid"):
                        frontend_context_uuid = event["context_uuid"]
                    if event.get("backend_uuid"):
                        last_backend_uuid = event["backend_uuid"]
                    if event.get("full_text"):
                        full_content = event["full_text"]
                    if event.get("done"):
                        break
                        
                return build_non_streaming_response(
                    content=full_content or tracker.get("full_text", ""),
                    model=model,
                    conversation_id=frontend_context_uuid,
                    message_id=last_backend_uuid
                )
                
            except (TokenExpiredError, HTTPError) as e:
                if attempt == 0:
                    log.warning("Perplexity session expired/blocked; resetting and retrying: %s", e)
                    await self._reset_session()
                    continue
                raise

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        messages: list[dict],
        conversation_id: str | None = None,
    ) -> str:
        """
        Convert messages to a single prompt string.
        If we are continuing an active thread, only send the last user message.
        Otherwise, concatenate prior turns to preserve context.
        """
        if conversation_id and messages:
            # Multi-turn: Perplexity tracks history server-side on the context_uuid.
            # Only send the latest user query.
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return self._extract_text(msg)
            return self._extract_text(messages[-1])
            
        # Fresh conversation: concatenate messages
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

    @staticmethod
    def _base_headers() -> dict:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
