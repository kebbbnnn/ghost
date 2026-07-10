"""
session.py — SurfSense anonymous chat provider with unlimited token quota bypass.

Uses a direct HTTP API approach to call api.surfsense.com. Since the FastAPI
backend trusts proxy headers for rate-limiting, captcha, and quota limits,
this provider bypasses the 500K limit and CAPTCHA requirements by rotating
both a generated anonymous session token and a spoofed client IP address
for every request.

Bypass Mechanics:
  1. IP Rotation: Generates a random public IP on every request, mapping it to
     `X-Forwarded-For`, `X-Real-IP`, and `CF-Connecting-IP`. This resets the
     backend's IP-based rate-limits, CAPTCHA counters, and token quotas.
  2. Session Rotation: Generates a new 43-character base64 URL-safe token
     for `surfsense_anon_session` on every request. This resets the session-
     based token quota, resulting in a fresh 500K limit per call.
  3. No Browser Required: Playwright and headless Chromium dependencies are
     completely bypassed, ensuring high performance, zero latency, and
     reduced container sizes.
"""
import asyncio
import json
import logging
import os
import random
import secrets
import typing
import uuid

import httpx

from src.core.provider import BaseProvider
from src.core.exceptions import RateLimitError, ProviderError
from .converter import (
    parse_surfsense_sse_line,
    surfsense_event_to_delta,
    is_done_event,
    is_error_event,
    build_openai_streaming_chunk,
    format_openai_chunk,
    build_non_streaming_response,
)

log = logging.getLogger(__name__)

API_BASE = "https://api.surfsense.com/api/v1/public/anon-chat"
MODELS_URL = f"{API_BASE}/models"
STREAM_URL = f"{API_BASE}/stream"

DEFAULT_MODEL_SLUG = os.environ.get(
    "SURFSENSE_MODEL_SLUG", "gpt-5.4-mini-no-login"
)


class SurfsenseProvider(BaseProvider):
    """
    Anonymous SurfSense chat provider with unlimited token bypass.

    Generates a unique session cookie and client IP address for each API call,
    completely bypassing Turnstile and token quotas.
    """

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def get_models(self) -> list[dict]:
        """Fetch models available on SurfSense's free tier."""
        try:
            client = await self._get_http()
            resp = await client.get(MODELS_URL)
            resp.raise_for_status()
            models = resp.json()
            return [
                {
                    "id": f"surfsense-{m.get('seo_slug', m.get('model_name', 'unknown'))}",
                    "object": "model",
                    "created": 1752192000,
                    "owned_by": f"surfsense/{m.get('provider', 'unknown')}",
                    "description": f"{m.get('name', '')} via SurfSense free tier",
                }
                for m in models
            ]
        except Exception as e:
            log.warning("Failed to fetch SurfSense models: %s", e)
            return [
                {
                    "id": "surfsense-gpt-5.4-mini-no-login",
                    "object": "model",
                    "created": 1752192000,
                    "owned_by": "surfsense/azure",
                },
                {
                    "id": "surfsense-gpt-o4-mini-no-login",
                    "object": "model",
                    "created": 1752192000,
                    "owned_by": "surfsense/azure",
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
        model_slug = self._resolve_model_slug(model)

        # Generate a unique IP and session token for this request
        spoofed_ip = self._generate_random_ip()
        anon_session = secrets.token_urlsafe(32)

        payload = {
            "model_slug": model_slug,
            "messages": self._build_messages(messages),
        }

        headers = self._build_headers(spoofed_ip, anon_session)

        try:
            client = await self._get_http()
            async with client.stream(
                "POST",
                STREAM_URL,
                json=payload,
                headers=headers,
                timeout=120.0,
            ) as resp:
                if resp.status_code == 429:
                    raise RateLimitError("Rate limited by SurfSense")
                if resp.status_code == 403:
                    body = await self._read_error_body(resp)
                    raise ProviderError(
                        f"SurfSense 403 (likely IP block/CAPTCHA): {body}"
                    )
                if resp.status_code >= 400:
                    body = await self._read_error_body(resp)
                    raise ProviderError(
                        f"SurfSense returned {resp.status_code}: {body}"
                    )

                async for line in resp.aiter_lines():
                    event = parse_surfsense_sse_line(line)
                    if event is None:
                        continue

                    # Check for errors
                    err = is_error_event(event)
                    if err:
                        yield format_openai_chunk({"error": err})
                        break

                    # Check for stream end
                    if is_done_event(event):
                        yield format_openai_chunk(
                            build_openai_streaming_chunk(
                                delta="",
                                model=model,
                                chunk_id=chunk_id,
                                finish_reason="stop",
                            )
                        )
                        break

                    # Extract text delta
                    delta = surfsense_event_to_delta(event)
                    if delta:
                        yield format_openai_chunk(
                            build_openai_streaming_chunk(
                                delta=delta,
                                model=model,
                                chunk_id=chunk_id,
                            )
                        )

        except RateLimitError as exc:
            yield format_openai_chunk(
                {"error": {"message": str(exc), "type": "rate_limit_error"}}
            )
        except Exception as exc:
            log.exception("SurfSense stream error")
            yield format_openai_chunk(
                {"error": {"message": str(exc), "type": "server_error"}}
            )

        yield "data: [DONE]\n\n"

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        model_slug = self._resolve_model_slug(model)
        full_content = ""

        # Generate a unique IP and session token for this request
        spoofed_ip = self._generate_random_ip()
        anon_session = secrets.token_urlsafe(32)

        payload = {
            "model_slug": model_slug,
            "messages": self._build_messages(messages),
        }

        headers = self._build_headers(spoofed_ip, anon_session)

        client = await self._get_http()
        async with client.stream(
            "POST",
            STREAM_URL,
            json=payload,
            headers=headers,
            timeout=120.0,
        ) as resp:
            if resp.status_code == 429:
                raise RateLimitError("Rate limited by SurfSense")
            if resp.status_code >= 400:
                body = await self._read_error_body(resp)
                raise ProviderError(
                    f"SurfSense returned {resp.status_code}: {body}"
                )

            async for line in resp.aiter_lines():
                event = parse_surfsense_sse_line(line)
                if event is None:
                    continue
                if is_done_event(event):
                    break
                delta = surfsense_event_to_delta(event)
                if delta:
                    full_content += delta

        return build_non_streaming_response(content=full_content, model=model)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                http2=True,
                timeout=30.0,
                follow_redirects=True,
            )
        return self._http

    async def start_background_tasks(self):
        """Pre-warm the HTTP client."""
        await self._get_http()
        log.info("SurfSense provider initialized (unlimited bypass mode active)")

    async def stop_background_tasks(self):
        if self._http:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_random_ip() -> str:
        """Generate a random public IPv4 address (excluding private ranges)."""
        while True:
            o1 = random.randint(1, 223)
            # Exclude private, loopback, and carrier-grade NAT space
            if o1 in (10, 100, 127, 169, 172, 192):
                continue
            break
        return f"{o1}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

    @staticmethod
    def _build_headers(ip: str, session_token: str) -> dict:
        """Build request headers with spoofed IP and session token cookies."""
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
            "Referer": "https://www.surfsense.com/",
            "Origin": "https://www.surfsense.com",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Ch-Ua-Mobile": "?0",
            # IP Spoofing Headers
            "CF-Connecting-IP": ip,
            "X-Real-IP": ip,
            "X-Forwarded-For": ip,
            # Session Spoofing Cookie
            "Cookie": f"surfsense_anon_session={session_token}",
        }

    @staticmethod
    def _resolve_model_slug(model: str) -> str:
        """Map GhostTunnel model names to SurfSense SEO slugs."""
        name = model.lower().strip()

        if name.startswith("surfsense-"):
            name = name[len("surfsense-"):]

        slug_map = {
            "gpt-5.4-mini": "gpt-5.4-mini-no-login",
            "gpt-5.4-mini-no-login": "gpt-5.4-mini-no-login",
            "o4-mini": "gpt-o4-mini-no-login",
            "gpt-o4-mini": "gpt-o4-mini-no-login",
            "gpt-o4-mini-no-login": "gpt-o4-mini-no-login",
        }

        return slug_map.get(name, DEFAULT_MODEL_SLUG)

    @staticmethod
    def _build_messages(messages: list[dict]) -> list[dict]:
        """Convert OpenAI messages to the format SurfSense expects.

        SurfSense expects: [{"role": "user"|"assistant", "content": "..."}]
        System messages are folded into the first user message.
        """
        out = []
        system_prefix = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract text from content arrays
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)

            if role == "system":
                system_prefix += f"[System]: {content}\n\n"
                continue

            if role == "user" and system_prefix:
                content = system_prefix + content
                system_prefix = ""

            out.append({"role": role, "content": content})

        if system_prefix and not out:
            out.append({"role": "user", "content": system_prefix.strip()})

        return out

    @staticmethod
    async def _read_error_body(resp) -> str:
        """Read the response body for error diagnostics."""
        try:
            chunks = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
                if len(chunks) > 10:
                    break
            return "".join(chunks)[:1000]
        except Exception:
            return "<unreadable>"
