"""
guest_session.py — Anonymous guest session lifecycle manager.

Handles:
  1. Initial page load to obtain oai-did cookie
  2. Calling /backend-anon/sentinel/chat-requirements
  3. Solving PoW and generating sentinel token
  4. Sending chat messages and streaming SSE responses
"""
import asyncio
import typing
import json
import time
import uuid
import logging
from typing import AsyncIterator

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import HTTPError

from src.core.provider import BaseProvider
from src.core.exceptions import RateLimitError, TokenExpiredError
from .converter import (
    map_model,
    parse_chatgpt_sse,
    chatgpt_event_to_openai_chunk,
    format_openai_chunk,
    build_non_streaming_response,
)

from .sentinel.token_generator import (
    generate_requirements_token,
    generate_enforcement_token,
    build_sentinel_token,
)

from .uploader import preprocess_images

log = logging.getLogger(__name__)

BASE_URL = "https://chatgpt.com"
REQUIREMENTS_URL = f"{BASE_URL}/backend-anon/sentinel/chat-requirements"
CONVERSATION_URL = f"{BASE_URL}/backend-anon/conversation"
CONVERSATION_URL_FILES = f"{BASE_URL}/backend-anon/f/conversation"
MODELS_URL = f"{BASE_URL}/backend-anon/models"

# Chat requirements token expires after ~540 seconds (9 min); refresh at 8 min
TOKEN_TTL_SECONDS = 480


class ChatGPTProvider(BaseProvider):
    """
    A single anonymous ChatGPT guest session.
    Manages device-id, chat-requirements token, and conversation state.
    """

    async def get_models(self) -> list[dict]:
        return [
            {"id": "gpt-5-5", "object": "model", "created": 1746057600, "owned_by": "openai"},
            {"id": "auto", "object": "model", "created": 1746057600, "owned_by": "openai"},
        ]

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> typing.AsyncGenerator[str, None]:
        target_model = map_model(model)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        content_tracker = []

        try:
            for attempt in range(2):
                try:
                    async for raw_line in self._stream_raw_chat(messages, target_model, conversation_id, parent_message_id):
                        event = parse_chatgpt_sse(raw_line)
                        if event is None:
                            continue

                        chunk = chatgpt_event_to_openai_chunk(event, target_model, chunk_id)
                        if chunk is None:
                            continue

                        choices = chunk.get("choices", [])
                        if choices:
                            delta_content = choices[0].get("delta", {}).get("content", "")
                            if delta_content:
                                new_text = self._get_incremental_text(content_tracker, delta_content)
                                if new_text:
                                    chunk["choices"][0]["delta"]["content"] = new_text
                                else:
                                    chunk["choices"][0]["delta"].pop("content", None)

                        yield format_openai_chunk(chunk)

                        if choices and choices[0].get("finish_reason"):
                            break

                    break  # success

                except TokenExpiredError:
                    if attempt == 0:
                        log.warning("Token expired, refreshing and retrying...")
                        await self.ensure_ready()
                        content_tracker.clear()
                    else:
                        raise
        except RateLimitError:
            error_chunk = {"error": {"message": "Rate limited by ChatGPT", "type": "rate_limit_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
        except Exception as exc:
            log.exception("Stream error")
            error_chunk = {"error": {"message": str(exc), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        target_model = map_model(model)
        full_content = ""
        finish_reason = "stop"

        try:
            for attempt in range(2):
                try:
                    async for raw_line in self._stream_raw_chat(messages, target_model, conversation_id, parent_message_id):
                        event = parse_chatgpt_sse(raw_line)
                        if not event or event.get("done"):
                            continue

                        msg = event.get("message", {})
                        parts = msg.get("content", {}).get("parts", [])
                        for part in parts:
                            if isinstance(part, str):
                                full_content = part

                        status = msg.get("status", "")
                        if status == "max_tokens":
                            finish_reason = "length"
                    break
                except TokenExpiredError:
                    if attempt == 0:
                        await self.ensure_ready()
                    else:
                        raise
        except RateLimitError:
            raise RateLimitError("Rate limited by ChatGPT")
        except Exception as exc:
            log.exception("Non-streaming error")
            raise

        return build_non_streaming_response(full_content, target_model, finish_reason)

    def _get_incremental_text(self, tracker: list, new_full_text: str) -> str:
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

    def __init__(self):
        self._session: AsyncSession | None = None
        self._oai_did: str = ""
        self._chat_req: dict = {}
        self._sentinel_token: str = ""
        self._token_fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> AsyncSession:
        if self._session is None:
            self._session = AsyncSession(impersonate="chrome131")
        return self._session

    async def _reset_session(self, hard: bool = True):
        """Tear down the current HTTP session and wipe all device state.

        Called on 403 so the next attempt gets completely fresh Cloudflare
        cookies (oai-sc, __cf_bm, _cfuvid) and a new oai-did, rather than
        reusing tainted cookies from the flagged session.
        """
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        self._sentinel_headers = None
        self._token_fetched_at = 0.0
        
        if hard:
            self._oai_did = ""
            log.warning("Session reset: all cookies and device state cleared")
        else:
            log.warning("Soft session reset: cookies and sentinel cleared, device state preserved")

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Device ID provisioning
    # ------------------------------------------------------------------

    async def _ensure_device_id(self):
        """Load the homepage to get oai-did cookie if we don't have one."""
        if self._oai_did:
            return
        session = await self._get_session()
        resp = await session.get(
            f"{BASE_URL}/",
            headers=self._base_headers(),
        )
        did = session.cookies.get("oai-did")
        if did:
            self._oai_did = did
            log.info(f"Got oai-did: {did}")
        else:
            # Fallback: generate a random UUID (server will reject it but
            # the chat-requirements endpoint will give us a real one)
            self._oai_did = str(uuid.uuid4())
            log.warning("oai-did not in cookies, using generated UUID")

    # ------------------------------------------------------------------
    # Chat requirements
    # ------------------------------------------------------------------

    def _is_token_expired(self) -> bool:
        return (time.time() - self._token_fetched_at) > TOKEN_TTL_SECONDS

    async def _refresh_chat_requirements(self):
        """
        POST /backend-anon/sentinel/chat-requirements to get a fresh
        chat-requirements response, then solve PoW and build sentinel token.
        """
        await self._ensure_device_id()
        session = await self._get_session()

        # generate_requirements_token() returns (p_token, sid) — sid must be
        # reused by generate_enforcement_token so both use the same config array.
        p_token, sid = generate_requirements_token()

        headers = {
            **self._base_headers(),
            "oai-device-id": self._oai_did,
            "oai-language": "en-US",
            "Content-Type": "application/json",
        }

        resp = await session.post(
            REQUIREMENTS_URL,
            headers=headers,
            json={"p": p_token},
        )
        resp.raise_for_status()
        self._chat_req = resp.json()
        log.debug(f"chat-requirements response: persona={self._chat_req.get('persona')}")

        # Update oai-did from response cookies if server issued a new one
        did = session.cookies.get("oai-did")
        if did:
            self._oai_did = did

        # Solve proof-of-work if required
        pow_data = self._chat_req.get("proofofwork", {})
        if pow_data.get("required"):
            seed = pow_data["seed"]
            difficulty = pow_data["difficulty"]
            log.info(f"Solving PoW: seed={seed!r} difficulty={difficulty!r}")
            enforcement_token = await asyncio.to_thread(
                generate_enforcement_token, seed, difficulty, sid
            )
        else:
            enforcement_token = p_token

        from .sentinel.token_generator import execute_turnstile_vm
        
        turnstile_token = ""
        turnstile = self._chat_req.get("turnstile", {})
        if turnstile.get("required") and turnstile.get("dx"):
            turnstile_token = execute_turnstile_vm(turnstile["dx"], p_token)

        self._sentinel_headers = {
            "Openai-Sentinel-Chat-Requirements-Token": self._chat_req.get("token", ""),
            "Openai-Sentinel-Proof-Token": enforcement_token,
            "Openai-Sentinel-Turnstile-Token": turnstile_token,
        }
        self._token_fetched_at = time.time()
        log.info("Sentinel token generated successfully")

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def ensure_ready(self):
        """Ensure we have valid sentinel headers; refresh if expired."""
        async with self._lock:
            if not getattr(self, "_sentinel_headers", None) or self._is_token_expired():
                await self._refresh_chat_requirements()

    async def start_background_tasks(self):
        # Pre-initialize the HTTP session
        await self._get_session()
        self._bg_task = asyncio.create_task(self._maintain_session_loop())

    async def stop_background_tasks(self):
        if hasattr(self, "_bg_task") and self._bg_task:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass

    async def _maintain_session_loop(self):
        """Periodically checks and refreshes the chat requirements in the background."""
        while True:
            try:
                async with self._lock:
                    if not getattr(self, "_sentinel_headers", None) or self._is_token_expired():
                        await self._refresh_chat_requirements()
            except Exception as e:
                log.error(f"ChatGPT background token refresh failed: {e}")
            await asyncio.sleep(10)

    async def _stream_raw_chat(
        self,
        messages: list[dict],
        model: str = "auto",
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> typing.AsyncGenerator[bytes, None]:
        """
        Send a chat message and yield raw SSE lines from the response.

        Args:
            messages:        OpenAI-format message list
            model:           ChatGPT model slug (e.g. "gpt-5-5", "auto")
            conversation_id: Optional existing conversation ID for multi-turn
            parent_message_id: Optional parent message ID for multi-turn

        Yields:
            Raw SSE data lines (e.g. 'data: {"message": {...}}')
        """
        await self.ensure_ready()
        
        max_retries = 1
        for attempt in range(max_retries + 1):
            await self.ensure_ready()
            session = await self._get_session()

            # Pre-process images inside retry loop: if session resets,
            # file IDs from the old session become stale and must be re-uploaded
            processed_messages, has_images = await preprocess_images(
                messages, session, self._base_headers(), self._oai_did
            )
            
            # Convert messages to ChatGPT format
            chatgpt_messages = _convert_messages(processed_messages)
            
            # If continuing a conversation, ChatGPT expects only the newest message
            if conversation_id and parent_message_id:
                chatgpt_messages = chatgpt_messages[-1:]

            body = {
                "action": "next",
                "messages": chatgpt_messages,
                "model": model,
                "parent_message_id": parent_message_id or "client-created-root",
                "timezone_offset_min": -480,
                "timezone": "Asia/Manila",
            }
            if conversation_id:
                body["conversation_id"] = conversation_id
                
            if has_images:
                body["conversation_mode"] = {"kind": "primary_assistant"}
                body["supported_encodings"] = ["v1"]
                body["client_contextual_info"] = {
                    "is_dark_mode": True,
                    "time_since_loaded": 320,
                    "page_height": 916,
                    "page_width": 816,
                    "pixel_ratio": 1,
                    "screen_height": 1050,
                    "screen_width": 1680,
                    "app_name": "chatgpt.com"
                }
                body["force_parallel_switch"] = "auto"
                body["client_prepare_state"] = "sent"
                body["enable_message_followups"] = True
                body["system_hints"] = []
                body["supports_buffering"] = True
                body["no_auth_ad_preferences"] = {
                    "personalization_enabled": True,
                    "history_enabled": True,
                    "bazaar_consent_set": False
                }
                body["paragen_cot_summary_display_override"] = "allow"
            
            headers = {
                **self._base_headers(),
                "oai-device-id": self._oai_did,
                "oai-language": "en-US",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **self._sentinel_headers,
            }
            
            if has_images:
                log.info("Sending multimodal payload: %s", json.dumps(body))

            try:
                conversation_url = CONVERSATION_URL_FILES if has_images else CONVERSATION_URL
                log.info("Requesting conversation from URL: %s", conversation_url)
                async with session.stream(
                    "POST",
                    conversation_url,
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status_code == 429:
                        raise RateLimitError("ChatGPT rate limit hit (429)")
                    if resp.status_code in (401, 403):
                        hard_reset = not bool(conversation_id)
                        await self._reset_session(hard=hard_reset)
                        if attempt < max_retries:
                            log.warning("Device flagged (%d). Resetting session and retrying...", resp.status_code)
                            continue
                        raise TokenExpiredError("Device flagged — session reset failed after retry")
                    if resp.status_code == 404 and conversation_id:
                        # Stale conversation — start fresh
                        log.warning("404 on conversation %s — resetting state", conversation_id)
                        conversation_id = None
                        parent_message_id = None
                        body.pop("conversation_id", None)
                        body["parent_message_id"] = str(uuid.uuid4())
                        # Re-include all messages for the fresh conversation
                        chatgpt_messages = _convert_messages(messages)
                        body["messages"] = chatgpt_messages
                        await self._reset_session(hard=True)
                        if attempt < max_retries:
                            log.warning("Retrying as new conversation...")
                            continue
                        raise TokenExpiredError("Conversation not found and retry failed")
                    
                    if resp.status_code >= 400:
                        body_text = ""
                        try:
                            async for chunk in resp.aiter_lines():
                                body_text += chunk.decode("utf-8") + "\n"
                        except:
                            pass
                        log.error(f"HTTP {resp.status_code} Error Body: {body_text}")
                        resp.raise_for_status()

                    async for chunk in resp.aiter_lines():
                        if chunk:
                            yield chunk.decode("utf-8")
                    break
            except HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403):
                    if attempt < max_retries:
                        log.warning("Device flagged (%s). Resetting session and retrying...", status)
                        hard_reset = not bool(conversation_id)
                        await self._reset_session(hard=hard_reset)
                        await asyncio.sleep(1)
                        continue
                raise TokenExpiredError("Session token expired or device blocked")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _base_headers(self) -> dict:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://chatgpt.com",
            "Referer": "https://chatgpt.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            # Client hint headers — must match impersonate="chrome131" + macOS UA
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }


# ------------------------------------------------------------------
# Message format conversion
# ------------------------------------------------------------------

def _convert_messages(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI Chat Completions messages to ChatGPT guest format.
    """
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # System messages become a user message with the instructions header
            out.append({
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {
                    "content_type": "text",
                    "parts": [f"[System]: {content}"],
                },
            })
        elif role == "user":
            parts = _parse_content(content)
            user_msg = {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "create_time": time.time(),
                "content": {
                    "content_type": "multimodal_text" if any(isinstance(p, dict) for p in parts) else "text",
                    "parts": parts,
                },
            }
            # Pass through attachment metadata from image preprocessing
            if msg.get("metadata"):
                user_msg["metadata"] = msg["metadata"]
            out.append(user_msg)
        elif role == "assistant":
            text = content if isinstance(content, str) else _extract_text(content)
            out.append({
                "id": str(uuid.uuid4()),
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": [text],
                },
            })
    return out


def _parse_content(content):
    """Parse content that may be a string or an array of content parts."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Pass through image_asset_pointer dicts from uploader
                if item.get("content_type") == "image_asset_pointer":
                    parts.append(item)
                    continue
                t = item.get("type")
                if t == "text":
                    parts.append(item.get("text", ""))
                elif t == "image_url":
                    url_data = item.get("image_url", {})
                    url = url_data.get("url", "") if isinstance(url_data, dict) else url_data
                    parts.append({"content_type": "image_url", "asset_pointer": url})
            elif isinstance(item, str):
                parts.append(item)
        return parts
    return [str(content)]


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------

