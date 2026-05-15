import uuid
import json
import asyncio
import logging
from typing import AsyncGenerator
import time
from curl_cffi.requests import AsyncSession
import os

from src.core.provider import BaseProvider
from src.core.exceptions import RateLimitError
from .rotator import AccountRotator

log = logging.getLogger(__name__)

BASE_URL = "https://claude.ai"


class ClaudeProvider(BaseProvider):
    def __init__(self, rotator_file_path: str = None):
        if rotator_file_path is None:
            rotator_file_path = os.path.join(os.path.dirname(__file__), "accounts.json")
        self.rotator = AccountRotator(rotator_file_path)
        # Cache: {email: {"org_id": str, "conv_uuid": str}}
        # Reuse conversations until they error out or context limit is hit.
        self._conv_cache: dict[str, dict] = {}
    
    async def get_models(self) -> list[dict]:
        return [
            {"id": "claude-sonnet-4-6", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-5-sonnet-20241022", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-opus-20240229", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-5-haiku-20241022", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
        ]

    def _base_headers(self, session_key: str) -> dict:
        return {
            "Cookie": f"sessionKey={session_key}",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://claude.ai/",
            "Origin": "https://claude.ai",
        }

    async def _get_org_id(self, session_key: str, client: AsyncSession) -> str:
        headers = self._base_headers(session_key)
        res = await client.get(f"{BASE_URL}/api/organizations", headers=headers)
        log.info(f"Org fetch status: {res.status_code}")
        if res.status_code == 429:
            raise RateLimitError("Rate limited when fetching organizations.")
        if res.status_code != 200:
            raise Exception(f"Failed to get org id: {res.status_code} {res.text}")
        orgs = res.json()
        if not orgs:
            raise Exception("No organizations found for this account.")
        org_id = orgs[0]["uuid"]
        log.info(f"Got org_id: {org_id}")
        return org_id

    async def _create_conversation(self, org_id: str, session_key: str, client: AsyncSession) -> str:
        """Create a new conversation and return its UUID."""
        headers = self._base_headers(session_key)
        conv_uuid = str(uuid.uuid4())
        payload = {"uuid": conv_uuid, "name": ""}
        res = await client.post(
            f"{BASE_URL}/api/organizations/{org_id}/chat_conversations",
            headers=headers,
            json=payload,
        )
        if res.status_code not in (200, 201):
            raise Exception(f"Failed to create conversation: {res.status_code} {res.text}")
        log.info(f"Created conversation: {conv_uuid}")
        return conv_uuid

    def _map_messages(self, messages: list[dict]) -> str:
        """
        Extract only the latest user message (and system prompt if present).
        Each request creates a fresh conversation, so we don't replay history.
        """
        system_prompt = ""
        last_user_msg = ""

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "system":
                system_prompt = content
            elif role == "user":
                last_user_msg = content  # Keep overwriting — last one wins

        prompt = last_user_msg.strip()
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        return prompt

    def _yield_error(self, message: str) -> str:
        """Format an error chunk matching the OpenAI error schema."""
        error_chunk = {"error": {"message": message, "type": "server_error"}}
        return f"data: {json.dumps(error_chunk)}\n\n"

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        
        from .rotator import AccountsExhaustedError

        max_retries = 3

        while True:
            try:
                account = await self.rotator.get_next_account()
            except AccountsExhaustedError:
                log.warning("All Claude accounts are exhausted or rate limited.")
                yield self._yield_error("All Claude accounts are currently rate limited. Please try again later.")
                yield "data: [DONE]\n\n"
                return

            session_key = account["sessionKey"]
            email = account["email"]

            for attempt in range(max_retries):
                try:
                    async with AsyncSession(impersonate="chrome120") as client:
                        # Reuse cached org_id + conversation, or create new ones
                        cached = self._conv_cache.get(email)
                        if cached:
                            org_id = cached["org_id"]
                            conv_uuid = cached["conv_uuid"]
                            log.info(f"Reusing conversation {conv_uuid} for {email}")
                        else:
                            org_id = await self._get_org_id(session_key, client)
                            conv_uuid = await self._create_conversation(org_id, session_key, client)
                            self._conv_cache[email] = {"org_id": org_id, "conv_uuid": conv_uuid}
                            log.info(f"Created new conversation {conv_uuid} for {email}")

                        # Send the message
                        prompt = self._map_messages(messages)
                        msg_payload = {
                            "prompt": prompt,
                            "timezone": "America/New_York",
                            "attachments": [],
                            "files": [],
                        }

                        headers = {
                            **self._base_headers(session_key),
                            "Accept": "text/event-stream",
                        }

                        req_url = f"{BASE_URL}/api/organizations/{org_id}/chat_conversations/{conv_uuid}/completion"
                        log.info(f"Sending completion (attempt {attempt + 1}) to {req_url}")

                        res = await client.post(req_url, json=msg_payload, headers=headers, stream=True)
                        log.info(f"Completion response status: {res.status_code}")
                        log.info(f"Response headers: {dict(res.headers)}")

                        if res.status_code == 429:
                            # Log the full 429 response for debugging
                            body = ""
                            try:
                                body = res.text if hasattr(res, 'text') else ""
                            except Exception:
                                pass
                            log.warning(f"[429] Account: {email} | Attempt: {attempt + 1}/{max_retries}")
                            log.warning(f"[429] Response body: {body[:2000]}")
                            log.warning(f"[429] Response headers: {dict(res.headers)}")

                            # Check if this is a real quota exhaustion or just throttling
                            try:
                                body_data = json.loads(body) if body else {}
                                msg_limit = body_data.get("messageLimit", {})
                                log.warning(f"[429] messageLimit: {json.dumps(msg_limit, indent=2)}")
                                if msg_limit.get("type") == "exceeded":
                                    log.warning(f"[429] QUOTA EXHAUSTED for {email}")
                                    await self.rotator.mark_rate_limited(email)
                                    break  # Try next account
                            except (json.JSONDecodeError, Exception) as parse_err:
                                log.warning(f"[429] Could not parse body as JSON: {parse_err}")

                            # Likely frequency throttling — short backoff and retry
                            wait_time = 5 * (attempt + 1)
                            log.info(f"[429] Treating as frequency throttle. Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                            continue

                        if res.status_code != 200:
                            err_text = ""
                            try:
                                err_text = res.text if hasattr(res, 'text') else ""
                            except Exception:
                                pass
                            log.error(f"Claude completion error: {res.status_code} - {err_text[:2000]}")
                            log.error(f"Response headers: {dict(res.headers)}")
                            raise Exception(f"Claude API error: {res.status_code} - {err_text[:500]}")

                        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

                        async for chunk in res.aiter_lines():
                            if not chunk:
                                continue
                            line = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk

                            if not line.startswith("data: "):
                                continue

                            data_str = line[len("data: "):].strip()
                            if not data_str:
                                continue

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Check for rate limit info in the response
                            msg_limit = data.get("messageLimit", {})
                            if msg_limit.get("type") == "exceeded":
                                windows = msg_limit.get("windows", {})
                                five_h = windows.get("5h", {})
                                resets_at = five_h.get("resets_at")
                                if resets_at:
                                    cooldown_secs = max(0, resets_at - int(time.time()))
                                else:
                                    cooldown_secs = 18000
                                await self.rotator.mark_rate_limited(email, cooldown_secs)
                                yield self._yield_error("Claude rate limit exceeded for this account.")
                                yield "data: [DONE]\n\n"
                                return

                            completion_text = data.get("completion", "")
                            stop_reason = data.get("stop_reason")

                            if completion_text:
                                openai_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": completion_text},
                                        "finish_reason": None,
                                    }],
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n"

                            if stop_reason:
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
                                yield f"data: {json.dumps(final_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                        # Stream ended without explicit stop_reason
                        yield "data: [DONE]\n\n"
                        return

                except RateLimitError:
                    self._conv_cache.pop(email, None)
                    await self.rotator.mark_rate_limited(email)
                    break  # Try next account
                except Exception as e:
                    self._conv_cache.pop(email, None)
                    log.exception(f"Claude provider error for account {email}")
                    yield self._yield_error(str(e))
                    yield "data: [DONE]\n\n"
                    return

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        full_text = ""
        async for chunk in self.stream_chat(messages, model, conversation_id, parent_message_id):
            if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                data_str = chunk[len("data: "):].strip()
                if data_str:
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                full_text += delta["content"]
                    except Exception:
                        pass

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
