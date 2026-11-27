"""
tests/test_sdk.py — Closed-loop tests for embeddable Ghost SDK & in-process transports.

Covers:
  1. Direct AsyncGhost & sync Ghost clients (ask, chat.completions, messages, models)
  2. Streaming + early-break cancellation cleanup
  3. OpenAI tool-calling & parameter emulation through GhostEngine
  4. Automatic failover vs RateLimitError (429) preservation
  5. Per-client provider isolation (no cross-client state leakage)
  6. Lifecycle management (warm_up / start / close / aclose)
  7. Pure-httpx in-process transports (AsyncGhostTransport & GhostTransport)
     for OpenAI, Anthropic, Models, Health, and error status codes
  8. SDK factory helpers (openai_client, anthropic_client) & actionable ImportError messages
  9. Import hygiene: zero filesystem writes and zero FastAPI/OpenCV requirement
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

import ghost
from ghost import (
    AsyncGhost,
    AsyncGhostTransport,
    BaseProvider,
    Ghost,
    GhostEngine,
    GhostTransport,
    ModelNotFoundError,
    ProviderError,
    ProviderRegistry,
    RateLimitError,
    async_httpx_client,
    httpx_client,
)


# ---------------------------------------------------------------------------
# Deterministic mock providers
# ---------------------------------------------------------------------------

class FakeProvider(BaseProvider):
    def __init__(
        self,
        *,
        name: str = "fake",
        models: list[dict] | None = None,
        reply: str = "hello from fake",
        chunks: list[str] | None = None,
        fail_with: Exception | None = None,
        stream_delay: float = 0.0,
    ):
        self.name = name
        self._models = models or [
            {"id": "auto", "object": "model", "created": 1700000000, "owned_by": name}
        ]
        self.reply = reply
        self.chunks = chunks
        self.fail_with = fail_with
        self.stream_delay = stream_delay
        self.calls: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.stream_closed = False

    async def get_models(self) -> list[dict]:
        return list(self._models)

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        self.calls.append(
            {
                "mode": "non_stream",
                "messages": messages,
                "model": model,
                "conversation_id": conversation_id,
                "parent_message_id": parent_message_id,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return {
            "id": f"chatcmpl-{self.name}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ):
        self.calls.append(
            {
                "mode": "stream",
                "messages": messages,
                "model": model,
                "conversation_id": conversation_id,
                "parent_message_id": parent_message_id,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        pieces = self.chunks if self.chunks is not None else [self.reply]
        try:
            for idx, piece in enumerate(pieces):
                if self.stream_delay:
                    await asyncio.sleep(self.stream_delay)
                delta = {"role": "assistant", "content": piece} if idx == 0 else {"content": piece}
                payload = {
                    "id": f"chatcmpl-{self.name}",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            final_payload = {
                "id": f"chatcmpl-{self.name}",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            self.stream_closed = True

    async def start_background_tasks(self):
        self.started = True

    async def stop_background_tasks(self):
        self.stopped = True

    async def close(self):
        self.closed = True


def _make_registry(chatgpt=None, gemini=None, spectre=None, default_provider=None):
    cg = chatgpt or FakeProvider(
        name="chatgpt",
        models=[{"id": "auto", "object": "model", "created": 1, "owned_by": "openai"}],
        reply="chatgpt reply",
        chunks=["chat", "gpt ", "reply"],
    )
    gm = gemini or FakeProvider(
        name="gemini",
        models=[{"id": "gemini", "object": "model", "created": 1, "owned_by": "google"}],
        reply="gemini reply",
        chunks=["gem", "ini ", "reply"],
    )
    sp = spectre or FakeProvider(
        name="spectre",
        models=[{"id": "spectre", "object": "model", "created": 1, "owned_by": "ghost"}],
        reply="spectre reply",
        chunks=["spec", "tre"],
    )
    return ProviderRegistry(
        default_provider=default_provider,
        chatgpt=cg,
        gemini=gm,
        spectre=sp,
    )


# ---------------------------------------------------------------------------
# 1. AsyncGhost tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_ghost_ask_and_models():
    reg = _make_registry()
    async with AsyncGhost(registry=reg) as client:
        assert await client.ask("Hi", model="auto") == "chatgpt reply"
        assert await client.ask("Hi", model="gemini", system="Be brief") == "gemini reply"

        models = await client.models.list()
        ids = {m["id"] for m in models["data"]}
        assert ids == {"auto", "gemini", "spectre"}

        gem = await client.models.retrieve("gemini")
        assert gem["owned_by"] == "google"

        with pytest.raises(ModelNotFoundError):
            await client.models.retrieve("nonexistent-model")

    assert reg.chatgpt.closed and reg.gemini.closed and reg.spectre.closed


@pytest.mark.asyncio
async def test_async_ghost_streaming_and_tool_emulation():
    tool_reply = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    reg = _make_registry(
        chatgpt=FakeProvider(name="chatgpt", reply=tool_reply, chunks=[tool_reply])
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get city weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    async with AsyncGhost(registry=reg) as client:
        resp = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=tools,
            stream=False,
        )
        msg = resp["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}

        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=tools,
            stream=True,
        )
        collected = [chunk async for chunk in stream]
        tool_deltas = [
            tc
            for c in collected
            for tc in (c["choices"][0]["delta"].get("tool_calls") or [])
        ]
        assert any(tc.get("function", {}).get("name") == "get_weather" for tc in tool_deltas)


@pytest.mark.asyncio
async def test_async_ghost_failover_and_rate_limit_preservation():
    # Generic failure fails over to fallback provider
    reg_failover = _make_registry(
        chatgpt=FakeProvider(name="chatgpt", fail_with=RuntimeError("boom")),
        gemini=FakeProvider(name="gemini", reply="saved by gemini", chunks=["saved ", "by gemini"]),
    )
    async with AsyncGhost(registry=reg_failover) as client:
        assert await client.ask("Hello", model="auto") == "saved by gemini"
        stream = await client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "Hello"}], stream=True
        )
        text = "".join([c["choices"][0]["delta"].get("content", "") async for c in stream])
        assert text == "saved by gemini"

    # RateLimitError MUST propagate without silent failover
    reg_429 = _make_registry(
        chatgpt=FakeProvider(name="chatgpt", fail_with=RateLimitError("429 slow down")),
        gemini=FakeProvider(name="gemini", reply="should not run"),
    )
    async with AsyncGhost(registry=reg_429) as client:
        with pytest.raises(RateLimitError, match="429 slow down"):
            await client.ask("Hello", model="auto")
        assert len(reg_429.gemini.calls) == 0


@pytest.mark.asyncio
async def test_async_ghost_anthropic_messages_and_warm_up():
    reg = _make_registry()
    async with AsyncGhost(registry=reg, warm_up=True) as client:
        assert reg.chatgpt.started and reg.gemini.started

        msg = await client.messages.create(
            model="gemini",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert msg["type"] == "message"
        assert msg["content"][0]["text"] == "gemini reply"

        stream = await client.messages.create(
            model="gemini",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        events = [ev async for ev in stream]
        event_types = [ev.get("type") for ev in events]
        assert "message_start" in event_types
        assert "message_stop" in event_types

        tok = await client.messages.count_tokens(
            model="gemini", messages=[{"role": "user", "content": "Hello world"}]
        )
        assert tok["input_tokens"] > 0

    assert reg.chatgpt.stopped and reg.chatgpt.closed


# ---------------------------------------------------------------------------
# 2. Sync Ghost tests (including early stream cancel & isolation)
# ---------------------------------------------------------------------------

def test_sync_ghost_full_flow_and_early_stream_cancel():
    slow_provider = FakeProvider(
        name="chatgpt",
        chunks=[f"tok-{i} " for i in range(50)],
        stream_delay=0.02,
    )
    reg = _make_registry(chatgpt=slow_provider)

    with Ghost(registry=reg, warm_up=True) as client:
        assert reg.chatgpt.started

        # Non-streaming
        assert client.ask("Hi", model="gemini") == "gemini reply"

        # Streaming full consumption
        chunks = list(
            client.chat.completions.create(
                model="gemini",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == "gemini reply"

        # Early break closes upstream async generator cleanly without hanging
        stream = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Slow stream"}],
            stream=True,
        )
        with stream:
            first = next(stream)
            assert first["choices"][0]["delta"]["content"] == "tok-0 "
        assert slow_provider.stream_closed

        # Anthropic sync
        msg = client.messages.create(
            model="gemini",
            max_tokens=128,
            messages=[{"role": "user", "content": "Ping"}],
        )
        assert msg["content"][0]["text"] == "gemini reply"

        # Models sync
        assert len(client.models.list()["data"]) == 3

    assert reg.chatgpt.stopped and reg.chatgpt.closed


def test_per_client_provider_isolation():
    c1 = Ghost(default_provider="chatgpt")
    c2 = Ghost(default_provider="gemini")
    try:
        assert c1.engine.registry.chatgpt is not c2.engine.registry.chatgpt
        assert c1.engine.registry.gemini is not c2.engine.registry.gemini
        assert c1.engine.registry.get_provider_for_model("auto") is c1.engine.registry.chatgpt
        assert c2.engine.registry.get_provider_for_model("auto") is c2.engine.registry.gemini
    finally:
        c1.close()
        c2.close()


# ---------------------------------------------------------------------------
# 3. Pure-httpx in-process transports (Async & Sync)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_httpx_transport_openai_and_anthropic():
    reg = _make_registry()
    async with async_httpx_client(registry=reg) as client:
        # Health
        r = await client.get("http://ghost.local/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}

        # Models
        r = await client.get("/models")
        assert r.status_code == 200
        assert {m["id"] for m in r.json()["data"]} == {"auto", "gemini", "spectre"}

        r = await client.get("/models/unknown-id")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

        # OpenAI non-streaming
        r = await client.post(
            "/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        assert r.headers["X-Request-ID"].startswith("req_")
        assert r.json()["choices"][0]["message"]["content"] == "chatgpt reply"

        # OpenAI streaming
        lines = []
        async with client.stream(
            "POST",
            "/chat/completions",
            json={"model": "gemini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            async for line in resp.aiter_lines():
                if line:
                    lines.append(line)
        assert lines[-1] == "data: [DONE]"
        assert any("gem" in ln for ln in lines)

        # Anthropic non-streaming & streaming
        r = await client.post(
            "http://ghost.local/v1/messages",
            json={
                "model": "gemini",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 200
        assert r.headers["Anthropic-Version"] == "2023-06-01"
        assert r.json()["content"][0]["text"] == "gemini reply"

        anth_lines = []
        async with client.stream(
            "POST",
            "http://ghost.local/v1/messages",
            json={
                "model": "gemini",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line:
                    anth_lines.append(line)
        assert any("event: message_start" in ln for ln in anth_lines)
        assert any("event: message_stop" in ln for ln in anth_lines)

    assert reg.chatgpt.closed and reg.gemini.closed


def test_sync_httpx_transport_streaming_and_rate_limit_429():
    reg = _make_registry(
        chatgpt=FakeProvider(name="chatgpt", fail_with=RateLimitError("too many requests")),
        gemini=FakeProvider(name="gemini", chunks=["hi ", "there"]),
    )
    with httpx_client(registry=reg) as client:
        # 429 mapped cleanly for official SDK retry handling
        r = client.post(
            "/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 429
        assert r.json()["error"]["type"] == "rate_limit_error"
        assert r.json()["error"]["code"] == "rate_limit_exceeded"

        # Streaming works synchronously through GhostTransport
        body_lines = []
        with client.stream(
            "POST",
            "/chat/completions",
            json={"model": "gemini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line:
                    body_lines.append(line)
        assert body_lines[-1] == "data: [DONE]"
        assert any("there" in ln for ln in body_lines)

        # Invalid JSON returns clean 400
        bad = client.post("/chat/completions", content=b"{bad json", headers={"Content-Type": "application/json"})
        assert bad.status_code == 400
        assert bad.json()["error"]["type"] == "invalid_request_error"

    assert reg.chatgpt.closed and reg.gemini.closed


# ---------------------------------------------------------------------------
# 4. SDK factory wiring (OpenAI & Anthropic drop-ins)
# ---------------------------------------------------------------------------

def test_sdk_factories_wire_http_client_and_handle_missing_packages(monkeypatch):
    # Case 1: Fake openai & anthropic modules verify exact wiring
    captured = {}

    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["openai"] = kwargs
            self.http_client = kwargs["http_client"]

        def close(self):
            self.http_client.close()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["async_openai"] = kwargs
            self.http_client = kwargs["http_client"]

    fake_openai.OpenAI = FakeOpenAI
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI

    fake_anthropic = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["anthropic"] = kwargs
            self.http_client = kwargs["http_client"]

        def close(self):
            self.http_client.close()

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured["async_anthropic"] = kwargs
            self.http_client = kwargs["http_client"]

    fake_anthropic.Anthropic = FakeAnthropic
    fake_anthropic.AsyncAnthropic = FakeAsyncAnthropic

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    reg = _make_registry()
    oa = ghost.openai_client(registry=reg)
    try:
        assert captured["openai"]["api_key"] == "ghost"
        assert captured["openai"]["base_url"] == "http://ghost.local/v1"
        r = oa.http_client.post(
            "/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "chatgpt reply"
    finally:
        oa.close()

    an = ghost.anthropic_client(registry=_make_registry())
    try:
        assert captured["anthropic"]["api_key"] == "ghost"
        assert captured["anthropic"]["base_url"] == "http://ghost.local"
        r = an.http_client.post(
            "/v1/messages",
            json={"model": "gemini", "max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["content"][0]["text"] == "gemini reply"
    finally:
        an.close()

    # Case 2: Clear ImportError when optional SDK isn't installed
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match="ghost\\[openai\\]"):
        ghost.openai_client()
    with pytest.raises(ImportError, match="ghost\\[anthropic\\]"):
        ghost.anthropic_client()


# ---------------------------------------------------------------------------
# 5. Library hygiene: no default disk logging in GhostEngine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_library_mode_disables_raw_file_logging_by_default(tmp_path, monkeypatch):
    import ghost.core.debug_logger as dl

    log_file = tmp_path / "should_not_exist.log"
    latest_file = tmp_path / "should_not_exist_latest.log"
    monkeypatch.setattr(dl, "RAW_LOG_FILE", str(log_file))
    monkeypatch.setattr(dl, "RAW_LOG_LATEST_FILE", str(latest_file))
    monkeypatch.setattr(dl, "RAW_LOGGING_ENABLED", True)

    reg = _make_registry()
    async with AsyncGhost(registry=reg) as client:
        assert await client.ask("Hello", model="auto") == "chatgpt reply"

    assert not log_file.exists()
    assert not latest_file.exists()

    # Opt-in raw_logging=True writes trace cleanly
    async with AsyncGhost(registry=_make_registry(), raw_logging=True) as client:
        assert await client.ask("Hello opt-in", model="auto") == "chatgpt reply"

    assert log_file.exists()
    assert "Hello opt-in" in log_file.read_text(encoding="utf-8")
