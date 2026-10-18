"""
tests/test_openai_response_fidelity.py — Closed-loop fidelity tests for OpenAI API responses.
Validates headers, SSE chunk schemas, stream options usage chunks, system fingerprints,
error schemas, and absence of non-standard body pollution.
"""
import json
import pytest
from starlette.testclient import TestClient

from src.server import app
from src.core.emulation.stream_parser import StreamEmulationProcessor
from src.core.exceptions import RateLimitError


@pytest.fixture
def client():
    return TestClient(app)


def test_openai_non_streaming_response_fidelity(client, monkeypatch):
    """Verify non-streaming OpenAI completions match official 1:1 schema and headers."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_non_streaming_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Hello there!"}, "finish_reason": "stop"}],
            "conversation_id": "test_conv_123",
            "parent_message_id": "test_parent_456",
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_non_streaming_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    # 1. Header checks
    assert "x-request-id" in resp.headers or "X-Request-ID" in resp.headers
    assert resp.headers.get("openai-organization") == "org-ghost"
    assert resp.headers.get("openai-version") == "2020-10-01"
    assert "openai-processing-ms" in resp.headers
    assert resp.headers.get("x-ghost-conversation-id") == "test_conv_123"
    assert resp.headers.get("x-ghost-parent-message-id") == "test_parent_456"

    # 2. Body schema checks
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int)
    assert data["model"] == "auto"
    assert "system_fingerprint" in data
    assert data["system_fingerprint"].startswith("fp_ghost_")

    # Verify no metadata leakage in body
    assert "conversation_id" not in data
    assert "parent_message_id" not in data

    # Choices checks
    choices = data["choices"]
    assert len(choices) == 1
    assert choices[0]["index"] == 0
    assert choices[0]["message"]["role"] == "assistant"
    assert choices[0]["message"]["content"] == "Hello there!"
    assert choices[0]["finish_reason"] == "stop"
    assert choices[0]["logprobs"] is None

    # Usage checks
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]


def test_openai_streaming_response_fidelity(client, monkeypatch):
    """Verify streaming OpenAI completions match official SSE chunk structure and headers."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_stream_chat(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id": "c1", "conversation_id": "test_conv_789", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": " world!"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    # 1. Header checks
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("openai-organization") == "org-ghost"
    assert resp.headers.get("openai-version") == "2020-10-01"

    # 2. SSE chunk checks
    lines = resp.text.strip().split("\n\n")
    chunks = []
    has_done = False
    for line in lines:
        if line.startswith("data: "):
            val = line[6:].strip()
            if val == "[DONE]":
                has_done = True
            else:
                chunks.append(json.loads(val))

    assert has_done is True
    assert len(chunks) >= 2

    for c in chunks:
        assert c["object"] == "chat.completion.chunk"
        assert c["id"].startswith("c1") or c["id"].startswith("chatcmpl-")
        assert "system_fingerprint" in c
        assert "conversation_id" not in c  # clean standard body
        assert "parent_message_id" not in c
        if c.get("choices"):
            assert c["choices"][0]["logprobs"] is None

    # Final chunk should have finish_reason="stop"
    final_chunk = chunks[-1]
    assert final_chunk["choices"][0]["finish_reason"] == "stop"


def test_openai_stream_options_include_usage(client, monkeypatch):
    """Verify stream_options: {'include_usage': True} emits final usage chunk before [DONE]."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_stream_chat(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"choices": [{"delta": {"content": "Test response"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Compute this"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    lines = [line[6:].strip() for line in resp.text.strip().split("\n\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    usage_chunk = json.loads(lines[-2])

    assert usage_chunk["object"] == "chat.completion.chunk"
    assert usage_chunk["choices"] == []
    assert "usage" in usage_chunk
    assert usage_chunk["usage"]["prompt_tokens"] > 0
    assert usage_chunk["usage"]["completion_tokens"] > 0
    assert usage_chunk["usage"]["total_tokens"] == usage_chunk["usage"]["prompt_tokens"] + usage_chunk["usage"]["completion_tokens"]


def test_openai_error_response_fidelity_non_streaming(client, monkeypatch):
    """Verify error responses on non-streaming /v1/chat/completions match OpenAI error schema."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_rate_limit(messages, model, conversation_id=None, parent_message_id=None):
        raise RateLimitError("Rate limit exceeded on ChatGPT")

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_rate_limit)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Spam"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 429

    data = resp.json()
    assert "error" in data
    assert data["error"]["type"] == "rate_limit_error"
    assert data["error"]["code"] == "rate_limit_exceeded"
    assert "Rate limit exceeded" in data["error"]["message"]


def test_openai_validation_error_response_fidelity(client):
    """Verify RequestValidationError produces OpenAI compliant error dictionary."""
    # Send request with invalid messages (missing role)
    payload = {
        "model": "auto",
        "messages": [{"content": "Missing role"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"
    assert "role" in data["error"]["message"]
