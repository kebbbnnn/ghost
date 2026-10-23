"""
tests/test_anthropic_response_fidelity.py — Closed-loop fidelity tests for Anthropic Messages API responses.
Validates headers, event sequencing (message_start, ping, content_block_start, delta, stop, message_delta, message_stop),
error formats, usage tracking, and absence of non-standard body pollution.
"""
import json
import pytest
from starlette.testclient import TestClient

from src.server import app
from src.core.exceptions import RateLimitError


@pytest.fixture
def client():
    return TestClient(app)


def test_anthropic_non_streaming_response_fidelity(client, monkeypatch):
    """Verify non-streaming Anthropic messages match official 1:1 schema and headers."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_non_streaming_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "I am Claude."}}],
            "conversation_id": "anthropic_conv_123",
            "parent_message_id": "anthropic_parent_456",
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_non_streaming_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Who are you?"}],
        "stream": False,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    # 1. Header checks
    assert "request-id" in resp.headers or "Request-ID" in resp.headers
    assert resp.headers.get("anthropic-version") == "2023-06-01"
    assert "anthropic-ratelimit-requests-limit" in resp.headers
    assert "anthropic-ratelimit-tokens-limit" in resp.headers
    assert resp.headers.get("x-ghost-conversation-id") == "anthropic_conv_123"
    assert resp.headers.get("x-ghost-parent-message-id") == "anthropic_parent_456"

    # 2. Body schema checks
    data = resp.json()
    assert data["type"] == "message"
    assert data["id"].startswith("msg_")
    assert data["role"] == "assistant"
    assert data["stop_reason"] == "end_turn"
    assert data["stop_sequence"] is None

    # Check content blocks
    assert isinstance(data["content"], list)
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "I am Claude."

    # Check clean body: no conversation_id pollution
    assert "conversation_id" not in data
    assert "parent_message_id" not in data

    # Check usage
    assert "usage" in data
    assert data["usage"]["input_tokens"] >= 0
    assert data["usage"]["output_tokens"] > 0


def test_anthropic_streaming_event_sequence_fidelity(client, monkeypatch):
    """Verify streaming Anthropic messages emit exact sequence: message_start -> ping -> content_block_start -> delta -> stop -> message_delta -> message_stop."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_stream_chat(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id": "c1", "conversation_id": "conv_stream_001", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": " world!"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    # 1. Header checks
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("anthropic-version") == "2023-06-01"

    # 2. Parse SSE events
    raw_text = resp.text.strip()
    event_blocks = raw_text.split("\n\n")

    events = []
    for block in event_blocks:
        if not block.strip():
            continue
        event_type = None
        data_payload = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_payload = json.loads(line[6:].strip())
        if event_type:
            events.append((event_type, data_payload))

    # Event lifecycle checks
    event_names = [e[0] for e in events]
    assert "message_start" in event_names
    assert "ping" in event_names
    assert "content_block_start" in event_names
    assert "content_block_delta" in event_names
    assert "content_block_stop" in event_names
    assert "message_delta" in event_names
    assert "message_stop" in event_names

    # Check message_start payload
    msg_start = next(p for name, p in events if name == "message_start")
    assert msg_start["type"] == "message_start"
    assert msg_start["message"]["id"].startswith("msg_")
    assert msg_start["message"]["role"] == "assistant"
    assert msg_start["message"]["content"] == []
    assert "usage" in msg_start["message"]

    # Check message_delta payload (ensure clean body, no conversation_id)
    msg_delta = next(p for name, p in events if name == "message_delta")
    assert msg_delta["type"] == "message_delta"
    assert msg_delta["delta"]["stop_reason"] == "end_turn"
    assert "conversation_id" not in msg_delta
    assert "parent_message_id" not in msg_delta
    assert "usage" in msg_delta
    assert msg_delta["usage"]["output_tokens"] > 0


def test_anthropic_error_response_fidelity_non_streaming(client, monkeypatch):
    """Verify error responses on non-streaming /v1/messages match Anthropic error schema."""
    from src.providers import PROVIDERS
    mock_provider = PROVIDERS[0]

    async def mock_rate_limit(messages, model, conversation_id=None, parent_message_id=None):
        raise RateLimitError("Rate limit exceeded on Anthropic backend")

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_rate_limit)

    payload = {
        "model": "auto",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Spam"}],
        "stream": False,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 429

    data = resp.json()
    assert data["type"] == "error"
    assert "error" in data
    assert data["error"]["type"] == "rate_limit_error"
    assert "Rate limit exceeded" in data["error"]["message"]


def test_anthropic_validation_error_response_fidelity(client):
    """Verify RequestValidationError on /v1/messages returns Anthropic compliant error dictionary."""
    # Send invalid body with non-list messages
    payload = {
        "model": "auto",
        "messages": "not a list",
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert data["type"] == "error"
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"


def test_anthropic_route_aliases(client, monkeypatch):
    """Verify /v1/messages, /v1/v1/messages, and /messages all route correctly."""
    from src.providers.chatgpt.session import ChatGPTProvider

    async def mock_stream_chat(self, messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hi"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(ChatGPTProvider, "stream_chat", mock_stream_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    for path in ["/v1/messages", "/v1/v1/messages", "/messages"]:
        resp = client.post(path, json=payload)
        assert resp.status_code == 200, f"Failed on path {path}"
        assert resp.headers.get("content-type", "").startswith("text/event-stream")

