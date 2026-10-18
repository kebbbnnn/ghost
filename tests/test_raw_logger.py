"""
tests/test_raw_logger.py — Closed-loop verification of raw request prompt and response logging.
"""
import json
import os
import tempfile
import pytest
from starlette.testclient import TestClient

from src.core.debug_logger import (
    RequestTrace,
    start_trace,
    get_current_trace,
    _sanitize_for_logging,
    _extract_readable_prompt,
)
import src.core.debug_logger as debug_logger_module
from src.server import app


def test_sanitize_base64_strings():
    """Verify base64 image URLs and blobs are safely summarized for logging."""
    fake_b64 = "A" * 2000
    payload = {
        "text": "Hello world",
        "image_data_uri": f"data:image/png;base64,{fake_b64}",
        "raw_b64": fake_b64,
        "nested": {
            "image": f"data:image/jpeg;base64,{fake_b64}"
        }
    }
    sanitized = _sanitize_for_logging(payload)
    assert sanitized["text"] == "Hello world"
    assert "data:image/png;base64,<base64 data omitted:" in sanitized["image_data_uri"]
    assert "<base64 data omitted:" in sanitized["raw_b64"]
    assert "data:image/jpeg;base64,<base64 data omitted:" in sanitized["nested"]["image"]


def test_extract_readable_prompt():
    """Verify human-readable prompt extraction for user and system turns."""
    payload = {
        "system": "You are a coding assistant.",
        "tools": [
            {
                "name": "get_weather",
                "description": "Fetch weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        ],
        "messages": [
            {"role": "user", "content": "What's the weather in Seattle?"}
        ]
    }
    readable = _extract_readable_prompt(payload)
    assert "=== System Instructions ===" in readable
    assert "You are a coding assistant." in readable
    assert "=== Tools Defined ===" in readable
    assert "Tool: get_weather" in readable
    assert "--- Turn 1 (user) ---" in readable
    assert "What's the weather in Seattle?" in readable


def test_request_trace_lifecycle_and_file_writing(tmp_path, monkeypatch):
    """Verify RequestTrace creates, buffers, formats, and writes complete traces to files."""
    log_file = tmp_path / "test_debug_raw.log"
    latest_file = tmp_path / "test_debug_raw_latest.log"

    monkeypatch.setattr(debug_logger_module, "RAW_LOG_FILE", str(log_file))
    monkeypatch.setattr(debug_logger_module, "RAW_LOG_LATEST_FILE", str(latest_file))
    monkeypatch.setattr(debug_logger_module, "RAW_LOGGING_ENABLED", True)

    trace = RequestTrace(
        request_id="test-req-123",
        endpoint="POST /v1/messages",
        model="auto",
        stream=True,
    )

    # 1. Log client request
    trace.log_client_request({
        "model": "auto",
        "system": "System instructions here",
        "messages": [{"role": "user", "content": "Test prompt"}]
    })

    # 2. Log upstream request
    trace.log_upstream_request(
        provider_name="ChatGPT",
        url="https://chatgpt.com/backend-anon/conversation",
        payload={
            "action": "next",
            "messages": [
                {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["[System Instructions]: Foo\n\n[USER REQUEST]: Test prompt"]}
                }
            ]
        }
    )

    # 3. Log upstream response lines
    trace.log_upstream_chunk('data: {"v": "Hello "}')
    trace.log_upstream_chunk('data: {"v": "world!"}')
    trace.log_upstream_chunk('data: [DONE]')

    # 4. Log client chunks
    trace.log_client_chunk('event: content_block_delta\ndata: {"delta": {"text": "Hello world!"}}\n\n')

    # Finish trace
    trace.finish()

    assert log_file.exists()
    assert latest_file.exists()

    content = log_file.read_text(encoding="utf-8")
    assert "GHOSTTUNNEL RAW TRACE" in content
    assert "ID: test-req-123" in content
    assert "1. INCOMING CLIENT REQUEST" in content
    assert "2. UPSTREAM REQUEST PAYLOAD" in content
    assert "2b. UPSTREAM PROMPT TEXT" in content
    assert "3. RAW UPSTREAM RESPONSE" in content
    assert "data: {\"v\": \"Hello \"}" in content
    assert "3b. ACCUMULATED RAW MODEL TEXT" in content
    assert "Hello world!" in content
    assert "4. FINAL CLIENT RESPONSE" in content

    # Test idempotence: calling finish again does not duplicate
    size_before = len(content)
    trace.finish()
    assert len(log_file.read_text(encoding="utf-8")) == size_before


def test_server_openai_completions_logs_trace(tmp_path, monkeypatch):
    """Verify /v1/chat/completions end-to-end writes to raw log files."""
    log_file = tmp_path / "test_server_raw.log"
    latest_file = tmp_path / "test_server_raw_latest.log"

    monkeypatch.setattr(debug_logger_module, "RAW_LOG_FILE", str(log_file))
    monkeypatch.setattr(debug_logger_module, "RAW_LOG_LATEST_FILE", str(latest_file))
    monkeypatch.setattr(debug_logger_module, "RAW_LOGGING_ENABLED", True)

    from src.providers import _chatgpt

    async def mock_stream_chat(messages, model, conversation_id=None, parent_message_id=None):
        trace = get_current_trace()
        if trace:
            trace.log_upstream_request("ChatGPT", "https://chatgpt.com/backend-anon/conversation", {"action": "next", "messages": messages})
            trace.log_upstream_chunk('data: {"v": "Mocked test response"}')
            trace.log_upstream_chunk('data: [DONE]')
        yield 'data: {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Mocked test response"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(_chatgpt, "stream_chat", mock_stream_chat)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello from OpenAI test"}],
            "stream": True,
        }
    )

    assert response.status_code == 200
    assert log_file.exists()
    assert latest_file.exists()

    log_content = log_file.read_text(encoding="utf-8")
    assert "POST /v1/chat/completions" in log_content
    assert "Hello from OpenAI test" in log_content
    assert "Mocked test response" in log_content


def test_server_anthropic_messages_logs_trace(tmp_path, monkeypatch):
    """Verify /v1/messages end-to-end writes to raw log files."""
    log_file = tmp_path / "test_server_anthropic_raw.log"
    latest_file = tmp_path / "test_server_anthropic_raw_latest.log"

    monkeypatch.setattr(debug_logger_module, "RAW_LOG_FILE", str(log_file))
    monkeypatch.setattr(debug_logger_module, "RAW_LOG_LATEST_FILE", str(latest_file))
    monkeypatch.setattr(debug_logger_module, "RAW_LOGGING_ENABLED", True)

    from src.providers import _chatgpt

    async def mock_stream_chat(messages, model, conversation_id=None, parent_message_id=None):
        trace = get_current_trace()
        if trace:
            trace.log_upstream_request("ChatGPT", "https://chatgpt.com/backend-anon/conversation", {"action": "next", "messages": messages})
            trace.log_upstream_chunk('data: {"v": "Paris"}')
            trace.log_upstream_chunk('data: [DONE]')
        yield 'data: {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Paris"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(_chatgpt, "stream_chat", mock_stream_chat)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": True,
        }
    )

    assert response.status_code == 200
    assert log_file.exists()
    assert latest_file.exists()

    log_content = log_file.read_text(encoding="utf-8")
    assert "POST /v1/messages" in log_content
    assert "What is the capital of France?" in log_content
    assert "Paris" in log_content
