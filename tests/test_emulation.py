"""
tests/test_emulation.py — Closed-loop tests for OpenAI parameter emulation and tool-calling engine.
"""
import json
import pytest
from starlette.testclient import TestClient

from src.core.emulation.pipeline import EmulationPipeline
from src.core.emulation.stream_parser import (
    StreamEmulationProcessor,
    parse_tool_payload,
    process_non_streaming_response,
)
from src.server import app, ChatCompletionRequest, ChatMessage


# ============================================================================
# 1. Pipeline Request Transpilation Tests
# ============================================================================

def test_system_prompt_tool_injection():
    """Verify tools schema is injected into system prompt."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    req = {
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "tools": tools,
    }
    messages, ctx = EmulationPipeline.prepare_request(req)

    assert ctx["has_tools"] is True
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Tool Calling Instructions" in messages[0]["content"]
    assert "get_weather" in messages[0]["content"]
    assert "<tool_call>" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What's the weather in Tokyo?"


def test_tool_choice_directives():
    """Verify tool_choice='required' and named tool_choice add mandatory directives."""
    req_required = {
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "test_func"}}],
        "tool_choice": "required",
    }
    msgs_req, _ = EmulationPipeline.prepare_request(req_required)
    assert "MANDATORY REQUIREMENT: You MUST invoke at least one tool" in msgs_req[0]["content"]

    req_named = {
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "fetch_data"}}],
        "tool_choice": {"type": "function", "function": {"name": "fetch_data"}},
    }
    msgs_named, _ = EmulationPipeline.prepare_request(req_named)
    assert "MANDATORY REQUIREMENT: You MUST invoke the tool 'fetch_data'" in msgs_named[0]["content"]


def test_temperature_steering():
    """Verify temperature maps to appropriate behavioral directives."""
    # Low temperature -> factual
    msgs_low, _ = EmulationPipeline.prepare_request({
        "messages": [{"role": "user", "content": "Query"}],
        "temperature": 0.1,
    })
    assert any("strictly factual, precise" in m["content"] for m in msgs_low)

    # High temperature -> creative
    msgs_high, _ = EmulationPipeline.prepare_request({
        "messages": [{"role": "user", "content": "Query"}],
        "temperature": 0.9,
    })
    assert any("creative, explorative" in m["content"] for m in msgs_high)


def test_multi_turn_tool_folding():
    """Verify assistant tool_calls and role='tool' results are folded for upstream."""
    messages = [
        {"role": "user", "content": "Calculate 2+2"},
        {
            "role": "assistant",
            "content": "Let me calculate that for you.",
            "tool_calls": [
                {
                    "id": "call_calc_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_calc_1",
            "content": "4",
        },
    ]

    folded, _ = EmulationPipeline.prepare_request({"messages": messages})

    # The folded messages should not contain any role='tool'
    assert all(m["role"] in ("system", "user", "assistant") for m in folded)

    # Check assistant message contains serialized <tool_call>
    assistant_msg = next(m for m in folded if m["role"] == "assistant")
    assert "Let me calculate" in assistant_msg["content"]
    assert "<tool_call>" in assistant_msg["content"]
    assert "calculator" in assistant_msg["content"]

    # Check tool result folded into user message
    tool_folded_user = [m for m in folded if m["role"] == "user"][-1]
    assert "[Tool Result for call_calc_1]" in tool_folded_user["content"]
    assert "4" in tool_folded_user["content"]


# ============================================================================
# 2. Sliding-Window Streaming Parser Tests
# ============================================================================

import asyncio


def test_streaming_normal_text_no_tools():
    """Verify standard text streams without delay or corruption."""
    async def run():
        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "<table is HTML> "}}]},
                {"choices": [{"delta": {"content": "world!"}}]},
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "test-model"})
        collected_deltas = []
        finish_reasons = []

        async for chunk_str in processor.process_stream(mock_stream()):
            if chunk_str.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk_str[6:])
            choice = data["choices"][0]
            if "content" in choice["delta"]:
                collected_deltas.append(choice["delta"]["content"])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])

        full_text = "".join(collected_deltas)
        assert full_text == "Hello <table is HTML> world!"
        assert finish_reasons == ["stop"]

    asyncio.run(run())


def test_streaming_tool_call_interception():
    """Verify <tool_call> is intercepted and emitted as delta.tool_calls."""
    async def run():
        async def mock_tool_stream():
            chunks = [
                {"choices": [{"delta": {"content": "I'll check. <"}}]},
                {"choices": [{"delta": {"content": "tool_call>\n"}}]},
                {"choices": [{"delta": {"content": '{"name": "get_stock", "arguments": {"ticker": "AAPL"}}\n'}}]},
                {"choices": [{"delta": {"content": "</tool_call>"}}]},
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "test-model"})
        content_parts = []
        tool_calls = []
        finish_reasons = []

        async for chunk_str in processor.process_stream(mock_tool_stream()):
            if chunk_str.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk_str[6:])
            choice = data["choices"][0]
            delta = choice["delta"]
            if "content" in delta:
                content_parts.append(delta["content"])
            if "tool_calls" in delta:
                tool_calls.extend(delta["tool_calls"])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])

        # Raw <tool_call> should NOT leak into content
        assert "".join(content_parts) == "I'll check. "
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_stock"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"ticker": "AAPL"}
        assert finish_reasons == ["tool_calls"]

    asyncio.run(run())


def test_streaming_multiple_tool_calls():
    """Verify multiple <tool_call> blocks are parsed with sequential indices."""
    async def run():
        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "<tool_call>{\"name\": \"func1\", \"arguments\": {\"a\": 1}}</tool_call>"}}]},
                {"choices": [{"delta": {"content": "\n<tool_call>{\"name\": \"func2\", \"arguments\": {\"b\": 2}}</tool_call>"}}]},
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "test-model"})
        tool_calls = []
        finish_reasons = []

        async for chunk_str in processor.process_stream(mock_stream()):
            if chunk_str.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk_str[6:])
            choice = data["choices"][0]
            if "tool_calls" in choice["delta"]:
                tool_calls.extend(choice["delta"]["tool_calls"])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])

        assert len(tool_calls) == 2
        assert tool_calls[0]["index"] == 0
        assert tool_calls[0]["function"]["name"] == "func1"
        assert tool_calls[1]["index"] == 1
        assert tool_calls[1]["function"]["name"] == "func2"
        assert finish_reasons == ["tool_calls"]

    asyncio.run(run())


def test_streaming_stop_sequence_truncation():
    """Verify stream terminates immediately when stop sequence is encountered."""
    async def run():
        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "Line 1\n"}}]},
                {"choices": [{"delta": {"content": "STOP_HERE\nShould not appear"}}]},
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "test-model", "stop": ["STOP_HERE"]})
        content_parts = []
        finish_reasons = []

        async for chunk_str in processor.process_stream(mock_stream()):
            if chunk_str.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk_str[6:])
            choice = data["choices"][0]
            if "content" in choice["delta"]:
                content_parts.append(choice["delta"]["content"])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])

        assert "".join(content_parts) == "Line 1\n"
        assert "stop" in finish_reasons

    asyncio.run(run())


# ============================================================================
# 3. Non-Streaming Response Processing Tests
# ============================================================================

def test_non_streaming_tool_call_extraction():
    """Verify non-streaming response extracts tool_calls and cleans content."""
    raw_resp = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 123456,
        "model": "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "Sure! I am querying the database.\n"
                        "<tool_call>\n"
                        '{"name": "query_db", "arguments": {"sql": "SELECT 1"}}\n'
                        "</tool_call>"
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }
    processed = process_non_streaming_response(raw_resp, {"model": "auto"})

    choice = processed["choices"][0]
    msg = choice["message"]
    assert msg["content"] == "Sure! I am querying the database."
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "query_db"
    assert choice["finish_reason"] == "tool_calls"


def test_non_streaming_stop_sequence():
    """Verify non-streaming truncates at stop sequence."""
    raw_resp = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "First line. END_OF_TEXT Extra text."},
                "finish_reason": "stop",
            }
        ]
    }
    processed = process_non_streaming_response(raw_resp, {"stop": ["END_OF_TEXT"]})
    assert processed["choices"][0]["message"]["content"] == "First line. "
    assert processed["choices"][0]["finish_reason"] == "stop"


# ============================================================================
# 4. FastAPI Server End-to-End Test
# ============================================================================

def test_server_endpoint_parameter_acceptance(monkeypatch):
    """Verify server accepts tools, temperature, max_tokens without 422 error."""
    client = TestClient(app)

    # Mock provider response
    async def mock_non_streaming(messages, model, conv_id=None, parent_id=None):
        return {
            "id": "cmpl-mock",
            "object": "chat.completion",
            "created": 123456,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Calling calculator:\n"
                            "<tool_call>\n"
                            '{"name": "add", "arguments": {"a": 1, "b": 2}}\n'
                            "</tool_call>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    from src.providers import get_provider_for_model
    provider = get_provider_for_model("auto")
    monkeypatch.setattr(provider, "non_streaming_chat", mock_non_streaming)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is 1+2?"}],
        "temperature": 0.2,
        "max_tokens": 500,
        "stop": ["END"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {"type": "object", "properties": {"a": {"type": "int"}, "b": {"type": "int"}}},
                },
            }
        ],
        "tool_choice": "auto",
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tool_calls = data["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "add"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"a": 1, "b": 2}


def test_server_endpoint_streaming_tool_call(monkeypatch):
    """Verify server streaming endpoint yields delta.tool_calls in SSE."""
    client = TestClient(app)

    async def mock_streaming(messages, model, conv_id=None, parent_id=None):
        chunks = [
            {"choices": [{"delta": {"content": "Thinking...\n<tool_call>{\"name\": \"get_time\", \"arguments\": {}}</tool_call>"}}]},
        ]
        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n"
        yield "data: [DONE]\n\n"

    from src.providers import get_provider_for_model
    provider = get_provider_for_model("auto")
    monkeypatch.setattr(provider, "stream_chat", mock_streaming)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What time is it?"}],
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "get_time"}}],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    sse_lines = response.text.split("\n\n")
    data_lines = [l for l in sse_lines if l.startswith("data: ") and not l.endswith("[DONE]")]

    found_tool_call = False
    found_finish_reason = False
    for line in data_lines:
        chunk = json.loads(line[6:])
        choice = chunk["choices"][0]
        if "tool_calls" in choice.get("delta", {}):
            found_tool_call = True
            assert choice["delta"]["tool_calls"][0]["function"]["name"] == "get_time"
        if choice.get("finish_reason") == "tool_calls":
            found_finish_reason = True

    assert found_tool_call
    assert found_finish_reason


def test_streaming_max_tokens_enforcement():
    """Verify streaming terminates with finish_reason='length' when max_tokens is exceeded."""
    async def run():
        async def mock_stream():
            # Send long text
            chunks = [
                {"choices": [{"delta": {"content": "The quick brown fox jumps over the lazy dog. " * 5}}]},
                {"choices": [{"delta": {"content": "Another long paragraph that should be truncated. " * 5}}]},
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "test-model", "max_tokens": 10})
        finish_reasons = []

        async for chunk_str in processor.process_stream(mock_stream()):
            if chunk_str.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk_str[6:])
            choice = data["choices"][0]
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])

        assert "length" in finish_reasons

    asyncio.run(run())


def test_streaming_malformed_json_fallback():
    """Verify malformed JSON inside tool tag falls back gracefully to content text."""
    async def run():
        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "<tool_call>not valid json at all</tool_call>"}}]},
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "test-model"})
        content_parts = []

        async for chunk_str in processor.process_stream(mock_stream()):
            if chunk_str.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk_str[6:])
            choice = data["choices"][0]
            if "content" in choice.get("delta", {}):
                content_parts.append(choice["delta"]["content"])

        full_content = "".join(content_parts)
        # Should not crash, and content should contain the fallback text
        assert "not valid json at all" in full_content

    asyncio.run(run())


def test_null_content_in_assistant_message():
    """Verify assistant messages with content=None and tool_calls fold without error."""
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
    )
    req = ChatCompletionRequest(
        model="auto",
        messages=[msg],
    )
    messages, ctx = EmulationPipeline.prepare_request(req)
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert "<tool_call>" in messages[0]["content"]
    assert "foo" in messages[0]["content"]
