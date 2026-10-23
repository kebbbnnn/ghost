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
    assert "HEADLESS API GATEWAY" in messages[0]["content"]
    assert "get_weather" in messages[0]["content"]
    assert "<tool_call>" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What's the weather in Tokyo?"


def test_tool_synthesis_phase_directive():
    """Verify tool results trigger SYNTHESIS PHASE ACTIVE directive rather than TOOL_CALL_REQUIRED."""
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    messages = [
        {"role": "user", "content": "Compute 5*5"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "calculator", "arguments": '{"expr": "5*5"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "25"},
    ]
    req = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    folded, ctx = EmulationPipeline.prepare_request(req)
    system_msg = next(m for m in folded if m["role"] == "system")
    assert "SYNTHESIS PHASE: External tool execution is COMPLETE" in system_msg["content"]
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" not in system_msg["content"]
    assert "DO NOT invoke or emit any <tool_call> tags" in system_msg["content"]
    assert ctx["is_synthesis_phase"] is True
    assert "calculator" in ctx["executed_tool_names"]


def test_second_turn_user_prompt_resets_synthesis_phase():
    """Verify that when a user asks a new question after a completed tool turn, synthesis phase is reset to False."""
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    messages = [
        {"role": "user", "content": "Compute 5*5"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "calculator", "arguments": '{"expr": "5*5"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "25"},
        {"role": "assistant", "content": "The answer is 25."},
        {"role": "user", "content": "Now compute 10*10"},
    ]
    req = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    folded, ctx = EmulationPipeline.prepare_request(req)
    system_msg = next(m for m in folded if m["role"] == "system")
    assert ctx["is_synthesis_phase"] is False
    assert ctx["executed_tool_names"] == []
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" in system_msg["content"]
    assert "SYNTHESIS PHASE: External tool execution is COMPLETE" not in system_msg["content"]


def test_synthesis_phase_suppresses_duplicate_tool_echo_non_streaming():
    """Verify echoed tool calls in synthesis phase are stripped, preserving content and setting finish_reason='stop'."""
    raw_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name": "calculator", "arguments": {"expr": "5*5"}}</tool_call> The answer is 25.',
                },
                "finish_reason": "stop",
            }
        ]
    }
    context = {
        "is_synthesis_phase": True,
        "executed_tool_names": ["calculator"],
    }
    processed = process_non_streaming_response(raw_resp, context)
    choice = processed["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "The answer is 25."


def test_synthesis_phase_suppresses_duplicate_tool_echo_streaming():
    """Verify streaming processor suppresses echoed tool calls in synthesis phase."""
    async def run():
        processor = StreamEmulationProcessor({
            "is_synthesis_phase": True,
            "executed_tool_names": ["calculator"],
        })
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "<tool_call>{\\"name\\": \\"calculator\\", \\"arguments\\": {}}</tool_call> The answer is 25."}}]}\n\n'
            yield 'data: [DONE]\n\n'

        chunks = []
        async for c in processor.process_stream(mock_stream()):
            chunks.append(c)

        # Parse SSE lines
        tool_deltas = []
        content_deltas = []
        finish_reasons = []
        for line in chunks:
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = json.loads(line[6:])
                delta = payload["choices"][0]["delta"]
                if "tool_calls" in delta:
                    tool_deltas.extend(delta["tool_calls"])
                if "content" in delta:
                    content_deltas.append(delta["content"])
                if payload["choices"][0].get("finish_reason"):
                    finish_reasons.append(payload["choices"][0]["finish_reason"])

        assert len(tool_deltas) == 0, "No duplicate tool calls should be emitted"
        assert "".join(content_deltas).strip() == "The answer is 25."
        assert "stop" in finish_reasons
        assert "tool_calls" not in finish_reasons

    asyncio.run(run())


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
    assert "[Tool Results]:" in tool_folded_user["content"]
    assert "call_calc_1" in tool_folded_user["content"]
    assert "calculator" in tool_folded_user["content"]
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


def test_stream_emulation_preserves_conversation_id_and_parent_message_id():
    """Verify StreamEmulationProcessor captures conversation_id and parent_message_id on processor,
    and only includes in chunk payload when expose_session_metadata_in_chunks is enabled."""
    async def run():
        async def mock_stream():
            yield 'data: {"id": "c1", "conversation_id": "conv_123", "parent_message_id": "resp_456", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
            yield 'data: [DONE]\n\n'

        # 1. Strict standard mode (default)
        processor = StreamEmulationProcessor({"model": "test-model"})
        chunks = []
        async for c in processor.process_stream(mock_stream()):
            if c.strip() != "data: [DONE]":
                chunks.append(json.loads(c[6:]))

        assert len(chunks) > 0
        assert processor.conversation_id == "conv_123"
        assert processor.parent_message_id == "resp_456"
        assert "conversation_id" not in chunks[0]
        assert "parent_message_id" not in chunks[0]

        # 2. Explicit chunk exposure mode
        proc_legacy = StreamEmulationProcessor({"model": "test-model", "expose_session_metadata_in_chunks": True})
        chunks_legacy = []
        async for c in proc_legacy.process_stream(mock_stream()):
            if c.strip() != "data: [DONE]":
                chunks_legacy.append(json.loads(c[6:]))

        assert len(chunks_legacy) > 0
        assert chunks_legacy[0]["conversation_id"] == "conv_123"
        assert chunks_legacy[0]["parent_message_id"] == "resp_456"

    import asyncio
    asyncio.run(run())


def test_stream_emulation_strips_google_card_urls():
    """Verify StreamEmulationProcessor strips Google card content URLs before tool calls or text."""
    async def run():
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "http://googleusercontent.com/card_content/0\\n<tool_call>{\\"name\\": \\"get_weather\\", \\"arguments\\": {\\"location\\": \\"Paris\\"}}</tool_call>"}}]}\n\n'
            yield 'data: [DONE]\n\n'

        processor = StreamEmulationProcessor({"model": "gemini"})
        tool_calls = []
        contents = []
        async for c in processor.process_stream(mock_stream()):
            if c.strip() != "data: [DONE]":
                delta = json.loads(c[6:])["choices"][0]["delta"]
                if "tool_calls" in delta:
                    tool_calls.extend(delta["tool_calls"])
                if "content" in delta and delta["content"]:
                    contents.append(delta["content"])

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        full_content = "".join(contents)
        assert "googleusercontent.com" not in full_content

    import asyncio
    asyncio.run(run())


def test_gemini_session_build_prompt_multi_turn_recency():
    """Verify GeminiProvider._build_prompt appends recency reinforcement for tool calls on multi-turn conversations."""
    from src.providers.gemini.session import GeminiProvider
    provider = GeminiProvider()
    messages = [
        {"role": "system", "content": "[SYSTEM: HEADLESS API GATEWAY]\nCRITICAL MODE: TOOL_CALL_REQUIRED.\nAvailable external RPC tools:\n- get_weather()"},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "Paris is sunny."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    prompt = provider._build_prompt(messages, conversation_id=None)
    assert "[ACTIVE DIRECTIVE FOR CURRENT TURN]" in prompt
    assert "MANDATORY: Emit fresh <tool_call> tags" in prompt


# ============================================================================
# 8. GenUI Tag Sanitization & Parallel Tool Calling Tests
# ============================================================================

def test_genui_tag_sanitization_non_streaming():
    """Verify GenUI React widget tags are sanitized into human-readable text in non-streaming responses."""
    from src.core.emulation.stream_parser import sanitize_genui_tags

    raw_text = (
        '<ClockWidget location="Tokyo, Japan" tz_name="Asia/Tokyo" time_format="24h" mode="live" />\n\n'
        '**Paris weather:** Sunny\n'
        '<WeatherLiveSource location="Paris" duration={1} locale="en-US" />'
    )
    cleaned = sanitize_genui_tags(raw_text)
    assert "<ClockWidget" not in cleaned
    assert "<WeatherLiveSource" not in cleaned
    assert "**Tokyo, Japan** (Asia/Tokyo)" in cleaned
    assert "**Paris**" in cleaned

    # Test arbitrary unfamiliar custom element (generalized, no hardcoded component guard)
    stock_text = '<StockCard symbol="AAPL" price="150.25" currency="USD" />'
    cleaned_stock = sanitize_genui_tags(stock_text)
    assert cleaned_stock == "**AAPL** (Price: 150.25, Currency: USD)"

    # Test element with inner content
    alert_text = '<AlertBox type="warning">Low disk space</AlertBox>'
    assert sanitize_genui_tags(alert_text) == "Low disk space"

    # Test code block and inline code preservation (must not be mangled)
    code_text = "```jsx\n<ClockWidget location=\"Tokyo\" />\n```\nAnd inline: `<ClockWidget />`\nAnd html: <div>text</div>"
    cleaned_code = sanitize_genui_tags(code_text)
    assert "<ClockWidget location=\"Tokyo\" />" in cleaned_code
    assert "`<ClockWidget />`" in cleaned_code
    assert "<div>text</div>" in cleaned_code

    # Test via process_non_streaming_response
    raw_resp = {
        "id": "chatcmpl-test",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '<ClockWidget location="Tokyo, Japan" tz_name="Asia/Tokyo" time_format="24h" mode="live" />\n\nAll good!',
            },
            "finish_reason": "stop"
        }]
    }
    processed = process_non_streaming_response(raw_resp, {"model": "auto"})
    final_content = processed["choices"][0]["message"]["content"]
    assert "<ClockWidget" not in final_content
    assert "**Tokyo, Japan** (Asia/Tokyo)" in final_content


def test_genui_tag_sanitization_streaming():
    """Verify GenUI React widget tags split across streaming chunks are held and cleanly formatted."""
    async def run():
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "<ClockWidget"}}]}\n\n'
            yield 'data: {"choices": [{"delta": {"content": " location=\\"Tokyo, Japan\\" tz_name=\\"Asia/Tokyo\\""}}]}\n\n'
            yield 'data: {"choices": [{"delta": {"content": " time_format=\\"24h\\" mode=\\"live\\" />\\n\\n**Paris weather:** Sunny"}}]}\n\n'
            yield 'data: [DONE]\n\n'

        processor = StreamEmulationProcessor({"model": "auto"})
        contents = []
        async for c in processor.process_stream(mock_stream()):
            if c.strip() != "data: [DONE]":
                delta = json.loads(c[6:])["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    contents.append(delta["content"])

        full_content = "".join(contents)
        assert "<ClockWidget" not in full_content
        assert "**Tokyo, Japan** (Asia/Tokyo)" in full_content
        assert "**Paris weather:** Sunny" in full_content

    import asyncio
    asyncio.run(run())


def test_parallel_tool_calling_system_prompt():
    """Verify system instructions include parallel tool calling rules and multi-intent examples."""
    tools = [
        {"type": "function", "function": {"name": "calculator"}},
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_current_time"}},
    ]
    req = {
        "messages": [{"role": "user", "content": "Time in Tokyo, weather in Paris, and calculate 450 * 1.15"}],
        "tools": tools,
    }
    messages, ctx = EmulationPipeline.prepare_request(req)
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "PARALLEL & MULTI-TOOL CALLS" in system_msg["content"]
    assert "Example of parallel tool calls:" in system_msg["content"]
    assert "get_current_time" in system_msg["content"]
    assert "calculator" in system_msg["content"]


def test_parallel_multi_tool_call_parsing():
    """Verify both streaming and non-streaming parsers handle multiple consecutive <tool_call> tags."""
    # 1. Non-streaming
    raw_content = (
        '<tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}}</tool_call>\n'
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>\n'
        '<tool_call>{"name": "calculator", "arguments": {"expression": "450 * 1.15"}}</tool_call>'
    )
    raw_resp = {
        "id": "chatcmpl-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": raw_content},
            "finish_reason": "stop"
        }]
    }
    processed = process_non_streaming_response(raw_resp, {"model": "auto"})
    tool_calls = processed["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 3
    assert tool_calls[0]["function"]["name"] == "get_current_time"
    assert tool_calls[1]["function"]["name"] == "get_weather"
    assert tool_calls[2]["function"]["name"] == "calculator"
    assert processed["choices"][0]["finish_reason"] == "tool_calls"

    # 2. Streaming
    async def run():
        async def mock_stream():
            chunk_json = json.dumps({"choices": [{"delta": {"content": raw_content}}]})
            yield f"data: {chunk_json}\n\n"
            yield "data: [DONE]\n\n"

        processor = StreamEmulationProcessor({"model": "auto"})
        streamed_calls = []
        async for c in processor.process_stream(mock_stream()):
            if c.strip() != "data: [DONE]":
                delta = json.loads(c[6:])["choices"][0]["delta"]
                if "tool_calls" in delta:
                    streamed_calls.extend(delta["tool_calls"])

        assert len(streamed_calls) == 3
        assert streamed_calls[0]["function"]["name"] == "get_current_time"
        assert streamed_calls[1]["function"]["name"] == "get_weather"
        assert streamed_calls[2]["function"]["name"] == "calculator"

    import asyncio
    asyncio.run(run())


def test_genui_suppression_prompt_injection():
    """Verify explicit GenUI suppression and desired response directives in pipeline and ChatGPT session."""
    from src.providers.chatgpt.session import _convert_messages, CHATGPT_BASE_DIRECTIVE

    # 1. Plain chat completions with ChatGPT session -> injects CHATGPT_BASE_DIRECTIVE
    guest_msgs = _convert_messages([{"role": "user", "content": "What is the time in Tokyo?"}])
    assert len(guest_msgs) == 1
    content_part = guest_msgs[0]["content"]["parts"][0]
    assert "Desired Response Format: Respond strictly in standard markdown" in content_part
    assert "Do NOT output GenUI components" in content_part
    assert "[USER REQUEST]:\nWhat is the time in Tokyo?" in content_part

    # 2. Emulation pipeline tool-calling phase -> explicit GenUI suppression in TOOL_CALL_REQUIRED
    req_tools = {
        "messages": [{"role": "user", "content": "Check weather in Tokyo"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    msgs_tool, _ = EmulationPipeline.prepare_request(req_tools)
    sys_tool = next(m for m in msgs_tool if m["role"] == "system")
    assert "DO NOT emit GenUI tags, interactive components, or custom XML/JSX tags" in sys_tool["content"]
    assert "Available external RPC tools:" in sys_tool["content"]

    # 3. Emulation pipeline synthesis phase -> explicit desired response format in SYNTHESIS PHASE
    req_synthesis = {
        "messages": [
            {"role": "user", "content": "Check weather in Tokyo"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"temp\": \"20C\", \"condition\": \"Sunny\"}"
            }
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    msgs_synth, ctx_synth = EmulationPipeline.prepare_request(req_synthesis)
    assert ctx_synth["is_synthesis_phase"] is True
    sys_synth = next(m for m in msgs_synth if m["role"] == "system")
    assert "DESIRED RESPONSE FORMAT: Respond strictly in standard markdown text only" in sys_synth["content"]
    assert "Do NOT output GenUI components" in sys_synth["content"]

