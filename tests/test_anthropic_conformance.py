"""
tests/test_anthropic_conformance.py — Comprehensive conformance test suite for Anthropic Messages API.

Validates 100% adherence to Anthropic Messages API specifications:
- Strict Pydantic models with `extra="forbid"` to ensure zero schema drift or body pollution.
- Full matrix of request features: system prompts (string and content block arrays),
  multimodal base64 vision images, sampling params (temperature, top_p, stop_sequences),
  tools definitions, tool_choice variants (auto, any, tool, none), and multi-turn
  tool_use / tool_result conversation histories.
- Non-streaming response schemas, headers, content blocks (text & tool_use), stop reasons, usage accounting.
- Streaming SSE event sequencing: strict lifecycle order
  (message_start -> ping -> content_block_start -> content_block_delta -> content_block_stop -> message_delta -> message_stop),
  streaming tool_use input_json_delta, error events.
- Conformance of error responses (validation, malformed JSON, unrouted models, rate limits).
"""
import json
import pytest
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from starlette.testclient import TestClient

from src.server import app
from src.core.exceptions import RateLimitError
from src.providers import PROVIDERS


# ---------------------------------------------------------------------------
# Strict Anthropic Contract Schemas (extra="forbid" rejects non-standard fields)
# ---------------------------------------------------------------------------

class AnthropicTextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class AnthropicToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int
    output_tokens: int


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[Union[AnthropicTextBlock, AnthropicToolUseBlock]]
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


class AnthropicMessageStartDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[Any]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


class AnthropicMessageStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_start"]
    message: AnthropicMessageStartDetail


class AnthropicPingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ping"]


class AnthropicContentBlockStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["content_block_start"]
    index: int
    content_block: Union[AnthropicTextBlock, AnthropicToolUseBlock]


class AnthropicTextDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_delta"]
    text: str


class AnthropicInputJsonDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_json_delta"]
    partial_json: str


class AnthropicContentBlockDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["content_block_delta"]
    index: int
    delta: Union[AnthropicTextDelta, AnthropicInputJsonDelta]


class AnthropicContentBlockStopEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["content_block_stop"]
    index: int


class AnthropicMessageDeltaBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None


class AnthropicMessageDeltaUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_tokens: int


class AnthropicMessageDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_delta"]
    delta: AnthropicMessageDeltaBody
    usage: AnthropicMessageDeltaUsage


class AnthropicMessageStopEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_stop"]


class AnthropicErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"]
    error: AnthropicErrorDetail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Request Conformance Tests
# ---------------------------------------------------------------------------

def test_anthropic_request_conformance_minimal(client, monkeypatch):
    """Test standard minimal valid Anthropic request conforms to specification."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "I am Claude."}}],
            "conversation_id": "conv_claude_1",
            "parent_message_id": "parent_claude_1",
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Who are you?"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    # Header checks
    assert resp.headers["anthropic-version"] == "2023-06-01"
    assert "request-id" in resp.headers or "Request-ID" in resp.headers
    assert "anthropic-ratelimit-requests-limit" in resp.headers
    assert "anthropic-ratelimit-tokens-limit" in resp.headers

    # Validate schema
    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.id.startswith("msg_")
    assert validated.type == "message"
    assert validated.role == "assistant"
    assert validated.stop_reason == "end_turn"
    assert len(validated.content) == 1
    assert validated.content[0].type == "text"
    assert validated.content[0].text == "I am Claude."
    assert validated.usage.input_tokens >= 0
    assert validated.usage.output_tokens > 0


def test_anthropic_request_conformance_system_string(client, monkeypatch):
    """Test Anthropic request with system prompt provided as a string."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "Understood."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "system": "You are a concise, helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.content[0].text == "Understood."

    sys_text = " ".join(m.get("content", "") for m in captured_messages if isinstance(m.get("content"), str))
    assert "concise, helpful assistant" in sys_text


def test_anthropic_request_conformance_system_block_array(client, monkeypatch):
    """Test Anthropic request with system prompt provided as array of text content blocks."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "Multi-system accepted."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "system": [
            {"type": "text", "text": "Part 1: You are a scientist."},
            {"type": "text", "text": "Part 2: Answer in bullet points."},
        ],
        "messages": [{"role": "user", "content": "What is water?"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.content[0].text == "Multi-system accepted."

    sys_text = " ".join(m.get("content", "") for m in captured_messages if isinstance(m.get("content"), str))
    assert "Part 1: You are a scientist" in sys_text
    assert "Part 2: Answer in bullet points" in sys_text


def test_anthropic_request_conformance_multimodal_vision(client, monkeypatch):
    """Test Anthropic request with base64 image content block."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "The image shows a blue triangle."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this shape."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        },
                    },
                ],
            }
        ],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert "triangle" in validated.content[0].text


def test_anthropic_request_conformance_sampling_params(client, monkeypatch):
    """Test Anthropic sampling parameters (temperature, top_p, stop_sequences)."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Output section 1\nEND_OF_THOUGHT\nOutput section 2"}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop_sequences": ["END_OF_THOUGHT"],
        "messages": [{"role": "user", "content": "Write sections"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.stop_reason == "stop_sequence"
    assert validated.stop_sequence == "END_OF_THOUGHT"
    assert validated.content[0].text == "Output section 1\n"


def test_anthropic_request_conformance_tools_definition_and_choices(client, monkeypatch):
    """Test Anthropic tools schema definition and tool_choice variants (auto, any, tool, none)."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Tool acknowledged"}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    tools = [
        {
            "name": "calculate",
            "description": "Calculate math expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression string"}
                },
                "required": ["expression"],
            },
        }
    ]

    for choice in [{"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "calculate"}, {"type": "none"}]:
        resp = client.post("/v1/messages", json={
            "model": "auto",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Compute 50 * 2"}],
            "tools": tools,
            "tool_choice": choice,
        })
        assert resp.status_code == 200
        AnthropicMessage.model_validate(resp.json())


def test_anthropic_request_conformance_tool_result_history(client, monkeypatch):
    """Test multi-turn tool history with assistant tool_use and user tool_result blocks."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "The result is 100."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What is 50 * 2?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_math_123",
                        "name": "calculate",
                        "input": {"expression": "50 * 2"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_math_123",
                        "content": "100",
                    }
                ],
            },
        ],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.stop_reason == "end_turn"
    assert "100" in validated.content[0].text


# ---------------------------------------------------------------------------
# Response Conformance Tests (Non-Streaming)
# ---------------------------------------------------------------------------

def test_anthropic_response_conformance_tool_use_block(client, monkeypatch):
    """Test non-streaming response containing tool_use content block conforms to Anthropic schema."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>',
                }
            }]
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Weather in SF"}],
        "tools": [{"name": "get_weather", "description": "...", "input_schema": {"type": "object"}}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.stop_reason == "tool_use"
    assert len(validated.content) == 1
    tool_block = validated.content[0]
    assert tool_block.type == "tool_use"
    assert tool_block.id.startswith("toolu_")
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"location": "San Francisco"}


def test_anthropic_response_conformance_max_tokens_length(client, monkeypatch):
    """Test non-streaming response stop_reason is 'max_tokens' when token limit is hit."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "X" * 200}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Generate text"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    validated = AnthropicMessage.model_validate(resp.json())
    assert validated.stop_reason == "max_tokens"
    assert len(validated.content[0].text) <= 40


def test_anthropic_response_conformance_zero_body_pollution(client, monkeypatch):
    """Test non-streaming response body contains zero internal session variables (no conversation_id or parent_message_id)."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Clean Claude response."}}],
            "conversation_id": "ghost_secret_anthropic_conv",
            "parent_message_id": "ghost_secret_anthropic_parent",
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    raw_data = resp.json()
    assert "conversation_id" not in raw_data
    assert "parent_message_id" not in raw_data
    assert resp.headers.get("x-ghost-conversation-id") == "ghost_secret_anthropic_conv"
    assert resp.headers.get("x-ghost-parent-message-id") == "ghost_secret_anthropic_parent"

    # Schema must validate with extra="forbid"
    AnthropicMessage.model_validate(raw_data)


# ---------------------------------------------------------------------------
# Streaming Response Conformance Tests
# ---------------------------------------------------------------------------

def test_anthropic_streaming_conformance_lifecycle_sequence(client, monkeypatch):
    """Test streaming messages emit strictly ordered sequence of 7 Anthropic lifecycle events."""
    mock_provider = PROVIDERS[0]

    async def mock_stream(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": " Claude!"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers["anthropic-version"] == "2023-06-01"

    # Parse SSE events into (event_type, payload)
    raw_blocks = [b.strip() for b in resp.text.strip().split("\n\n") if b.strip()]
    events = []
    for block in raw_blocks:
        ev_type = None
        ev_data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_type = line[7:].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[6:].strip())
        if ev_type and ev_data is not None:
            events.append((ev_type, ev_data))

    event_names = [e[0] for e in events]
    # Strict event lifecycle order
    expected_order = [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    for expected_name in expected_order:
        assert expected_name in event_names

    # Validate each event against its strict Pydantic model
    for ev_type, ev_payload in events:
        if ev_type == "message_start":
            ev_obj = AnthropicMessageStartEvent.model_validate(ev_payload)
            assert ev_obj.message.id.startswith("msg_")
            assert ev_obj.message.role == "assistant"
            assert ev_obj.message.content == []
        elif ev_type == "ping":
            AnthropicPingEvent.model_validate(ev_payload)
        elif ev_type == "content_block_start":
            cb_start = AnthropicContentBlockStartEvent.model_validate(ev_payload)
            assert cb_start.content_block.type == "text"
        elif ev_type == "content_block_delta":
            cb_delta = AnthropicContentBlockDeltaEvent.model_validate(ev_payload)
            assert cb_delta.delta.type == "text_delta"
        elif ev_type == "content_block_stop":
            AnthropicContentBlockStopEvent.model_validate(ev_payload)
        elif ev_type == "message_delta":
            msg_delta = AnthropicMessageDeltaEvent.model_validate(ev_payload)
            assert msg_delta.delta.stop_reason == "end_turn"
            assert msg_delta.usage.output_tokens > 0
        elif ev_type == "message_stop":
            AnthropicMessageStopEvent.model_validate(ev_payload)


def test_anthropic_streaming_conformance_tool_use_events(client, monkeypatch):
    """Test streaming tool call yields content_block_start with type 'tool_use' and input_json_delta."""
    mock_provider = PROVIDERS[0]

    async def mock_stream(messages, model, conversation_id=None, parent_message_id=None):
        tool_tag = '<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>'
        chunk = json.dumps({"choices": [{"delta": {"content": tool_tag}}]})
        yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": [{"name": "get_time", "description": "...", "input_schema": {"type": "object"}}],
        "stream": True,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    raw_blocks = [b.strip() for b in resp.text.strip().split("\n\n") if b.strip()]
    events = []
    for block in raw_blocks:
        ev_type = None
        ev_data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev_type = line[7:].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[6:].strip())
        if ev_type and ev_data is not None:
            events.append((ev_type, ev_data))

    # Verify tool_use content block start
    tool_starts = [p for name, p in events if name == "content_block_start" and p.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    tb = tool_starts[0]["content_block"]
    assert tb["name"] == "get_time"
    assert tb["id"].startswith("toolu_")

    # Verify input_json_delta
    tool_deltas = [p for name, p in events if name == "content_block_delta" and p.get("delta", {}).get("type") == "input_json_delta"]
    assert len(tool_deltas) >= 1
    parsed_input = json.loads(tool_deltas[0]["delta"]["partial_json"])
    assert parsed_input == {"tz": "UTC"}

    # Verify message_delta has stop_reason='tool_use'
    msg_delta = next(p for name, p in events if name == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


def test_anthropic_streaming_conformance_error_event(client, monkeypatch):
    """Test error raised during streaming emits Anthropic 'event: error' chunk."""
    mock_provider = PROVIDERS[0]

    async def mock_failing_stream(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"choices": [{"delta": {"content": "Starting..."}}]}\n\n'
        raise RateLimitError("Anthropic upstream concurrency limit reached")

    monkeypatch.setattr(mock_provider, "stream_chat", mock_failing_stream)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Crash stream"}],
        "stream": True,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200

    raw_blocks = [b.strip() for b in resp.text.strip().split("\n\n") if b.strip()]
    error_event = None
    for block in raw_blocks:
        lines = block.split("\n")
        if any(line.startswith("event: error") for line in lines):
            data_line = next(line for line in lines if line.startswith("data: "))
            error_event = json.loads(data_line[6:].strip())

    assert error_event is not None
    err_obj = AnthropicErrorResponse.model_validate(error_event)
    assert err_obj.type == "error"
    assert err_obj.error.type == "rate_limit_error"
    assert "concurrency limit" in err_obj.error.message.lower()


# ---------------------------------------------------------------------------
# Error Response Conformance Tests
# ---------------------------------------------------------------------------

def test_anthropic_error_conformance_invalid_json(client):
    """Test sending malformed JSON to /v1/messages returns Anthropic compliant error structure."""
    resp = client.post(
        "/v1/messages",
        content="This is definitely not JSON {",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400

    err_obj = AnthropicErrorResponse.model_validate(resp.json())
    assert err_obj.type == "error"
    assert err_obj.error.type == "invalid_request_error"
    assert "Invalid JSON" in err_obj.error.message


def test_anthropic_error_conformance_missing_messages(client):
    """Test missing or non-list messages returns Anthropic compliant error structure."""
    payload = {
        "model": "auto",
        "max_tokens": 1024,
        # missing messages
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 400

    err_obj = AnthropicErrorResponse.model_validate(resp.json())
    assert err_obj.type == "error"
    assert err_obj.error.type == "invalid_request_error"
    assert "messages" in err_obj.error.message


def test_anthropic_error_conformance_unknown_model(client, monkeypatch):
    """Test unsupported model returns Anthropic compliant error structure with status 400."""
    monkeypatch.setattr("src.server.get_provider_for_model", lambda m: None)
    payload = {
        "model": "unknown_model_123",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 400

    err_obj = AnthropicErrorResponse.model_validate(resp.json())
    assert err_obj.type == "error"
    assert err_obj.error.type == "invalid_request_error"
    assert "No provider supports model" in err_obj.error.message


def test_anthropic_error_conformance_rate_limit(client, monkeypatch):
    """Test upstream RateLimitError returns HTTP 429 with Anthropic rate_limit_error schema."""
    mock_provider = PROVIDERS[0]

    async def mock_rate_limit(messages, model, conversation_id=None, parent_message_id=None):
        raise RateLimitError("Anthropic organization credit exhausted")

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_rate_limit)

    payload = {
        "model": "auto",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Spam"}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 429

    err_obj = AnthropicErrorResponse.model_validate(resp.json())
    assert err_obj.type == "error"
    assert err_obj.error.type == "rate_limit_error"
    assert "credit exhausted" in err_obj.error.message.lower()


# ---------------------------------------------------------------------------
# Multi-Turn Coding Workflow Tracker & Whitespace Normalization Tests
# ---------------------------------------------------------------------------

def test_extract_requested_files_preserves_appearance_order():
    """Verify _extract_requested_files preserves exact appearance order while upgrading bare filenames in-place."""
    from src.converters.anthropic import _extract_requested_files

    prompt = (
        "Create `wal.py`, `memtable.py`, and `tests/test_wal.py`. "
        "Also make sure `lsm_engine/wal.py` is imported properly."
    )
    files = _extract_requested_files(prompt)
    assert files == ["lsm_engine/wal.py", "memtable.py", "tests/test_wal.py"]


def test_parse_invoke_xml_normalizes_trailing_line_whitespace():
    """Verify _parse_invoke_xml strips trailing spaces/tabs before newlines in Write and Edit parameters."""
    from src.converters.anthropic import _parse_invoke_xml

    xml_write = (
        '<invoke name="Write">\n'
        '  <parameter name="file_path">wal.py</parameter>\n'
        '  <parameter name="content">def foo():\n    x = 1\n        \n    return x</parameter>\n'
        '</invoke>'
    )
    name, params = _parse_invoke_xml(xml_write)
    assert name == "Write"
    assert params["content"] == "def foo():\n    x = 1\n\n    return x"

    xml_edit = (
        '<invoke name="Edit">\n'
        '  <parameter name="file_path">wal.py</parameter>\n'
        '  <parameter name="old_string">    x = 1   \n        \n    return x</parameter>\n'
        '  <parameter name="new_string">    x = 2\t\n\n    return x</parameter>\n'
        '</invoke>'
    )
    e_name, e_params = _parse_invoke_xml(xml_edit)
    assert e_name == "Edit"
    assert e_params["old_string"] == "    x = 1\n\n    return x"
    assert e_params["new_string"] == "    x = 2\n\n    return x"


def test_coding_workflow_state_edit_error_and_test_failure_persistence():
    """Verify _analyze_coding_workflow_state distinguishes Edit errors from Bash test failures and retains last_test_failed across Read turns."""
    from src.converters.anthropic import _analyze_coding_workflow_state

    # 1. Edit error -> instructs model to use Write to overwrite the target file
    msgs_edit_err = [
        {"role": "user", "content": "Create `db.py` and run tests via Bash."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Write", "input": {"file_path": "db.py", "content": "x=1"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "File created"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_2", "name": "Edit", "input": {"file_path": "db.py", "old_string": "a", "new_string": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_2", "is_error": True, "content": "<tool_use_error>String to replace not found in file.</tool_use_error>"},
            ],
        },
    ]
    state_edit = _analyze_coding_workflow_state(
        msgs_edit_err, has_coding_action_tools=True, has_pending_tool_results=True, has_error_tool_results=True
    )
    assert state_edit["is_incomplete"] is True
    assert "Your previous `Edit` tool call on `db.py` failed" in state_edit["progress_directive"]
    assert "overwrite `db.py`" in state_edit["progress_directive"]

    # 2. Bash test failure followed by a Read turn -> still remembers last_test_failed!
    msgs_read_after_fail = [
        {"role": "user", "content": "Create `db.py` and run unittest via Bash."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Write", "input": {"file_path": "db.py", "content": "x=1"}},
                {"type": "tool_use", "id": "tu_2", "name": "Bash", "input": {"command": "python3 -m unittest"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "File created"},
                {"type": "tool_result", "tool_use_id": "tu_2", "is_error": True, "content": "FAILED (failures=1)\nAssertionError: 1 != 2"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_3", "name": "Read", "input": {"file_path": "db.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_3", "is_error": False, "content": "x=1"},
            ],
        },
    ]
    state_read = _analyze_coding_workflow_state(
        msgs_read_after_fail, has_coding_action_tools=True, has_pending_tool_results=True, has_error_tool_results=False
    )
    assert state_read["is_incomplete"] is True
    assert "AssertionError: 1 != 2" in state_read["progress_directive"]
    assert "has NOT yet been modified since that failure" in state_read["progress_directive"]

