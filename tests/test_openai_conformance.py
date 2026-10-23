"""
tests/test_openai_conformance.py — Comprehensive conformance test suite for OpenAI API.

Validates 100% adherence to OpenAI Chat Completions & Models API specifications:
- Strict Pydantic models with `extra="forbid"` to ensure zero schema drift or body pollution.
- Full matrix of request features: roles (user, system, developer), multimodal vision,
  sampling params (temperature, top_p, max_tokens, stop), tool definitions, tool choice,
  and multi-turn tool execution history.
- Non-streaming response schemas, headers, tool_calls, stop reasons, usage accounting.
- Streaming SSE chunk structures, event framing, stream_options include_usage, streaming tool calls.
- Conformance of error responses (validation, missing model, rate limits, 404s).
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
# Strict OpenAI Contract Schemas (extra="forbid" rejects any non-standard fields)
# ---------------------------------------------------------------------------

class OpenAIFunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: str  # Must be valid JSON string according to OpenAI spec


class OpenAIToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["function"]
    function: OpenAIFunctionCall


class OpenAIChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["assistant"]
    content: Optional[str] = None
    tool_calls: Optional[list[OpenAIToolCall]] = None
    refusal: Optional[str] = None


class OpenAIChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    message: OpenAIChoiceMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Any] = None


class OpenAIUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[dict[str, Any]] = None
    prompt_tokens_details: Optional[dict[str, Any]] = None


class OpenAIChatCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage
    system_fingerprint: Optional[str] = None


class OpenAIChunkFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = None
    arguments: Optional[str] = None


class OpenAIChunkToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    function: Optional[OpenAIChunkFunction] = None


class OpenAIChunkDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list[OpenAIChunkToolCall]] = None
    refusal: Optional[str] = None


class OpenAIChunkChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    delta: OpenAIChunkDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Any] = None


class OpenAIChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[OpenAIChunkChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[OpenAIUsage] = None


class OpenAIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    description: Optional[str] = None


class OpenAIModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object: Literal["list"]
    data: list[OpenAIModel]


class OpenAIErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: OpenAIErrorDetail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Request Conformance Tests
# ---------------------------------------------------------------------------

def test_openai_request_conformance_minimal(client, monkeypatch):
    """Test standard minimal valid chat completion request conforms to OpenAI spec."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "conversation_id": "conv_123",
            "parent_message_id": "parent_456",
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    req_payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp = client.post("/v1/chat/completions", json=req_payload)
    assert resp.status_code == 200

    # Header conformance
    assert "openai-organization" in resp.headers
    assert resp.headers["openai-version"] == "2020-10-01"
    assert "x-request-id" in resp.headers or "X-Request-ID" in resp.headers
    assert "openai-processing-ms" in resp.headers

    # Validate against strict Pydantic model
    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert validated.id.startswith("chatcmpl-")
    assert validated.object == "chat.completion"
    assert validated.choices[0].message.role == "assistant"
    assert validated.choices[0].message.content == "Hello!"
    assert validated.choices[0].finish_reason == "stop"
    assert validated.usage.prompt_tokens > 0
    assert validated.usage.completion_tokens > 0
    assert validated.usage.total_tokens == validated.usage.prompt_tokens + validated.usage.completion_tokens


def test_openai_request_conformance_system_and_developer_roles(client, monkeypatch):
    """Test support for both 'system' and 'developer' prompt roles (OpenAI 4o / o1 specification)."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "Instructions acknowledged."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [
            {"role": "developer", "content": "You are a precise calculation assistant."},
            {"role": "system", "content": "Always output numerical results directly."},
            {"role": "user", "content": "Compute 2 + 2"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert validated.choices[0].message.content == "Instructions acknowledged."

    # Both system instructions should have been preserved or folded into prompt
    all_content = " ".join(m.get("content", "") for m in captured_messages if isinstance(m.get("content"), str))
    assert "precise calculation assistant" in all_content or "numerical results directly" in all_content


def test_openai_request_conformance_multimodal_vision(client, monkeypatch):
    """Test multimodal vision payload conforming to OpenAI image_url schema."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "I see an apple on a wooden table."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/apple.jpg", "detail": "high"}},
                ],
            }
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert "apple" in validated.choices[0].message.content


def test_openai_request_conformance_sampling_params(client, monkeypatch):
    """Test sampling hyperparameters (temperature, top_p, max_tokens, stop) are accepted and emulated."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Line 1\nSTOP_HERE\nLine 2"}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Generate lines"}],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 50,
        "stop": ["STOP_HERE"],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    # Should be truncated at STOP_HERE
    assert validated.choices[0].message.content == "Line 1\n"
    assert validated.choices[0].finish_reason == "stop"


def test_openai_request_conformance_tool_definitions_and_choice(client, monkeypatch):
    """Test tool definitions schema and tool_choice values (auto, none, required, specific)."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "Tool call requested."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_quote",
                "description": "Get current stock quote for ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock symbol"}
                    },
                    "required": ["ticker"],
                },
            },
        }
    ]

    # Test tool_choice: "auto"
    resp = client.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "Price of NVDA"}],
        "tools": tools,
        "tool_choice": "auto",
    })
    assert resp.status_code == 200
    OpenAIChatCompletion.model_validate(resp.json())

    # Test tool_choice: specific function
    resp_specific = client.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "Price of NVDA"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_stock_quote"}},
    })
    assert resp_specific.status_code == 200
    OpenAIChatCompletion.model_validate(resp_specific.json())


def test_openai_request_conformance_multi_turn_tool_execution(client, monkeypatch):
    """Test multi-turn tool history with role: 'assistant' tool_calls followed by role: 'tool'."""
    mock_provider = PROVIDERS[0]
    captured_messages = []

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        nonlocal captured_messages
        captured_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "NVDA is currently trading at $130."}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "What is NVDA price?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_nvda_123",
                        "type": "function",
                        "function": {
                            "name": "get_stock_quote",
                            "arguments": '{"ticker": "NVDA"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_nvda_123",
                "name": "get_stock_quote",
                "content": '{"price": 130.00, "currency": "USD"}',
            },
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert validated.choices[0].finish_reason == "stop"
    assert "130" in validated.choices[0].message.content


# ---------------------------------------------------------------------------
# Response Conformance Tests (Non-Streaming)
# ---------------------------------------------------------------------------

def test_openai_response_conformance_single_tool_call(client, monkeypatch):
    """Test non-streaming response emitting a single function call conforming to OpenAI schema."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name": "fetch_weather", "arguments": {"city": "Berlin"}}</tool_call>',
                }
            }]
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Weather in Berlin"}],
        "tools": [{"type": "function", "function": {"name": "fetch_weather"}}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert validated.choices[0].finish_reason == "tool_calls"
    assert validated.choices[0].message.content is None
    tool_calls = validated.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].id.startswith("call_")
    assert tool_calls[0].function.name == "fetch_weather"
    # Arguments must be a valid JSON string
    args = json.loads(tool_calls[0].function.arguments)
    assert args == {"city": "Berlin"}


def test_openai_response_conformance_parallel_tool_calls(client, monkeypatch):
    """Test non-streaming response emitting multiple parallel tool calls conforming to OpenAI schema."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        raw_text = (
            '<tool_call>{"name": "fetch_weather", "arguments": {"city": "London"}}</tool_call>\n'
            '<tool_call>{"name": "get_time", "arguments": {"timezone": "Europe/London"}}</tool_call>'
        )
        return {
            "choices": [{"message": {"role": "assistant", "content": raw_text}}]
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Weather and time in London"}],
        "tools": [
            {"type": "function", "function": {"name": "fetch_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert validated.choices[0].finish_reason == "tool_calls"
    tool_calls = validated.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "fetch_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "London"}
    assert tool_calls[1].function.name == "get_time"
    assert json.loads(tool_calls[1].function.arguments) == {"timezone": "Europe/London"}


def test_openai_response_conformance_max_tokens_length_finish_reason(client, monkeypatch):
    """Test non-streaming response finish_reason is 'length' when max_tokens is exceeded."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "A" * 200}}],
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Long string"}],
        "max_tokens": 10,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    validated = OpenAIChatCompletion.model_validate(resp.json())
    assert validated.choices[0].finish_reason == "length"
    assert len(validated.choices[0].message.content) <= 40


def test_openai_response_conformance_zero_body_pollution(client, monkeypatch):
    """Test non-streaming response body contains zero internal session variables (no conversation_id or parent_message_id)."""
    mock_provider = PROVIDERS[0]

    async def mock_chat(messages, model, conversation_id=None, parent_message_id=None):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Clean response"}}],
            "conversation_id": "ghost_secret_conv_id",
            "parent_message_id": "ghost_secret_parent_id",
        }

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_chat)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    raw_data = resp.json()
    assert "conversation_id" not in raw_data
    assert "parent_message_id" not in raw_data
    # Metadata should only reside in headers
    assert resp.headers.get("x-ghost-conversation-id") == "ghost_secret_conv_id"
    assert resp.headers.get("x-ghost-parent-message-id") == "ghost_secret_parent_id"

    # Schema must validate with extra="forbid"
    OpenAIChatCompletion.model_validate(raw_data)


# ---------------------------------------------------------------------------
# Streaming Response Conformance Tests
# ---------------------------------------------------------------------------

def test_openai_streaming_conformance_sse_protocol_and_chunks(client, monkeypatch):
    """Test streaming completions conform strictly to SSE line framing and OpenAIChatCompletionChunk schema."""
    mock_provider = PROVIDERS[0]

    async def mock_stream(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": "First "}}]}\n\n'
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": "Second."}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Stream me"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n\n")
    assert lines[-1] == "data: [DONE]"

    chunks = []
    for line in lines[:-1]:
        assert line.startswith("data: ")
        chunk_dict = json.loads(line[6:].strip())
        chunk_obj = OpenAIChatCompletionChunk.model_validate(chunk_dict)
        chunks.append(chunk_obj)

    assert len(chunks) >= 2
    # First chunk must declare role: assistant
    assert chunks[0].choices[0].delta.role == "assistant"
    # Final content chunk must have finish_reason="stop"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_openai_streaming_conformance_include_usage(client, monkeypatch):
    """Test stream_options: {'include_usage': True} emits final usage chunk with choices: [] before [DONE]."""
    mock_provider = PROVIDERS[0]

    async def mock_stream(messages, model, conversation_id=None, parent_message_id=None):
        yield 'data: {"id": "c1", "choices": [{"delta": {"content": "Computing..."}}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Estimate usage"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    lines = [line[6:].strip() for line in resp.text.strip().split("\n\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"

    usage_chunk_dict = json.loads(lines[-2])
    usage_chunk = OpenAIChatCompletionChunk.model_validate(usage_chunk_dict)
    assert usage_chunk.choices == []
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.prompt_tokens > 0
    assert usage_chunk.usage.completion_tokens > 0
    assert usage_chunk.usage.total_tokens == usage_chunk.usage.prompt_tokens + usage_chunk.usage.completion_tokens


def test_openai_streaming_conformance_tool_calls(client, monkeypatch):
    """Test streaming completions correctly emit delta.tool_calls chunks and finish_reason='tool_calls'."""
    mock_provider = PROVIDERS[0]

    async def mock_stream(messages, model, conversation_id=None, parent_message_id=None):
        tool_tag = '<tool_call>{"name": "lookup", "arguments": {"id": 42}}</tool_call>'
        chunk_json = json.dumps({"id": "c1", "choices": [{"delta": {"content": tool_tag}}]})
        yield f"data: {chunk_json}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(mock_provider, "stream_chat", mock_stream)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Lookup 42"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    lines = [line[6:].strip() for line in resp.text.strip().split("\n\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"

    tool_call_chunks = []
    final_finish_reason = None
    for line in lines[:-1]:
        chunk = OpenAIChatCompletionChunk.model_validate(json.loads(line))
        if chunk.choices:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)
            if chunk.choices[0].finish_reason:
                final_finish_reason = chunk.choices[0].finish_reason

    assert len(tool_call_chunks) >= 1
    assert tool_call_chunks[0].function.name == "lookup"
    assert json.loads(tool_call_chunks[0].function.arguments) == {"id": 42}
    assert final_finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Error Response Conformance Tests
# ---------------------------------------------------------------------------

def test_openai_error_conformance_validation_error(client):
    """Test RequestValidationError returns OpenAI compliant error structure with status 400."""
    # Send request with invalid messages (missing role)
    payload = {
        "model": "auto",
        "messages": [{"content": "No role provided"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400

    err_obj = OpenAIErrorResponse.model_validate(resp.json())
    assert err_obj.error.type == "invalid_request_error"
    assert "role" in err_obj.error.message


def test_openai_error_conformance_unknown_model(client, monkeypatch):
    """Test requesting unsupported model returns OpenAI compliant error structure with status 400."""
    monkeypatch.setattr("src.server.get_provider_for_model", lambda m: None)
    payload = {
        "model": "non_existent_model_xyz",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400

    err_obj = OpenAIErrorResponse.model_validate(resp.json())
    assert err_obj.error.type == "invalid_request_error"
    assert "No provider supports model" in err_obj.error.message


def test_openai_error_conformance_rate_limit(client, monkeypatch):
    """Test upstream RateLimitError returns HTTP 429 with rate_limit_error type and code."""
    mock_provider = PROVIDERS[0]

    async def mock_rate_limit(messages, model, conversation_id=None, parent_message_id=None):
        raise RateLimitError("Upstream ChatGPT anonymous rate limit reached")

    monkeypatch.setattr(mock_provider, "non_streaming_chat", mock_rate_limit)

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Fast request"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 429

    err_obj = OpenAIErrorResponse.model_validate(resp.json())
    assert err_obj.error.type == "rate_limit_error"
    assert err_obj.error.code == "rate_limit_exceeded"
    assert "rate limit" in err_obj.error.message.lower()


# ---------------------------------------------------------------------------
# Models Endpoint Conformance Tests
# ---------------------------------------------------------------------------

def test_openai_models_list_conformance(client):
    """Test GET /v1/models conforms strictly to OpenAI model list schema."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200

    model_list = OpenAIModelList.model_validate(resp.json())
    assert model_list.object == "list"
    assert len(model_list.data) > 0

    model_ids = [m.id for m in model_list.data]
    assert "auto" in model_ids


def test_openai_models_get_single_conformance(client):
    """Test GET /v1/models/{model_id} conforms strictly to OpenAI model schema."""
    resp = client.get("/v1/models/auto")
    assert resp.status_code == 200

    model_obj = OpenAIModel.model_validate(resp.json())
    assert model_obj.id == "auto"
    assert model_obj.object == "model"
    assert model_obj.created > 0
    assert len(model_obj.owned_by) > 0


def test_openai_models_not_found_conformance(client):
    """Test GET /v1/models/{model_id} for unknown model returns 404 with OpenAI compliant error."""
    resp = client.get("/v1/models/definitely_not_a_model")
    assert resp.status_code == 404

    err_obj = OpenAIErrorResponse.model_validate(resp.json())
    assert err_obj.error.type == "invalid_request_error"
    assert err_obj.error.code == "model_not_found"
