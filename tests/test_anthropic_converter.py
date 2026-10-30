import pytest
import json
import asyncio
from src.converters.anthropic import (
    anthropic_to_openai_request,
    openai_to_anthropic_stream,
    openai_to_anthropic_json,
    _parse_invoke_xml,
    _build_tools_xml,
)

VERBOSE_ANTHROPIC_REQUEST = {
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 4096,
  "system": [
    {
      "type": "text",
      "text": "You are an expert AI systems engineer."
    },
    {
      "type": "text",
      "text": "Be concise unless deep detail is requested."
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this architecture diagram."
        },
        {
          "type": "image",
          "source": {
            "type": "url",
            "url": "https://example.com/diagram.png"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "I can analyze the system topology."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Also check current weather in Davao and estimate latency implications."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01ABCXYZ",
          "name": "get_weather",
          "input": {
            "location": "Davao",
            "unit": "celsius"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01ABCXYZ",
          "content": [
            {
              "type": "text",
              "text": "29°C, moderate rain"
            }
          ]
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    },
    {
      "name": "search_docs",
      "description": "Search engineering documentation",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string"
          }
        },
        "required": ["query"]
      }
    }
  ],
  "tool_choice": {
    "type": "auto"
  },
  "thinking": {
    "type": "enabled",
    "budget_tokens": 2048
  },
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "stop_sequences": [
    "</final_answer>",
    "[END]"
  ],
  "stream": True,
  "metadata": {
    "user_id": "user_12345",
    "session_id": "sess_98765",
    "trace_id": "trace_abcxyz"
  },
  "service_tier": "priority",
  "container": "container_01XYZ",
  "mcp_servers": [
    {
      "type": "url",
      "url": "https://mcp.example.com/server",
      "name": "internal-docs"
    }
  ]
}

def test_anthropic_to_openai_verbose_request():
    openai_req = anthropic_to_openai_request(VERBOSE_ANTHROPIC_REQUEST)
    
    assert openai_req["model"] == "claude-sonnet-4-20250514"
    assert openai_req["max_tokens"] == 4096
    assert openai_req["temperature"] == 0.7
    assert openai_req["top_p"] == 0.95
    assert openai_req["stop"] == ["</final_answer>", "[END]"]
    assert openai_req["stream"] is True
    
    # System prompt verification
    sys_msg = openai_req["messages"][0]
    assert sys_msg["role"] == "system"
    assert "You are an expert AI systems engineer." in sys_msg["content"]
    assert "Be concise unless deep detail is requested." in sys_msg["content"]
    assert '<tool name="get_weather">' in sys_msg["content"]
    assert '<tool name="search_docs">' in sys_msg["content"]
    
    # Messages verification
    msgs = openai_req["messages"][1:]
    assert len(msgs) == 5
    
    # User message 1 with text and image url
    u1 = msgs[0]
    assert u1["role"] == "user"
    assert isinstance(u1["content"], list)
    assert u1["content"][0] == {"type": "text", "text": "Analyze this architecture diagram."}
    assert u1["content"][1] == {"type": "image_url", "image_url": {"url": "https://example.com/diagram.png"}}
    
    # Assistant message 1
    a1 = msgs[1]
    assert a1["role"] == "assistant"
    assert a1["content"] == "I can analyze the system topology."
    
    # User message 2
    u2 = msgs[2]
    assert u2["role"] == "user"
    assert u2["content"] == "Also check current weather in Davao and estimate latency implications."
    
    # Assistant message 2 (tool_use block converted to <invoke>)
    a2 = msgs[3]
    assert a2["role"] == "assistant"
    assert '<invoke name="get_weather">' in a2["content"]
    assert '<parameter name="location">Davao</parameter>' in a2["content"]
    assert '<parameter name="unit">celsius</parameter>' in a2["content"]
    
    # User message 3 (tool_result block converted to <tool_result>)
    u3 = msgs[4]
    assert u3["role"] == "user"
    assert '<tool_result id="toolu_01ABCXYZ">' in u3["content"]
    assert '29°C, moderate rain' in u3["content"]

def test_base64_image_and_tool_choice():
    req = {
        "model": "auto",
        "system": "System prompt",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at image"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "ABC123XYZ=="
                        }
                    }
                ]
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather tool"
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "get_weather"
        }
    }
    
    openai_req = anthropic_to_openai_request(req)
    sys_content = openai_req["messages"][0]["content"]
    assert 'MUST call the tool "get_weather"' in sys_content
    
    user_msg = openai_req["messages"][1]
    assert user_msg["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,ABC123XYZ=="

def test_tool_result_is_error():
    req = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_err",
                        "is_error": True,
                        "content": "Network timeout error"
                    }
                ]
            }
        ]
    }
    
    openai_req = anthropic_to_openai_request(req)
    msg_content = openai_req["messages"][0]["content"]
    assert '<tool_result id="toolu_err" is_error="true">' in msg_content
    assert "Network timeout error" in msg_content

def test_parse_invoke_xml():
    xml = """
    <invoke name="get_weather">
      <parameter name="location">Davao</parameter>
      <parameter name="unit">"celsius"</parameter>
      <parameter name="count">5</parameter>
      <parameter name="flag">true</parameter>
    </invoke>
    """
    name, params = _parse_invoke_xml(xml)
    assert name == "get_weather"
    assert params["location"] == "Davao"
    assert params["unit"] == "celsius"
    assert params["count"] == 5
    assert params["flag"] is True

def test_openai_to_anthropic_json_tool_call():
    openai_resp = {
        "model": "gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": 'Checking weather...\n<invoke name="get_weather">\n  <parameter name="location">Davao</parameter>\n</invoke>'
                }
            }
        ]
    }
    
    res = asyncio.run(openai_to_anthropic_json(openai_resp))
    assert res["type"] == "message"
    assert res["role"] == "assistant"
    assert res["stop_reason"] == "tool_use"
    assert len(res["content"]) == 2
    assert res["content"][0] == {"type": "text", "text": "Checking weather..."}
    assert res["content"][1]["type"] == "tool_use"
    assert res["content"][1]["name"] == "get_weather"
    assert res["content"][1]["input"] == {"location": "Davao"}
    assert res["content"][1]["id"].startswith("toolu_")

async def _fake_openai_stream(chunks: list[str]):
    for c in chunks:
        d = {"choices": [{"delta": {"content": c}}]}
        yield f"data: {json.dumps(d)}\n\n"
    yield "data: [DONE]\n\n"

def test_openai_to_anthropic_stream_tool_call():
    chunks = [
        "Let me check ",
        "the weather for you.\n",
        '<invoke name="get_weather">\n',
        '  <parameter name="location">Davao</parameter>\n',
        '</invoke>'
    ]
    
    events = []
    async def run_stream():
        async for evt in openai_to_anthropic_stream(_fake_openai_stream(chunks), model="claude-3"):
            events.append(evt)
            
    asyncio.run(run_stream())
    
    full_output = "".join(events)
    assert "event: message_start" in full_output
    assert "event: content_block_start" in full_output
    assert "event: content_block_delta" in full_output
    assert "get_weather" in full_output
    assert "input_json_delta" in full_output
    assert "tool_use" in full_output
    assert "event: message_stop" in full_output


def test_openai_to_anthropic_stream_multi_tool_calls():
    """Verify multiple tool calls in stream are yielded as distinct sequential content blocks."""
    chunks = [
        '<tool_call>{"name": "get_current_time", "arguments": {"location": "Tokyo"}}</tool_call>\n',
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>\n',
        '<tool_call>{"name": "calculator", "arguments": {"expression": "450 * 1.15"}}</tool_call>'
    ]

    events = []
    async def run_stream():
        async for evt in openai_to_anthropic_stream(_fake_openai_stream(chunks), model="gemini"):
            events.append(evt)

    asyncio.run(run_stream())
    full_output = "".join(events)
    assert "get_current_time" in full_output
    assert "get_weather" in full_output
    assert "calculator" in full_output
    assert full_output.count('"type": "tool_use"') == 3
    assert "stop_reason\": \"tool_use\"" in full_output


def test_anthropic_headless_sandbox_prompt():
    """Verify headless sandbox and multi-tool permissions are injected to suppress provider web cards."""
    req = {
        "model": "gemini",
        "messages": [{"role": "user", "content": "Tell me Tokyo time, Paris weather, and 450*1.15"}],
        "tools": [
            {"name": "get_current_time", "description": "Current time"},
            {"name": "get_weather", "description": "Weather info"},
            {"name": "calculator", "description": "Do math"}
        ]
    }
    openai_req = anthropic_to_openai_request(req)
    sys_content = openai_req["messages"][0]["content"]
    assert "HEADLESS API GATEWAY" in sys_content
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" in sys_content
    assert "sandbox environment with NO web access" in sys_content
    assert "DO NOT attempt to search the web" in sys_content
    assert "invoke multiple tools in succession" in sys_content
    assert "You cannot call multiple tools at once" not in sys_content


def test_temperature_steering():
    # Low temperature steering
    req_low = {
        "model": "auto",
        "system": "Base system.",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
    }
    low_res = anthropic_to_openai_request(req_low)
    low_sys = low_res["messages"][0]["content"]
    assert "strictly factual, precise, concise" in low_sys

    # High temperature steering
    req_high = {
        "model": "auto",
        "system": "Base system.",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.9,
    }
    high_res = anthropic_to_openai_request(req_high)
    high_sys = high_res["messages"][0]["content"]
    assert "creative, explorative" in high_sys


def test_synthesis_phase_detection():
    req = {
        "model": "auto",
        "system": "You are a helpful assistant.",
        "tools": [{"name": "calculator", "description": "Calc tool"}],
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "calculator", "input": {"x": 2}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "4"}]},
        ],
    }
    res = anthropic_to_openai_request(req)
    assert res["_is_synthesis_phase"] is True
    sys_content = res["messages"][0]["content"]
    assert "[SYNTHESIS PHASE]" in sys_content
    assert "External tool execution is COMPLETE" in sys_content


def test_anthropic_second_turn_user_prompt_resets_synthesis_phase():
    """Verify that when a user asks a new question after a completed tool turn in Anthropic mode, synthesis phase is reset."""
    req = {
        "model": "auto",
        "system": "You are a helpful assistant.",
        "tools": [{"name": "calculator", "description": "Calc tool"}],
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "calculator", "input": {"x": 2}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "4"}]},
            {"role": "assistant", "content": "2+2 is 4."},
            {"role": "user", "content": "Now compute 10*10."},
        ],
    }
    res = anthropic_to_openai_request(req)
    assert res["_is_synthesis_phase"] is False
    sys_content = res["messages"][0]["content"]
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" in sys_content
    assert "[SYNTHESIS PHASE]" not in sys_content


def test_stream_stop_sequences():
    chunks = [
        "The quick brown ",
        "fox jumps over ",
        "the lazy dog",
    ]
    events = []

    async def run():
        async for evt in openai_to_anthropic_stream(
            _fake_openai_stream(chunks),
            model="claude-3-5-sonnet",
            stop_sequences=["jumps"],
        ):
            events.append(evt)

    asyncio.run(run())
    full_output = "".join(events)
    assert "lazy dog" not in full_output
    assert '"stop_reason": "stop_sequence"' in full_output
    assert '"stop_sequence": "jumps"' in full_output


def test_stream_max_tokens():
    chunks = [
        "First sentence here. ",
        "Second sentence that exceeds the token limit.",
    ]
    events = []

    async def run():
        async for evt in openai_to_anthropic_stream(
            _fake_openai_stream(chunks),
            model="claude-3-5-sonnet",
            max_tokens=4,
        ):
            events.append(evt)

    asyncio.run(run())
    full_output = "".join(events)
    assert '"stop_reason": "max_tokens"' in full_output


def test_stream_token_usage_reporting():
    chunks = ["Hello world, this is a test stream."]
    events = []

    async def run():
        async for evt in openai_to_anthropic_stream(
            _fake_openai_stream(chunks),
            model="claude-3-5-sonnet",
            input_tokens=55,
        ):
            events.append(evt)

    asyncio.run(run())
    full_output = "".join(events)
    assert '"input_tokens": 55' in full_output
    assert '"output_tokens":' in full_output


def test_tool_call_tag_support_streaming():
    chunks = [
        "Calling tool now:\n",
        '<tool_call>{"name": "fetch_db", "arguments": {"table": "users"}}</tool_call>',
    ]
    events = []

    async def run():
        async for evt in openai_to_anthropic_stream(
            _fake_openai_stream(chunks),
            model="claude-3-5-sonnet",
        ):
            events.append(evt)

    asyncio.run(run())
    full_output = "".join(events)
    assert "fetch_db" in full_output
    assert "table" in full_output
    assert '"stop_reason": "tool_use"' in full_output


def test_invoke_with_raw_json_streaming():
    chunks = [
        '<invoke name="calc">{"query": "sqrt(16)"}</invoke>'
    ]
    events = []

    async def run():
        async for evt in openai_to_anthropic_stream(
            _fake_openai_stream(chunks),
            model="claude-3-5-sonnet",
        ):
            events.append(evt)

    asyncio.run(run())
    full_output = "".join(events)
    assert "calc" in full_output
    assert "sqrt(16)" in full_output
    assert '"stop_reason": "tool_use"' in full_output


def test_non_streaming_stop_and_max_tokens():
    resp = {
        "model": "gpt-5-5",
        "choices": [
            {"message": {"role": "assistant", "content": "Line 1\nSTOP_HERE\nLine 2"}}
        ]
    }
    res = asyncio.run(openai_to_anthropic_json(resp, stop_sequences=["STOP_HERE"], input_tokens=20))
    assert res["stop_reason"] == "stop_sequence"
    assert res["stop_sequence"] == "STOP_HERE"
    assert "Line 2" not in res["content"][0]["text"]
    assert res["usage"]["input_tokens"] == 20
    assert res["usage"]["output_tokens"] > 0

    # Max tokens test
    res_max = asyncio.run(openai_to_anthropic_json(resp, max_tokens=2, input_tokens=10))
    assert res_max["stop_reason"] == "max_tokens"


def test_server_post_v1_messages_endpoint(monkeypatch):
    from starlette.testclient import TestClient
    from src.server import app
    from src.providers.chatgpt.session import ChatGPTProvider

    async def mock_stream_chat(*args, **kwargs):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello from Claude emulation!"}}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    async def mock_non_streaming_chat(*args, **kwargs):
        return {
            "model": "gpt-5-5",
            "choices": [{"message": {"role": "assistant", "content": "Non-streaming answer"}}]
        }

    from src.providers import get_provider_for_model
    from unittest.mock import AsyncMock
    provider = get_provider_for_model("auto")
    monkeypatch.setattr(provider, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(provider, "non_streaming_chat", mock_non_streaming_chat)
    monkeypatch.setattr("src.server.get_all_models", AsyncMock(return_value=[{"id": "auto"}]))

    client = TestClient(app)

    # 1. Non-streaming request with native model
    res = client.post("/v1/messages", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "temperature": 0.2,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["text"] == "Non-streaming answer"
    assert data["usage"]["input_tokens"] > 0

    # 2. Streaming request with native model
    res_stream = client.post("/v1/messages", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    })
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers["content-type"]
    text = res_stream.text
    assert "event: message_start" in text
    assert "event: content_block_delta" in text
    assert "Hello from Claude emulation!" in text
    assert "event: message_stop" in text

    # 3. Verify /v1/models returns provider models
    models_res = client.get("/v1/models")
    assert models_res.status_code == 200
    model_ids = [m["id"] for m in models_res.json().get("data", [])]
    assert "auto" in model_ids

    # 4. Verify GUI Dashboard serves protocol switcher controls
    dash_res = client.get("/dashboard/")
    assert dash_res.status_code == 200
    html = dash_res.text
    assert "protocol-segmented-control" in html
    assert "proto-openai-btn" in html
    assert "proto-anthropic-btn" in html
    assert "/v1/messages" in html

    app_js_res = client.get("/dashboard/app.js")
    assert app_js_res.status_code == 200
    js = app_js_res.text
    assert "setupProtocolSwitcher" in js
    assert "formatMessagesForAnthropic" in js
    assert "getActiveProtocol" in js

    style_css_res = client.get("/dashboard/style.css")
    assert style_css_res.status_code == 200
    css = style_css_res.text
    assert "protocol-segmented-control" in css
    assert "protocol-segment" in css


# ============================================================================
# Claude Code & Coding Harness Regression Tests
# ============================================================================

def test_claude_code_massive_tool_schema_compaction():
    """Verify 24KB+ tool descriptions (e.g. Claude Code Artifact tool) are compacted without losing schema fidelity."""
    huge_desc = "Artifact creation guide. " + ("Detailed instructions on claude.ai artifacts. " * 600)
    tools = [
        {
            "name": "Artifact",
            "description": huge_desc,
            "input_schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique identifier. " * 30},
                    "command": {"type": "string", "enum": ["create", "update", "rewrite"]},
                },
                "required": ["id", "command"],
            },
        },
        {
            "name": "Write",
            "description": "Writes a file to the local filesystem.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute path to the file to write"},
                    "content": {"type": "string", "description": "The content to write to the file"},
                },
                "required": ["file_path", "content"],
            },
        },
    ]
    xml = _build_tools_xml(tools)
    assert len(xml) < 2000, f"Expected compacted XML under 2000 chars, got {len(xml)}"
    assert '<tool name="Artifact">' in xml
    assert '<tool name="Write">' in xml
    assert 'enum="create|update|rewrite"' in xml
    assert 'name="file_path" type="string" required="required"' in xml


def test_claude_code_mid_conversation_system_message_consolidation():
    """Verify mid-conversation role=='system' messages (Claude Code # Environment block) are consolidated
    before the Ghost Kernel Override so they never bury TOOL_CALL_REQUIRED."""
    req = {
        "model": "gemini",
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.114"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "create island.html"}]},
            {"role": "system", "content": "# System Prompt\nYou are an interactive CLI tool.\n# Environment\nWorking directory: /tmp"},
        ],
        "tools": [
            {
                "name": "Write",
                "description": "Writes a file to the local filesystem.",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["file_path", "content"],
                },
            }
        ],
    }
    openai_req = anthropic_to_openai_request(req)
    # Only one system message at index 0; no mid-conversation system message after user turn
    sys_msgs = [m for m in openai_req["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert openai_req["messages"][0]["role"] == "system"
    assert openai_req["messages"][1]["role"] == "user"

    sys_content = openai_req["messages"][0]["content"]
    # Client environment block comes BEFORE the Ghost Directive & TOOL_CALL_REQUIRED block
    env_pos = sys_content.find("# Environment")
    override_pos = sys_content.find("GHOST WORKSPACE DIRECTIVE")
    if override_pos == -1:
        override_pos = sys_content.find("GHOST")
    tool_req_pos = sys_content.find("CRITICAL MODE: TOOL_CALL_REQUIRED")
    assert env_pos != -1 and override_pos != -1 and tool_req_pos != -1
    assert env_pos < override_pos < tool_req_pos


def test_claude_code_suggestion_mode_and_away_recap_suppress_tool_calls():
    """Verify Claude Code background [SUGGESTION MODE] and away-recap requests do not force TOOL_CALL_REQUIRED."""
    tools = [{"name": "Write", "description": "Write file"}]

    req_suggestion = {
        "model": "gemini",
        "messages": [
            {"role": "user", "content": "create island.html"},
            {"role": "assistant", "content": "Created island.html."},
            {
                "role": "user",
                "content": '[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code. Reply with ONLY the suggestion, no quotes or explanation.]',
            },
        ],
        "tools": tools,
    }
    out_sug = anthropic_to_openai_request(req_suggestion)
    sys_sug = out_sug["messages"][0]["content"]
    assert "[AUXILIARY TEXT MODE]" in sys_sug
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" not in sys_sug

    req_recap = {
        "model": "gemini",
        "messages": [
            {"role": "user", "content": "create island.html"},
            {"role": "assistant", "content": "Created island.html."},
            {
                "role": "user",
                "content": "The user stepped away and is coming back. Recap in under 40 words, 1-2 plain sentences, no markdown, what was last worked on.",
            },
        ],
        "tools": tools,
    }
    out_recap = anthropic_to_openai_request(req_recap)
    sys_recap = out_recap["messages"][0]["content"]
    assert "[AUXILIARY TEXT MODE]" in sys_recap
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" not in sys_recap


def test_claude_code_discovery_tools_keep_tool_call_required_active():
    """Verify that after calling a discovery/read tool (ToolSearch, Skill, Read) when Write/Edit/Bash are available,
    the next turn stays in TOOL_CALL_REQUIRED mode so the agent chains into Write/Edit/Bash."""
    tools = [
        {"name": "ToolSearch", "description": "Search deferred tools"},
        {"name": "Skill", "description": "Load skill"},
        {"name": "Read", "description": "Read file"},
        {"name": "Write", "description": "Write file"},
        {"name": "Bash", "description": "Run bash"},
    ]
    req = {
        "model": "gemini",
        "tools": tools,
        "messages": [
            {"role": "user", "content": "Create a 3D island simulation and save it in island.html"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "artifact-design"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Skill guide loaded."}],
            },
        ],
    }
    res = anthropic_to_openai_request(req)
    assert res["_is_synthesis_phase"] is False
    sys_content = res["messages"][0]["content"]
    assert "CRITICAL MODE: TOOL_CALL_REQUIRED" in sys_content
    assert "[SYNTHESIS PHASE]" not in sys_content


def test_parse_invoke_xml_stray_bracket_and_autolink_unwrapping():
    """Verify _parse_invoke_xml handles stray trailing ']' (Trace #1012) and unwraps Gemini [url](url) auto-links (Trace #1075)."""
    # 1. Stray trailing ']' from Trace #1012
    raw_trace_1012 = '<tool_call>{"name": "ToolSearch", "arguments": {"max_results": 5, "query": "select:Artifact"}}]</tool_call>'
    name, args = _parse_invoke_xml(raw_trace_1012)
    assert name == "ToolSearch"
    assert args == {"max_results": 5, "query": "select:Artifact"}

    # 2. Gemini markdown auto-link corruption inside HTML <script src="[url](url)"> from Trace #1075
    raw_trace_1075 = (
        '<tool_call>{"name": "Write", "arguments": {'
        '"file_path": "/tmp/island.html", '
        '"content": "<!DOCTYPE html>\\n<script src=\\"[https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js](https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js)\\"></script>\\n'
        'See [Documentation](https://threejs.org/docs) for details."'
        '}}</tool_call>'
    )
    w_name, w_args = _parse_invoke_xml(raw_trace_1075)
    assert w_name == "Write"
    assert w_args["file_path"] == "/tmp/island.html"
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>' in w_args["content"]
    # Normal markdown links where label != url must be preserved
    assert "[Documentation](https://threejs.org/docs)" in w_args["content"]



