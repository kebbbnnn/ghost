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
