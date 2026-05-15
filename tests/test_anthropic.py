import asyncio
import httpx
import json

URL = "http://127.0.0.1:8100/v1/messages"

async def test_text_stream():
    print("--- Test Text Streaming ---")
    payload = {
        "model": "auto",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Say hello world briefly."}
        ],
        "stream": True,
        "max_tokens": 100
    }
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", URL, json=payload, timeout=30.0) as resp:
            if resp.status_code != 200:
                print(f"Error: {resp.status_code}")
                text = await resp.aread()
                print(text)
                return
                
            async for line in resp.aiter_lines():
                if line:
                    print(line)
    print("\n")

async def test_tool_use_stream():
    print("--- Test Tool Use Streaming ---")
    payload = {
        "model": "auto",
        "system": "You are a helpful assistant.",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "What is the weather in Boston right now?"}
        ],
        "stream": True,
        "max_tokens": 100
    }
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", URL, json=payload, timeout=30.0) as resp:
            if resp.status_code != 200:
                print(f"Error: {resp.status_code}")
                text = await resp.aread()
                print(text)
                return
                
            async for line in resp.aiter_lines():
                if line:
                    print(line)
    print("\n")

if __name__ == "__main__":
    asyncio.run(test_text_stream())
    asyncio.run(test_tool_use_stream())
