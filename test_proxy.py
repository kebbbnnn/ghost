import asyncio
import json
import sys
from curl_cffi.requests import AsyncSession

PROXY_URL = "http://127.0.0.1:8100/v1/chat/completions"

async def test_proxy(stream: bool = True):
    print(f"Testing Proxy Server ({'Streaming' if stream else 'Non-Streaming'})...")
    
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Write a haiku about artificial intelligence."}
        ],
        "stream": stream
    }

    async with AsyncSession() as s:
        try:
            # We use stream=stream in the request call so curl_cffi doesn't close the connection early
            resp = await s.post(PROXY_URL, json=payload, stream=stream)
            
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                return

            if not stream:
                print("\nResponse:")
                data = resp.json()
                print(data["choices"][0]["message"]["content"])
                return

            print("\nResponse Stream:")
            async for chunk in resp.aiter_lines():
                if chunk:
                    line = chunk.decode("utf-8")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                # Print incrementally without newlines
                                sys.stdout.write(delta["content"])
                                sys.stdout.flush()
                        except json.JSONDecodeError:
                            pass
            print("\n\n[Stream Finished]")
            
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Make sure the proxy server is running! (python3 -m uvicorn src.server:app --port 8100)")

async def main():
    print("====================================")
    await test_proxy(stream=True)
    print("====================================")
    await test_proxy(stream=False)
    print("====================================")

if __name__ == "__main__":
    asyncio.run(main())
