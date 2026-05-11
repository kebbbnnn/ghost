import asyncio
import json
from curl_cffi.requests import AsyncSession

async def main():
    async with AsyncSession() as s:
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 2+2? Reply only with the number."}],
            "stream": True
        }
        print("Sending request to local proxy...")
        resp = await s.post("http://127.0.0.1:8100/v1/chat/completions", json=body)
        print(f"Status: {resp.status_code}")
        async for chunk in resp.aiter_lines():
            if chunk:
                print(chunk.decode('utf-8'))

asyncio.run(main())
