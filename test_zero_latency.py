import asyncio
import time
from httpx import AsyncClient
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_zero_latency")

async def test_latency():
    log.info("Starting test. The server should be running on localhost:8100.")
    async with AsyncClient() as client:
        # Wait a moment to ensure background tasks are started by the server
        log.info("Sleeping for 5 seconds to allow background warmup...")
        await asyncio.sleep(5)
        
        # Test Gemini
        start_time = time.time()
        log.info("Sending first request to Gemini...")
        resp = await client.post(
            "http://localhost:8100/v1/chat/completions",
            json={
                "model": "gemini",
                "messages": [{"role": "user", "content": "Hi, reply 'OK'"}],
                "stream": False
            },
            timeout=30.0
        )
        gemini_latency = time.time() - start_time
        log.info(f"Gemini response in {gemini_latency:.2f}s: {resp.status_code}")
        
        # Test ChatGPT
        start_time = time.time()
        log.info("Sending first request to ChatGPT (auto)...")
        resp = await client.post(
            "http://localhost:8100/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hi, reply 'OK'"}],
                "stream": False
            },
            timeout=30.0
        )
        chatgpt_latency = time.time() - start_time
        log.info(f"ChatGPT response in {chatgpt_latency:.2f}s: {resp.status_code}")
        
        # Check if latencies are reasonable (e.g. less than 5 seconds for a simple request,
        # which means PoW/token fetching didn't dominate the time)
        if gemini_latency > 8.0:
            log.warning("Gemini latency seems high, background task might not have finished or worked.")
        if chatgpt_latency > 8.0:
            log.warning("ChatGPT latency seems high, background task might not have finished or worked.")
        
        log.info("Test complete.")

if __name__ == "__main__":
    asyncio.run(test_latency())
