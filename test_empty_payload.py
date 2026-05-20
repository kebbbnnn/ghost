import asyncio
import json
from curl_cffi.requests import AsyncSession

async def main():
    session = AsyncSession(impersonate="chrome131")
    
    # 1. Get chat requirements
    req = await session.post(
        "https://chatgpt.com/backend-anon/sentinel/chat-requirements",
        headers={"Content-Type": "application/json", "oai-language": "en-US"},
        json={"p": "gAAAAAB...dummy"}
    )
    
    oai_did = session.cookies.get("oai-did") or "dummy"
    print(f"oai-did: {oai_did}")
    
    # 2. Send empty payload to /f/conversation
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "oai-device-id": oai_did,
        "oai-language": "en-US",
        "origin": "https://chatgpt.com",
        "referer": "https://chatgpt.com/"
    }
    
    resp = await session.post(
        "https://chatgpt.com/backend-anon/f/conversation",
        headers=headers,
        json={"action": "next", "messages": [], "parent_message_id": "client-created-root", "model": "auto"}
    )
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text}")

asyncio.run(main())
