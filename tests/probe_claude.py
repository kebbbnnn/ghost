"""
Probe the 429 response to understand what Claude is actually rejecting.
"""
import asyncio
import json
import os
import uuid
from curl_cffi.requests import AsyncSession

try:
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.core.config import ACCOUNTS_FILE
except ImportError:
    ACCOUNTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "accounts.json")
BASE_URL = "https://claude.ai"

def load_session_key():
    with open(ACCOUNTS_FILE) as f:
        data = json.load(f)
    acc = data["accounts"][0]
    return acc["sessionKey"], acc["email"]

def base_headers(session_key):
    return {
        "Cookie": f"sessionKey={session_key}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://claude.ai/",
        "Origin": "https://claude.ai",
    }

async def probe():
    sk, email = load_session_key()
    print(f"Using account: {email}\n")

    async with AsyncSession(impersonate="chrome120") as client:
        h = base_headers(sk)

        # 1. Get org
        res = await client.get(f"{BASE_URL}/api/organizations", headers=h)
        print(f"Org fetch: {res.status_code}")
        orgs = res.json()
        org_id = orgs[0]["uuid"]
        print(f"Org ID: {org_id}\n")

        # 2. Create conversation
        conv_uuid = str(uuid.uuid4())
        res2 = await client.post(
            f"{BASE_URL}/api/organizations/{org_id}/chat_conversations",
            headers=h,
            json={"uuid": conv_uuid, "name": ""},
        )
        print(f"Create conversation: {res2.status_code}")
        print(f"Conv UUID: {conv_uuid}\n")

        # 3. Send completion
        h_stream = {**h, "Accept": "text/event-stream"}
        msg_payload = {
            "prompt": "Say hello in one word.",
            "timezone": "America/New_York",
            "attachments": [],
            "files": [],
        }
        
        print(f"--- Sending completion ---")
        print(f"URL: {BASE_URL}/api/organizations/{org_id}/chat_conversations/{conv_uuid}/completion")
        print(f"Headers: {json.dumps({k: v[:30] + '...' if len(str(v)) > 30 else v for k, v in h_stream.items()}, indent=2)}")
        print(f"Payload: {json.dumps(msg_payload, indent=2)}\n")
        
        res3 = await client.post(
            f"{BASE_URL}/api/organizations/{org_id}/chat_conversations/{conv_uuid}/completion",
            headers=h_stream,
            json=msg_payload,
        )
        
        print(f"Completion status: {res3.status_code}")
        print(f"Response headers: {dict(res3.headers)}")
        print(f"\nResponse body:\n{res3.text[:3000]}")

if __name__ == "__main__":
    asyncio.run(probe())
