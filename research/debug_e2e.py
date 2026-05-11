"""
debug_e2e.py — Step-by-step debug script that prints every raw response.
Run: python3 research/debug_e2e.py
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from curl_cffi.requests import AsyncSession
from src.sentinel.token_generator import (
    generate_requirements_token,
    generate_enforcement_token,
    build_sentinel_token,
)
import uuid

BASE_URL = "https://chatgpt.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

async def main():
    print("=== STEP 1: Get oai-did from homepage ===")
    async with AsyncSession(impersonate="chrome131") as s:
        resp = await s.get(BASE_URL + "/", headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        print(f"Homepage status: {resp.status_code}")
        oai_did = s.cookies.get("oai-did")
        print(f"oai-did: {oai_did}")
        all_cookies = {k: v for k, v in s.cookies.items()}
        print(f"All cookies: {list(all_cookies.keys())}")

        print("\n=== STEP 2: Generate requirements token (gAAAAAC) ===")
        p_token, sid = generate_requirements_token()
        print(f"p_token prefix: {p_token[:30]}...")
        print(f"session id (sid): {sid}")

        print("\n=== STEP 3: POST /backend-anon/sentinel/chat-requirements ===")
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/event-stream",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "oai-device-id": oai_did or str(uuid.uuid4()),
            "oai-language": "en-US",
            "Origin": "https://chatgpt.com",
            "Referer": "https://chatgpt.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }
        resp2 = await s.post(
            BASE_URL + "/backend-anon/sentinel/chat-requirements",
            headers=headers,
            json={"p": p_token},
        )
        print(f"chat-requirements status: {resp2.status_code}")
        chat_req = resp2.json()
        print(f"persona: {chat_req.get('persona')}")
        print(f"token prefix: {str(chat_req.get('token',''))[:30]}...")
        print(f"turnstile.required: {chat_req.get('turnstile', {}).get('required')}")
        print(f"proofofwork.required: {chat_req.get('proofofwork', {}).get('required')}")
        pow_data = chat_req.get("proofofwork", {})
        seed = pow_data.get("seed", "0.0")
        difficulty = pow_data.get("difficulty", "ffffff")
        print(f"PoW seed: {seed!r}  difficulty: {difficulty!r}")

        # Update oai-did if refreshed
        new_did = s.cookies.get("oai-did")
        if new_did:
            oai_did = new_did

        print("\n=== STEP 4: Solve PoW ===")
        enforcement_token = generate_enforcement_token(seed, difficulty, sid)
        print(f"enforcement_token prefix: {enforcement_token[:30]}...")

        print("\n=== STEP 5: Build sentinel token ===")
        sentinel_token = build_sentinel_token(
            chat_req=chat_req,
            enforcement_token=enforcement_token,
            p_token=p_token,
            oai_did=oai_did,
        )
        import base64
        try:
            parsed = json.loads(base64.b64decode(sentinel_token).decode("utf-8"))
            print(f"sentinel token keys: {list(parsed.keys())}")
            print(f"  p prefix: {str(parsed.get('p',''))[:30]}")
            print(f"  t:        {parsed.get('t')}")
            print(f"  c prefix: {str(parsed.get('c',''))[:30]}")
            print(f"  id:       {parsed.get('id')}")
            print(f"  flow:     {parsed.get('flow')}")
        except Exception as e:
            print(f"Error parsing sentinel token: {e}")

        import base64, json
        print("\n=== STEP 6: POST /backend-anon/conversation (single message) ===")
        s_payload = json.loads(base64.b64decode(sentinel_token).decode("utf-8"))
        
        conv_headers = {
            **headers,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Openai-Sentinel-Chat-Requirements-Token": s_payload.get("c", ""),
            "Openai-Sentinel-Proof-Token": s_payload.get("p", ""),
            "Openai-Sentinel-Turnstile-Token": s_payload.get("t", ""),
        }
        body = {
            "action": "next",
            "messages": [{
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Reply with the word PONG only"]},
            }],
            "model": "auto",
            "parent_message_id": str(uuid.uuid4()),
            "stream": True,
            "timezone_offset_min": -480,
        }
        print(f"Request body (messages): {json.dumps(body['messages'])[:100]}")
        print(f"Sending to: {BASE_URL}/backend-anon/conversation")

        resp3 = await s.post(
            BASE_URL + "/backend-anon/conversation",
            headers=conv_headers,
            json=body,
        )
        print(f"\nConversation status: {resp3.status_code}")
        print(f"Response headers: Content-Type={resp3.headers.get('content-type')}")
        print(f"Request headers: {resp3.request.headers}")
        print(f"Session cookies: {s.cookies.get_dict()}")

        # Print raw body (first 1000 chars)
        raw = resp3.text
        print(f"Raw response (first 1000 chars):\n{raw[:1000]}")

        if resp3.status_code == 403:
            print("\n[ERROR] 403 received. Trying /backend-anon/sentinel/req as fallback...")
            # Try the /req endpoint like the SDK does in its iframe
            resp_req = await s.post(
                BASE_URL + "/backend-anon/sentinel/req",
                headers={
                    **headers,
                    "Content-Type": "application/json",
                },
                json={"p": p_token, "flow": "__default__", "id": oai_did},
            )
            print(f"/sentinel/req status: {resp_req.status_code}")
            print(f"/sentinel/req body: {resp_req.text[:500]}")

if __name__ == "__main__":
    asyncio.run(main())
