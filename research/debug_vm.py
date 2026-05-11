"""
debug_vm.py — Inspect the Turnstile VM dx bytecode decryption.
Run: python3 research/debug_vm.py
"""
import asyncio
import base64
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from curl_cffi.requests import AsyncSession
from src.sentinel.token_generator import generate_requirements_token, generate_enforcement_token
import uuid

BASE_URL = "https://chatgpt.com"

def xor_decrypt(data: str, key: str) -> str:
    return "".join(chr(ord(d) ^ ord(key[i % len(key)])) for i, d in enumerate(data))

async def main():
    async with AsyncSession(impersonate="chrome131") as s:
        # Get device ID
        await s.get(BASE_URL + "/", headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        oai_did = s.cookies.get("oai-did") or str(uuid.uuid4())

        # Get p_token
        p_token, sid = generate_requirements_token()
        print(f"p_token: {p_token[:50]}...")

        # Get chat requirements
        resp = await s.post(
            BASE_URL + "/backend-anon/sentinel/chat-requirements",
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Content-Type": "application/json",
                "oai-device-id": oai_did,
                "oai-language": "en-US",
                "Origin": "https://chatgpt.com",
                "Referer": "https://chatgpt.com/",
            },
            json={"p": p_token},
        )
        chat_req = resp.json()
        dx = chat_req.get("turnstile", {}).get("dx", "")
        pow_data = chat_req.get("proofofwork", {})
        seed = pow_data.get("seed", "0")
        difficulty = pow_data.get("difficulty", "ffffff")

        print(f"dx length: {len(dx)}")
        print(f"pow seed: {seed!r}, difficulty: {difficulty!r}")

        # Solve PoW
        enforcement_token = generate_enforcement_token(seed, difficulty, sid)
        print(f"enforcement_token: {enforcement_token[:50]}...")

        # Try decrypting dx with p_token (gAAAAAC)
        print(f"\n--- Decrypting dx with p_token (gAAAAAC) ---")
        try:
            raw = base64.b64decode(dx).decode("latin-1")
            decrypted = xor_decrypt(raw, p_token)
            print(f"Decrypted (first 200 chars): {decrypted[:200]!r}")
            try:
                parsed = json.loads(decrypted)
                print(f"Parsed as JSON: type={type(parsed).__name__}, len={len(parsed) if isinstance(parsed, list) else 'N/A'}")
                if isinstance(parsed, list) and len(parsed) > 0:
                    print(f"First instruction: {parsed[0]}")
                    print(f"Second instruction: {parsed[1] if len(parsed) > 1 else 'N/A'}")
                    print(f"Total instructions: {len(parsed)}")
            except json.JSONDecodeError as e:
                print(f"Not valid JSON: {e}")
        except Exception as e:
            print(f"Error decrypting with p_token: {e}")

        # Try decrypting dx with enforcement_token (gAAAAAB)
        print(f"\n--- Decrypting dx with enforcement_token (gAAAAAB) ---")
        try:
            raw = base64.b64decode(dx).decode("latin-1")
            decrypted = xor_decrypt(raw, enforcement_token)
            print(f"Decrypted (first 200 chars): {decrypted[:200]!r}")
            try:
                parsed = json.loads(decrypted)
                print(f"Parsed as JSON: type={type(parsed).__name__}")
            except json.JSONDecodeError as e:
                print(f"Not valid JSON: {e}")
        except Exception as e:
            print(f"Error: {e}")

        # Try with just the raw p_token without the gAAAAAC prefix
        print(f"\n--- Decrypting dx with raw p_token (strip gAAAAAC prefix) ---")
        try:
            raw_p = p_token.replace("gAAAAAC", "")
            raw = base64.b64decode(dx).decode("latin-1")
            decrypted = xor_decrypt(raw, raw_p)
            print(f"Decrypted (first 200 chars): {decrypted[:200]!r}")
            try:
                parsed = json.loads(decrypted)
                print(f"Parsed as JSON!")
            except:
                print("Not valid JSON")
        except Exception as e:
            print(f"Error: {e}")

        # Try with chat_req.token as key
        print(f"\n--- Decrypting dx with chat_req.token ---")
        try:
            token = chat_req.get("token", "")
            raw = base64.b64decode(dx).decode("latin-1")
            decrypted = xor_decrypt(raw, token)
            print(f"Decrypted (first 200 chars): {decrypted[:200]!r}")
            try:
                parsed = json.loads(decrypted)
                print(f"Parsed as JSON! type={type(parsed).__name__}")
            except:
                print("Not valid JSON")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
