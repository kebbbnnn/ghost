"""Fetch requirements, decrypt bytecode, and check for cfIp keys"""
import base64
import json
from curl_cffi.requests import Session

from src.providers.chatgpt.sentinel.token_generator import generate_requirements_token
from src.providers.chatgpt.sentinel.vm import _xor_decrypt

s = Session(impersonate="chrome131")
p_token, sid = generate_requirements_token()

resp = s.post(
    "https://chatgpt.com/backend-anon/sentinel/chat-requirements",
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "oai-client-build-number": "8338682",
        "oai-client-version": "prod-e8f67765f24e1c6773b4b61b4310d5ff9297439a",
        "Content-Type": "application/json",
    },
    json={"p": p_token},
)
chat_req = resp.json()
turnstile = chat_req.get("turnstile", {})
dx = turnstile.get("dx")

if dx:
    raw = base64.b64decode(dx).decode("latin-1")
    decrypted = _xor_decrypt(raw, p_token)
    instructions = json.loads(decrypted)
    
    # Dump all strings in instructions
    all_strings = set()
    def collect_strings(x):
        if isinstance(x, str):
            all_strings.add(x)
        elif isinstance(x, list):
            for item in x:
                collect_strings(item)
        elif isinstance(x, dict):
            for k, v in x.items():
                collect_strings(k)
                collect_strings(v)
                
    collect_strings(instructions)
    print("--- Decrypted VM strings ---")
    for s in sorted(all_strings):
        if any(term in s.lower() for term in ["cf", "ip", "lat", "lon", "country", "city", "vendor", "platform"]):
            print(f"Match: {s}")
else:
    print("No dx found in turnstile!")
