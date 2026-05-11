import sys, base64, json
sys.path.insert(0, '.')
from src.sentinel.vm import _VM, _xor_decrypt
from src.sentinel.token_generator import generate_requirements_token
from curl_cffi.requests import AsyncSession
import asyncio
import uuid

async def main():
    async with AsyncSession(impersonate='chrome131') as s:
        # Use a random uuid for oai-did to avoid block if possible
        oai_did = str(uuid.uuid4())
        
        p_token, sid = generate_requirements_token()
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Origin": "https://chatgpt.com",
            "Referer": "https://chatgpt.com/",
            "oai-device-id": oai_did,
            "oai-language": "en-US",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }
        resp = await s.post('https://chatgpt.com/backend-anon/sentinel/chat-requirements', headers=headers, json={'p': p_token})
        print(f"Status: {resp.status_code}")
        data = resp.json()
        dx = data.get('turnstile', {}).get('dx')
        if dx:
            raw = base64.b64decode(dx).decode("latin-1")
            decrypted = _xor_decrypt(raw, p_token)
            instructions = json.loads(decrypted)
            vm = _VM()
            
            orig_call_fn = vm._call_fn
            def _call_fn_hook(fn, call_args):
                if fn == "function":
                    print(f"TRACED CALL 'function': args={call_args}")
                elif callable(fn) and "_resolve" in str(fn):
                    print(f"TRACED CALL _resolve: args={call_args}")
                return orig_call_fn(fn, call_args)
            vm._call_fn = _call_fn_hook
            
            orig_set = vm._set
            def _set_hook(reg, val):
                if val == "function":
                    print(f"TRACED SET 'function' into reg {reg}")
                orig_set(reg, val)
            vm._set = _set_hook
            
            res = vm.execute(instructions)
            print(f'VM finished. Result = {res}')
        else:
            print(f"No dx: {data}")

asyncio.run(main())
