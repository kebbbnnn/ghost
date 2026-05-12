"""
test_vm_live.py — Test the VM against a live dx blob.
"""
import asyncio, base64, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from curl_cffi.requests import AsyncSession
from src.sentinel.token_generator import generate_requirements_token, generate_enforcement_token
from src.sentinel.vm import execute_turnstile_vm
import uuid

BASE_URL = "https://chatgpt.com"

async def main():
    async with AsyncSession(impersonate="chrome131") as s:
        await s.get(BASE_URL + "/", headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        })
        oai_did = s.cookies.get("oai-did") or str(uuid.uuid4())
        p_token, sid = generate_requirements_token()
        print(f"p_token prefix: {p_token[:40]}...")

        resp = await s.post(
            BASE_URL + "/backend-anon/sentinel/chat-requirements",
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Content-Type": "application/json",
                "oai-device-id": oai_did,
                "oai-language": "en-US",
                "Origin": "https://chatgpt.com",
                "Referer": "https://chatgpt.com/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
            },
            json={"p": p_token},
        )
        chat_req = resp.json()
        dx = chat_req.get("turnstile", {}).get("dx", "")
        seed = chat_req.get("proofofwork", {}).get("seed", "0")
        difficulty = chat_req.get("proofofwork", {}).get("difficulty", "ffffff")
        print(f"dx length: {len(dx)}")
        print(f"PoW: seed={seed!r} difficulty={difficulty!r}")

        print("\nSolving PoW...")
        enforcement_token = generate_enforcement_token(seed, difficulty, sid)
        print(f"enforcement_token prefix: {enforcement_token[:40]}...")

        print("\nRunning Turnstile VM...")
        vm_result = execute_turnstile_vm(dx, p_token)
        print(f"VM result: {vm_result!r}")

        if vm_result:
            print(f"  → SUCCESS! VM returned {len(vm_result)} chars")
        else:
            print(f"  → FAILED: VM returned None")

            # Detailed VM trace
            def xor(d, k): return "".join(chr(ord(a)^ord(k[i%len(k)])) for i, a in enumerate(d))
            raw = base64.b64decode(dx).decode("latin-1")
            instructions = json.loads(xor(raw, p_token))

            from src.sentinel.vm import _VM, OP_IP
            vm2 = _VM()
            vm2._build_alias_map(instructions)
            vm2._set(OP_IP, instructions)
            op_names = {0:'LOAD',1:'XOR',2:'SET',3:'RESOLVE',4:'REJECT',5:'ADD',6:'GET_IDX',7:'CALL',8:'COPY',9:'IP',10:'WINDOW',11:'SCRIPT',12:'GET_VM',13:'TRY_CALL',14:'JSON_PARSE',15:'JSON_STR',16:'KEY',17:'TRY_CALL_R',18:'ATOB',19:'BTOA',20:'IF_EQ',21:'IF_NOT_CLOSE',22:'SUBROUTINE',23:'IF_DEF',24:'BIND',27:'SUB',29:'LT',30:'DEF_FN',33:'MUL',34:'AWAIT',35:'DIV'}
            
            ip = 0
            step = 0
            while True:
                cur_inst = vm2._get(OP_IP)
                if not isinstance(cur_inst, list) or ip >= len(cur_inst): break
                instr = cur_inst[ip]
                ip += 1
                if not isinstance(instr, list) or len(instr) < 1: continue
                raw_op = instr[0]
                op = vm2._resolve_op(raw_op)
                args = instr[1:]
                
                op_name = op_names.get(op, f'OP_{op}')
                if op == 7:  # CALL
                    fn = vm2._get(args[0])
                    print(f"STEP {step} [CALL] fn={fn} args={[vm2._get(a) for a in args[1:]]}")
                
                old_id = id(vm2._get(OP_IP))
                try: vm2._dispatch(op, args)
                except Exception: pass
                if id(vm2._get(OP_IP)) != old_id: ip = 0
                if vm2.result is not None: 
                    print(f"STEP {step} -> RESULT = {vm2.result}")
                    break
                step += 1

if __name__ == "__main__":
    asyncio.run(main())
