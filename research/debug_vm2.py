"""
debug_vm2.py — Dump full decoded instruction array and simulate execution.
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
        await s.get(BASE_URL + "/", headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        })
        oai_did = s.cookies.get("oai-did") or str(uuid.uuid4())
        p_token, sid = generate_requirements_token()

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

        # Decrypt with p_token
        raw = base64.b64decode(dx).decode("latin-1")
        decrypted = xor_decrypt(raw, p_token)
        instructions = json.loads(decrypted)

        print(f"Total instructions: {len(instructions)}")
        print(f"\nAll instructions:")
        for i, instr in enumerate(instructions):
            print(f"  [{i:3d}] {instr}")

        # Map opcodes  
        # First instruction: [8, 76.24, 8] — this is COPY with register 76.24 being created
        # The opcode IDs in the first array are the VM's PROGRAM (opcode → handler) IDs
        # Register 76.24 is assigned the opcode handler for 8 (which is COPY)
        # Then [76.24, 66.15, 15] means: use register 76.24 (COPY op) with args 66.15, 15
        #   i.e., register 66.15 = register 15 (copy opcode = 15 = JSON_STRINGIFY... wait)

        # Let's trace the pattern
        print(f"\n--- Opcode resolution ---")
        print(f"Instruction 0: {instructions[0]}")
        print(f"  → COPY opcode definition: reg[{instructions[0][1]}] = COPY (opcode {instructions[0][2]})")
        copy_reg = instructions[0][1]
        opcode_aliases = {copy_reg: 8}  # 8 = COPY

        for instr in instructions[1:]:
            if instr[0] == copy_reg and len(instr) == 3:
                target, source = instr[1], instr[2]
                opcode_aliases[target] = opcode_aliases.get(source, source)
                
        print(f"\nOpcode aliases resolved:")
        for reg, op in sorted(opcode_aliases.items()):
            names = {
                0: 'LOAD_PROGRAM', 1: 'XOR_REG', 2: 'SET', 3: 'RESOLVE', 4: 'REJECT',
                5: 'ADD', 6: 'GET_INDEX', 7: 'CALL', 8: 'COPY', 9: 'IP', 10: 'window',
                11: 'SCRIPT_MATCH', 12: 'GET_VM', 13: 'TRY_CALL', 14: 'JSON_PARSE',
                15: 'JSON_STRINGIFY', 16: 'KEY', 17: 'TRY_CALL_RES', 18: 'ATOB',
                19: 'BTOA', 20: 'IF_EQUAL', 21: 'IF_NOT_CLOSE', 22: 'SUBROUTINE',
                23: 'IF_DEFINED', 24: 'BIND', 25: 'noop1', 26: 'noop2', 27: 'SUB',
                28: 'noop3', 29: 'LESS_THAN', 30: 'DEFINE_FUNC', 33: 'MULTIPLY',
                34: 'AWAIT', 35: 'DIVIDE'
            }
            print(f"  reg[{reg}] = opcode {op} ({names.get(int(op), 'UNKNOWN')})")

        print(f"\n--- Annotated execution trace ---")
        for i, instr in enumerate(instructions):
            op = instr[0]
            args = instr[1:]
            real_op = opcode_aliases.get(op, op)
            names = {
                0: 'LOAD_PROGRAM', 1: 'XOR_REG', 2: 'SET', 3: 'RESOLVE', 4: 'REJECT',
                5: 'ADD', 6: 'GET_INDEX', 7: 'CALL', 8: 'COPY', 9: 'IP', 10: 'window',
                11: 'SCRIPT_MATCH', 12: 'GET_VM', 13: 'TRY_CALL', 14: 'JSON_PARSE',
                15: 'JSON_STRINGIFY', 16: 'KEY', 17: 'TRY_CALL_RES', 18: 'ATOB',
                19: 'BTOA', 20: 'IF_EQUAL', 21: 'IF_NOT_CLOSE', 22: 'SUBROUTINE',
                23: 'IF_DEFINED', 24: 'BIND', 25: 'noop1', 26: 'noop2', 27: 'SUB',
                28: 'noop3', 29: 'LESS_THAN', 30: 'DEFINE_FUNC', 33: 'MULTIPLY',
                34: 'AWAIT', 35: 'DIVIDE'
            }
            op_name = names.get(int(real_op), f'OP_{real_op}')
            print(f"  [{i:3d}] {op_name:20s} {args}")

if __name__ == "__main__":
    asyncio.run(main())
