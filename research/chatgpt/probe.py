import asyncio
import json
from curl_cffi.requests import AsyncSession

async def probe():
    with open("research/probe_output.txt", "w") as f:
        f.write("=== Phase 1: Probe chatgpt.com ===\n")
        async with AsyncSession(impersonate="chrome131") as s:
            resp = await s.get("https://chatgpt.com/", headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })
            f.write(f"Status: {resp.status_code}\n")
            f.write("Cookies:\n")
            for name, value in s.cookies.items():
                f.write(f"  {name}: {value}\n")
            
            oai_did = s.cookies.get("oai-did")
            if oai_did:
                f.write(f"Found oai-did: {oai_did}\n")

            f.write("\n=== Phase 1.1: Try backend-anon chat-requirements ===\n")
            req_headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Content-Type": "application/json",
                "oai-language": "en-US",
                "origin": "https://chatgpt.com",
                "referer": "https://chatgpt.com/",
            }
            if oai_did:
                req_headers["oai-device-id"] = oai_did

            resp2 = await s.post("https://chatgpt.com/backend-anon/sentinel/chat-requirements", headers=req_headers, json={"p": "invalid_token"})
            f.write(f"Status: {resp2.status_code}\n")
            try:
                f.write(f"Response: {json.dumps(resp2.json(), indent=2)}\n")
            except:
                f.write(f"Response: {resp2.text[:200]}\n")

if __name__ == "__main__":
    asyncio.run(probe())
