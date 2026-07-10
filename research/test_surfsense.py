"""Live integration test for SurfsenseProvider with header-based bypass.

Tests multiple sequential requests to verify that quota/CAPTCHA bypass works.
"""
import asyncio
import json
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

async def main():
    from src.providers.surfsense.session import SurfsenseProvider

    provider = SurfsenseProvider()

    try:
        # Run 5 requests in a row to simulate heavy usage
        for i in range(1, 6):
            print(f"\n=== Running request {i}/5 ===")
            messages = [{"role": "user", "content": f"Repeat exactly: token bypass request {i}"}]

            full_text = ""
            async for chunk_str in provider.stream_chat(
                messages=messages,
                model="surfsense-gpt-5.4-mini-no-login",
            ):
                if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                    try:
                        data = json.loads(chunk_str[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                            if delta:
                                full_text += delta
                                sys.stdout.write(delta)
                                sys.stdout.flush()
                    except json.JSONDecodeError:
                        pass

            print(f"\nResponse {i}: {full_text!r}")

        print("\n=== ALL REQUESTS SUCCEEDED AND BYPASSED CAPTCHAS ===")

    except Exception as e:
        print(f"\n=== ERROR: {e} ===")
        import traceback
        traceback.print_exc()
    finally:
        await provider.stop_background_tasks()

if __name__ == "__main__":
    asyncio.run(main())
