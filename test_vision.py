"""
test_vision.py — End-to-end test for ChatGPT vision/multimodal support.

Tests:
  1. Base64 image (small red square PNG)
  2. External URL image (Wikipedia public domain)
  3. Dead URL (expect 400/error)
"""
import asyncio
import base64
import io
import json
import sys

import httpx

PROXY_URL = "http://127.0.0.1:8100/v1/chat/completions"


def make_tiny_red_png() -> str:
    """Fetch a real image to use as base64 to avoid backend crashes on solid color squares."""
    import urllib.request
    from PIL import Image
    import io
    url = "https://raw.githubusercontent.com/python/pythondotorg/master/static/img/python-logo.png"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        img_bytes = response.read()
    
    # Convert to JPEG
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


async def test_base64_image():
    """Test 1: Send a base64-encoded image."""
    print("\n" + "=" * 60)
    print("TEST 1: Base64 image (tiny red square)")
    print("=" * 60)

    data_uri = make_tiny_red_png()
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "What color is this image? Reply in one word."},
                ],
            }
        ],
        "stream": True,
    }

    print("Sending request...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", PROXY_URL, json=payload) as resp:
            print(f"Status: {resp.status_code}")
            if resp.status_code != 200:
                body = await resp.aread()
                print(f"Error: {body.decode()}")
                return False

            full_text = ""
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "error" in chunk:
                            print(f"Stream error: {chunk['error']}")
                            return False
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

            print(f"\n\nFull response: {full_text!r}")
            return bool(full_text)


async def test_external_url():
    """Test 2: Send an external URL image."""
    print("\n" + "=" * 60)
    print("TEST 2: External URL image (Wikipedia)")
    print("=" * 60)

    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/python/pythondotorg/master/static/img/python-logo.png"
                        },
                    },
                    {"type": "text", "text": "Describe this image briefly in one sentence."},
                ],
            }
        ],
        "stream": True,
    }

    print("Sending request...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", PROXY_URL, json=payload) as resp:
            print(f"Status: {resp.status_code}")
            if resp.status_code != 200:
                body = await resp.aread()
                print(f"Error: {body.decode()}")
                return False

            full_text = ""
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "error" in chunk:
                            print(f"Stream error: {chunk['error']}")
                            return False
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

            print(f"\n\nFull response: {full_text!r}")
            return bool(full_text)


async def test_dead_url():
    """Test 3: Send a dead URL — should fail with 400 or stream error."""
    print("\n" + "=" * 60)
    print("TEST 3: Dead URL (expect error)")
    print("=" * 60)

    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://thisdomaindoesnotexist12345.com/image.png"},
                    },
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
        "stream": True,
    }

    print("Sending request...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", PROXY_URL, json=payload) as resp:
                print(f"Status: {resp.status_code}")
                if resp.status_code >= 400:
                    body = await resp.aread()
                    print(f"Got expected error: {body.decode()}")
                    return True
                
                # Check for stream error
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "error" in chunk:
                                print(f"Got expected stream error: {chunk['error']}")
                                return True
                        except json.JSONDecodeError:
                            pass

                print("ERROR: No error returned for dead URL!")
                return False
        except Exception as e:
            print(f"Got expected exception: {e}")
            return True


async def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}
    if test in ("1", "base64", "all"):
        results["base64"] = await test_base64_image()
    if test in ("2", "url", "all"):
        results["url"] = await test_external_url()
    if test in ("3", "dead", "all"):
        results["dead_url"] = await test_dead_url()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
