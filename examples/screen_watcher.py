#!/usr/bin/env python3
"""
screen_watcher.py — GhostTunnel Client Example

This example script captures a screenshot of your screen at a regular interval (default: 30 seconds)
and sends it to ChatGPT Vision via a running GhostTunnel proxy server to get a description.

Requirements:
  - PIL (Pillow) — already in requirements.txt
  - httpx — already in requirements.txt
  - A running GhostTunnel proxy server (default: http://127.0.0.1:8100)

Usage:
  python examples/screen_watcher.py --interval 30 --output watcher_log.jsonl
"""

import argparse
import asyncio
import base64
import datetime
import io
import json
import os
import sys
import time
import httpx
from PIL import ImageGrab, Image

def images_are_similar(img1: Image.Image, img2: Image.Image, threshold: float = 2.0) -> bool:
    """
    Compare two PIL Images. Returns True if they are very similar.
    Downscales to a 64x64 thumbnail and computes Mean Absolute Error (MAE).
    """
    try:
        size = (64, 64)
        g1 = img1.convert("L").resize(size, Image.Resampling.BILINEAR)
        g2 = img2.convert("L").resize(size, Image.Resampling.BILINEAR)
        p1 = list(g1.tobytes())
        p2 = list(g2.tobytes())
        mae = sum(abs(a - b) for a, b in zip(p1, p2)) / len(p1)
        return mae < threshold
    except Exception:
        return False

async def capture_screenshot() -> tuple[str, Image.Image]:
    """
    Capture a screenshot of the screen and return it as a tuple: (JPEG base64 Data URI, PIL Image).
    Optimizes quality and resolution to keep payloads light but recognizable.
    """
    try:
        # ImageGrab.grab() grabs the primary screen.
        screenshot = ImageGrab.grab()
        
        # Keep the original screenshot for similarity comparison (pre-downscaling for API)
        comparison_img = screenshot.copy()
        
        # Downscale slightly if the image is extremely large (e.g. high-DPI retina display)
        # to save bandwidth and speed up request processing, while keeping details legible.
        max_size = 1920
        width, height = screenshot.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save to buffer as JPEG with reasonable quality. Convert to RGB if format has alpha layer.
        if screenshot.mode in ("RGBA", "P"):
            screenshot = screenshot.convert("RGB")
        buf = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=75)
        raw_bytes = buf.getvalue()
        
        # Convert to base64 Data URI
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        size_kb = len(raw_bytes) / 1024
        print(f"📸 Captured screenshot ({screenshot.width}x{screenshot.height}, {size_kb:.1f} KB)")
        return f"data:image/jpeg;base64,{b64_data}", comparison_img
        
    except OSError as e:
        print(f"❌ OSError during screen capture: {e}")
        print("💡 TIP: On macOS, ensure your terminal/editor has Screen Recording permission under System Settings.")
        raise
    except Exception as e:
        print(f"❌ Failed to capture screenshot: {e}")
        raise

async def get_description_stream(server_url: str, model: str, image_uri: str, prompt: str):
    """
    Send the base64 image URI to GhostTunnel completions endpoint and stream the result.
    """
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_uri}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "stream": True
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("🤖 Prompting ChatGPT Vision...", end="", flush=True)
    
    # We use a long timeout since image processing/Turnstile solving on backend can take time
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    try:
                        err_body = await resp.aread()
                        err_msg = err_body.decode()
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}"
                    print(f"\n❌ Server error: {err_msg}")
                    return None
                
                print(" Receiving stream:\n")
                full_text = ""
                
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "error" in chunk:
                                print(f"\n❌ Stream error: {chunk['error']}")
                                return None
                            
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                print("\n")
                return full_text
                
        except httpx.ConnectError:
            print(f"\n❌ Connection failed! Is GhostTunnel server running at {server_url}?")
            return None
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return None

def save_to_log(log_path: str, timestamp: str, description: str):
    """Save the description to a local JSONL log file."""
    log_entry = {
        "timestamp": timestamp,
        "description": description
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write to log file: {e}")

async def main():
    parser = argparse.ArgumentParser(
        description="GhostTunnel screen watcher example using ChatGPT Vision."
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=30.0,
        help="Capture interval in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--server", "-s", type=str, default="http://127.0.0.1:8100",
        help="GhostTunnel proxy server URL (default: http://127.0.0.1:8100)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="auto",
        help="Model slug to use (default: auto)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional path to a JSONL file to log the descriptions"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=0,
        help="Number of captures to run before exiting (default: 0 for unlimited)"
    )
    parser.add_argument(
        "--skip", action="store_true",
        help="Enable skipping API calls when the screen is unchanged"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=2.0,
        help="Similarity threshold for skipping captures (default: 2.0, lower is more sensitive)"
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        default="Describe what's on my screen right now in a single, concise paragraph.",
        help="Prompt to pass alongside the screenshot"
    )
    
    args = parser.parse_args()
    
    print("==================================================")
    print("      GhostTunnel ChatGPT Vision Screen Watcher   ")
    print("==================================================")
    print(f"Interval:  {args.interval}s")
    print(f"Server:    {args.server}")
    print(f"Model:     {args.model}")
    print(f"Skip Mode: {args.skip}")
    if args.skip:
        print(f"Threshold: {args.threshold}")
    if args.output:
        print(f"Log File:  {args.output}")
    print("Press Ctrl+C to stop.")
    print("==================================================")
    
    loop_count = 0
    last_screenshot = None
    try:
        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] (Iteration {loop_count + 1})")
            
            try:
                img_uri, screenshot_img = await capture_screenshot()
                
                # Check for screen similarity if skip mode is enabled
                if args.skip and last_screenshot is not None and images_are_similar(last_screenshot, screenshot_img, args.threshold):
                    print("📷 Screen unchanged. Skipping API call.")
                else:
                    description = await get_description_stream(
                        server_url=args.server,
                        model=args.model,
                        image_uri=img_uri,
                        prompt=args.prompt
                    )
                    
                    if description:
                        last_screenshot = screenshot_img
                        if args.output:
                            save_to_log(args.output, timestamp, description)
                    
            except Exception as e:
                print(f"❌ Error during iteration: {e}")
                import traceback
                traceback.print_exc()
                
            loop_count += 1
            if args.count > 0 and loop_count >= args.count:
                print(f"Target count of {args.count} iterations reached. Exiting.")
                break
                
            print(f"Sleeping for {args.interval} seconds...")
            await asyncio.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nWatcher stopped by user. Goodbye!")
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
