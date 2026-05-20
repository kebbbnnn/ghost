import asyncio
import base64
import json
import os
import sys
from curl_cffi.requests import AsyncSession

PROXY_URL = "http://127.0.0.1:8100/v1/chat/completions"

MODEL_LABELS = {
    "gemini": "Gemini",
    "auto":   "ChatGPT",
    "claude-3-5-sonnet-20241022": "Claude Sonnet",
    "claude-3-opus-20240229": "Claude Opus",
    "claude-3-5-haiku-20241022": "Claude Haiku",
}

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

def resolve_model(arg: str | None) -> tuple[str, str]:
    """Return (model_id, display_label) from a CLI arg."""
    model = (arg or "auto").strip().lower()
    label = MODEL_LABELS.get(model, model.upper())
    return model, label


def image_to_data_uri(path: str) -> str:
    """Read a local image file and return a data URI."""
    path = path.strip().strip("'\"")  # handle drag-and-drop quoted paths
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def build_content(text: str, image_urls: list[str]) -> list[dict] | str:
    """Build the content field for a message, with optional images."""
    if not image_urls:
        return text
    
    parts = []
    for url in image_urls:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    parts.append({"type": "text", "text": text})
    return parts


async def interactive_chat(model: str = "auto", label: str = "ChatGPT"):
    print("==================================================")
    print(f"{label} Proxy Interactive Test")
    print("Type your message and press Enter. Type 'exit' to quit.")
    print("")
    print("  Vision commands:")
    print("  /image <path>   — attach a local image file")
    print("  /url <url>      — attach an image from URL")
    print("  /clear          — clear attached images")
    print("  (then type your prompt on the next line)")
    print("==================================================\n")

    # Keep a simple conversation history and conversation tracking IDs
    messages = []
    conversation_id = None
    parent_message_id = None
    pending_images: list[str] = []  # data URIs or URLs to attach

    async with AsyncSession() as s:
        while True:
            try:
                # Show pending image count in prompt
                if pending_images:
                    user_input = input(f"\nYou [{len(pending_images)} image(s) attached]: ")
                else:
                    user_input = input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                break
                
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ["exit", "quit"]:
                break

            # --- Handle /image command ---
            if user_input.strip().lower().startswith("/image "):
                path = user_input.strip()[7:].strip()
                try:
                    data_uri = image_to_data_uri(path)
                    pending_images.append(data_uri)
                    size_kb = len(data_uri) * 3 // 4 // 1024  # approximate decoded size
                    print(f"  ✅ Image attached ({size_kb} KB). Type your prompt next.")
                except FileNotFoundError as e:
                    print(f"  ❌ {e}")
                except Exception as e:
                    print(f"  ❌ Failed to read image: {e}")
                continue

            # --- Handle /url command ---
            if user_input.strip().lower().startswith("/url "):
                url = user_input.strip()[5:].strip()
                pending_images.append(url)
                print(f"  ✅ Image URL attached. Type your prompt next.")
                continue

            # --- Handle /clear command ---
            if user_input.strip().lower() == "/clear":
                pending_images.clear()
                print("  🗑️  Cleared all attached images.")
                continue

            # --- Check if input looks like a dragged file path ---
            clean_input = user_input.strip().strip("'\"")
            if os.path.isfile(clean_input) and os.path.splitext(clean_input)[1].lower() in SUPPORTED_IMAGE_EXTS:
                try:
                    data_uri = image_to_data_uri(clean_input)
                    pending_images.append(data_uri)
                    size_kb = len(data_uri) * 3 // 4 // 1024
                    print(f"  ✅ Image attached ({size_kb} KB). Type your prompt next.")
                except Exception as e:
                    print(f"  ❌ Failed to read image: {e}")
                continue

            # --- Build and send message ---
            content = build_content(user_input, pending_images)
            messages.append({"role": "user", "content": content})
            images_sent = len(pending_images)
            pending_images.clear()

            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            if conversation_id:
                payload["conversation_id"] = conversation_id
            if parent_message_id:
                payload["parent_message_id"] = parent_message_id

            if images_sent:
                print(f"\n{label} (🖼️ analyzing {images_sent} image(s)): ", end="")
            else:
                print(f"\n{label}: ", end="")
            sys.stdout.flush()
            
            full_response = ""
            last_msg_id = None
            got_parent_from_chunk = False

            try:
                resp = await s.post(PROXY_URL, json=payload, stream=True)
                
                if resp.status_code != 200:
                    print(f"\n[Error {resp.status_code}: {resp.text}]")
                    messages.pop() # remove the failed message
                    continue

                async for chunk in resp.aiter_lines():
                    if chunk:
                        line = chunk.decode("utf-8")
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if "error" in data:
                                    print(f"\n[Proxy Error: {data['error'].get('message', data['error'])}]")
                                    break
                                    
                                if data.get("conversation_id"):
                                    conversation_id = data["conversation_id"]
                                # Gemini returns parent_message_id in the chunk envelope
                                if data.get("parent_message_id"):
                                    parent_message_id = data["parent_message_id"]
                                    got_parent_from_chunk = True
                                if data.get("id"):
                                    last_msg_id = data["id"]
                                    
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    full_response += content
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
            except Exception as e:
                print(f"\n[Connection failed: {e}]")
                break
            
            print() # Print newline after response finishes
            
            if full_response:
                messages.append({"role": "assistant", "content": full_response})
                # Only fall back to last_msg_id for ChatGPT (which doesn't embed
                # parent_message_id in chunks). For Gemini, it was already captured
                # correctly from the chunk envelope above — don't overwrite it.
                if last_msg_id and not got_parent_from_chunk:
                    parent_message_id = last_msg_id

if __name__ == "__main__":
    _model, _label = resolve_model(sys.argv[1] if len(sys.argv) > 1 else None)
    asyncio.run(interactive_chat(model=_model, label=_label))
