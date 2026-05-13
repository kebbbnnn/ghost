import asyncio
import json
import sys
from curl_cffi.requests import AsyncSession

PROXY_URL = "http://127.0.0.1:8100/v1/chat/completions"

MODEL_LABELS = {
    "gemini": "Gemini",
    "auto":   "ChatGPT",
}

def resolve_model(arg: str | None) -> tuple[str, str]:
    """Return (model_id, display_label) from a CLI arg."""
    model = (arg or "auto").strip().lower()
    label = MODEL_LABELS.get(model, model.upper())
    return model, label

async def interactive_chat(model: str = "auto", label: str = "ChatGPT"):
    print("==================================================")
    print(f"{label} Proxy Interactive Test")
    print("Type your message and press Enter. Type 'exit' to quit.")
    print("==================================================\n")

    # Keep a simple conversation history and conversation tracking IDs
    messages = []
    conversation_id = None
    parent_message_id = None

    async with AsyncSession() as s:
        while True:
            try:
                user_input = input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                break
                
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ["exit", "quit"]:
                break
                
            messages.append({"role": "user", "content": user_input})

            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            if conversation_id:
                payload["conversation_id"] = conversation_id
            if parent_message_id:
                payload["parent_message_id"] = parent_message_id

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
