import asyncio
import json
import sys
from curl_cffi.requests import AsyncSession

PROXY_URL = "http://127.0.0.1:8100/v1/chat/completions"

async def interactive_chat():
    print("==================================================")
    print("ChatGPT Proxy Interactive Test")
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
                "model": "auto",
                "messages": messages,
                "stream": True
            }
            if conversation_id and parent_message_id:
                payload["conversation_id"] = conversation_id
                payload["parent_message_id"] = parent_message_id

            print("\nChatGPT: ", end="")
            sys.stdout.flush()
            
            full_response = ""
            last_msg_id = None

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
                                    
                                if "conversation_id" in data and data["conversation_id"]:
                                    conversation_id = data["conversation_id"]
                                if "id" in data and data["id"]:
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
                if last_msg_id:
                    parent_message_id = last_msg_id

if __name__ == "__main__":
    asyncio.run(interactive_chat())
