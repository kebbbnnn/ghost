"""
converter.py — Perplexity SSE to OpenAI format converter.

Handles parsing Perplexity SSE messages, tracking incremental text chunks,
and formatting output to match OpenAI's JSON schema for streaming and non-streaming.
"""
import json
import time
import uuid
import logging

log = logging.getLogger(__name__)

def parse_perplexity_event(line: str, tracker: dict) -> dict | None:
    """
    Parse a single SSE event line from Perplexity rest/sse/perplexity_ask.
    
    The tracker dict is mutated to keep track of yielded chunk indices:
      tracker = {
          "last_yielded_idx": -1,
          "full_text": ""
      }
      
    Returns a dict with:
      - "delta": str (new incremental text)
      - "backend_uuid": str
      - "context_uuid": str
      - "done": bool
      - "full_text": str
    or None if the event carried no new text/completion info.
    """
    if not line.strip():
        return None
        
    if line.startswith("event: end_of_stream"):
        return {"done": True, "delta": "", "backend_uuid": "", "context_uuid": ""}
        
    if not line.startswith("event: message\r\ndata: ") and not line.startswith("event: message\ndata: "):
        # Check if it starts with data: directly
        if not line.startswith("data: "):
            return None
        payload = line[6:].strip()
    else:
        # Strip event prefix
        idx = line.find("data: ")
        if idx == -1:
            return None
        payload = line[idx + 6:].strip()
        
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
        
    backend_uuid = data.get("backend_uuid", "")
    context_uuid = data.get("context_uuid", "")
    is_final = data.get("final", False)
    
    delta_text = ""
    full_answer = ""
    
    # Process blocks
    blocks = data.get("blocks", [])
    for block in blocks:
        if block.get("intended_usage") == "ask_text":
            md = block.get("markdown_block", {})
            progress = md.get("progress", "")
            chunks = md.get("chunks", [])
            start_offset = md.get("chunk_starting_offset", 0)
            answer = md.get("answer", "")
            
            if answer:
                full_answer = answer
                
            # Process incremental chunks
            new_chunks = []
            last_idx = tracker.get("last_yielded_idx", -1)
            for i, chunk in enumerate(chunks):
                current_idx = start_offset + i
                if current_idx > last_idx:
                    new_chunks.append(chunk)
                    last_idx = current_idx
            
            if new_chunks:
                delta_text = "".join(new_chunks)
                tracker["last_yielded_idx"] = last_idx
                tracker["full_text"] = tracker.get("full_text", "") + delta_text
                
            if progress == "DONE" and answer:
                tracker["full_text"] = answer
                
    if not delta_text and not is_final and not full_answer:
        return None
        
    return {
        "delta": delta_text,
        "backend_uuid": backend_uuid,
        "context_uuid": context_uuid,
        "done": is_final,
        "full_text": full_answer or tracker.get("full_text", "")
    }

def format_openai_chunk(chunk: dict) -> str:
    """Format a chunk dict as an SSE data line."""
    return f"data: {json.dumps(chunk)}\n\n"

def build_openai_streaming_chunk(
    delta: str,
    model: str,
    chunk_id: str,
    conversation_id: str = "",
    message_id: str = "",
    finish_reason: str | None = None
) -> dict:
    """Build a single OpenAI-compatible streaming completion chunk."""
    choice = {
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason
    }
    
    if delta:
        choice["delta"] = {"content": delta}
        
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice]
    }
    
    if conversation_id:
        chunk["conversation_id"] = conversation_id
    if message_id:
        chunk["parent_message_id"] = message_id
        
    return chunk

def build_non_streaming_response(
    content: str,
    model: str,
    conversation_id: str = "",
    message_id: str = "",
    finish_reason: str = "stop"
) -> dict:
    """Build a complete (non-streaming) OpenAI chat completion response."""
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    
    if conversation_id:
        resp["conversation_id"] = conversation_id
    if message_id:
        resp["parent_message_id"] = message_id
        
    return resp
