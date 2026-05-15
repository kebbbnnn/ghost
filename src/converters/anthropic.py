import json
import uuid
from typing import AsyncGenerator

def _build_tools_xml(tools: list) -> str:
    xml_parts = ["<tools>"]
    for t in tools:
        if "name" not in t:
            continue
        xml_parts.append(f'  <tool name="{t["name"]}">')
        if "description" in t:
            xml_parts.append(f'    <description>{t["description"]}</description>')
        if "input_schema" in t:
            props = t["input_schema"].get("properties", {})
            reqs = t["input_schema"].get("required", [])
            xml_parts.append('    <parameters>')
            for pname, pdef in props.items():
                ptype = pdef.get("type", "string")
                pdesc = pdef.get("description", "")
                req_str = "required" if pname in reqs else "optional"
                xml_parts.append(f'      <parameter name="{pname}" type="{ptype}" required="{req_str}">{pdesc}</parameter>')
            xml_parts.append('    </parameters>')
        xml_parts.append('  </tool>')
    xml_parts.append("</tools>")
    return "\n".join(xml_parts)

def anthropic_to_openai_request(req: dict) -> dict:
    """Convert Anthropic Messages API request to OpenAI chat completions format."""
    messages = []
    
    # 1. Handle System Prompt & Tools
    system_text = ""
    
    # Anthropic system can be a string or array of blocks
    sys_field = req.get("system")
    if sys_field:
        if isinstance(sys_field, str):
            system_text += sys_field
        elif isinstance(sys_field, list):
            system_text += "\n".join(
                b.get("text", "") for b in sys_field if b.get("type") == "text"
            )
            
    # Inject tools into system prompt
    tools = req.get("tools", [])
    if tools:
        tools_xml = _build_tools_xml(tools)
        tool_prompt = (
            "\n\nYou have access to the following tools:\n"
            f"{tools_xml}\n\n"
            "To use a tool, you MUST output an XML block like this:\n"
            '<invoke name="tool_name">\n'
            '  <parameter name="param_name">value</parameter>\n'
            '</invoke>\n\n'
            "When you use a tool, output ONLY the <invoke> block. "
            "You cannot call multiple tools at once. "
            "If you do not need to use a tool, simply answer the user directly."
        )
        system_text += tool_prompt
        
    if system_text:
        messages.append({"role": "system", "content": system_text.strip()})
        
    # 2. Handle Messages
    for msg in req.get("messages", []):
        role = msg.get("role")
        content_field = msg.get("content", "")
        
        text_content = ""
        if isinstance(content_field, str):
            text_content = content_field
        elif isinstance(content_field, list):
            # Handle user text and tool results
            for block in content_field:
                btype = block.get("type")
                if btype == "text":
                    text_content += block.get("text", "") + "\n"
                elif btype == "tool_use":
                    # Assistant used a tool (in history)
                    tool_name = block.get("name")
                    tool_args = block.get("input", {})
                    
                    # Convert to XML so history matches what model is instructed to output
                    args_xml = ""
                    for k, v in tool_args.items():
                        args_xml += f'  <parameter name="{k}">{json.dumps(v) if not isinstance(v, str) else v}</parameter>\n'
                        
                    text_content += f'<invoke name="{tool_name}">\n{args_xml}</invoke>\n'
                elif btype == "tool_result":
                    # User provided tool result
                    tool_id = block.get("tool_use_id")
                    res_content = block.get("content", "")
                    text_content += f'<tool_result id="{tool_id}">\n{res_content}\n</tool_result>\n'
                    
        messages.append({"role": role, "content": text_content.strip()})
        
    # 3. Build OpenAI Request
    openai_req = {
        "model": req.get("model", "auto"),
        "messages": messages,
        "stream": req.get("stream", False)
    }
    
    if "max_tokens" in req:
        openai_req["max_tokens"] = req["max_tokens"]
    if "temperature" in req:
        openai_req["temperature"] = req["temperature"]
        
    return openai_req

async def openai_to_anthropic_stream(openai_stream: AsyncGenerator[str, None], model: str) -> AsyncGenerator[str, None]:
    """Convert OpenAI SSE stream to Anthropic SSE stream, intercepting XML tool calls."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    # 1. Message Start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    # 2. Content Block Start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    buffer = ""
    in_invoke = False
    
    async for chunk_line in openai_stream:
        if not chunk_line.startswith("data: "):
            continue
            
        data_str = chunk_line[6:].strip()
        if data_str == "[DONE]":
            break
            
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
            
        if "error" in data:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': data['error'].get('message', 'Unknown error')}})}\n\n"
            return
            
        choices = data.get("choices", [])
        if not choices:
            continue
            
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        
        if content:
            buffer += content
            
            # Streaming XML parser state machine
            if not in_invoke:
                invoke_idx = buffer.find("<invoke")
                if invoke_idx != -1:
                    # Flush text before <invoke as standard text block
                    text_before = buffer[:invoke_idx]
                    if text_before:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text_before}})}\n\n"
                    
                    buffer = buffer[invoke_idx:]
                    in_invoke = True
                    
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    
                else:
                    # Safe to flush everything except the very end which might be partial "<inv"
                    safe_len = max(0, len(buffer) - 7)
                    if safe_len > 0:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': buffer[:safe_len]}})}\n\n"
                        buffer = buffer[safe_len:]
                        
            if in_invoke:
                # Look for </invoke>
                end_idx = buffer.find("</invoke>")
                if end_idx != -1:
                    xml_block = buffer[:end_idx + 9]
                    buffer = buffer[end_idx + 9:]
                    in_invoke = False
                    
                    # Parse the XML to extract tool name and params
                    tool_name = "unknown"
                    name_start = xml_block.find('name="')
                    if name_start != -1:
                        name_end = xml_block.find('"', name_start + 6)
                        tool_name = xml_block[name_start+6:name_end]
                        
                    params = {}
                    param_parts = xml_block.split("<parameter")
                    for p in param_parts[1:]:
                        n_start = p.find('name="')
                        if n_start != -1:
                            n_end = p.find('"', n_start + 6)
                            p_name = p[n_start+6:n_end]
                            v_start = p.find('>') + 1
                            v_end = p.find('</parameter>')
                            p_val_raw = p[v_start:v_end].strip()
                            try:
                                p_val = json.loads(p_val_raw)
                            except Exception:
                                p_val = p_val_raw
                            params[p_name] = p_val
                            
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'tool_use', 'id': tool_use_id, 'name': tool_name, 'input': {}}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(params)}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 1})}\n\n"

    # Flush remaining buffer
    if buffer:
        if in_invoke:
            # Malformed, just dump it as text
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 2, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'text_delta', 'text': buffer}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 2})}\n\n"
        else:
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': buffer}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
    # Message Stop
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

async def openai_to_anthropic_json(openai_resp: dict) -> dict:
    """Convert OpenAI JSON response to Anthropic JSON response."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = openai_resp.get("model", "unknown")
    
    if "error" in openai_resp:
        return {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": openai_resp["error"].get("message", "Unknown error")
            }
        }
        
    choices = openai_resp.get("choices", [])
    if not choices:
        return {"type": "error", "error": {"type": "api_error", "message": "No choices returned"}}
        
    content = choices[0].get("message", {}).get("content", "")
    
    blocks = []
    
    invoke_idx = content.find("<invoke")
    if invoke_idx != -1:
        text_before = content[:invoke_idx].strip()
        if text_before:
            blocks.append({"type": "text", "text": text_before})
            
        end_idx = content.find("</invoke>")
        if end_idx != -1:
            xml_block = content[invoke_idx:end_idx + 9]
            
            tool_name = "unknown"
            name_start = xml_block.find('name="')
            if name_start != -1:
                name_end = xml_block.find('"', name_start + 6)
                tool_name = xml_block[name_start+6:name_end]
                
            params = {}
            param_parts = xml_block.split("<parameter")
            for p in param_parts[1:]:
                n_start = p.find('name="')
                if n_start != -1:
                    n_end = p.find('"', n_start + 6)
                    p_name = p[n_start+6:n_end]
                    v_start = p.find('>') + 1
                    v_end = p.find('</parameter>')
                    p_val_raw = p[v_start:v_end].strip()
                    try:
                        p_val = json.loads(p_val_raw)
                    except Exception:
                        p_val = p_val_raw
                    params[p_name] = p_val
                    
            blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tool_name,
                "input": params
            })
    else:
        blocks.append({"type": "text", "text": content})
        
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "tool_use" if invoke_idx != -1 else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }
