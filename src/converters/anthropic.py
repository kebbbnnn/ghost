import json
import re
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

def _parse_invoke_xml(xml_block: str) -> tuple[str, dict]:
    """Parse an XML <invoke name="...">...</invoke> block into (tool_name, parameters)."""
    tool_name = "unknown"
    m_name = re.search(r'name=["\']([^"\']+)["\']', xml_block)
    if m_name:
        tool_name = m_name.group(1)
        
    params = {}
    param_matches = re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s*>(.*?)</parameter>', xml_block, re.DOTALL)
    for match in param_matches:
        p_name = match.group(1)
        p_val_raw = match.group(2).strip()
        try:
            p_val = json.loads(p_val_raw)
        except Exception:
            p_val = p_val_raw
        params[p_name] = p_val
        
    return tool_name, params

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
                b.get("text", "") for b in sys_field if isinstance(b, dict) and b.get("type") == "text"
            )
            
    # Inject tools into system prompt
    tools = req.get("tools", [])
    tool_choice = req.get("tool_choice", {})
    t_choice_type = tool_choice.get("type") if isinstance(tool_choice, dict) else None

    if tools and t_choice_type != "none":
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
        if t_choice_type == "any":
            tool_prompt += "\nIMPORTANT: You MUST call at least one of the available tools."
        elif t_choice_type == "tool":
            target_name = tool_choice.get("name", "")
            if target_name:
                tool_prompt += f'\nIMPORTANT: You MUST call the tool "{target_name}".'
                
        system_text += tool_prompt
        
    if system_text:
        messages.append({"role": "system", "content": system_text.strip()})
        
    # 2. Handle Messages
    for msg in req.get("messages", []):
        role = msg.get("role")
        content_field = msg.get("content", "")
        
        text_content = ""
        image_parts = []
        
        if isinstance(content_field, str):
            text_content = content_field
        elif isinstance(content_field, list):
            for block in content_field:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_content += block.get("text", "") + "\n"
                elif btype == "image":
                    source = block.get("source", {})
                    stype = source.get("type")
                    if stype == "url":
                        img_url = source.get("url", "")
                        if img_url:
                            image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                    elif stype == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        if data:
                            img_url = f"data:{media_type};base64,{data}"
                            image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                elif btype == "tool_use":
                    # Assistant used a tool (in history)
                    tool_name = block.get("name")
                    tool_args = block.get("input", {})
                    
                    # Convert to XML so history matches what model is instructed to output
                    args_xml = ""
                    if isinstance(tool_args, dict):
                        for k, v in tool_args.items():
                            val_str = json.dumps(v) if not isinstance(v, str) else v
                            args_xml += f'  <parameter name="{k}">{val_str}</parameter>\n'
                        
                    text_content += f'<invoke name="{tool_name}">\n{args_xml}</invoke>\n'
                elif btype == "tool_result":
                    # User provided tool result
                    tool_id = block.get("tool_use_id", "")
                    is_err = block.get("is_error", False)
                    err_attr = ' is_error="true"' if is_err else ''
                    res_content = block.get("content", "")
                    
                    res_text = ""
                    if isinstance(res_content, str):
                        res_text = res_content
                    elif isinstance(res_content, list):
                        for rblock in res_content:
                            if not isinstance(rblock, dict):
                                continue
                            rb_type = rblock.get("type")
                            if rb_type == "text":
                                res_text += rblock.get("text", "") + "\n"
                            elif rb_type == "image":
                                source = rblock.get("source", {})
                                stype = source.get("type")
                                if stype == "url":
                                    img_url = source.get("url", "")
                                    if img_url:
                                        image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                                elif stype == "base64":
                                    media_type = source.get("media_type", "image/png")
                                    data = source.get("data", "")
                                    if data:
                                        img_url = f"data:{media_type};base64,{data}"
                                        image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                                        
                    text_content += f'<tool_result id="{tool_id}"{err_attr}>\n{res_text.strip()}\n</tool_result>\n'
                    
        # Construct message content
        if image_parts:
            msg_content = []
            if text_content.strip():
                msg_content.append({"type": "text", "text": text_content.strip()})
            msg_content.extend(image_parts)
            messages.append({"role": role, "content": msg_content})
        else:
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
    if "top_p" in req:
        openai_req["top_p"] = req["top_p"]
    if "stop_sequences" in req:
        openai_req["stop"] = req["stop_sequences"]
        
    return openai_req

async def openai_to_anthropic_stream(openai_stream: AsyncGenerator[str, None], model: str) -> AsyncGenerator[str, None]:
    """Convert OpenAI SSE stream to Anthropic SSE stream, intercepting XML tool calls."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    # Message Start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    block_index = 0
    active_block_type = None  # None, "text", or "tool_use"
    buffer = ""
    tool_called = False
    
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
            
            while buffer:
                if active_block_type != "tool_use":
                    invoke_idx = buffer.find("<invoke")
                    if invoke_idx != -1:
                        text_before = buffer[:invoke_idx]
                        if text_before:
                            if active_block_type != "text":
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                active_block_type = "text"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text_before}})}\n\n"
                            
                        if active_block_type == "text":
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                            active_block_type = None
                            block_index += 1
                            
                        buffer = buffer[invoke_idx:]
                        active_block_type = "tool_use"
                        tool_called = True
                    else:
                        safe_len = max(0, len(buffer) - 7)
                        if safe_len > 0:
                            text_chunk = buffer[:safe_len]
                            buffer = buffer[safe_len:]
                            if active_block_type != "text":
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                active_block_type = "text"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text_chunk}})}\n\n"
                        break
                        
                if active_block_type == "tool_use":
                    end_idx = buffer.find("</invoke>")
                    if end_idx != -1:
                        xml_block = buffer[:end_idx + 9]
                        buffer = buffer[end_idx + 9:]
                        
                        tool_name, params = _parse_invoke_xml(xml_block)
                        tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_use_id, 'name': tool_name, 'input': {}}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(params)}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        
                        active_block_type = None
                        block_index += 1
                    else:
                        break

    # Flush remaining buffer
    if buffer:
        if active_block_type != "text":
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            active_block_type = "text"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': buffer}})}\n\n"
        
    if active_block_type == "text":
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
        active_block_type = None

    stop_reason = "tool_use" if tool_called else "end_turn"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}})}\n\n"
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
        
    content = choices[0].get("message", {}).get("content", "") or ""
    
    blocks = []
    has_tool_call = False
    
    curr = content
    while curr:
        invoke_idx = curr.find("<invoke")
        if invoke_idx != -1:
            text_before = curr[:invoke_idx].strip()
            if text_before:
                blocks.append({"type": "text", "text": text_before})
                
            end_idx = curr.find("</invoke>", invoke_idx)
            if end_idx != -1:
                xml_block = curr[invoke_idx:end_idx + 9]
                curr = curr[end_idx + 9:]
                
                tool_name, params = _parse_invoke_xml(xml_block)
                blocks.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": tool_name,
                    "input": params
                })
                has_tool_call = True
            else:
                # Unclosed <invoke, treat rest as text
                blocks.append({"type": "text", "text": curr})
                break
        else:
            if curr.strip():
                blocks.append({"type": "text", "text": curr.strip()})
            break
            
    if not blocks:
        blocks.append({"type": "text", "text": ""})
        
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "tool_use" if has_tool_call else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }
