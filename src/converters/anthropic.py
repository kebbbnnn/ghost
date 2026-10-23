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
    """Parse an XML <invoke name="...">...</invoke> or <tool_call>...</tool_call> block into (tool_name, parameters)."""
    # 1. Handle <tool_call>{"name": "...", "arguments": {...}}</tool_call> format
    if "<tool_call>" in xml_block:
        clean = xml_block.replace("<tool_call>", "").replace("</tool_call>", "").strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        clean = re.sub(r",\s*(\}|\])", r"\1", clean)

        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                name = data.get("name", "unknown")
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                return name, args if isinstance(args, dict) else {}
        except Exception:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', clean)
            if name_match:
                func_name = name_match.group(1)
                args_match = re.search(r'"arguments"\s*:\s*(\{.*\}|\[.*\]|"[^"]*")', clean, re.DOTALL)
                if args_match:
                    try:
                        args = json.loads(args_match.group(1))
                        return func_name, args if isinstance(args, dict) else {}
                    except Exception:
                        return func_name, {}
                return func_name, {}

    # 2. Handle <invoke name="...">...</invoke> format
    tool_name = "unknown"
    m_name = re.search(r'name=["\']([^"\']+)["\']', xml_block)
    if m_name:
        tool_name = m_name.group(1)

    params: dict = {}
    param_matches = list(re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s*>(.*?)</parameter>', xml_block, re.DOTALL))
    if param_matches:
        for match in param_matches:
            p_name = match.group(1)
            p_val_raw = match.group(2).strip()
            try:
                p_val = json.loads(p_val_raw)
            except Exception:
                p_val = p_val_raw
            params[p_name] = p_val
    else:
        # Check if raw JSON was embedded inside <invoke ...>{...}</invoke>
        inner_match = re.search(r'>\s*(\{.*\})\s*</invoke>', xml_block, re.DOTALL)
        if inner_match:
            try:
                parsed_json = json.loads(inner_match.group(1))
                if isinstance(parsed_json, dict):
                    params = parsed_json
            except Exception:
                pass

    return tool_name, params


def anthropic_to_openai_request(req: dict) -> dict:
    """Convert Anthropic Messages API request to OpenAI chat completions format with parameter emulation."""
    if "messages" not in req or not isinstance(req["messages"], list):
        raise ValueError("'messages' is a required field and must be a list")

    messages = []
    total_input_chars = 0

    # 1. Handle System Prompt & Tools
    system_text = ""

    # Anthropic system can be a string or array of blocks
    sys_field = req.get("system")
    if sys_field:
        if isinstance(sys_field, str):
            system_text += sys_field
            total_input_chars += len(sys_field)
        elif isinstance(sys_field, list):
            for b in sys_field:
                if isinstance(b, dict) and b.get("type") == "text":
                    txt = b.get("text", "")
                    system_text += txt + "\n"
                    total_input_chars += len(txt)

    # Detect if the conversation is currently in synthesis phase awaiting answer to pending tool results
    raw_msgs = req.get("messages", [])
    last_msg = raw_msgs[-1] if raw_msgs else {}
    last_content = last_msg.get("content", "")

    has_pending_tool_results = False
    if last_msg.get("role") == "user":
        if isinstance(last_content, list):
            has_pending_tool_results = any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in last_content
            )
        elif isinstance(last_content, str):
            has_pending_tool_results = "<tool_result" in last_content

    tools = req.get("tools", [])
    tool_choice = req.get("tool_choice", {})
    t_choice_type = tool_choice.get("type") if isinstance(tool_choice, dict) else None
    is_synthesis_phase = bool(tools and has_pending_tool_results and t_choice_type not in ("any", "tool"))

    if tools and t_choice_type != "none":
        tools_xml = _build_tools_xml(tools)
        if is_synthesis_phase:
            tool_prompt = (
                "\n\n[SYSTEM: HEADLESS API GATEWAY]\n"
                "[SYNTHESIS PHASE]\n"
                "External tool execution is COMPLETE. The results are provided above in <tool_result> blocks.\n"
                "CRITICAL RULES:\n"
                "1. Provide the direct, complete, and final answer to the user's request based on the tool results.\n"
                "2. DO NOT invoke or emit duplicate tools unless additional tool execution is strictly required.\n"
                "3. Output the final answer directly in clear, helpful text.\n\n"
                f"{tools_xml}"
            )
        else:
            tool_prompt = (
                "\n\n[SYSTEM: HEADLESS API GATEWAY]\n"
                "CRITICAL MODE: TOOL_CALL_REQUIRED.\n"
                "You are an automated API proxy running in a sandbox environment with NO web access, NO python access, and NO UI widgets.\n"
                "DO NOT attempt to search the web, execute internal scripts, or render widgets/card links. Any direct markdown answer before required tool execution is invalid.\n"
                "You have access to the following tools:\n"
                f"{tools_xml}\n\n"
                "Your task: For any user request that requires calculations, external data, weather, time, or actions, you MUST emit tool invocation blocks.\n"
                "MANDATORY PROTOCOL RULES:\n"
                "1. Even if the user repeats a query or asks something already discussed in the chat history, you MUST emit fresh tool invocation blocks.\n"
                "2. NEVER answer directly using previous conversation context or Google search cards when tools are available.\n"
                "3. You may emit either:\n"
                '<tool_call>{"name": "tool_name", "arguments": {"param_name": "value"}}</tool_call>\n'
                "or:\n"
                '<invoke name="tool_name">\n'
                '  <parameter name="param_name">value</parameter>\n'
                '</invoke>\n\n'
                "If the user query requires multiple operations or questions, you can invoke multiple tools in succession.\n"
                "Output ONLY the tool invocation blocks, nothing else."
            )
            if t_choice_type == "any":
                tool_prompt += "\nIMPORTANT: You MUST call at least one of the available tools."
            elif t_choice_type == "tool":
                target_name = tool_choice.get("name", "")
                if target_name:
                    tool_prompt += f'\nIMPORTANT: You MUST call the tool "{target_name}".'

        system_text += tool_prompt

    # Behavioral steering from temperature parameter
    temperature = req.get("temperature")
    if temperature is not None:
        if temperature <= 0.3:
            system_text += "\n\nBehavioral Directive: Be strictly factual, precise, concise, and logically grounded."
        elif temperature >= 0.8:
            system_text += "\n\nBehavioral Directive: Be creative, explorative, and offer diverse alternatives."

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
            total_input_chars += len(content_field)
        elif isinstance(content_field, list):
            for block in content_field:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    txt = block.get("text", "")
                    text_content += txt + "\n"
                    total_input_chars += len(txt)
                elif btype == "image":
                    source = block.get("source", {})
                    stype = source.get("type")
                    if stype == "url":
                        img_url = source.get("url", "")
                        if img_url:
                            image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                            total_input_chars += 400
                    elif stype == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        if data:
                            img_url = f"data:{media_type};base64,{data}"
                            image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                            total_input_chars += 400
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

                    tool_repr = f'<invoke name="{tool_name}">\n{args_xml}</invoke>\n'
                    text_content += tool_repr
                    total_input_chars += len(tool_repr)
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
                                        total_input_chars += 400
                                elif stype == "base64":
                                    media_type = source.get("media_type", "image/png")
                                    data = source.get("data", "")
                                    if data:
                                        img_url = f"data:{media_type};base64,{data}"
                                        image_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                                        total_input_chars += 400

                    result_repr = f'<tool_result id="{tool_id}"{err_attr}>\n{res_text.strip()}\n</tool_result>\n'
                    text_content += result_repr
                    total_input_chars += len(result_repr)

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
        "stream": req.get("stream", False),
        "_estimated_input_tokens": max(1, total_input_chars // 4),
        "_is_synthesis_phase": is_synthesis_phase,
    }

    if "max_tokens" in req:
        openai_req["max_tokens"] = req["max_tokens"]
    if "temperature" in req:
        openai_req["temperature"] = req["temperature"]
    if "top_p" in req:
        openai_req["top_p"] = req["top_p"]
    if "stop_sequences" in req:
        openai_req["stop"] = req["stop_sequences"]
    if "tools" in req:
        openai_req["tools"] = req["tools"]
    if "tool_choice" in req:
        openai_req["tool_choice"] = req["tool_choice"]

    return openai_req


async def openai_to_anthropic_stream(
    openai_stream: AsyncGenerator[str, None],
    model: str,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    input_tokens: int = 0,
    session_metadata: dict | None = None,
    expose_session_metadata_in_body: bool = False,
) -> AsyncGenerator[str, None]:
    """Convert OpenAI SSE stream to Anthropic SSE stream, intercepting tool calls, enforcing max_tokens and stop_sequences."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Normalize stop sequences
    normalized_stops: list[str] = []
    if stop_sequences:
        normalized_stops = [s for s in stop_sequences if isinstance(s, str) and s]

    # Message Start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 1}}})}\n\n"
    yield 'event: ping\ndata: {"type": "ping"}\n\n'

    block_index = 0
    active_block_type = None  # None, "text", or "tool_use"
    buffer = ""
    tool_called = False
    stop_triggered = False
    length_triggered = False
    matched_stop_sequence = None
    output_tokens_accumulated = 0
    captured_conversation_id: str | None = None
    captured_parent_message_id: str | None = None

    OPEN_TAGS = ["<invoke", "<tool_call>"]

    async for chunk_line in openai_stream:
        if stop_triggered or length_triggered:
            break

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

        if "conversation_id" in data and data["conversation_id"]:
            captured_conversation_id = data["conversation_id"]
            if session_metadata is not None:
                session_metadata["conversation_id"] = captured_conversation_id
        if "parent_message_id" in data and data["parent_message_id"]:
            captured_parent_message_id = data["parent_message_id"]
            if session_metadata is not None:
                session_metadata["parent_message_id"] = captured_parent_message_id

        choices = data.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})

        # Handle provider stream delivering choices[0].delta.tool_calls directly
        if "tool_calls" in delta and delta["tool_calls"]:
            for tc in delta["tool_calls"]:
                func = tc.get("function", {})
                fn_name = func.get("name", "unknown")
                args_str = func.get("arguments", "{}")
                try:
                    args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args_obj = {}

                tool_use_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                if active_block_type == "text":
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    active_block_type = None
                    block_index += 1

                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_use_id, 'name': fn_name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(args_obj)}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                block_index += 1
                tool_called = True
                output_tokens_accumulated += max(1, len(args_str) // 4)
            continue

        content = delta.get("content")
        if content:
            buffer += content

            while buffer:
                if active_block_type != "tool_use":
                    # Check for earliest opening tool tag
                    found_tag = None
                    tag_idx = -1
                    for t in OPEN_TAGS:
                        idx = buffer.find(t)
                        if idx != -1 and (tag_idx == -1 or idx < tag_idx):
                            tag_idx = idx
                            found_tag = t

                    if tag_idx != -1:
                        text_before = buffer[:tag_idx]
                        if text_before:
                            # Enforce stop sequences on text_before
                            for s in normalized_stops:
                                if s in text_before:
                                    s_idx = text_before.find(s)
                                    text_before = text_before[:s_idx]
                                    stop_triggered = True
                                    matched_stop_sequence = s
                                    break

                            # Enforce max_tokens on text_before
                            if max_tokens is not None and not stop_triggered:
                                est_tokens = max(1, len(text_before) // 4)
                                if output_tokens_accumulated + est_tokens >= max_tokens:
                                    rem = max(0, max_tokens - output_tokens_accumulated)
                                    text_before = text_before[:rem * 4]
                                    length_triggered = True

                            if text_before and text_before.strip():
                                if active_block_type != "text":
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                    active_block_type = "text"
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text_before}})}\n\n"
                                output_tokens_accumulated += max(1, len(text_before) // 4)

                        if active_block_type == "text":
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                            active_block_type = None
                            block_index += 1

                        if stop_triggered or length_triggered:
                            buffer = ""
                            break

                        buffer = buffer[tag_idx:]
                        active_block_type = "tool_use"
                    else:
                        # Check lookahead prefix match across all open tags
                        max_prefix_len = 0
                        for t in OPEN_TAGS:
                            for l in range(min(len(t) - 1, len(buffer)), 0, -1):
                                if t.startswith(buffer[-l:]):
                                    if l > max_prefix_len:
                                        max_prefix_len = l

                        if max_prefix_len > 0:
                            safe_text = buffer[:-max_prefix_len]
                            buffer = buffer[-max_prefix_len:]
                        else:
                            safe_text = buffer
                            buffer = ""

                        if safe_text:
                            # Stop sequences enforcement
                            for s in normalized_stops:
                                if s in safe_text:
                                    s_idx = safe_text.find(s)
                                    safe_text = safe_text[:s_idx]
                                    stop_triggered = True
                                    matched_stop_sequence = s
                                    break

                            # Max tokens enforcement
                            if max_tokens is not None and not stop_triggered:
                                est_tokens = max(1, len(safe_text) // 4)
                                if output_tokens_accumulated + est_tokens >= max_tokens:
                                    rem = max(0, max_tokens - output_tokens_accumulated)
                                    safe_text = safe_text[:rem * 4]
                                    length_triggered = True

                            if safe_text:
                                if active_block_type != "text":
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                    active_block_type = "text"
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': safe_text}})}\n\n"
                                output_tokens_accumulated += max(1, len(safe_text) // 4)

                            if stop_triggered or length_triggered:
                                if active_block_type == "text":
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    active_block_type = None
                                    block_index += 1
                                buffer = ""
                                break

                        break

                if active_block_type == "tool_use":
                    # Determine matching closing tag
                    if buffer.startswith("<tool_call>"):
                        close_tag = "</tool_call>"
                    else:
                        close_tag = "</invoke>"

                    end_idx = buffer.find(close_tag)
                    if end_idx != -1:
                        xml_block = buffer[:end_idx + len(close_tag)]
                        buffer = buffer[end_idx + len(close_tag):]

                        tool_name, params = _parse_invoke_xml(xml_block)
                        tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"

                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_use_id, 'name': tool_name, 'input': {}}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(params)}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

                        active_block_type = None
                        block_index += 1
                        tool_called = True
                        output_tokens_accumulated += max(1, len(xml_block) // 4)
                    else:
                        break

    # Flush remaining buffer
    if buffer and not stop_triggered and not length_triggered:
        if active_block_type == "tool_use":
            # Attempt to parse unclosed tool block
            tool_name, params = _parse_invoke_xml(buffer)
            if tool_name != "unknown":
                tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_use_id, 'name': tool_name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(params)}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                tool_called = True
                active_block_type = None
            else:
                active_block_type = "text"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': buffer}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                active_block_type = None
        else:
            if active_block_type != "text":
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                active_block_type = "text"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': buffer}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            active_block_type = None
    elif active_block_type == "text":
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
        active_block_type = None

    if stop_triggered:
        stop_reason = "stop_sequence"
    elif length_triggered:
        stop_reason = "max_tokens"
    elif tool_called:
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    final_output_tokens = max(1, output_tokens_accumulated)
    delta_payload = {
        'type': 'message_delta',
        'delta': {'stop_reason': stop_reason, 'stop_sequence': matched_stop_sequence},
        'usage': {'output_tokens': final_output_tokens}
    }
    if expose_session_metadata_in_body:
        if captured_conversation_id:
            delta_payload['conversation_id'] = captured_conversation_id
        if captured_parent_message_id:
            delta_payload['parent_message_id'] = captured_parent_message_id
    yield f"event: message_delta\ndata: {json.dumps(delta_payload)}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


async def openai_to_anthropic_json(
    openai_resp: dict,
    model: str | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    input_tokens: int = 0,
    session_metadata: dict | None = None,
    expose_session_metadata_in_body: bool = False,
) -> dict:
    """Convert OpenAI JSON response to Anthropic JSON response with support for tools, stop_sequences, and max_tokens."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = model or openai_resp.get("model", "unknown")

    if "error" in openai_resp:
        err = openai_resp["error"]
        err_type = err.get("type", "api_error")
        valid_anthropic_error_types = {
            "invalid_request_error", "authentication_error", "permission_error",
            "not_found_error", "request_too_large", "rate_limit_error",
            "api_error", "overloaded_error"
        }
        if err_type not in valid_anthropic_error_types:
            err_type = "rate_limit_error" if "rate" in str(err).lower() else "api_error"
        return {
            "type": "error",
            "error": {
                "type": err_type,
                "message": err.get("message", "Unknown error")
            }
        }

    choices = openai_resp.get("choices", [])
    if not choices:
        return {"type": "error", "error": {"type": "api_error", "message": "No choices returned"}}

    msg_obj = choices[0].get("message", {})
    content = msg_obj.get("content", "") or ""
    tool_calls = msg_obj.get("tool_calls", [])

    blocks = []
    has_tool_call = False
    stop_triggered = False
    length_triggered = False
    matched_stop_sequence = None

    normalized_stops = [s for s in (stop_sequences or []) if isinstance(s, str) and s]

    # 1. Handle native OpenAI tool_calls if already parsed by provider
    if tool_calls and isinstance(tool_calls, list):
        if content.strip():
            blocks.append({"type": "text", "text": content.strip()})
        for tc in tool_calls:
            tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
            fn = tc.get("function", {})
            fn_name = fn.get("name", "unknown")
            args_str = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                parsed_args = {}
            blocks.append({
                "type": "tool_use",
                "id": tc_id,
                "name": fn_name,
                "input": parsed_args
            })
            has_tool_call = True

    # 2. Handle XML <invoke> or <tool_call> in content string
    elif "<invoke" in content or "<tool_call>" in content:
        curr = content
        while curr:
            invoke_idx = curr.find("<invoke")
            tool_idx = curr.find("<tool_call>")
            earliest_idx = -1
            chosen_tag = None
            if invoke_idx != -1 and (earliest_idx == -1 or invoke_idx < earliest_idx):
                earliest_idx = invoke_idx
                chosen_tag = "invoke"
            if tool_idx != -1 and (earliest_idx == -1 or tool_idx < earliest_idx):
                earliest_idx = tool_idx
                chosen_tag = "tool_call"

            if earliest_idx != -1 and chosen_tag:
                text_before = curr[:earliest_idx].strip()
                if text_before:
                    blocks.append({"type": "text", "text": text_before})

                close_tag = "</invoke>" if chosen_tag == "invoke" else "</tool_call>"
                end_idx = curr.find(close_tag, earliest_idx)
                if end_idx != -1:
                    xml_block = curr[earliest_idx:end_idx + len(close_tag)]
                    curr = curr[end_idx + len(close_tag):]

                    tool_name, params = _parse_invoke_xml(xml_block)
                    blocks.append({
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": tool_name,
                        "input": params
                    })
                    has_tool_call = True
                else:
                    blocks.append({"type": "text", "text": curr})
                    break
            else:
                if curr.strip():
                    blocks.append({"type": "text", "text": curr.strip()})
                break
    else:
        # Normal text response: check stop_sequences and max_tokens
        text_resp = content
        for s in normalized_stops:
            if s in text_resp:
                idx = text_resp.find(s)
                text_resp = text_resp[:idx]
                stop_triggered = True
                matched_stop_sequence = s
                break

        if max_tokens is not None and not stop_triggered:
            if len(text_resp) // 4 >= max_tokens:
                text_resp = text_resp[:max_tokens * 4]
                length_triggered = True

        blocks.append({"type": "text", "text": text_resp})

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if stop_triggered:
        stop_reason = "stop_sequence"
    elif length_triggered:
        stop_reason = "max_tokens"
    elif has_tool_call:
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    total_chars = sum(len(b.get("text", "")) + len(json.dumps(b.get("input", {}))) for b in blocks)
    output_tokens = max(1, total_chars // 4)

    resp = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
    }
    if session_metadata is not None:
        if openai_resp.get("conversation_id"):
            session_metadata["conversation_id"] = openai_resp["conversation_id"]
        if openai_resp.get("parent_message_id"):
            session_metadata["parent_message_id"] = openai_resp["parent_message_id"]

    if expose_session_metadata_in_body:
        if openai_resp.get("conversation_id"):
            resp["conversation_id"] = openai_resp["conversation_id"]
        if openai_resp.get("parent_message_id"):
            resp["parent_message_id"] = openai_resp["parent_message_id"]
    return resp

