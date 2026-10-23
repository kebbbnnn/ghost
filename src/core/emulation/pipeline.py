"""
src/core/emulation/pipeline.py — Request preparation and transpilation for OpenAI parameter emulation.
Handles:
  1. Multi-turn tool folding (assistant tool_calls & role: 'tool' results)
  2. System prompt tool injection & behavioral steering (temperature)
  3. Execution context assembly for downstream streaming & non-streaming interception
"""
from __future__ import annotations

import json
import logging
import typing
from typing import Any, AsyncGenerator

from .stream_parser import StreamEmulationProcessor, process_non_streaming_response

log = logging.getLogger(__name__)


class EmulationPipeline:
    """Universal emulation pipeline wrapping standard OpenAI chat completions."""

    @classmethod
    def prepare_request(cls, req: Any) -> tuple[list[dict], dict]:
        """
        Prepares and transpiles an incoming ChatCompletionRequest into upstream-compatible
        messages and an emulation context dictionary.

        Args:
            req: ChatCompletionRequest (Pydantic model or dict)

        Returns:
            (transpiled_messages, emulation_context)
        """
        # Support both Pydantic models and raw dicts
        req_dict = req.model_dump() if hasattr(req, "model_dump") else dict(req)

        raw_messages: list[dict] = req_dict.get("messages", [])
        tools: list[dict] | None = req_dict.get("tools")
        tool_choice: Any = req_dict.get("tool_choice")
        temperature: float | None = req_dict.get("temperature")
        stop: str | list[str] | None = req_dict.get("stop")
        max_tokens: int | None = req_dict.get("max_tokens") or req_dict.get("max_completion_tokens")
        model: str = req_dict.get("model", "auto")

        # 1. Fold multi-turn tool messages
        folded_messages, executed_tool_names = cls._fold_messages(raw_messages)

        # A turn is in synthesis phase IF AND ONLY IF the conversation currently ends
        # with tool results awaiting assistant synthesis (not a new user inquiry).
        has_pending_tool_results = False
        active_executed_tools: set[str] = set()

        if folded_messages:
            last_msg = folded_messages[-1]
            last_content = last_msg.get("content", "")
            if isinstance(last_content, str):
                if last_content.startswith("[Tool Results]:") or "<tool_result" in last_content:
                    has_pending_tool_results = True
                    active_executed_tools = executed_tool_names
            elif isinstance(last_content, list):
                if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in last_content):
                    has_pending_tool_results = True
                    active_executed_tools = executed_tool_names

        is_synthesis_phase = bool(tools and has_pending_tool_results and tool_choice != "required")

        # 2. Inject tool instructions and behavioral steering into system prompt
        transpiled_messages = cls._inject_system_instructions(
            folded_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            is_synthesis_phase=is_synthesis_phase,
        )

        # Normalize stop sequences to list[str]
        normalized_stop: list[str] = []
        if isinstance(stop, str) and stop:
            normalized_stop = [stop]
        elif isinstance(stop, list):
            normalized_stop = [s for s in stop if isinstance(s, str) and s]

        prompt_chars = 0
        for m in raw_messages:
            c = m.get("content")
            if isinstance(c, str):
                prompt_chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and "text" in part:
                        prompt_chars += len(part["text"])
        est_prompt_tokens = max(1, prompt_chars // 4) if prompt_chars else 1

        emulation_context = {
            "model": model,
            "has_tools": bool(tools and tool_choice != "none"),
            "tools": tools,
            "tool_choice": tool_choice,
            "stop": normalized_stop,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "is_synthesis_phase": is_synthesis_phase,
            "executed_tool_names": list(active_executed_tools) if is_synthesis_phase else [],
            "prompt_tokens": est_prompt_tokens,
            "stream_options": req_dict.get("stream_options"),
            "expose_session_metadata_in_chunks": req_dict.get("expose_session_metadata_in_chunks", False),
        }

        return transpiled_messages, emulation_context

    @classmethod
    def _fold_messages(cls, messages: list[dict]) -> tuple[list[dict], set[str]]:
        """
        Fold agentic message sequences into standard user/assistant/system roles.
        - Assistant messages with tool_calls serialize into <tool_call> tags.
        - Consecutive tool messages fold into a single cohesive [Tool Results] user turn.
        Returns (folded_messages, executed_tool_names).
        """
        folded: list[dict] = []
        pending_tool_results: list[dict] = []
        call_id_to_name: dict[str, str] = {}
        executed_tool_names: set[str] = set()

        def flush_tool_results():
            nonlocal pending_tool_results
            if not pending_tool_results:
                return
            formatted_parts = ["[Tool Results]:"]
            for tr in pending_tool_results:
                call_id = tr.get("tool_call_id") or "call"
                fn_name = tr.get("name") or call_id_to_name.get(call_id) or "tool"
                if fn_name != "tool":
                    executed_tool_names.add(fn_name)
                raw_content = tr.get("content", "")
                content_str = raw_content if isinstance(raw_content, str) else json.dumps(raw_content)
                formatted_parts.append(f"- Function {fn_name} ({call_id}):\n{content_str}")
            formatted_parts.append("\nPlease review the tool results above and provide the final answer to the user's request.")
            folded.append({
                "role": "user",
                "content": "\n\n".join(formatted_parts)
            })
            pending_tool_results = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "tool":
                pending_tool_results.append(msg)
                continue

            # Non-tool message: flush any accumulated tool results first
            flush_tool_results()

            if role == "assistant":
                text_content = ""
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, list):
                    # Multimodal content extraction
                    text_parts = [
                        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    text_content = " ".join(text_parts)

                # Append tool calls if present
                if tool_calls and isinstance(tool_calls, list):
                    serialized_calls = []
                    for tc in tool_calls:
                        tc_id = tc.get("id")
                        func = tc.get("function", {})
                        func_name = func.get("name", "")
                        if tc_id and func_name:
                            call_id_to_name[tc_id] = func_name

                        func_args = func.get("arguments", "{}")
                        if isinstance(func_args, str):
                            try:
                                parsed_args = json.loads(func_args)
                            except Exception:
                                parsed_args = func_args
                        else:
                            parsed_args = func_args

                        serialized_calls.append(
                            f'<tool_call>\n{{"name": "{func_name}", "arguments": {json.dumps(parsed_args)}}}\n</tool_call>'
                        )
                    if serialized_calls:
                        tool_block = "\n".join(serialized_calls)
                        text_content = f"{text_content}\n{tool_block}".strip()

                folded.append({"role": "assistant", "content": text_content})

            else:
                # Normal user or system message
                folded.append(dict(msg))

        flush_tool_results()
        return folded, executed_tool_names

    @classmethod
    def _inject_system_instructions(
        cls,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        temperature: float | None = None,
        is_synthesis_phase: bool = False,
    ) -> list[dict]:
        """
        Appends tool calling rules and temperature behavioral steering to the system prompt.
        """
        instructions: list[str] = []

        # 1. Behavioral steering from temperature
        if temperature is not None:
            if temperature <= 0.3:
                instructions.append(
                    "Behavioral Directive: Be strictly factual, precise, concise, and logically grounded."
                )
            elif temperature >= 0.8:
                instructions.append(
                    "Behavioral Directive: Be creative, explorative, and offer diverse alternatives."
                )

        # 2. Tool instructions
        if tools and tool_choice != "none":
            tools_spec = []
            for t in tools:
                if t.get("type") == "function" and "function" in t:
                    tools_spec.append(t["function"])
                elif "name" in t:
                    tools_spec.append(t)

            compact_signatures = []
            for t in tools_spec:
                fn_name = t.get("name", "")
                params = t.get("parameters", {}).get("properties", {})
                param_strs = []
                for p_name, p_info in params.items():
                    p_type = p_info.get("type", "string")
                    param_strs.append(f"{p_name}: {p_type}")
                sig = f"- {fn_name}({', '.join(param_strs)})"
                compact_signatures.append(sig)
            signatures_text = "\n".join(compact_signatures)

            if is_synthesis_phase:
                tool_prompt = (
                    "[SYSTEM: HEADLESS API GATEWAY]\n"
                    "SYNTHESIS PHASE: External tool execution is COMPLETE. The results are provided above in [Tool Results].\n"
                    "DESIRED RESPONSE FORMAT: Respond strictly in standard markdown text only.\n"
                    "Do NOT output GenUI components, interactive React/JSX widgets, or custom XML/HTML tags.\n"
                    "CRITICAL PROTOCOL RULES:\n"
                    "1. Provide the direct, complete, and final answer to the user's inquiry based on the [Tool Results] provided in the conversation.\n"
                    "2. DO NOT invoke or emit any <tool_call> tags. Tool execution has already finished.\n"
                    "3. Output the final answer directly in clear, helpful markdown text."
                )
                instructions.append(tool_prompt)
            else:
                tool_prompt = (
                    "[SYSTEM: HEADLESS API GATEWAY]\n"
                    "CRITICAL MODE: TOOL_CALL_REQUIRED.\n"
                    "You are an automated API proxy that converts user input into JSON RPC tool calls.\n"
                    "You are running in a sandbox environment with NO web access, NO python access, and NO UI widgets.\n"
                    "DO NOT attempt to call web.run or functions.exec or render widgets. Any markdown output is invalid.\n"
                    "DO NOT emit GenUI tags, interactive components, or custom XML/JSX tags.\n"
                    "Available external RPC tools:\n"
                    f"{signatures_text}\n\n"
                    "PARALLEL & MULTI-TOOL CALLS:\n"
                    "If the user request contains multiple tasks or inquiries, you MUST emit multiple <tool_call> tags in sequence.\n"
                    "Example of parallel tool calls:\n"
                    '<tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}}</tool_call>\n'
                    '<tool_call>{"name": "calculator", "arguments": {"expression": "450 * 1.15"}}</tool_call>\n\n'
                    "Your task: Emit the necessary <tool_call> tags for the user's request.\n"
                    "MANDATORY PROTOCOL RULES:\n"
                    "1. Even if the user query repeats or asks for information already answered earlier in the chat history, you MUST emit fresh <tool_call> tags.\n"
                    "2. NEVER answer directly using previous conversation context, web search, or markdown text when tools are available.\n"
                    "3. Output ONLY <tool_call> tags, nothing else:\n"
                    '<tool_call>{"name": "tool_name", "arguments": {"arg": "val"}}</tool_call>'
                )
                instructions.append(tool_prompt)

                # Mandatory tool calling directive
                if tool_choice == "required":
                    instructions.append(
                        "MANDATORY REQUIREMENT: You MUST invoke at least one tool using the <tool_call> tag."
                    )
                elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    target_func = tool_choice.get("function", {}).get("name")
                    if target_func:
                        instructions.append(
                            f"MANDATORY REQUIREMENT: You MUST invoke the tool '{target_func}' using the <tool_call> tag."
                        )

        if not instructions:
            return messages

        combined_instructions = "\n\n".join(instructions)

        # Inject into existing system message or prepend a new one
        result_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system" and not system_found:
                existing_content = msg.get("content", "")
                if isinstance(existing_content, str):
                    new_content = f"{existing_content}\n\n{combined_instructions}".strip()
                else:
                    new_content = f"{combined_instructions}"
                result_messages.append({"role": "system", "content": new_content})
                system_found = True
            else:
                result_messages.append(msg)

        if not system_found:
            result_messages.insert(0, {"role": "system", "content": combined_instructions})

        return result_messages

    @classmethod
    async def wrap_stream(
        cls,
        raw_stream: AsyncGenerator[str, None],
        context: dict,
    ) -> AsyncGenerator[str, None]:
        """
        Intercepts provider SSE stream, extracting tool calls and enforcing parameter limits.
        """
        processor = StreamEmulationProcessor(context)
        async for chunk in processor.process_stream(raw_stream):
            yield chunk

    @classmethod
    def wrap_non_streaming(cls, raw_response: dict, context: dict) -> dict:
        """
        Formats a non-streaming response dictionary, populating message.tool_calls if detected.
        """
        return process_non_streaming_response(raw_response, context)
