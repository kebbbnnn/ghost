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
        folded_messages = cls._fold_messages(raw_messages)

        # 2. Inject tool instructions and behavioral steering into system prompt
        transpiled_messages = cls._inject_system_instructions(
            folded_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

        # Normalize stop sequences to list[str]
        normalized_stop: list[str] = []
        if isinstance(stop, str) and stop:
            normalized_stop = [stop]
        elif isinstance(stop, list):
            normalized_stop = [s for s in stop if isinstance(s, str) and s]

        emulation_context = {
            "model": model,
            "has_tools": bool(tools and tool_choice != "none"),
            "tools": tools,
            "tool_choice": tool_choice,
            "stop": normalized_stop,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        return transpiled_messages, emulation_context

    @classmethod
    def _fold_messages(cls, messages: list[dict]) -> list[dict]:
        """
        Fold agentic message sequences into standard user/assistant/system roles.
        - Assistant messages with tool_calls serialize into <tool_call> tags.
        - Tool role messages fold into user messages formatted as [Tool Result for ...].
        """
        folded: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

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
                        func = tc.get("function", {})
                        func_name = func.get("name", "")
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

            elif role == "tool":
                # Upstream providers reject role='tool'. Fold into user role.
                if tool_call_id:
                    call_tag = tool_call_id if tool_call_id.startswith("call_") else f"call_{tool_call_id}"
                else:
                    call_tag = "unknown_call"
                tool_text = content if isinstance(content, str) else json.dumps(content)
                folded.append({
                    "role": "user",
                    "content": f"[Tool Result for {call_tag}]:\n{tool_text}",
                })

            else:
                # Normal user or system message
                folded.append(dict(msg))

        return folded

    @classmethod
    def _inject_system_instructions(
        cls,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        temperature: float | None = None,
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

            tool_prompt = (
                "# Tool Calling Instructions\n"
                "You have access to the following tools:\n"
                f"{json.dumps(tools_spec, indent=2)}\n\n"
                "To call one or more tools, you MUST format your call using this XML tag:\n"
                "<tool_call>\n"
                '{"name": "<function_name>", "arguments": <arguments_json_object>}\n'
                "</tool_call>\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. If a user query can be answered or assisted by any of the tools above (e.g. calculation, weather, time, search), "
                "you MUST invoke the appropriate tool using the <tool_call> tag. Do NOT answer directly with your internal knowledge or execute python/search internally.\n"
                "2. When invoking tools, your response should be the <tool_call> block (you may include brief reasoning before it).\n"
                "3. If calling multiple tools, provide each inside its own <tool_call>...</tool_call> block.\n"
                "4. When no tool applies, respond with standard text directly."
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
