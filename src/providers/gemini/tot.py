"""
tot.py — Multi-turn Tree of Thoughts (ToT) engine for GeminiProvider.

Implements a Beam Search (B=3 breadth, D=3 max depth) reasoning pipeline.
Streams intermediate candidate thoughts and evaluations live wrapped in <thought> tags,
and synthesizes the final solution upon search completion.
"""
import json
import logging
import re
import time
import typing

from src.core.exceptions import RateLimitError, ProviderError
from .converter import parse_gemini_response, format_openai_chunk, _split_into_tokens
from src.core.emulation.stream_parser import (
    CLOSED_TOOL_CALL_RE,
    synthesize_workflow_recovery_tool_call,
    transpile_fenced_json_tool_calls,
)
from .converter import (
    is_gemini_canned_refusal,
    parse_gemini_response,
    format_openai_chunk,
    _split_into_tokens,
    transpile_gemini_code_reference,
)

log = logging.getLogger(__name__)


class TreeOfThoughtsEngine:
    """
    Orchestrates a multi-turn Beam Search (B=3, D=3) Tree of Thoughts exploration.
    """

    def __init__(
        self,
        send_func: typing.Callable[..., typing.Awaitable[str]],
        breadth: int = 3,
        max_depth: int = 3,
    ):
        """
        :param send_func: Async callable matching GeminiProvider._send_message(prompt) -> raw_response_str
        :param breadth: Number of candidate thoughts per step (B)
        :param max_depth: Maximum tree depth (D)
        """
        self._send_func = send_func
        self.breadth = breadth
        self.max_depth = max_depth

    async def run(
        self,
        messages: list[dict],
        chunk_id: str,
        model: str = "gemini-tree-of-thoughts",
    ) -> typing.AsyncGenerator[str, None]:
        """
        Executes the Tree of Thoughts search and yields OpenAI-compatible SSE chunks.
        """
        # Yield initial role announcement chunk
        yield self._make_chunk(chunk_id, model, role="assistant")

        user_query = self._extract_user_query(messages)
        selected_path: list[str] = []

        try:
            for depth in range(1, self.max_depth + 1):
                log.info("ToT Step %d/%d starting...", depth, self.max_depth)

                # --- 1. Candidate Generation Phase ---
                gen_prompt = self._build_generation_prompt(
                    user_query, selected_path, depth
                )

                try:
                    raw_gen = await self._send_func(gen_prompt)
                    parsed_gen = parse_gemini_response(raw_gen)
                    candidates_text = parsed_gen.get("content", "").strip()
                except RateLimitError:
                    if depth > 1:
                        msg = f"[ToT] Rate limit reached at step {depth}. Synthesizing best partial solution...\n\n"
                        async for chunk in self._stream_thought(msg, chunk_id, model):
                            yield chunk
                        break
                    raise
                except Exception as exc:
                    if depth > 1:
                        log.warning("ToT step %d gen failed (%s), proceeding to synthesis", depth, exc)
                        msg = f"[ToT] Encountered error at step {depth}. Synthesizing partial solution...\n\n"
                        async for chunk in self._stream_thought(msg, chunk_id, model):
                            yield chunk
                        break
                    raise ProviderError(f"ToT step {depth} generation failed: {exc}")

                candidates = self._parse_candidates(candidates_text)

                # Stream candidate thoughts live as reasoning_content
                thought_gen_text = (
                    f"[Step {depth}/{self.max_depth}] Candidate Thoughts:\n"
                    f"{candidates_text}\n\n"
                )
                async for chunk in self._stream_thought(thought_gen_text, chunk_id, model):
                    yield chunk

                # --- 2. Candidate Evaluation Phase ---
                eval_prompt = self._build_evaluation_prompt(
                    user_query, selected_path, candidates, depth
                )

                try:
                    raw_eval = await self._send_func(eval_prompt)
                    parsed_eval = parse_gemini_response(raw_eval)
                    eval_text = parsed_eval.get("content", "").strip()
                except RateLimitError:
                    if depth > 1:
                        # Fallback to the first parsed candidate
                        best_candidate = candidates[0] if candidates else candidates_text
                        selected_path.append(f"Step {depth}: {best_candidate}")
                        msg = f"[ToT] Rate limit on step {depth} evaluation. Defaulting to top candidate.\n\n"
                        async for chunk in self._stream_thought(msg, chunk_id, model):
                            yield chunk
                        break
                    raise
                except Exception as exc:
                    log.warning("ToT step %d eval failed (%s), selecting first candidate", depth, exc)
                    eval_text = "Evaluation skipped due to temporary response parsing issue."

                best_idx, best_thought = self._select_winning_candidate(
                    candidates, eval_text
                )
                selected_path.append(f"Step {depth}: {best_thought}")

                # Stream evaluation live as reasoning_content
                thought_eval_text = (
                    f"[Step {depth}/{self.max_depth}] Evaluation & Selection:\n"
                    f"{eval_text}\n\n"
                    f"Selected Branch: {best_thought}\n\n"
                )
                async for chunk in self._stream_thought(thought_eval_text, chunk_id, model):
                    yield chunk

            # --- 3. Final Answer Synthesis Phase ---
            synthesis_prompt = self._build_synthesis_prompt(user_query, selected_path)
            full_context, requires_tool_call = self._extract_full_synthesis_context(messages)
            synthesis_prompt = self._build_synthesis_prompt(
                full_context, selected_path, requires_tool_call=requires_tool_call
            )
            try:
                raw_synth = await self._send_func(synthesis_prompt)
                parsed_synth = parse_gemini_response(raw_synth)
                final_answer = parsed_synth.get("content", "").strip()
                final_answer = transpile_fenced_json_tool_calls(final_answer)
                if requires_tool_call and "```python?code_reference" in final_answer:
                    final_answer = transpile_gemini_code_reference(final_answer, synthesis_prompt)
                if requires_tool_call and (
                    is_gemini_canned_refusal(final_answer)
                    or not CLOSED_TOOL_CALL_RE.search(final_answer)
                ):
                    recovered = synthesize_workflow_recovery_tool_call(synthesis_prompt, messages)
                    if recovered:
                        final_answer = recovered
            except Exception as exc:
                log.error("ToT synthesis call failed: %s", exc)
                # Fallback to presenting the accumulated reasoning steps directly
                final_answer = (
                    "Based on the reasoning path explored:\n\n"
                    + "\n\n".join(selected_path)
                )

            # Stream final answer content chunks (outside <thought> block)
            async for chunk in self._stream_text(final_answer, chunk_id, model):
                yield chunk

        except RateLimitError:
            error_chunk = {
                "error": {
                    "message": "Rate limited by Gemini during Tree of Thoughts search",
                    "type": "rate_limit_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        except Exception as exc:
            log.exception("Tree of Thoughts execution error")
            error_chunk = {
                "error": {"message": str(exc), "type": "server_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # Final finish chunk
        yield self._make_chunk(chunk_id, model, finish_reason="stop")

    # ------------------------------------------------------------------
    # Prompt Construction Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_content_to_str(content: typing.Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content or "")

    def _extract_user_query(self, messages: list[dict]) -> str:
        """Extract the full conversation context or primary user question."""
        """Extract the conversation context for thought generation/evaluation (stripping tool-call enforcement)."""
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                user_parts.append(content)
            elif role == "system":
                user_parts.insert(0, f"System Instruction: {content}")
            content_str = self._msg_content_to_str(msg.get("content", ""))
            if not content_str:
                continue
            if role == "system":
                # Strip [SYSTEM: HEADLESS API GATEWAY] tool enforcement during intermediate thought steps
                gw_idx = content_str.find("[SYSTEM: HEADLESS API GATEWAY]")
                clean_sys = content_str[:gw_idx].strip() if gw_idx != -1 else content_str.strip()
                if clean_sys:
                    user_parts.insert(0, f"System Instruction: {clean_sys}")
            elif role == "assistant":
                user_parts.append(f"[Previous Assistant Turn]:\n{content_str[-2000:]}")
            elif role == "user":
                user_parts.append(content_str[-4000:] if len(content_str) > 4000 else content_str)
        return "\n\n".join(user_parts) if user_parts else "Hello"

    def _extract_full_synthesis_context(self, messages: list[dict]) -> tuple[str, bool]:
        """Extract full context (including tool directives) for the final synthesis phase."""
        parts = []
        requires_tool_call = False
        for msg in messages:
            role = msg.get("role", "user")
            content_str = self._msg_content_to_str(msg.get("content", ""))
            if not content_str:
                continue
            if "CRITICAL MODE: TOOL_CALL_REQUIRED" in content_str:
                requires_tool_call = True
            if role == "system":
                parts.insert(0, f"System Instruction: {content_str}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content_str}")
            elif role == "user":
                parts.append(f"[User]: {content_str}" if len(messages) > 2 else content_str)
        return ("\n\n".join(parts) if parts else "Hello"), requires_tool_call

    def _build_generation_prompt(
        self, query: str, path: list[str], depth: int
    ) -> str:
        path_str = "\n".join(path) if path else "None (starting initial step)."
        return (
            f"You are executing a Tree of Thoughts problem-solving algorithm.\n"
            f"Original Question:\n{query}\n\n"
            f"Selected Reasoning Path So Far:\n{path_str}\n\n"
            f"Task for Step {depth}/{self.max_depth}:\n"
            f"Propose exactly {self.breadth} distinct, logical next-step thoughts to advance toward solving the question.\n"
            f"Do NOT emit <tool_call> or <invoke> XML tags in this planning step.\n"
            f"Format your response strictly as:\n"
            f"Thought 1: <description of first thought option>\n"
            f"Thought 2: <description of second thought option>\n"
            f"Thought 3: <description of third thought option>\n"
        )

    def _build_evaluation_prompt(
        self, query: str, path: list[str], candidates: list[str], depth: int
    ) -> str:
        candidates_str = "\n\n".join(
            f"Candidate {i+1}:\n{cand}" for i, cand in enumerate(candidates)
        )
        return (
            f"You are evaluating candidate thought steps in a Tree of Thoughts algorithm.\n"
            f"Original Question:\n{query}\n\n"
            f"Candidates for Step {depth}/{self.max_depth}:\n{candidates_str}\n\n"
            f"Task:\n"
            f"1. Evaluate each candidate for logical correctness, feasibility, and usefulness.\n"
            f"2. Assign a numerical score from 1 to 10 for each candidate.\n"
            f"3. State which candidate is the best choice and why.\n"
            f"Do NOT emit <tool_call> or <invoke> XML tags in this evaluation step.\n"
        )

    def _build_synthesis_prompt(self, query: str, path: list[str]) -> str:
    def _build_synthesis_prompt(
        self, query: str, path: list[str], requires_tool_call: bool = False
    ) -> str:
        path_str = "\n\n".join(path)
        if requires_tool_call:
            return (
                f"{query}\n\n"
                f"[TREE OF THOUGHTS WINNING REASONING PATH]:\n{path_str}\n\n"
                f"[FINAL SYNTHESIS & TOOL EXECUTION DIRECTIVE]:\n"
                f"Guided by the Winning Reasoning Path above, execute the next required action for the user's request "
                f"right now by emitting the required `<invoke>` or `<tool_call>` XML blocks directly."
            )
        return (
            f"You have completed a Tree of Thoughts reasoning search for the following question.\n"
            f"Original Question:\n{query}\n\n"
            f"Winning Reasoning Path:\n{path_str}\n\n"
            f"Task:\n"
            f"Synthesize a clear, detailed, and complete final answer to the user's original question "
            f"based on the winning reasoning path above."
        )

    # ------------------------------------------------------------------
    # Parsing Helpers
    # ------------------------------------------------------------------

    def _parse_candidates(self, text: str) -> list[str]:
        """Extract candidate thoughts from response text."""
        pattern = r"(?:Thought|Candidate)\s*\d+:\s*(.*?)(?=(?:Thought|Candidate)\s*\d+:|$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        cleaned = [m.strip() for m in matches if m.strip()]
        if len(cleaned) >= self.breadth:
            return cleaned[: self.breadth]
        # Fallback: split by lines or numbered lists if pattern didn't match cleanly
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) >= self.breadth:
            return lines[: self.breadth]
        return [text]

    def _select_winning_candidate(
        self, candidates: list[str], eval_text: str
    ) -> tuple[int, str]:
        """Determine the index and text of the top-rated candidate."""
        if not candidates:
            return 0, "Proceed with logical analysis."

        # Search for scores like Candidate 1: 9/10, Score: 8/10, etc.
        scores = {}
        for i in range(len(candidates)):
            cand_num = i + 1
            # Look for scores near Candidate N mention
            cand_pattern = rf"(?:Candidate|Thought)\s*{cand_num}.*?(\b[1-9]|10\b)\s*/\s*10"
            match = re.search(cand_pattern, eval_text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    scores[i] = int(match.group(1))
                except ValueError:
                    pass

        if scores:
            best_idx = max(scores, key=scores.get)
            return best_idx, candidates[best_idx]

        # Default to the first candidate if scoring couldn't be parsed
        return 0, candidates[0]

    # ------------------------------------------------------------------
    # Streaming Chunk Formatting Helpers
    # ------------------------------------------------------------------

    async def _stream_text(
        self, text: str, chunk_id: str, model: str
    ) -> typing.AsyncGenerator[str, None]:
        """Yield word-by-word streaming chunks for smooth real-time delivery."""
        tokens = _split_into_tokens(text)
        for token in tokens:
            yield self._make_chunk(chunk_id, model, content=token)

    async def _stream_thought(
        self, text: str, chunk_id: str, model: str
    ) -> typing.AsyncGenerator[str, None]:
        """Yield word-by-word streaming chunks with reasoning_content."""
        tokens = _split_into_tokens(text)
        for token in tokens:
            yield self._make_chunk(chunk_id, model, reasoning_content=token)

    def _stream_thought_header(
        self, chunk_id: str, model: str, message: str
    ) -> str:
        return self._make_chunk(chunk_id, model, reasoning_content=f"{message}\n\n")

    def _make_chunk(
        self,
        chunk_id: str,
        model: str,
        content: str | None = None,
        reasoning_content: str | None = None,
        role: str | None = None,
        finish_reason: str | None = None,
    ) -> str:
        delta = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content

        chunk_dict = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return format_openai_chunk(chunk_dict)
