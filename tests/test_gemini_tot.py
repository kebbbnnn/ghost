"""
test_gemini_tot.py — Unit tests for Gemini Tree of Thoughts (gemini-tree-of-thoughts) model.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from src.providers.gemini.session import GeminiProvider
from src.providers.gemini.tot import TreeOfThoughtsEngine
from src.core.exceptions import RateLimitError


def helper_make_raw_rpc_response(text: str, conv_id: str = "c_123", resp_id: str = "r_456") -> str:
    """Helper to mock Gemini's batch-RPC raw response text."""
    inner_payload = json.dumps([
        None,
        [conv_id, resp_id],
        None,
        None,
        [["rc_1", [text]]]
    ])
    outer_payload = [["wrb.fr", None, inner_payload]]
    return f")]}}'\n\n100\n{json.dumps(outer_payload)}\n"


def extract_stream_text(raw_sse: str) -> str:
    text = ""
    for line in raw_sse.split("\n"):
        line = line.strip()
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            try:
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    text += content
            except Exception:
                pass
    return text


def test_gemini_provider_get_models_includes_tot():
    async def _test():
        provider = GeminiProvider()
        models = await provider.get_models()
        model_ids = [m["id"] for m in models]
        assert "gemini" in model_ids
        assert "gemini-tree-of-thoughts" in model_ids

    asyncio.run(_test())


def test_tot_engine_full_run():
    async def _test():
        mock_send = AsyncMock()
        mock_send.side_effect = [
            helper_make_raw_rpc_response("Thought 1: Analyze problem statement.\nThought 2: Check prerequisites.\nThought 3: Brainstorm solutions."),
            helper_make_raw_rpc_response("Thought 1: Score 7/10.\nThought 2: Score 6/10.\nThought 3: Score 9/10 (Best approach)."),
            helper_make_raw_rpc_response("Thought 1: Formulate equation.\nThought 2: Test edge cases.\nThought 3: Deduce steps."),
            helper_make_raw_rpc_response("Thought 1: Score 5/10.\nThought 2: Score 9/10 (Highest accuracy).\nThought 3: Score 7/10."),
            helper_make_raw_rpc_response("Thought 1: Compute result.\nThought 2: Double check.\nThought 3: Summarize."),
            helper_make_raw_rpc_response("Thought 1: Score 9/10.\nThought 2: Score 8/10.\nThought 3: Score 7/10."),
            helper_make_raw_rpc_response("Final Solution: The computed answer is 42."),
        ]

        engine = TreeOfThoughtsEngine(send_func=mock_send, breadth=3, max_depth=3)
        messages = [{"role": "user", "content": "What is the answer to life?"}]

        chunks = []
        async for chunk in engine.run(messages, chunk_id="test-chunk-123"):
            chunks.append(chunk)

        full_stream_raw = "".join(chunks)
        reconstructed_text = extract_stream_text(full_stream_raw)

        assert "<thought>" in reconstructed_text
        assert "Step 1/3" in reconstructed_text
        assert "Step 2/3" in reconstructed_text
        assert "Step 3/3" in reconstructed_text
        assert "The computed answer is 42" in reconstructed_text
        assert '"finish_reason": "stop"' in full_stream_raw

    asyncio.run(_test())


def test_tot_engine_rate_limit_fallback():
    async def _test():
        mock_send = AsyncMock()
        mock_send.side_effect = [
            helper_make_raw_rpc_response("Thought 1: Initial step A.\nThought 2: Initial step B.\nThought 3: Initial step C."),
            helper_make_raw_rpc_response("Thought 1: Score 9/10. Best step."),
            RateLimitError("Gemini rate limit hit"),
            helper_make_raw_rpc_response("Partial Synthesis Solution: Result based on Step 1."),
        ]

        engine = TreeOfThoughtsEngine(send_func=mock_send, breadth=3, max_depth=3)
        messages = [{"role": "user", "content": "Solve math puzzle."}]

        chunks = []
        async for chunk in engine.run(messages, chunk_id="test-chunk-fallback"):
            chunks.append(chunk)

        full_stream_raw = "".join(chunks)
        reconstructed_text = extract_stream_text(full_stream_raw)

        assert "<thought>" in reconstructed_text
        assert "Rate limit reached at step 2" in reconstructed_text
        assert "Partial Synthesis Solution" in reconstructed_text

    asyncio.run(_test())


def test_candidate_parsing():
    engine = TreeOfThoughtsEngine(send_func=AsyncMock())
    text = (
        "Thought 1: Identify all prime factors of the number.\n"
        "Thought 2: Calculate the square root first.\n"
        "Thought 3: Use brute force search."
    )
    candidates = engine._parse_candidates(text)
    assert len(candidates) == 3
    assert candidates[0] == "Identify all prime factors of the number."
    assert candidates[1] == "Calculate the square root first."
    assert candidates[2] == "Use brute force search."


def test_select_winning_candidate():
    engine = TreeOfThoughtsEngine(send_func=AsyncMock())
    candidates = ["Option A", "Option B", "Option C"]
    eval_text = (
        "Candidate 1: Score 6/10.\n"
        "Candidate 2: Score 9/10 - Excellent approach.\n"
        "Candidate 3: Score 4/10."
    )
    idx, winner = engine._select_winning_candidate(candidates, eval_text)
    assert idx == 1
    assert winner == "Option B"


def test_gemini_provider_non_streaming_tot():
    """Verify GeminiProvider.non_streaming_chat correctly executes Tree of Thoughts when requested."""
    async def _test():
        provider = GeminiProvider()
        mock_send = AsyncMock()
        mock_send.side_effect = [
            helper_make_raw_rpc_response("Thought 1: Analyze problem statement.\nThought 2: Check prerequisites.\nThought 3: Brainstorm solutions."),
            helper_make_raw_rpc_response("Thought 1: Score 7/10.\nThought 2: Score 6/10.\nThought 3: Score 9/10 (Best approach)."),
            helper_make_raw_rpc_response("Final Solution: 42"),
        ]
        provider._send_message = mock_send

        # Use 1 step for quick test
        from unittest.mock import patch
        with patch.object(TreeOfThoughtsEngine, "__init__", lambda self, send_func, breadth=3, max_depth=1: None):
            engine = TreeOfThoughtsEngine(send_func=mock_send, breadth=3, max_depth=1)
            engine._send_func = mock_send
            engine.breadth = 3
            engine.max_depth = 1

            with patch("src.providers.gemini.session.TreeOfThoughtsEngine", return_value=engine):
                resp = await provider.non_streaming_chat(
                    messages=[{"role": "user", "content": "What is the answer?"}],
                    model="gemini-tree-of-thoughts",
                )

        assert resp["object"] == "chat.completion"
        assert resp["model"] == "gemini-tree-of-thoughts"
        content = resp["choices"][0]["message"]["content"]
        assert "<thought>" in content
        assert "Final Solution: 42" in content
        assert resp["choices"][0]["finish_reason"] == "stop"

    asyncio.run(_test())


def test_gemini_converter_multipart_extraction():
    """Verify _extract_content concatenates multiple text parts instead of dropping parts 1..N."""
    from src.providers.gemini.converter import parse_gemini_response

    inner = json.dumps([
        None,
        ["c_1", "r_1"],
        None,
        None,
        [["rc_1", ["First part of message. ", "Second part of message."], []]]
    ])
    outer = json.dumps([["wrb.fr", None, inner]])
    raw_rpc = f")]}}'\n100\n{outer}\n"

    parsed = parse_gemini_response(raw_rpc)
    assert parsed["content"] == "First part of message. Second part of message."


def test_gemini_converter_streaming_is_final():
    """Verify parse_streaming_chunks yields is_final=True on the terminal chunk during iteration."""
    from src.providers.gemini.converter import parse_streaming_chunks

    inner_1 = json.dumps([None, ["c_1", "r_1"], None, None, [["rc_1", ["Chunk 1"], []]]])
    inner_2 = json.dumps([None, ["c_1", "r_1"], None, None, [["rc_1", ["Chunk 1 and Chunk 2"], []]]])
    outer_1 = json.dumps([["wrb.fr", None, inner_1]])
    outer_2 = json.dumps([["wrb.fr", None, inner_2]])
    raw_rpc = f")]}}'\n100\n{outer_1}\n100\n{outer_2}\n"

    chunks = []
    for chunk in parse_streaming_chunks(raw_rpc):
        chunks.append(dict(chunk))

    assert len(chunks) == 2
    assert chunks[0]["is_final"] is False
    assert chunks[0]["content"] == "Chunk 1"
    assert chunks[1]["is_final"] is True
    assert chunks[1]["content"] == "Chunk 1 and Chunk 2"

