"""Unit tests for the Spectre multi-agent orchestration engine.

Tests:
  - Fast-mode routing with coordinator success and coordinator failure fallback
  - Fast-mode execution with worker failure fallback
  - Fast-mode streaming with worker failure fallback
  - Ultra-mode workflow DAG execution with step worker fallback
  - Coordinator JSON extraction and json_repair tolerance
  - Top-level provider failover mapping
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.provider import BaseProvider
from src.core.exceptions import TokenExpiredError, ProviderError
from src.providers.spectre.session import SpectreProvider
from src.providers.spectre.orchestrator.coordinator import Coordinator
from src.providers.spectre.orchestrator.executor import WorkflowExecutor
from src.providers.spectre.orchestrator.types import (
    OrchestratorMode,
    RoutingDecision,
    Workflow,
    WorkflowStep,
    WorkerResult,
)
from src.providers import get_fallback_provider, _spectre, _gemini, _chatgpt


class MockProvider(BaseProvider):
    """Mock provider with configurable streaming and non-streaming responses."""

    def __init__(self, name: str, should_fail: bool = False, fail_error: Exception | None = None):
        self.name = name
        self.should_fail = should_fail
        self.fail_error = fail_error or ProviderError(f"{name} upstream error")

    async def get_models(self) -> list[dict]:
        return [{"id": self.name, "object": "model"}]

    async def stream_chat(self, messages, model, conversation_id=None, parent_message_id=None):
        if self.should_fail:
            raise self.fail_error
        yield f'data: {{"choices": [{{"delta": {{"content": "Streamed from {self.name}"}}}}]}}\n\n'
        yield 'data: [DONE]\n\n'

    async def non_streaming_chat(self, messages, model, conversation_id=None, parent_message_id=None):
        if self.should_fail:
            raise self.fail_error
        return {
            "choices": [{"message": {"content": f"Answer from {self.name}"}}],
        }


class TestSpectreResilience(unittest.IsolatedAsyncioTestCase):
    """Verify Spectre orchestration fallback and error recovery."""

    def test_top_level_fallback(self):
        """Top-level provider failover should map Spectre to Gemini."""
        self.assertIs(get_fallback_provider(_spectre), _gemini)
        self.assertIs(get_fallback_provider(_chatgpt), _gemini)
        self.assertIs(get_fallback_provider(_gemini), _chatgpt)

    async def test_coordinator_route_fallback_on_failure(self):
        """When coordinator fails, it should pick an alternate worker rather than repeating failing model."""
        failing_coord = MockProvider("auto", should_fail=True, fail_error=TokenExpiredError("Device flagged 403"))
        coordinator = Coordinator(
            coordinator_provider=failing_coord,
            coordinator_model="auto",
            available_workers=["auto", "gemini"],
        )

        decision = await coordinator.route([{"role": "user", "content": "Hello"}])
        # Should fallback to "gemini", NOT the failing "auto"
        self.assertEqual(decision.selected_model, "gemini")
        self.assertIn("Fallback due to coordinator error", decision.reasoning)

    async def test_coordinator_json_repair_tolerance(self):
        """Coordinator should repair unquoted keys or trailing commas in model routing JSON."""
        class MalformedJSONCoordProvider(BaseProvider):
            async def get_models(self):
                return []
            async def stream_chat(self, *args, **kwargs):
                yield ""
            async def non_streaming_chat(self, *args, **kwargs):
                # Malformed JSON: unquoted keys and trailing comma
                return {
                    "choices": [{"message": {"content": "{model: 'gemini', reasoning: 'better math',}"}}]
                }

        coordinator = Coordinator(
            coordinator_provider=MalformedJSONCoordProvider(),
            coordinator_model="auto",
            available_workers=["auto", "gemini"],
        )

        decision = await coordinator.route([{"role": "user", "content": "What is 2+2?"}])
        self.assertEqual(decision.selected_model, "gemini")
        self.assertEqual(decision.reasoning, "better math")

    async def test_executor_step_fallback(self):
        """When an assigned step worker fails, executor should try fallback workers."""
        failing_worker = MockProvider("auto", should_fail=True)
        healthy_worker = MockProvider("gemini", should_fail=False)

        def resolver(model):
            return failing_worker if model == "auto" else healthy_worker

        executor = WorkflowExecutor(
            provider_resolver=resolver,
            available_workers=["auto", "gemini"],
        )

        workflow = Workflow(
            steps=[
                WorkflowStep(
                    step_id=0,
                    subtask="Generate code",
                    worker_model="auto",  # Primary is failing
                    access_list=[],
                )
            ],
            original_query="Write a script",
            mode=OrchestratorMode.ULTRA,
        )

        output = await executor.execute(workflow)
        self.assertEqual(output, "Answer from gemini")

    async def test_spectre_fast_mode_worker_fallback(self):
        """SpectreProvider _execute_fast should fallback to next worker when primary fails."""
        spectre = SpectreProvider()
        failing_worker = MockProvider("auto", should_fail=True)
        healthy_worker = MockProvider("gemini", should_fail=False)

        mock_coord = MagicMock()
        mock_coord.route = AsyncMock(return_value=RoutingDecision(selected_model="auto", reasoning="test"))
        mock_coord._available_workers = ["auto", "gemini"]

        spectre._coordinator = mock_coord
        spectre._initialized = True

        def resolver(model):
            return failing_worker if model == "auto" else healthy_worker

        with patch("src.providers.get_provider_for_model", side_effect=resolver):
            result = await spectre._execute_fast([{"role": "user", "content": "test"}])
            self.assertEqual(result, "Answer from gemini")

    async def test_spectre_fast_mode_streaming_fallback(self):
        """SpectreProvider _stream_fast should fallback to next worker before yield."""
        spectre = SpectreProvider()
        failing_worker = MockProvider("auto", should_fail=True)
        healthy_worker = MockProvider("gemini", should_fail=False)

        mock_coord = MagicMock()
        mock_coord.route = AsyncMock(return_value=RoutingDecision(selected_model="auto", reasoning="test"))
        mock_coord._available_workers = ["auto", "gemini"]

        spectre._coordinator = mock_coord
        spectre._initialized = True

        def resolver(model):
            return failing_worker if model == "auto" else healthy_worker

        with patch("src.providers.get_provider_for_model", side_effect=resolver):
            chunks = []
            async for chunk in spectre._stream_fast([{"role": "user", "content": "test"}], "cid"):
                chunks.append(chunk)

            full = "".join(chunks)
            self.assertIn("Streamed from gemini", full)


if __name__ == "__main__":
    unittest.main()
