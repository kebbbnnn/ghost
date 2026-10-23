"""
session.py — SpectreProvider: Fugu-style multi-agent orchestration
exposed as a standard Ghost provider.

This is the integration layer that wires the orchestrator core into
Ghost's provider registry. It presents two virtual models:

  - "spectre"       → Fast mode: route to the single best worker
  - "spectre-ultra" → Ultra mode: generate and execute a multi-step workflow

From the outside, Spectre looks like any other provider. Internally,
it uses the coordinator (ChatGPT by default) to make decisions, and
the executor to dispatch work to other providers (Gemini by default).

Configuration:
  - SPECTRE_COORDINATOR_MODEL: env var to override the coordinator model
    (default: "auto" → ChatGPT)
  - SPECTRE_WORKERS: env var with comma-separated worker model IDs
    (default: "auto,gemini")
"""
import asyncio
import json
import logging
import os
import time
import typing
import uuid

from src.core.provider import BaseProvider
from .orchestrator.coordinator import Coordinator
from .orchestrator.executor import WorkflowExecutor
from .orchestrator.types import OrchestratorMode

log = logging.getLogger(__name__)

# Configurable defaults
DEFAULT_COORDINATOR_MODEL = "auto"  # ChatGPT
DEFAULT_WORKERS = [
    "auto",
    "gemini",
]



class SpectreProvider(BaseProvider):
    """
    Multi-agent orchestration provider.

    Wraps the Spectre orchestrator as a standard BaseProvider so it
    plugs into Ghost's routing without any server changes.
    """

    def __init__(self):
        self._coordinator: Coordinator | None = None
        self._executor: WorkflowExecutor | None = None
        self._initialized = False

    def _ensure_initialized(self):
        """
        Lazy initialization — we can't resolve providers at import time
        because the provider registry hasn't been fully built yet.
        This runs on first request.
        """
        if self._initialized:
            return

        from src.providers import get_provider_for_model

        # Read config from environment
        coordinator_model = os.environ.get(
            "SPECTRE_COORDINATOR_MODEL", DEFAULT_COORDINATOR_MODEL
        )
        workers_str = os.environ.get("SPECTRE_WORKERS", "")
        workers = (
            [w.strip() for w in workers_str.split(",") if w.strip()]
            if workers_str
            else list(DEFAULT_WORKERS)
        )

        # Resolve the coordinator provider
        coordinator_provider = get_provider_for_model(coordinator_model)
        if not coordinator_provider:
            raise RuntimeError(
                f"Spectre: cannot find provider for coordinator model '{coordinator_model}'"
            )

        self._coordinator = Coordinator(
            coordinator_provider=coordinator_provider,
            coordinator_model=coordinator_model,
            available_workers=workers,
        )

        self._executor = WorkflowExecutor(
            provider_resolver=get_provider_for_model,
        )

        self._initialized = True
        log.info(
            "Spectre initialized: coordinator=%s, workers=%s",
            coordinator_model, workers,
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def get_models(self) -> list[dict]:
        return [
            {
                "id": "spectre",
                "object": "model",
                "created": 1751673600,
                "owned_by": "ghost",
                "description": "Spectre — fast single-model routing orchestrator",
            },
            {
                "id": "spectre-ultra",
                "object": "model",
                "created": 1751673600,
                "owned_by": "ghost",
                "description": "Spectre-Ultra — multi-step workflow orchestrator",
            },
        ]

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> typing.AsyncGenerator[str, None]:
        self._ensure_initialized()
        mode = self._resolve_mode(model)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        try:
            if mode == OrchestratorMode.FAST:
                async for chunk in self._stream_fast(messages, chunk_id):
                    yield chunk
            else:
                async for chunk in self._stream_ultra(messages, chunk_id):
                    yield chunk
        except Exception as exc:
            log.exception("Spectre stream error")
            error_chunk = {
                "error": {"message": str(exc), "type": "server_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        self._ensure_initialized()
        mode = self._resolve_mode(model)

        if mode == OrchestratorMode.FAST:
            content = await self._execute_fast(messages)
        else:
            content = await self._execute_ultra(messages)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ------------------------------------------------------------------
    # Fast mode (Spectre)
    # ------------------------------------------------------------------

    async def _execute_fast(self, messages: list[dict]) -> str:
        """Fast mode: route to a single worker and return its response."""
        from src.providers import get_provider_for_model

        decision = await self._coordinator.route(messages)
        provider = get_provider_for_model(decision.selected_model)
        if not provider:
            raise RuntimeError(f"No provider for routed model '{decision.selected_model}'")

        response = await provider.non_streaming_chat(messages, decision.selected_model)
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    async def _stream_fast(self, messages: list[dict], chunk_id: str):
        """Fast mode streaming: route to a single worker and stream its output."""
        from src.providers import get_provider_for_model

        decision = await self._coordinator.route(messages)
        provider = get_provider_for_model(decision.selected_model)
        if not provider:
            raise RuntimeError(f"No provider for routed model '{decision.selected_model}'")

        log.info("Spectre fast-mode: routing to %s", decision.selected_model)
        async for chunk in provider.stream_chat(messages, decision.selected_model):
            yield chunk

    # ------------------------------------------------------------------
    # Ultra mode (Spectre-Ultra)
    # ------------------------------------------------------------------

    async def _execute_ultra(self, messages: list[dict]) -> str:
        """Ultra mode: generate and execute a multi-step workflow."""
        workflow = await self._coordinator.generate_workflow(messages)
        return await self._executor.execute(workflow)

    async def _stream_ultra(self, messages: list[dict], chunk_id: str):
        """Ultra mode streaming: execute workflow, stream the final step."""
        workflow = await self._coordinator.generate_workflow(messages)
        log.info(
            "Spectre-Ultra: executing %d-step workflow (%s)",
            len(workflow.steps),
            workflow.metadata.get("reasoning", ""),
        )
        async for chunk in self._executor.execute_streaming(workflow):
            yield chunk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_mode(model: str) -> OrchestratorMode:
        """Determine orchestration mode from the model name."""
        if "ultra" in model.lower():
            return OrchestratorMode.ULTRA
        return OrchestratorMode.FAST
