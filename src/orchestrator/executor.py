"""
executor.py — Workflow execution engine for Spectre-Ultra.

Executes a Workflow DAG against the worker pool:
  - Topologically sorts steps into parallelizable layers
  - Runs independent steps concurrently via asyncio.gather
  - Manages context assembly from access lists (worker isolation)
  - Handles errors with fallback to alternate workers
  - Supports both streaming and non-streaming output

This mirrors Fugu-Ultra's execution model:
  - Intra-workflow agent isolation: each worker only sees outputs
    specified in its access_list
  - DAG-based scheduling: steps with no mutual dependencies run in parallel
"""
import asyncio
import json
import logging
import time
import uuid

from src.core.provider import BaseProvider
from .types import Workflow, WorkflowStep, WorkerResult
from .prompts import build_worker_prompt

log = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Executes Spectre-Ultra workflows against a pool of worker providers.

    The executor takes a Workflow (a DAG of WorkflowSteps) and runs it
    by dispatching each step to its assigned worker model, respecting
    the dependency structure defined by access_lists.
    """

    def __init__(self, provider_resolver):
        """
        Args:
            provider_resolver: Callable that maps a model name to a
                               BaseProvider instance.
                               Signature: (model_name: str) -> BaseProvider | None
        """
        self._resolve_provider = provider_resolver

    async def execute(self, workflow: Workflow) -> str:
        """
        Execute a workflow and return the final step's output.

        Steps are grouped into layers (topological sort). All steps in a
        layer run concurrently. Each step's worker only sees outputs from
        steps listed in its access_list (intra-workflow agent isolation).

        Returns:
            The content from the final step's execution.
        """
        layers = workflow.get_execution_layers()
        results: dict[int, WorkerResult] = {}

        log.info(
            "Executing workflow: %d steps in %d layers",
            len(workflow.steps), len(layers),
        )

        for layer_idx, layer in enumerate(layers):
            log.info(
                "Layer %d: running %d step(s) in parallel: %s",
                layer_idx, len(layer),
                [f"step {s.step_id} → {s.worker_model}" for s in layer],
            )

            # Run all steps in this layer concurrently
            tasks = [
                self._execute_step(step, workflow.original_query, results)
                for step in layer
            ]
            layer_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(layer, layer_results):
                if isinstance(result, Exception):
                    log.error("Step %d failed: %s", step.step_id, result)
                    results[step.step_id] = WorkerResult(
                        step_id=step.step_id,
                        worker_model=step.worker_model,
                        content=f"[Step {step.step_id} failed: {result}]",
                        success=False,
                        error=str(result),
                    )
                else:
                    results[step.step_id] = result
                    log.info(result.summary)

        # Return the final step's output
        final_step = workflow.steps[-1]
        final_result = results.get(final_step.step_id)
        if final_result and final_result.success:
            return final_result.content
        else:
            # If the final step failed, try to return the last successful output
            for step in reversed(workflow.steps):
                r = results.get(step.step_id)
                if r and r.success:
                    return r.content
            return "[All workflow steps failed]"

    async def execute_streaming(self, workflow: Workflow):
        """
        Execute a workflow and stream the final step's output.

        All steps except the last run non-streaming (we need their full
        output for context assembly). The last step streams its response
        back to the caller.

        Yields:
            SSE-formatted strings (OpenAI streaming format).
        """
        layers = workflow.get_execution_layers()
        results: dict[int, WorkerResult] = {}
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Find the final step
        final_step = workflow.steps[-1]

        # Execute all layers except the one containing the final step
        for layer_idx, layer in enumerate(layers):
            # Check if this layer contains the final step
            non_final_steps = [s for s in layer if s.step_id != final_step.step_id]
            has_final = any(s.step_id == final_step.step_id for s in layer)

            # Execute non-final steps in this layer
            if non_final_steps:
                tasks = [
                    self._execute_step(step, workflow.original_query, results)
                    for step in non_final_steps
                ]
                layer_results = await asyncio.gather(*tasks, return_exceptions=True)

                for step, result in zip(non_final_steps, layer_results):
                    if isinstance(result, Exception):
                        log.error("Step %d failed: %s", step.step_id, result)
                        results[step.step_id] = WorkerResult(
                            step_id=step.step_id,
                            worker_model=step.worker_model,
                            content=f"[Step {step.step_id} failed: {result}]",
                            success=False,
                            error=str(result),
                        )
                    else:
                        results[step.step_id] = result
                        log.info(result.summary)

            # If this layer contains the final step, stream it
            if has_final:
                async for chunk in self._stream_step(
                    final_step, workflow.original_query, results, chunk_id
                ):
                    yield chunk
                return

        # Shouldn't reach here, but safety net
        yield f"data: {json.dumps({'error': {'message': 'Workflow produced no output'}})}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        step: WorkflowStep,
        original_query: str,
        prior_results: dict[int, WorkerResult],
    ) -> WorkerResult:
        """Execute a single workflow step (non-streaming)."""
        start = time.monotonic()

        provider = self._resolve_provider(step.worker_model)
        if not provider:
            raise ValueError(f"No provider found for model '{step.worker_model}'")

        # Assemble context from access list (intra-workflow agent isolation)
        prior_outputs = {}
        for dep_id in step.access_list:
            dep_result = prior_results.get(dep_id)
            if dep_result:
                prior_outputs[dep_id] = dep_result.content

        messages = build_worker_prompt(
            subtask=step.subtask,
            original_query=original_query,
            prior_outputs=prior_outputs if prior_outputs else None,
        )

        response = await provider.non_streaming_chat(messages, step.worker_model)
        content = self._extract_content(response)
        elapsed = (time.monotonic() - start) * 1000

        return WorkerResult(
            step_id=step.step_id,
            worker_model=step.worker_model,
            content=content,
            latency_ms=elapsed,
            success=True,
        )

    async def _stream_step(
        self,
        step: WorkflowStep,
        original_query: str,
        prior_results: dict[int, WorkerResult],
        chunk_id: str,
    ):
        """Execute a workflow step with streaming output."""
        provider = self._resolve_provider(step.worker_model)
        if not provider:
            yield f"data: {json.dumps({'error': {'message': f'No provider for {step.worker_model}'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Assemble context from access list
        prior_outputs = {}
        for dep_id in step.access_list:
            dep_result = prior_results.get(dep_id)
            if dep_result:
                prior_outputs[dep_id] = dep_result.content

        messages = build_worker_prompt(
            subtask=step.subtask,
            original_query=original_query,
            prior_outputs=prior_outputs if prior_outputs else None,
        )

        async for chunk in provider.stream_chat(messages, step.worker_model):
            yield chunk

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(response: dict) -> str:
        """Extract text content from an OpenAI-format completion response."""
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "")
        return ""
