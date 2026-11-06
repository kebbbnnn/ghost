"""
coordinator.py — The brain of Spectre.

Implements two coordination modes:
  1. Fast (Spectre): Analyze the query and route to the single best worker.
  2. Ultra (Spectre-Ultra): Generate a multi-step workflow DAG with subtask
     decomposition, model assignment, and communication topologies.

This module handles the coordination *decision*. Execution of the resulting
workflow is handled by executor.py.
"""
import json
import logging
from typing import Any

from src.core.provider import BaseProvider
from .types import (
    OrchestratorMode,
    RoutingDecision,
    Workflow,
    WorkflowStep,
)
from .prompts import (
    build_fast_routing_prompt,
    build_ultra_workflow_prompt,
)

log = logging.getLogger(__name__)


class Coordinator:
    """
    Spectre coordinator — uses a coordinator model (LLM) to make routing
    decisions and generate workflows.

    Approximates Fugu's trained coordinator through prompt engineering:
    - Fast mode → analogous to Fugu's lightweight prediction head
    - Ultra mode → analogous to the Conductor's RL-trained workflow generator
    """

    def __init__(
        self,
        coordinator_provider: BaseProvider,
        coordinator_model: str,
        available_workers: list[str],
    ):
        """
        Args:
            coordinator_provider: The provider instance to use for coordination
                                  decisions (e.g. ChatGPTProvider).
            coordinator_model: The model name to pass to the coordinator provider
                               (e.g. "auto").
            available_workers: List of worker model IDs that can be assigned to
                               workflow steps (e.g. ["auto", "gemini"]).
        """
        self._provider = coordinator_provider
        self._model = coordinator_model
        self._available_workers = available_workers

    async def route(self, messages: list[dict]) -> RoutingDecision:
        """
        Fast-mode routing: select the single best worker for this query.

        Sends the user's query to the coordinator model with a routing
        prompt and parses the JSON response to extract a model selection.

        Falls back to the first available worker if parsing fails.
        """
        user_query = self._extract_user_query(messages)
        if len(messages) > 2 and any(
            isinstance(m.get("content"), str)
            and any(tag in m["content"] for tag in ("<tool_result", "<invoke", "<tool_call"))
            for m in messages
        ):
            target = self._model if self._model in self._available_workers else self._available_workers[0]
            log.info("Spectre routing (multi-turn tool fast-path): → %s", target)
            return RoutingDecision(
                selected_model=target,
                confidence=1.0,
                reasoning="Multi-turn tool execution continuation fast-path",
            )

        system_prompt = build_fast_routing_prompt(self._available_workers)

        coord_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            response = await self._provider.non_streaming_chat(
                coord_messages, self._model
            )
            raw = self._extract_content(response)
            decision = self._parse_routing_decision(raw)
            log.info("Spectre routing: %s → %s (%s)",
                     user_query[:80], decision.selected_model, decision.reasoning)
            return decision

        except Exception as e:
            log.warning("Coordinator routing failed (%s), falling back to alternate worker", e)
            fallback_model = next(
                (w for w in self._available_workers if w != self._model),
                self._available_workers[0],
            )
            return RoutingDecision(
                selected_model=fallback_model,
                confidence=0.0,
                reasoning=f"Fallback due to coordinator error: {e}",
            )

    async def generate_workflow(self, messages: list[dict]) -> Workflow:
        """
        Ultra-mode workflow generation: decompose the query into a multi-step
        workflow DAG with model assignments and communication topologies.

        Sends the user's query to the coordinator model with the Conductor-style
        workflow generation prompt and parses the structured JSON output.

        Falls back to a simple 2-step workflow if generation fails.
        """
        user_query = self._extract_user_query(messages)
        system_prompt = build_ultra_workflow_prompt(self._available_workers)

        coord_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            response = await self._provider.non_streaming_chat(
                coord_messages, self._model
            )
            raw = self._extract_content(response)
            workflow = self._parse_workflow(raw, user_query)
            log.info(
                "Spectre-Ultra workflow generated: %d steps, topology: %s",
                len(workflow.steps),
                workflow.metadata.get("reasoning", "unknown"),
            )
            return workflow

        except Exception as e:
            log.warning("Coordinator workflow generation failed (%s), using fallback", e)
            return self._fallback_workflow(user_query)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_routing_decision(self, raw: str) -> RoutingDecision:
        """Parse the coordinator's JSON routing response."""
        data = self._extract_json(raw)

        model = data.get("model", self._available_workers[0])
        # Validate model is in available workers
        if model not in self._available_workers:
            log.warning("Coordinator selected unknown model '%s', falling back", model)
            model = self._available_workers[0]

        return RoutingDecision(
            selected_model=model,
            reasoning=data.get("reasoning", ""),
        )

    def _parse_workflow(self, raw: str, original_query: str) -> Workflow:
        """Parse the coordinator's JSON workflow response."""
        data = self._extract_json(raw)

        steps_data = data.get("steps", [])
        if not steps_data:
            raise ValueError("Workflow has no steps")

        steps = []
        valid_ids = set()
        for s in steps_data:
            step_id = s.get("step_id", len(steps))
            worker = s.get("worker_model", self._available_workers[0])

            # Validate worker model
            if worker not in self._available_workers:
                log.warning("Step %d has unknown model '%s', remapping to first worker",
                            step_id, worker)
                worker = self._available_workers[0]

            # Validate access list references
            access_list = s.get("access_list", [])
            access_list = [a for a in access_list if a in valid_ids]

            steps.append(WorkflowStep(
                step_id=step_id,
                subtask=s.get("subtask", ""),
                worker_model=worker,
                access_list=access_list,
            ))
            valid_ids.add(step_id)

        return Workflow(
            steps=steps,
            original_query=original_query,
            mode=OrchestratorMode.ULTRA,
            metadata={"reasoning": data.get("reasoning", "")},
        )

    def _fallback_workflow(self, user_query: str) -> Workflow:
        """
        Simple 2-step fallback workflow: answer + verify.
        Used when the coordinator fails to generate a proper workflow.
        """
        primary = self._available_workers[0]
        verifier = self._available_workers[-1] if len(self._available_workers) > 1 else primary

        return Workflow(
            steps=[
                WorkflowStep(
                    step_id=0,
                    subtask=f"Answer the following question thoroughly:\n{user_query}",
                    worker_model=primary,
                    access_list=[],
                ),
                WorkflowStep(
                    step_id=1,
                    subtask=(
                        "Review the following answer for accuracy and completeness. "
                        "If it is correct and complete, reproduce it verbatim. "
                        "If not, provide the corrected and improved answer."
                    ),
                    worker_model=verifier,
                    access_list=[0],
                ),
            ],
            original_query=user_query,
            mode=OrchestratorMode.ULTRA,
            metadata={"reasoning": "Fallback: simple answer + verify workflow"},
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_user_query(messages: list[dict]) -> str:
        """Extract the latest user message text from an OpenAI-format message list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            parts.append(item)
                    return " ".join(parts)
        # Fallback: concatenate all messages
        return " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))

    @staticmethod
    def _extract_content(response: dict) -> str:
        """Extract text content from an OpenAI-format completion response."""
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "")
        return ""

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """
        Extract a JSON object from a string that may contain markdown fences
        or surrounding text.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = text.index("\n") if "\n" in text else 3
            text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Try json_repair fallback for malformed LLM outputs
        try:
            import json_repair
            repaired = json_repair.repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

        raise ValueError(f"Could not parse JSON from coordinator response: {text[:200]}")
