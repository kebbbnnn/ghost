"""
types.py — Data types for the Spectre orchestration engine.

Models the Fugu/Conductor workflow structure:
  - WorkflowStep: a single subtask with model assignment and access list
  - Workflow: a DAG of steps to execute
  - WorkerResult: output from a completed step
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrchestratorMode(str, Enum):
    """Spectre operating mode, mirroring Fugu variants."""
    FAST = "fast"    # Single-model routing (Fugu)
    ULTRA = "ultra"  # Multi-step workflow generation (Fugu-Ultra / Conductor)


@dataclass
class WorkflowStep:
    """
    A single step in a Spectre-Ultra workflow.

    Maps directly to the Conductor's workflow step concept:
      - subtask: natural-language description of what this step should do
      - worker_model: which model to assign (e.g. "auto", "gemini")
      - access_list: indices of prior steps whose outputs this step can see
      - step_id: unique identifier for dependency resolution

    Steps with no overlapping dependencies can run in parallel.
    """
    step_id: int
    subtask: str
    worker_model: str
    access_list: list[int] = field(default_factory=list)


@dataclass
class Workflow:
    """
    A complete Spectre-Ultra workflow — a DAG of WorkflowSteps.

    The coordinator generates this, and the executor runs it.
    Steps are ordered by step_id. The executor uses access_list
    to determine dependency edges and parallelization opportunities.
    """
    steps: list[WorkflowStep]
    original_query: str
    mode: OrchestratorMode = OrchestratorMode.ULTRA
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: int) -> WorkflowStep | None:
        """Look up a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_dependencies(self, step_id: int) -> list[int]:
        """Return the access list (dependency IDs) for a given step."""
        step = self.get_step(step_id)
        return step.access_list if step else []

    def get_execution_layers(self) -> list[list[WorkflowStep]]:
        """
        Topologically sort steps into layers for parallel execution.

        Steps in the same layer have no mutual dependencies and can
        run concurrently via asyncio.gather. Steps in layer N+1 depend
        only on steps in layers 0..N.

        Returns:
            List of layers, where each layer is a list of steps that
            can execute in parallel.
        """
        completed: set[int] = set()
        remaining = list(self.steps)
        layers: list[list[WorkflowStep]] = []

        while remaining:
            # Find steps whose dependencies are all completed
            ready = [s for s in remaining if all(d in completed for d in s.access_list)]
            if not ready:
                # Circular dependency or bad access list — break cycle
                # by forcing the first remaining step
                ready = [remaining[0]]
            layers.append(ready)
            for s in ready:
                completed.add(s.step_id)
                remaining.remove(s)

        return layers


@dataclass
class WorkerResult:
    """Output from a completed workflow step."""
    step_id: int
    worker_model: str
    content: str
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def summary(self) -> str:
        """Short summary for logging."""
        status = "✓" if self.success else "✗"
        return f"Step {self.step_id} [{self.worker_model}] {status} ({self.latency_ms:.0f}ms)"


@dataclass
class RoutingDecision:
    """Output from Spectre fast-mode: which single model to use."""
    selected_model: str
    confidence: float = 1.0
    reasoning: str = ""
