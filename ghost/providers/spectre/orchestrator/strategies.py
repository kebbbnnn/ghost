"""
strategies.py — Built-in fallback workflow strategies for Spectre-Ultra.

These are pre-defined workflow templates inspired by the emergent patterns
observed in Fugu's trained Conductor (Section 4.4 of the technical report):

  1. Debate & Aggregation: Multiple workers answer independently, a
     synthesizer merges the best elements.
  2. Build & Debug: One worker generates, another critiques, then a
     revision step incorporates feedback.
  3. Specialist Routing: Direct the query to the most appropriate single
     worker (essentially Spectre fast-mode as a strategy).

These serve as fallbacks when the coordinator's dynamic workflow generation
fails, and as building blocks for the coordinator's few-shot examples.
"""
from .types import Workflow, WorkflowStep, OrchestratorMode


def debate_and_aggregate(
    query: str,
    workers: list[str],
    synthesizer: str | None = None,
) -> Workflow:
    """
    Debate & Aggregation strategy.

    Pattern: All workers answer independently → synthesizer merges.
    This is the most common emergent pattern from Fugu's training.

    Args:
        query: The user's original query.
        workers: List of worker model IDs to participate.
        synthesizer: Model ID for the synthesis step. Defaults to first worker.
    """
    if not workers:
        raise ValueError("At least one worker is required")

    synth = synthesizer or workers[0]
    steps = []

    # Each worker answers independently (no access to others' output)
    for i, worker in enumerate(workers):
        steps.append(WorkflowStep(
            step_id=i,
            subtask=query,
            worker_model=worker,
            access_list=[],  # Independent — no access to other steps
        ))

    # Synthesis step sees all worker outputs
    steps.append(WorkflowStep(
        step_id=len(workers),
        subtask=(
            "You have received multiple independent responses to the same question. "
            "Compare them carefully, identify the strongest elements from each, "
            "correct any errors, and synthesize a single superior response."
        ),
        worker_model=synth,
        access_list=list(range(len(workers))),
    ))

    return Workflow(
        steps=steps,
        original_query=query,
        mode=OrchestratorMode.ULTRA,
        metadata={"strategy": "debate_and_aggregate", "reasoning": "Multiple workers answer independently, then best elements are merged"},
    )


def build_and_debug(
    query: str,
    builder: str,
    reviewer: str,
    reviser: str | None = None,
) -> Workflow:
    """
    Build & Debug strategy.

    Pattern: Builder generates → Reviewer critiques → Reviser incorporates feedback.
    Particularly effective for coding and technical tasks.

    Args:
        query: The user's original query.
        builder: Model ID for the initial generation step.
        reviewer: Model ID for the review/critique step.
        reviser: Model ID for the final revision. Defaults to builder.
    """
    rev = reviser or builder

    return Workflow(
        steps=[
            WorkflowStep(
                step_id=0,
                subtask=f"Produce a thorough and complete response to:\n{query}",
                worker_model=builder,
                access_list=[],
            ),
            WorkflowStep(
                step_id=1,
                subtask=(
                    "Carefully review the following response. Identify any errors, "
                    "missing information, edge cases, or areas for improvement. "
                    "Be specific and constructive in your critique."
                ),
                worker_model=reviewer,
                access_list=[0],
            ),
            WorkflowStep(
                step_id=2,
                subtask=(
                    "Given the original response and the review feedback, produce "
                    "the final improved response. Incorporate all valid suggestions "
                    "and fix any identified issues."
                ),
                worker_model=rev,
                access_list=[0, 1],
            ),
        ],
        original_query=query,
        mode=OrchestratorMode.ULTRA,
        metadata={"strategy": "build_and_debug", "reasoning": "Builder generates, reviewer critiques, reviser incorporates feedback"},
    )


def specialist_routing(
    query: str,
    specialist: str,
) -> Workflow:
    """
    Specialist Routing strategy.

    Pattern: Direct the query to a single specialist worker.
    This is essentially Spectre fast-mode wrapped as a 1-step workflow.

    Args:
        query: The user's original query.
        specialist: Model ID of the specialist worker.
    """
    return Workflow(
        steps=[
            WorkflowStep(
                step_id=0,
                subtask=query,
                worker_model=specialist,
                access_list=[],
            ),
        ],
        original_query=query,
        mode=OrchestratorMode.ULTRA,
        metadata={"strategy": "specialist_routing", "reasoning": "Direct routing to specialist worker"},
    )
