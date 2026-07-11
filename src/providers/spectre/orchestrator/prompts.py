"""
prompts.py — Coordinator prompt templates for Spectre.

All system prompts, few-shot examples, and output format specifications
used by the coordinator model to make routing decisions (fast mode) and
generate workflows (ultra mode).

Design note: These prompts approximate the behavior of Fugu's trained
coordinator. Where Fugu uses a learned prediction head or RL-trained
Conductor, we use structured prompt engineering with JSON output.
"""

# ---------------------------------------------------------------------------
# Model capability profiles
# ---------------------------------------------------------------------------

# These profiles inform the coordinator about each worker's known strengths.
# They're injected into the system prompt so the coordinator can make
# informed routing/delegation decisions.

MODEL_PROFILES = {
    "auto": {
        "provider": "chatgpt",
        "display_name": "ChatGPT (auto)",
        "strengths": [
            "Strong general reasoning and planning",
            "Excellent at structured output and following complex instructions",
            "Good at code generation, debugging, and software architecture",
            "Strong mathematical reasoning with chain-of-thought",
            "Reliable at multi-step logical deduction",
        ],
        "weaknesses": [
            "Can be verbose in explanations",
            "May over-engineer simple problems",
        ],
    },
    "gemini": {
        "provider": "gemini",
        "display_name": "Gemini",
        "strengths": [
            "Excellent at creative writing and open-ended generation",
            "Strong general knowledge and factual recall",
            "Good at summarization and information synthesis",
            "Fast at simple Q&A and conversational tasks",
            "Strong multilingual capabilities",
        ],
        "weaknesses": [
            "Less reliable at strict structured output formats",
            "May miss edge cases in complex reasoning chains",
        ],
    },
    "perplexity": {
        "provider": "perplexity",
        "display_name": "Perplexity",
        "strengths": [
            "Real-time web search and current events lookup",
            "Fact-checking and finding up-to-date information",
            "Locating and citing source URLs",
            "Answering questions about news, weather, and live data",
        ],
        "weaknesses": [
            "May lack deep offline coding and mathematical reasoning compared to ChatGPT",
            "Not suited for large-scale creative generation or open-ended chats",
        ],
    },
    "surfsense-gpt-5.4-mini-no-login": {
        "provider": "surfsense",
        "display_name": "SurfSense GPT 5.4 Mini",
        "strengths": [
            "Extremely strong conversational reasoning and general coding",
            "High general knowledge and multi-turn planning",
            "Strong logical deduction capabilities",
        ],
        "weaknesses": [
            "Cannot perform live web searching",
        ],
    },
    "surfsense-gpt-o4-mini-no-login": {
        "provider": "surfsense",
        "display_name": "SurfSense GPT O4 Mini",
        "strengths": [
            "Ultrafast lightweight reasoning and quick response generation",
            "Great at simple coding and logical tasks",
        ],
        "weaknesses": [
            "Lacks deep complex analytical reasoning of larger models",
        ],
    },
}


def _build_model_profiles_text(available_models: list[str]) -> str:
    """Build a formatted text block describing available worker models."""
    lines = []
    for model_id in available_models:
        profile = MODEL_PROFILES.get(model_id)
        if not profile:
            lines.append(f"- {model_id}: No profile available")
            continue
        lines.append(f"### {profile['display_name']} (id: \"{model_id}\")")
        lines.append("Strengths:")
        for s in profile["strengths"]:
            lines.append(f"  - {s}")
        if profile.get("weaknesses"):
            lines.append("Weaknesses:")
            for w in profile["weaknesses"]:
                lines.append(f"  - {w}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fast mode (Spectre): single-model routing
# ---------------------------------------------------------------------------

FAST_ROUTING_SYSTEM = """\
You are Spectre, an intelligent LLM router. Your job is to analyze an incoming \
user query and select the single best worker model to handle it.

You have access to the following worker models:

{model_profiles}

## Your Task
Analyze the user's query and select which worker model would handle it best. \
Consider the query's domain (code, math, creative writing, factual Q&A, etc.), \
complexity, and the known strengths of each model.

## Output Format
Respond with ONLY a JSON object (no markdown fences, no explanation):
{{"model": "<model_id>", "reasoning": "<one-sentence explanation>"}}

Examples:
- User asks to write a Python function → {{"model": "auto", "reasoning": "Code generation task suits ChatGPT's strong coding capabilities"}}
- User asks to summarize a long article → {{"model": "gemini", "reasoning": "Summarization and synthesis is a Gemini strength"}}
- User asks a complex math proof → {{"model": "auto", "reasoning": "Complex mathematical reasoning benefits from ChatGPT's chain-of-thought"}}
"""


def build_fast_routing_prompt(available_models: list[str]) -> str:
    """Build the system prompt for fast-mode routing."""
    profiles = _build_model_profiles_text(available_models)
    return FAST_ROUTING_SYSTEM.format(model_profiles=profiles)


# ---------------------------------------------------------------------------
# Ultra mode (Spectre-Ultra): workflow generation
# ---------------------------------------------------------------------------

ULTRA_WORKFLOW_SYSTEM = """\
You are Spectre-Ultra, an intelligent LLM orchestrator inspired by the Conductor \
framework. Your job is to analyze a user query and design a multi-step agentic \
workflow that coordinates multiple worker models to produce the best possible answer.

You have access to the following worker models:

{model_profiles}

## Workflow Design Principles

1. **Decompose** the problem into subtasks that leverage different model strengths.
2. **Assign** each subtask to the best-suited worker model.
3. **Define access lists** — each step can see outputs from specified prior steps. \
Steps that don't need to see each other's output can run in parallel.
4. **Always include a final synthesis step** that combines prior outputs into a \
coherent final answer.

### Topology Patterns You Can Use:
- **Sequential chain**: Step 1 → Step 2 → Step 3 (each sees the previous)
- **Parallel fan-out**: Steps 1,2,3 run independently → Step 4 synthesizes
- **Debate**: Two models answer independently → a third evaluates and picks/merges
- **Build & debug**: One model generates → another critiques → first revises
- **Specialist routing**: Direct a subtask to the model with domain expertise

### Key Rules:
- Use 2-5 steps total (prefer fewer for simpler queries).
- The LAST step's output becomes the final response to the user.
- The last step's subtask should instruct the worker to produce a polished, \
complete answer — not a meta-commentary about the process.
- access_list items refer to step_id values (0-indexed).
- A step with an empty access_list sees only the original user query.
- Worker isolation: each worker only sees outputs listed in its access_list, \
NOT all prior outputs. This prevents orchestration collapse.

## Output Format
Respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "steps": [
    {{
      "step_id": 0,
      "subtask": "<clear natural-language instruction for the worker>",
      "worker_model": "<model_id>",
      "access_list": []
    }},
    {{
      "step_id": 1,
      "subtask": "<instruction that references step 0's output>",
      "worker_model": "<model_id>",
      "access_list": [0]
    }}
  ],
  "reasoning": "<brief explanation of the chosen topology>"
}}

## Examples

### Simple factual question → 2-step verify
{{
  "steps": [
    {{"step_id": 0, "subtask": "Answer the following question thoroughly: {{query}}", "worker_model": "auto", "access_list": []}},
    {{"step_id": 1, "subtask": "Review the following answer for accuracy and completeness. If correct, reproduce it. If not, provide the corrected answer.", "worker_model": "gemini", "access_list": [0]}}
  ],
  "reasoning": "Simple question benefits from independent verification by a second model"
}}

### Complex coding task → build & debug
{{
  "steps": [
    {{"step_id": 0, "subtask": "Write a complete solution for: {{query}}", "worker_model": "auto", "access_list": []}},
    {{"step_id": 1, "subtask": "Review this code for bugs, edge cases, and improvements. List all issues found.", "worker_model": "gemini", "access_list": [0]}},
    {{"step_id": 2, "subtask": "Given the original code and the review feedback, produce the final corrected and improved solution.", "worker_model": "auto", "access_list": [0, 1]}}
  ],
  "reasoning": "Build-and-debug pattern: ChatGPT writes, Gemini reviews, ChatGPT incorporates feedback"
}}

### Creative or subjective task → debate & aggregate
{{
  "steps": [
    {{"step_id": 0, "subtask": "{{query}}", "worker_model": "auto", "access_list": []}},
    {{"step_id": 1, "subtask": "{{query}}", "worker_model": "gemini", "access_list": []}},
    {{"step_id": 2, "subtask": "You have two responses to the same question. Compare them, take the best elements from each, and synthesize a superior final answer.", "worker_model": "auto", "access_list": [0, 1]}}
  ],
  "reasoning": "Debate pattern: both models answer independently, then best elements are merged"
}}
"""


def build_ultra_workflow_prompt(available_models: list[str]) -> str:
    """Build the system prompt for ultra-mode workflow generation."""
    profiles = _build_model_profiles_text(available_models)
    return ULTRA_WORKFLOW_SYSTEM.format(model_profiles=profiles)


# ---------------------------------------------------------------------------
# Worker step execution
# ---------------------------------------------------------------------------

WORKER_STEP_SYSTEM = """\
You are a worker model in a multi-agent system. You have been assigned a \
specific subtask as part of a larger workflow. Focus exclusively on completing \
your assigned subtask to the best of your ability.

Do NOT reference the multi-agent system, other workers, or the orchestration \
process in your response. Simply produce the best possible output for the \
subtask you've been given."""


def build_worker_prompt(
    subtask: str,
    original_query: str,
    prior_outputs: dict[int, str] | None = None,
) -> list[dict]:
    """
    Build the message list for a worker model executing a workflow step.

    Args:
        subtask: The subtask description from the workflow step.
        original_query: The user's original query (always included as context).
        prior_outputs: Dict mapping step_id → output content from prior steps
                       (only those in this step's access_list).
    """
    messages = [{"role": "system", "content": WORKER_STEP_SYSTEM}]

    # Build context from prior step outputs
    context_parts = []
    if prior_outputs:
        for step_id in sorted(prior_outputs.keys()):
            output = prior_outputs[step_id]
            context_parts.append(
                f"--- Output from Step {step_id} ---\n{output}\n--- End Step {step_id} ---"
            )

    # Construct the user message
    user_parts = []
    user_parts.append(f"Original user query:\n{original_query}")
    if context_parts:
        user_parts.append("\nContext from prior steps:\n" + "\n\n".join(context_parts))
    user_parts.append(f"\nYour assigned subtask:\n{subtask}")

    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


# ---------------------------------------------------------------------------
# Synthesis (used when the coordinator itself needs to merge outputs)
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You are synthesizing multiple worker outputs into a single coherent response. \
Produce a polished final answer that takes the best elements from all inputs. \
Do NOT mention the multi-agent process, workers, or synthesis steps — just \
deliver the answer as if you produced it yourself."""


def build_synthesis_prompt(
    original_query: str,
    outputs: dict[int, str],
) -> list[dict]:
    """Build messages for a synthesis/merge step."""
    messages = [{"role": "system", "content": SYNTHESIS_SYSTEM}]

    parts = [f"User's question:\n{original_query}\n"]
    for step_id in sorted(outputs.keys()):
        parts.append(f"--- Response {step_id + 1} ---\n{outputs[step_id]}\n")
    parts.append(
        "Synthesize these into a single, high-quality response. "
        "Take the best elements from each."
    )

    messages.append({"role": "user", "content": "\n".join(parts)})
    return messages
