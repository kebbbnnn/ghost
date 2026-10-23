# Ghost Multi-Platform Proxy — Architecture

> **Purpose:** This document outlines the high-level architecture of Ghost, a modular, multi-platform proxy that exposes a unified OpenAI-compatible API while sandboxing individual AI provider implementations (ChatGPT, Gemini, etc.).

---

## 1. Project Overview

Ghost is an **OpenAI-compatible HTTP proxy** that routes incoming requests to various undocumented/anonymous AI APIs. 

**Core Paradigm:** 
- The proxy exclusively uses the **OpenAI JSON schema** as its "lingua franca" (`POST /v1/chat/completions`).
- All provider-specific complexities (e.g., bot-protection bypasses, headless browsers, token generation, custom data mappings) are completely isolated within strictly sandboxed "Provider" modules.
- The core server acts simply as a "dumb router," delegating requests to the correct provider without sharing state, cookies, or session locks.

---

## 2. Global Architecture

```text
Client (curl / chat.py / any OpenAI SDK)
        │
        │  POST /v1/chat/completions  (OpenAI format, optional streaming)
        ▼
┌─────────────────────┐
│   src/server.py     │  FastAPI Router (Stateless)
└────────┬────────────┘
         │
         │  Looks up provider in ProviderRegistry
         ▼
┌─────────────────────┐
│ src/core/provider.py│  BaseProvider Interface
└────────┬────────────┘
         │
         ├──► ┌────────────────────────────────────────┐
         │    │ src/providers/chatgpt/                 │  ◄── ChatGPT Sandbox
         │    │   ├── session.py (ChatGPTProvider)     │
         │    │   ├── converter.py                     │
         │    │   └── sentinel/                        │
         │    └────────────────────────────────────────┘
         │
         ├──► ┌────────────────────────────────────────┐
         │    │ src/providers/gemini/                  │  ◄── Gemini Sandbox
         │    └────────────────────────────────────────┘
         │
         └──► ┌────────────────────────────────────────────────────┐
              │ src/providers/spectre/                             │  ◄── Spectre Orchestrator
              │   ├── session.py (SpectreProvider)                │
              │   │                                               │
              │   └── orchestrator/                               │
              │       ├── coordinator.py  (routing + workflow)    │
              │       ├── executor.py     (DAG execution)         │
              │       ├── strategies.py   (built-in patterns)     │
              │       ├── prompts.py      (prompt templates)      │
              │       └── types.py        (Workflow, Step)        │
              └────────────────────────────────────────────────────┘
```

### Data Flow

1. **Request:** A client sends a standard OpenAI `ChatCompletionRequest` to the server.
2. **Routing:** `server.py` reads the requested `model` and queries the `ProviderRegistry` (`src/providers/__init__.py`) to find the matching provider.
3. **Delegation:** The server invokes `provider.stream_chat(...)` or `provider.non_streaming_chat(...)`.
4. **Execution:** The isolated provider handles all authentication, rate limits, proof-of-work, and API calls independently.
5. **Conversion:** The provider converts its native response format back into the standard OpenAI format.
6. **Response:** The server returns the OpenAI-formatted response back to the client.

### Spectre Data Flow (models: `spectre`, `spectre-ultra`)

1. **Coordination:** SpectreProvider sends the user query to the coordinator model (ChatGPT by default) to make a routing or workflow decision.
2. **Fast mode (`spectre`):** The coordinator selects the single best worker model. The query is forwarded directly to that worker.
3. **Ultra mode (`spectre-ultra`):** The coordinator generates a multi-step workflow DAG — decomposing the query into subtasks, assigning worker models, and defining communication topologies via access lists.
4. **DAG execution:** The executor topologically sorts the workflow into layers. Steps in the same layer run in parallel via `asyncio.gather`. Each worker only sees outputs from steps in its access list (intra-workflow agent isolation).
5. **Streaming:** All steps except the final one run non-streaming. The final step streams its output back to the client.

---

## 3. The Provider Abstraction

All providers inherit from `BaseProvider` (`src/core/provider.py`), which mandates the following interface:

- `async def get_models(self) -> list[dict]`: Returns a list of supported model dictionaries.
- `async def stream_chat(self, messages, model, conversation_id, parent_message_id)`: Yields strictly formatted OpenAI SSE string chunks (`data: {...}\n\n`).
- `async def non_streaming_chat(self, messages, model, conversation_id, parent_message_id)`: Returns a fully assembled OpenAI JSON completion dictionary.

**Strict Isolation Rule:** Providers must not share session state. If `ProviderA` gets rate-limited or banned, it should have zero impact on `ProviderB`. Error handling is translated into generic exceptions (`RateLimitError`, `TokenExpiredError`) defined in `src/core/exceptions.py`.

---

## 4. Provider Implementation Journals

Because each AI platform has drastically different bot-protection mechanisms, data schemas, and session lifecycle rules, each provider directory contains its own `ARCHITECTURE.md`. 

These local architecture files serve as implementation journals, explaining the reverse-engineering discoveries, the API structures, and any specific bug fixes for that platform.

- **ChatGPT:** Read the [ChatGPT Provider Architecture](./src/providers/chatgpt/ARCHITECTURE.md) for details on Cloudflare Turnstile evasion, Proof of Work solving, and the SSE conversion logic.

---

## 5. Spectre Orchestrator

Spectre is a Fugu-style multi-agent orchestration system that coordinates multiple LLM providers to exceed the performance of any individual model. Based on SakanaAI's [TRINITY](https://arxiv.org/abs/2512.04695) and [Conductor](https://arxiv.org/abs/2512.04388) papers.

### Architecture

| Component | File | Role |
|-----------|------|------|
| **SpectreProvider** | `src/providers/spectre/session.py` | BaseProvider wrapper — exposes `spectre` and `spectre-ultra` models |
| **Coordinator** | `src/providers/spectre/orchestrator/coordinator.py` | Analyzes queries and generates routing decisions or workflow DAGs |
| **Executor** | `src/providers/spectre/orchestrator/executor.py` | Runs workflows — topological sort, parallel execution, context assembly |
| **Strategies** | `src/providers/spectre/orchestrator/strategies.py` | Built-in workflow patterns (debate, build-and-debug, specialist routing) |
| **Prompts** | `src/providers/spectre/orchestrator/prompts.py` | All coordinator prompt templates and model capability profiles |
| **Types** | `src/providers/spectre/orchestrator/types.py` | `Workflow`, `WorkflowStep`, `WorkerResult`, `RoutingDecision` |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SPECTRE_COORDINATOR_MODEL` | `auto` (ChatGPT) | Which model serves as coordinator |
| `SPECTRE_WORKERS` | `auto,gemini` | Comma-separated list of worker model IDs |
