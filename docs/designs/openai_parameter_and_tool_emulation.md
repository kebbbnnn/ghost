# OpenAI Parameter & Tool-Calling Emulation Engine

## 1. Overview & Understanding Summary

- **What:** A decoupled, universal parameter emulation and tool-calling engine for Ghost.
- **Why:** Reverse-engineered anonymous AI web endpoints (ChatGPT, Gemini) do not support OpenAI JSON schema tool calling or hyperparameter tuning (`temperature`, `top_p`, `stop`, `max_tokens`). Modern agentic tools (Cursor, Claude Code, Aider, LangChain, OpenAI SDKs) require these features to function without throwing `422 Unprocessable Entity` or failing multi-turn agent loops.
- **Who:** Developers and automated coding agents utilizing Ghost as an OpenAI-compatible drop-in proxy.
- **Key Constraints:** 
  - Must not break fragile reverse-engineered web payloads.
  - Zero added latency for regular (non-tool) text streaming.
  - Multi-turn tool execution history (`role: "tool"` and assistant `tool_calls`) must fold cleanly into upstream web prompts.
- **Explicit Non-Goals:**
  - True hardware-level GPU sampling control.
  - Proxy-side tool execution (clients execute tools locally).
  - Authenticated official API key proxies.

---

## 2. Assumptions

1. **Decoupled Architecture:** Emulation logic is isolated in `src/core/emulation/`, acting as a reusable pipeline wrapped around `BaseProvider`.
2. **Tool Protocol:** An XML-style tag format (`<tool_call>{"name": "...", "arguments": {...}}</tool_call>`) is injected into system prompts and parsed from responses.
3. **Streaming Performance:** A sliding-window lookahead buffer (max 15 chars) prevents raw markup leakage while streaming standard text with $< 5\text{ms}$ latency overhead.
4. **Behavioral Steering:** High ($\ge 0.8$) and low ($\le 0.3$) `temperature` values are mapped into subtle behavioral steering cues injected into the system prompt.
5. **Client-Side Enforcement:** `stop` sequences and `max_tokens` limits are enforced directly by the proxy's stream processor.

---

## 3. Risks & Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Model outputs conversational text before a tool call (preamble) | Streaming parser might discard preamble or fail to detect tool call | The parser streams preamble text as `delta.content`, then transitions seamlessly into `delta.tool_calls` once `<tool_call>` is encountered. |
| Model emits malformed JSON inside `<tool_call>` | Client-side agent crash | A tolerant regex-based JSON cleaner sanitizes common formatting errors; if unrecoverable, raw text is emitted to `delta.content` so the agent sees the error. |
| Upstream web endpoint rejects `role: "tool"` | Immediate 400/500 error from upstream provider | The request transpiler folds `role: "tool"` and prior `tool_calls` into standard user/assistant text blocks before dispatching to the provider. |
| Leaking `<tool_call>` tags into client UI | Degraded UX in chat interfaces | A sliding-window lookahead buffer intercepts potential tag prefixes and suppresses them before they are emitted to `delta.content`. |

---

## 4. Decision Log

| Decision | Choice | Alternatives Considered | Rationale |
| :--- | :--- | :--- | :--- |
| **Emulation Strategy** | Emulate via Prompt Engineering & Response Parsing | Silent Passthrough; Official API Hybrid | Web endpoints lack native tool schemas; prompt-based emulation gives full drop-in compatibility with tools like Cursor, Claude Code, and LangChain. |
| **Tool Calling Fidelity** | Full Agentic Tool Calling | Single-turn only; General parameters first | Modern dev workflows require multi-turn agent loops (`tool_calls` $\rightarrow$ `role: "tool"` $\rightarrow$ final answer). |
| **Streaming Interception** | Streaming Window Buffer | Full Chunk Buffering; Post-process Passthrough | Zero added latency for regular text generation; cleanly suppresses raw `<tool_call>` tags from leaking into `content`. |
| **Hyperparameter Strategy** | Prompt Steering + Truncation | Silent acceptance; Strict schema rejection | Steers model behavior according to `temperature` bounds while strictly enforcing `stop` and `max_tokens` client-side. |
| **Architecture Placement** | Core Proxy Emulation Pipeline | Provider Inheritance Hooks; Two-Phase ReAct Orchestrator | Fully decoupled from fragile reverse-engineered session logic; reusable across ChatGPT, Gemini, and Spectre with zero code duplication. |

---

## 5. Architectural Specification

### 5.1 Request Pipeline (`src/core/emulation/pipeline.py`)

#### `ChatCompletionRequest` Updates
The Pydantic request model accepts all standard OpenAI parameters:
- `tools: list[dict] | None = None`
- `tool_choice: str | dict | None = None`
- `temperature: float | None = None`
- `top_p: float | None = None`
- `max_tokens: int | None = None`
- `max_completion_tokens: int | None = None`
- `stop: str | list[str] | None = None`
- `presence_penalty: float | None = None`
- `frequency_penalty: float | None = None`

#### Message Folding & Prompt Injection
```text
Client Request:
  [System: "You are a coding helper."]
  [User: "What's the weather in Tokyo?"]
  [Assistant: tool_calls=[get_weather(city="Tokyo")]]
  [Tool: "22°C, sunny"]

        │
        ▼ (Transpiled by EmulationPipeline.prepare_messages)
        │

Upstream Request Messages:
  [System:
    "You are a coding helper.
     # Tools Available:
     ...
     To call a tool, respond ONLY with:
     <tool_call>
     {\"name\": \"function_name\", \"arguments\": {...}}
     </tool_call>"
  ]
  [User: "What's the weather in Tokyo?"]
  [Assistant: "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}</tool_call>"]
  [User: "[Tool Result for call_get_weather]:\n22°C, sunny"]
```

### 5.2 Streaming Response Parser (`src/core/emulation/stream_parser.py`)

1. **Window Buffer:** Inspects incoming text deltas from provider SSE chunks.
   - Buffer holds up to 15 characters matching the prefix of `<tool_call>`.
   - Non-matching text is immediately yielded to `choices[0].delta.content`.
2. **Tool Mode Activation:**
   - When `<tool_call>` is confirmed, text output is suppressed.
   - Captured JSON is parsed into standard OpenAI `choices[0].delta.tool_calls` chunks.
   - Upon closing `</tool_call>`, finishes tool segment and emits `"finish_reason": "tool_calls"`.
3. **Parameter Enforcers:**
   - **Stop Sequences:** Matches stream text against `stop` string list; truncates and sets `"finish_reason": "stop"`.
   - **Max Tokens:** Running token estimator ($4\text{ chars/token}$); truncates and sets `"finish_reason": "length"`.

### 5.3 Non-Streaming Formatter
In `EmulationPipeline.wrap_non_streaming(raw_resp, req)`:
- Extracts full generated content.
- Scans for `<tool_call>...</tool_call>` blocks.
- If found:
  - Strips tags from `message.content`.
  - Populates `message.tool_calls` with properly formatted function dictionaries.
  - Sets `finish_reason: "tool_calls"`.

---

## 6. Verification & Closed-Loop Testing Strategy

Following John Carmack's principle of self-contained, closed-loop verification, the test suite (`tests/test_emulation.py`) operates without external network dependencies:

1. **`test_tool_call_streaming`**: Feeds synthetic SSE streams containing `<tool_call>` tags and asserts client receives valid `delta.tool_calls` and `finish_reason: "tool_calls"`.
2. **`test_preamble_with_tool_call`**: Asserts that reasoning text before `<tool_call>` is streamed as `delta.content` without leaking the tag.
3. **`test_multi_turn_folding`**: Asserts that `role: "tool"` and assistant messages with `tool_calls` are converted into upstream-compatible messages.
4. **`test_stop_sequences`**: Asserts stream stops immediately when matching stop tokens.
5. **`test_temperature_steering`**: Asserts appropriate prompt guidance is injected when temperature is low/high.
6. **`test_non_streaming_tool_call`**: Asserts single-turn JSON response contains `tool_calls` and clean content.
