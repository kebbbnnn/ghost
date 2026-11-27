# ChatGPT Reverse-Engineered Proxy — Architecture & Implementation Journal

> **Purpose:** This document captures the full design rationale, implementation
> decisions, bugs encountered, root-cause analyses, and fixes applied. It is the
> single source of truth for anyone maintaining or extending this codebase.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [How the ChatGPT Guest Session Works](#3-how-the-chatgpt-guest-session-works)
4. [Implementation Plan](#4-implementation-plan)
5. [Multi-Turn Conversation Design](#5-multi-turn-conversation-design)
6. [Bug Journal](#6-bug-journal)
7. [File Reference](#7-file-reference)
8. [Regression Test Checklist](#8-regression-test-checklist)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)

---

## 1. Project Overview

This project is an **OpenAI-compatible HTTP proxy** that routes requests to
ChatGPT's undocumented anonymous (`/backend-anon`) API, requiring no user
account or API key.

**Goal:** Allow any client that speaks the OpenAI Chat Completions API
(`POST /v1/chat/completions`) to transparently use ChatGPT as the backend.

**Non-goals:**
- This is a study/research project, not intended for production use at scale.
- It does not bypass rate limits or circumvent paid features.

---

## 2. Architecture

```
Client (curl / chat.py / any OpenAI SDK)
        │
        │  POST /v1/chat/completions  (OpenAI format, optional streaming)
        ▼
┌─────────────────────┐
│   src/server.py     │  FastAPI proxy — request validation, SSE bridging
└────────┬────────────┘
         │
         │  calls stream_chat() / non-streaming variant
         ▼
┌─────────────────────────────────┐
│  src/session/guest_session.py   │  Session lifecycle, PoW, streaming
│  ┌──────────────────────────┐   │
│  │  GuestSession            │   │
│  │  - _oai_did              │   │  ◄── persistent device identity
│  │  - _sentinel_headers     │   │  ◄── PoW token (expires ~8 min)
│  │  - _session (curl_cffi)  │   │  ◄── HTTP client with TLS fingerprint
│  └──────────────────────────┘   │
└────────┬────────────────────────┘
         │
         │  raw SSE lines
         ▼
┌─────────────────────┐
│  src/converter.py   │  SSE parser — ChatGPT format → OpenAI format
└─────────────────────┘
```

### Data flow (streaming request)

```
1.  Client sends POST /v1/chat/completions {messages, stream: true,
                                            conversation_id?, parent_message_id?}
2.  server.py validates & calls _stream_response()
3.  GuestSession.ensure_ready() fetches/refreshes PoW sentinel
4.  GuestSession.stream_chat() POSTs to /backend-anon/conversation
5.  Raw SSE lines from ChatGPT are yielded to server.py
6.  converter.py parses each line → OpenAI chunk dict
7.  server.py re-serialises as SSE and streams to client
8.  Client extracts conversation_id + message id from chunks
9.  On next turn, client sends these IDs back → server re-attaches them
```

---

## 3. How the ChatGPT Guest Session Works

Understanding the undocumented protocol is essential before making any changes.

### 3.1 Device Identity (`oai-did`)

ChatGPT uses a UUID cookie (`oai-did`) to identify the browser device. A fresh
anonymous session requires visiting `https://chatgpt.com/` to receive this
cookie. The proxy replicates this by issuing a GET to the homepage on first use.

**Critical:** The `oai-did` must remain stable across multiple conversation
turns. If it changes between turn 1 and turn 2, the backend treats the second
request as a different (unknown) device and returns `403`.

### 3.2 PoW Sentinel (`Openai-Sentinel-*` headers)

Every chat request must include three sentinel headers:

| Header | Purpose |
|---|---|
| `Openai-Sentinel-Chat-Requirements-Token` | Session token from `/backend-anon/sentinel/chat-requirements` |
| `Openai-Sentinel-Proof-Token` | Proof-of-work solution (solved locally via JS VM) |
| `Openai-Sentinel-Turnstile-Token` | Cloudflare Turnstile token (optional, solved if dx present) |

The PoW is obtained by:
1. POST `/backend-anon/sentinel/chat-requirements` with a `p` token
2. Receive `{proofofwork: {required: true, seed, difficulty}, token, turnstile}`
3. Solve the PoW by iterating SHA-3 hashes until the prefix matches `difficulty`
4. Encode the solution as a sentinel token

**Token TTL:** Chat-requirements tokens expire after ~540s. The proxy refreshes
at 480s (8 minutes) to maintain a safety margin. See `TOKEN_TTL_SECONDS`.

### 3.3 Conversation Continuity (Multi-Turn)

ChatGPT's guest API uses two IDs to chain turns:

- **`conversation_id`** — UUID identifying the conversation thread. Assigned by
  the backend on the first turn and returned in every SSE event.
- **`parent_message_id`** — UUID of the most recent assistant message. Each new
  turn must reference the previous turn's assistant message ID.

**Key invariant:** On continuation turns, the backend expects **only the newest
user message** in the `messages` array, not the full history. The full history
is stored server-side and looked up via `conversation_id`.

### 3.4 SSE Event Format

The ChatGPT backend emits SSE events of varying types:

```json
// Type 1: message events (content streaming)
{"message": {"id": "...", "author": {"role": "assistant"}, "content": {"parts": ["Hello"]}, "status": "in_progress"}, "conversation_id": "..."}

// Type 2: input_message (echoes the user's message — IGNORE for output)
{"type": "input_message", "input_message": {...}}

// Type 3: message_marker (internal — IGNORE)
{"type": "message_marker", ...}

// Type 4: stream complete
{"type": "message_stream_complete", "conversation_id": "..."}

// Type 5: done
[DONE]
```

**Critical:** Only `message` events where `author.role == "assistant"` contain
the model's response. System and user message echoes have `finished_successfully`
status with no content, which can produce false `finish_reason: "stop"` signals.

---

## 4. Implementation Plan

### Phase 1 — Core Proxy (Completed)

**Goal:** Single-turn stateless proxy that routes OpenAI-format requests to
ChatGPT and returns OpenAI-format responses.

- [x] `GuestSession`: homepage load → `oai-did` → PoW → conversation
- [x] `server.py`: FastAPI with `/v1/chat/completions` (streaming + non-streaming)
- [x] `converter.py`: SSE parser + OpenAI chunk formatter
- [x] Model map: OpenAI model names → ChatGPT slugs
- [x] Token TTL refresh loop

### Phase 2 — Multi-Turn State (Completed)

**Goal:** Maintain conversation context across multiple turns without re-sending
full history (which causes empty responses from the backend).

- [x] `ChatCompletionRequest` model extended with `conversation_id` + `parent_message_id`
- [x] `stream_chat()` accepts and propagates these IDs
- [x] When IDs are present, slice `messages` to only the last user message
- [x] Inject `conversation_id` into outgoing SSE chunks (extracted from backend events)
- [x] `chat.py` tracks `conversation_id` and `last_msg_id` across turns

### Phase 3 — Session Resilience (Completed)

**Goal:** Survive 403 PoW expiry mid-conversation without losing context.

- [x] Differentiate hard vs soft session reset
- [x] Soft reset on 403 during active conversation (preserves `oai-did`)
- [x] Hard reset on 403 for fresh sessions (wipes everything)
- [x] 404 stale-conversation recovery (falls back to fresh session, replays all messages)
- [x] Import `HTTPError` from `curl_cffi` (was missing, causing `NameError`)

---

## 5. Multi-Turn Conversation Design

### State the client must maintain

```python
conversation_id    = None   # Set after turn 1 from SSE chunk
parent_message_id  = None   # Set after each turn from the last SSE chunk's message id
messages           = []     # Full OpenAI-format history (for fallback)
```

### What the proxy sends to ChatGPT

**Turn 1 (no conversation_id):**
```json
{
  "action": "next",
  "messages": [<all messages>],
  "model": "auto",
  "parent_message_id": "<random UUID>",
  "stream": true
}
```

**Turn 2+ (with conversation_id):**
```json
{
  "action": "next",
  "messages": [<ONLY the last user message>],
  "model": "auto",
  "conversation_id": "<from turn 1>",
  "parent_message_id": "<assistant msg id from turn 1>",
  "stream": true
}
```

### Why slicing is necessary

If the full history is re-sent on turn 2, the backend either:
- Returns an empty stream (no content), OR
- Creates a parallel branch from the wrong parent message

The backend already has the history stored — it only needs the new message.

### What the proxy returns to the client

Each SSE chunk includes a non-standard `conversation_id` field:

```json
{
  "id": "<assistant message UUID>",
  "object": "chat.completion.chunk",
  "model": "auto",
  "conversation_id": "6a005...",
  "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
}
```

The client uses `id` as the next turn's `parent_message_id` and
`conversation_id` directly.

---

## 6. Bug Journal

### Bug 1: Empty responses on every turn

**Symptom:** The proxy returns only `data: [DONE]` with no content chunks,
even though the backend is sending data.

**Root cause:** `chatgpt_event_to_openai_chunk()` in `converter.py` was
processing *all* message events, including the echoed system message and user
message echo. Both have `status: "finished_successfully"` and empty content
parts `[""]`. This caused:

1. The system message echo → empty content + `finish_reason: "stop"` chunk
2. `server.py` sees `finish_reason` → **breaks the stream loop immediately**
3. All subsequent assistant content chunks are discarded

**Fix:** Added role guard in `converter.py`:
```python
if msg.get("author", {}).get("role") != "assistant":
    return None
```

**File:** `src/converter.py` → `chatgpt_event_to_openai_chunk()`

---

### Bug 2: 403 on turn 2 wiping `oai-did` and causing 404

**Symptom:** Turn 1 succeeds. Turn 2 returns 403, session resets, then 404
on retry.

**Root cause:** The PoW sentinel token expires after ~8 minutes, but the
`oai-did` device identity must be stable. The old `_reset_session()` wiped
**both** the sentinel and the `oai-did`. After the wipe, the retry used a new
`oai-did`, which the backend doesn't recognise as owning the conversation,
returning 404.

**Fix:** Made `_reset_session(hard: bool = True)` configurable:
- `hard=False` (soft reset): clears only `_sentinel_headers` and cookies.
  `_oai_did` is preserved. Used when a `conversation_id` is active.
- `hard=True` (full reset): clears everything including `_oai_did`.
  Used for fresh sessions (no `conversation_id` present).

Decision logic:
```python
hard_reset = not bool(conversation_id)
await self._reset_session(hard=hard_reset)
```

**File:** `src/session/guest_session.py` → `_reset_session()`, `stream_chat()`

---

### Bug 3: `NameError: name 'HTTPError' is not defined`

**Symptom:** On 401/403, the `except HTTPError` block itself raises a
`NameError`, crashing the handler and masking the real error.

**Root cause:** `HTTPError` was used in an `except` clause but never imported.

**Fix:**
```python
from curl_cffi.requests.exceptions import HTTPError
```

**File:** `src/session/guest_session.py` (line 18)

---

### Bug 4: Stale conversation_id causes 404 with no recovery

**Symptom:** If the backend drops the conversation (e.g. after a full session
wipe or extended inactivity), turn 2 returns 404 and the proxy raises
`TokenExpiredError` with no useful retry.

**Fix:** Added explicit 404 handling in `stream_chat()`:
```python
if resp.status_code == 404 and conversation_id:
    # Stale conversation — start fresh
    conversation_id = None
    parent_message_id = None
    body.pop("conversation_id", None)
    body["parent_message_id"] = str(uuid.uuid4())
    chatgpt_messages = _convert_messages(messages)  # re-include all messages
    body["messages"] = chatgpt_messages
    await self._reset_session(hard=True)
    if attempt < max_retries:
        continue  # retry as new conversation
    raise TokenExpiredError("Conversation not found and retry failed")
```

**File:** `src/session/guest_session.py` → `stream_chat()`

---

### Bug 5: Turn 1 data sent as bytes not str (TypeError on startswith)

**Symptom:** Test scripts checking `if line.startswith(b"data:")` raised
`TypeError: startswith first arg must be str`.

**Root cause:** `stream_chat()` yields `chunk.decode("utf-8")` — i.e., already
decoded strings. The test was checking for a bytes prefix.

**Fix:** Use string prefixes: `if line.startswith('data:')`.

**File:** No production code change needed; only affected ad-hoc test scripts.

---

## 7. File Reference

### `src/server.py`
FastAPI application. Exposes:
- `GET /v1/models` — list available models
- `GET /v1/models/{id}` — single model
- `POST /v1/chat/completions` — streaming and non-streaming
- `GET /health` — liveness probe

**Key functions:**
- `_stream_response()` — async generator that bridges backend SSE → client SSE
- `_get_incremental_text()` — computes delta text since last emission (ChatGPT
  sends full text each chunk, not deltas)
- `_non_streaming_response()` — collects full response into one JSON response

**Important:** `_stream_response()` breaks after the first chunk with
`finish_reason` set. Do not emit a `finish_reason` prematurely (see Bug 1).

---

### `src/converter.py`
Pure functions for SSE parsing and format conversion. No HTTP, no state.

- `parse_chatgpt_sse(line)` — parses one SSE line into a dict or `None`
- `chatgpt_event_to_openai_chunk(event, model, chunk_id)` — converts a parsed
  event into an OpenAI chunk. Returns `None` for non-assistant events.
- `format_openai_chunk(chunk)` — serialises a chunk dict to `"data: {...}\n\n"`
- `build_non_streaming_response(content, model, finish_reason)` — builds the
  final non-streaming completion JSON

**Key invariant:** Only `author.role == "assistant"` messages produce output
chunks. All other roles return `None`.

---

### `src/session/guest_session.py`
The core session manager. All ChatGPT HTTP communication goes through here.

**`GuestSession` class:**
- `_oai_did` — device UUID, must be stable within a conversation
- `_sentinel_headers` — the three PoW headers (dict), valid for ~8 min
- `_token_fetched_at` — timestamp to track expiry
- `_lock` — asyncio lock for concurrent refresh protection

**Key methods:**
- `ensure_ready()` — calls `_refresh_chat_requirements()` if token is expired
- `_reset_session(hard=True)` — tears down session; if `hard=False`, preserves
  `_oai_did`
- `stream_chat(messages, model, conversation_id, parent_message_id)` — main
  streaming entry point

**Retry logic in `stream_chat()`:**
1. Attempt 0: send request
2. On 401/403: do a hard or soft reset, retry once
3. On 404 with `conversation_id`: strip conversation state, retry as new
4. On 429: raise `RateLimitError` immediately (no retry)
5. On success: break the retry loop

---

### `src/sentinel/token_generator.py`
Implements the PoW solve pipeline:
- `generate_requirements_token()` — builds the `p` token sent to chat-requirements
- `generate_enforcement_token(seed, difficulty, sid)` — solves PoW
- `build_sentinel_token(...)` — encodes result
- `execute_turnstile_vm(dx, p_token)` — runs Cloudflare Turnstile JS in a VM

---

### `src/sentinel/pow_solver.py`
Low-level PoW solver. Iterates SHA-3 hashes until the hex digest starts with
the required prefix (defined by `difficulty` field).

---

### `chat.py`
Reference interactive client. Demonstrates correct stateful usage:
1. Sends messages to the proxy
2. Extracts `conversation_id` and message `id` from SSE chunks
3. Passes them back on subsequent turns via `payload["conversation_id"]` and
   `payload["parent_message_id"]`
4. Maintains local `messages` list for display and fallback

---

## 8. Regression Test Checklist

Run these scenarios manually or automate them before any change to the session
or converter logic.

### ✅ Turn 1 produces content
```bash
curl -s -N -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"stream":true}'
```
**Expected:** Multiple `data: {...}` SSE lines with `delta.content` tokens, ending with `data: [DONE]`.

### ✅ Turn 2 continues the conversation
```python
# Use chat.py or manually pass conversation_id + parent_message_id from turn 1.
# Ask a question that requires prior context, e.g. "What did I just ask you?"
# Expected: The model recalls the first message.
```

### ✅ System message is handled
Include `{"role": "system", "content": "You are a pirate."}` in messages.
The model should respond in pirate style.

### ✅ 403 soft reset doesn't break conversation
Manually corrupt the PoW token between turns; the proxy should retry with a
refreshed token and preserve the conversation.

### ✅ Stale conversation recovers gracefully
After a full session reset, send a turn 2 with the old `conversation_id`.
Expected: proxy retries as a new conversation (not an error to the client).

### ✅ Non-streaming works
```bash
curl -s -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"ping"}],"stream":false}'
```
**Expected:** A single JSON response with `choices[0].message.content`.

---

## 9. Known Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|---|---|
| Single `GuestSession` instance | Concurrent requests from multiple clients share one device identity; heavy load may trigger rate limits |
| Guest-tier only | Guest sessions have a rate limit; no way to use paid models (requires logged-in session) |
| PoW solver is synchronous | Solved in a thread pool via `asyncio.to_thread`; adds ~50–200ms latency |
| No conversation persistence | If the proxy restarts, `conversation_id` from a previous session will 404 |
| Incremental text diffing | `_get_incremental_text()` assumes the backend always sends monotonically growing text; edge cases may produce doubled tokens |

### Potential Improvements

1. **Session pool** — maintain N independent `GuestSession` instances and load-
   balance across them to increase throughput and reduce per-device flagging.

2. **Logged-in session** — extend `GuestSession` to accept an `Authorization`
   cookie/token for accessing paid models (`/backend-api/conversation`).

3. **Conversation store** — persist `(conversation_id, parent_message_id)` to
   disk or Redis so conversations survive proxy restarts.

4. **WebSocket upgrade** — ChatGPT backend can `resume_with_websockets`; using
   websockets instead of HTTP streaming may improve reliability for long
   conversations.

5. **Adaptive TTL** — instead of a fixed 480s refresh, parse the expiry from
   the chat-requirements response if available.
