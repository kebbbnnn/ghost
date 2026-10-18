# Perplexity Provider Architecture

This provider exposes the Perplexity concise/auto search model to Ghost clients using an anonymous guest session.

## Discoveries & Protocol Analysis

Perplexity's web interface uses an internal SSE endpoint for queries. We analyzed two communication paths:

### 1. Socket.IO Transport (Playground / Labs)
- **Path:** `https://www.perplexity.ai/socket.io/`
- **Protocol:** Socket.IO / Engine.IO long-polling upgraded to WebSocket.
- **Authentication:** Authenticates anonymously using the JWT `"anonymous-ask-user"`.
- **Limitation:** Cloudflare blocks WebSocket upgrade handshakes because standard python WebSocket clients (like `websocket-client` or `websockets`) do not match Chrome's TLS fingerprint. Re-wrapping sockets or staying on polling is complex and prone to latency/timeout issues.

### 2. SSE REST Endpoint (Main client)
- **Path:** `POST https://www.perplexity.ai/rest/sse/perplexity_ask`
- **Authentication:** Works completely anonymously without account login, but requires active session cookies (specifically Cloudflare verification cookies like `__cf_bm` and `__cflb`).
- **Warming up:** Before queries, a `GET https://www.perplexity.ai/api/auth/session` request impersonating Chrome (`curl_cffi`) obtains these cookies.
- **Payload:**
  - `query_str`: The query string.
  - `params.frontend_context_uuid`: The thread ID (used to maintain multi-turn chat).
  - `params.last_backend_uuid`: The backend message UUID of the last assistant reply.
  - `params.is_incognito`: Must be set to `False` to allow server-side thread persistence for multi-turn conversations.
  - `params.model_preference`: Set to `"turbo"`.
  - `params.mode`: Set to `"concise"`.

## Implementation Details

### Incremental Streaming Conversion
The SSE response yields JSON messages under the `event: message` format. The answer text is contained inside `blocks`:

```json
{
  "blocks": [
    {
      "intended_usage": "ask_text",
      "markdown_block": {
        "progress": "IN_PROGRESS",
        "chunks": ["A", "B", "C"],
        "chunk_starting_offset": 0
      }
    }
  ]
}
```

Since the `chunks` array in progressive events is sent with `chunk_starting_offset`, we match each incoming chunk offset to yield only the newly generated text deltas. The final event provides `progress: DONE` with the complete joined string under the `answer` key.
