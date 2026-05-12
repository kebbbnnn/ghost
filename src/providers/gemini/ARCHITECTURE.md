# Gemini Provider Architecture — Reverse Engineering Journal

> **Purpose:** Documents the reverse-engineering discoveries for anonymous Gemini web access, the API protocol, and implementation decisions.

---

## 1. Anonymous Access Discovery

**Key finding:** As of May 2026, `gemini.google.com` allows fully anonymous guest chat without any Google login. The homepage shows an "Ask Gemini" input with a "Sign in" button — the chat works without signing in.

### What Differs from Authenticated Access

| Aspect | Anonymous | Authenticated (cookie-based) |
|--------|-----------|------------------------------|
| CSRF token | `thykhd` field in `WIZ_global_data` | `SNlM0e` field |
| Cookies required | None (gets `NID` + `COMPASS` automatically) | `__Secure-1PSID`, `__Secure-1PSIDTS` |
| Multi-turn | Server-side via conversation_id | Same |
| Model selection | "Fast" only (default) | Fast, Pro, etc. |
| Rate limits | Aggressive (unknown exact limits) | More generous |
| Image upload | Untested | Supported |

---

## 2. Protocol

### Step 1: Token Extraction

**Request:** `GET https://gemini.google.com/app`

**Response:** HTML page (~660KB) containing embedded `WIZ_global_data` JavaScript object with key fields:

| Field | Example | Purpose |
|-------|---------|---------|
| `thykhd` | `AFWLbD2J8SWs0vZSkmDj4NRM...` | CSRF "at" token for POST requests |
| `cfb2h` | `boq_assistant-bard-web-server_20260510.18_p0` | Versioned bot server string (changes with deployments) |
| `FdrFJe` | `-3134948019431665950` | Session ID (captured but currently unused) |

**Cookies set by response:**
- `NID` — Google-wide session cookie
- `COMPASS` — Gemini-specific session cookie (prefixed `gemini-pd=`)

**Extraction:** Simple regex on the HTML:
```python
thykhd = re.search(r'"thykhd":"(.*?)"', html).group(1)
cfb2h  = re.search(r'"cfb2h":"(.*?)"', html).group(1)
```

### Step 2: Send Message

**Request:**
```
POST https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate
    ?bl={cfb2h}
    &_reqid={counter}
    &rt=c
```

**Headers:**
- `Content-Type: application/x-www-form-urlencoded;charset=UTF-8`
- `X-Same-Domain: 1`
- `Origin: https://gemini.google.com`
- `Referer: https://gemini.google.com/`

**Body (form-encoded):**
```
f.req=[null, "[[\"message text\"],null,[\"conv_id\",\"resp_id\",\"choice_id\"]]"]
at={thykhd}
```

The `f.req` value is `json.dumps([None, json.dumps([[message], None, [conv_id, resp_id, choice_id]])])`.

**`_reqid` counter:** Starts at a random 6-digit number, increments by 100,000 after each request.

### Step 3: Parse Response

The response is Google's proprietary **batch-RPC format**, NOT standard SSE:

```
)]}'

12345
[["wrb.fr","...","[[...deeply nested JSON string...]]", ...]]
```

**Parsing algorithm:**
1. Split response by `\n`
2. Find the first line starting with `[[` — this is the JSON payload
3. Parse as JSON: `outer = json.loads(line)`
4. The real data is a JSON string at `outer[0][2]` — parse again: `inner = json.loads(outer[0][2])`
5. Content: `inner[4][0][1][0]`
6. Conversation ID: `inner[1][0]`
7. Response ID: `inner[1][1]`
8. Choice ID: search for `rc_`-prefixed string in the data tree

---

## 3. Multi-Turn Conversations

Gemini tracks conversation context **server-side**. After each response, we capture:
- `conversation_id` (e.g., `c_559d96609a13f5af`)
- `response_id` (e.g., `r_b71c29f9428f7a9b`)
- `choice_id` (e.g., `rc_...`)

These are passed back in the next request's `f.req` payload. When starting a fresh conversation, all three are empty strings.

---

## 4. Known Limitations

1. **No true streaming:** The endpoint returns the full response in one HTTP response body. We simulate streaming by chunking the text into word-sized pieces.
2. **Token TTL unknown:** We conservatively refresh every 5 minutes. The actual expiry may be longer.
3. **Rate limits:** Anonymous sessions are rate-limited, but exact thresholds are unknown.
4. **Bot server version:** The `cfb2h` value changes with Google deployments. We extract it dynamically from the page, never hardcode it.
5. **CAPTCHA:** Under heavy use, Google may serve a CAPTCHA page instead of the app. We detect and surface this as `RateLimitError`.
