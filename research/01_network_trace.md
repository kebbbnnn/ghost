# Phase 1: Network Trace (Guest Session)

## Initial Page Load
- URL: `GET https://chatgpt.com/`
- Key Cookies Received:
  - `oai-did`: UUID (e.g. `267d82cf-7b25-4843-9bb9-89176d2467f7`)
  - `__cf_bm`, `_cfuvid`

## Requirements Provisioning
- URL: `POST https://chatgpt.com/backend-anon/sentinel/chat-requirements`
- Headers:
  - `oai-device-id`: `<oai-did>`
  - `oai-language`: `en-US`
- Body: `{"p": ""}` (or some proof of work token)
- Response: 200 OK
- Body:
```json
{
  "persona": "chatgpt-noauth",
  "token": "gAAAAABp___5...",
  "turnstile": {"required": true, "dx": "MjVOTUxfUHFF..."},
  "proofofwork": {"required": true, "seed": "...", "difficulty": "070871"},
  "so": {"required": true, "collector_dx": "..."}
}
```

The Sentinel SDK generates a combined token which resolves the requirements, returning `{p: <proofofwork>, t: <turnstile_token>, c: <chat_requirements_token>}`.

## Models Check
- URL: `GET https://chatgpt.com/backend-anon/models`
- Response lists `gpt-5-5` (Instant) and `auto` (Legacy).
