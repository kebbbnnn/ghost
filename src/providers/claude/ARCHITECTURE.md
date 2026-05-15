# Claude Provider Architecture

## Purpose
This document outlines the design and architecture of the Claude Provider for GhostTunnel. It allows users to proxy OpenAI-formatted requests to Claude's undocumented web API while overcoming account rate limits through a multi-account rotation system.

## Understanding Summary
*   **What is being built:** A new `ClaudeProvider` for GhostTunnel that uses logged-in Claude accounts, supported by a CLI login manager and a multi-account rotator.
*   **Why it exists:** To integrate Claude into the proxy while overcoming account rate limits by rotating through a pool of available authenticated accounts.
*   **Who it is for:** Users of GhostTunnel and coding agents (like Claude Code or Cursor) who want to use Claude's web API through a standard OpenAI API schema.
*   **Key Constraints:** 
    *   Claude's conversation history is tied to specific accounts. We use a stateless approach: upon rotation, the full chat history provided by the client is sent as a new conversation to the new account.
    *   The account rotator must be thread-safe/async-safe to handle concurrent incoming requests.
*   **Explicit Non-Goals:** 
    *   No automated headless login flow (no email OTP handling in code).
    *   No web UI for account management.

## Assumptions
1.  **Storage:** Credentials (`sessionKey`, `email`) are stored in a local flat JSON file (`src/providers/claude/accounts.json`).
2.  **Exhaustion Behavior:** If all accounts are rate-limited or have expired tokens, the proxy returns an OpenAI-compatible HTTP error (e.g., 429 Too Many Requests).

## Decision Log
1.  **Login Flow:** Chosen **Option A** (Playwright CLI). We will use a non-headless Playwright script to let the user log in manually, automatically extracting the `sessionKey` cookie and asking for the email.
    *   *Alternative considered:* Web UI or manual cookie extraction. Rejected due to high friction for the user and higher complexity for a simple proxy.
2.  **Account Rotation Strategy:** Chosen **Stateless**. The proxy acts as a dumb router. When rotating accounts, it does not attempt to resume an old conversation ID. Instead, it sends the full context array as a brand new conversation.
    *   *Alternative considered:* Stateful resuming. Rejected because it breaks the OpenAI API proxy paradigm and is difficult to synchronize across multiple Claude accounts.
3.  **Hot-Reloading:** The `AccountRotator` will use a file watcher (or modified timestamp checks) on `accounts.json`.
    *   *Why:* To allow users to add new accounts via the CLI without restarting the proxy.
4.  **Rate Limit Cooldown:** We will attempt to parse Claude's explicit rate limit reset timestamp from its API. If unavailable, we will default to a 5-hour cooldown for that specific account.

## Final Design

### 1. The CLI Login Tool (`manage_claude.py`)
A standalone script executed via `python src/providers/claude/manage_claude.py --add`. It launches an incognito Chromium window. The user logs in. The script polls for the `sessionKey` cookie, prompts for the account email, and appends the data to `accounts.json`.

### 2. The Account Rotator (`rotator.py`)
An async-safe class that manages the pool of accounts.
*   Watches `accounts.json` for changes.
*   Provides the next available account for incoming requests (Round-Robin).
*   Maintains a temporary `cooldown` state for accounts that hit a 429 error, respecting either the API-provided timestamp or a 5-hour default.
*   Throws an `AccountsExhaustedError` if no valid accounts are available.

### 3. The Provider Implementation (`session.py`)
Implements `BaseProvider`.
*   Fetches an account from the Rotator and injects the `sessionKey` into the headers.
*   Maps standard OpenAI `messages` to Claude's internal format, treating every request as a new conversation to ensure statelessness.
*   Streams Claude's SSE back into OpenAI's SSE chunk format.
*   Catches rate limits. If caught before streaming, retries automatically on the next account. If caught mid-stream, aborts gracefully.
