"""
token_generator.py — Top-level Sentinel token generation interface.

Builds the three-part token blob that the backend-anon/conversation endpoint
expects as the openai-sentinel-chat-requirements-token header value.

Token structure (JSON-stringified, passed as request body):
  {
    "p": <enforcement_token>,   # gAAAAAB... proof-of-work
    "t": <turnstile_token>,     # base64 from VM execution on dx blob
    "c": <chat_req_token>,      # the raw token from chat-requirements response
    "id": <oai-did cookie>,
    "flow": <flow_id>
  }
"""
import base64
import json
import random
import time
import uuid
import math

from .browser_env import BrowserEnv
from .pow_solver import solve_pow, build_requirements_token
from .vm import execute_turnstile_vm


def _build_config(client_version: str | None = None, sid: str | None = None) -> list:
    """
    Build the 18-element config array that getConfig() returns in the SDK.

    Positions:
      0  screen.width + screen.height
      1  new Date() string
      2  performance.memory.jsHeapSizeLimit
      3  Math.random()         (overwritten by PoW solver to attempt index)
      4  navigator.userAgent
      5  random script src from document.scripts (None / empty in 18-element version)
      6  data-build attribute (or null)
      7  navigator.language
      8  navigator.languages.join(",")
      9  Math.random()         (overwritten by PoW solver to elapsed ms)
      10 webkitGetUserMedia
      11 random document key
      12 random window key
      13 performance.now()
      14 sid (UUID)
      15 URLSearchParams keys joined (empty)
      16 hardwareConcurrency
      17 start time epoch ms
    """
    if client_version is not None and sid is None:
        if len(client_version) == 36 and "-" in client_version:
            sid = client_version
            client_version = "prod-abc"
        else:
            sid = str(uuid.uuid4())
    elif client_version is None:
        client_version = "prod-abc"
    if sid is None:
        sid = str(uuid.uuid4())
    # Simulate realistic browser timing:
    # - The browser loaded the page ~1-3s before calling getConfig()
    # - performance.now() counts ms since page load (NOT since system boot)
    # - Jitter prevents all tokens looking identical
    import random
    page_load_ms = random.uniform(1200.0, 3500.0)   # ms since page load
    now_ms = page_load_ms                           # performance.now()
    # performance.timeOrigin = epoch_ms - perf.now()
    epoch_ms = time.time() * 1000
    time_origin = epoch_ms - now_ms

    # Plausible random picks for the "pick random key from object" calls
    doc_keys = ["body", "head", "cookie", "title", "URL", "domain", "referrer"]
    win_keys = ["location", "navigator", "history", "document", "screen",
                "performance", "crypto", "fetch", "console", "sessionStorage"]

    return [
        BrowserEnv.screen_width + BrowserEnv.screen_height,           # 0
        time.strftime("%a %b %d %Y %H:%M:%S GMT+0800 (Philippine Standard Time)"), # 1
        4 * 1024 * 1024 * 1024,                                        # 2 jsHeapSizeLimit ~4GB
        random.random(),                                                # 3 (overwritten by PoW)
        BrowserEnv.user_agent,                                          # 4
        None,                                                           # 5
        client_version,                                                 # 6 data-build
        BrowserEnv.language,                                            # 7
        BrowserEnv.languages,                                           # 8
        random.random(),                                                # 9 (overwritten by PoW)
        "webkitGetUserMedia−function webkitGetUserMedia() { [native code] }", # 10
        random.choice(doc_keys),                                        # 11
        random.choice(win_keys),                                        # 12
        now_ms,                                                         # 13 performance.now()
        sid,                                                            # 14 this.sid
        "",                                                             # 15
        BrowserEnv.hardware_concurrency,                                # 16
        time_origin,                                                    # 17 performance.timeOrigin
    ], now_ms


def generate_requirements_token(client_version: str = "prod-abc", sid: str | None = None) -> tuple[str, str]:
    """
    Generate the lightweight gAAAAAC requirements/proof token.
    Returns:
        (p_token, sid)
    """
    if sid is None:
        sid = str(uuid.uuid4())
    config, now_ms = _build_config(client_version, sid)
    return build_requirements_token(config, now_ms), sid


def generate_enforcement_token(client_version: str, seed: str, difficulty: str, sid: str) -> str:
    """
    Solve the PoW for the given seed/difficulty and return a gAAAAAB token.
    Must use the SAME sid that generated the requirements token.
    """
    config, now_ms = _build_config(client_version, sid)
    return solve_pow(seed, difficulty, config, page_load_ms=now_ms)


def assemble_sentinel_headers(
    chat_req: dict,
    enforcement_token: str,
    p_token: str,
    oai_did: str,
    flow: str = "__default__",
    geo_info: dict = None
) -> str:
    """
    Assemble the final Turnstile + PoW payload, and Base64-encode it.

    Args:
        chat_req:           Response from /backend-anon/sentinel/chat-requirements
        enforcement_token:  The PoW-solved gAAAAAB token (proof of work answer)
        p_token:            The gAAAAAC requirements token (used as VM decryption key)
        oai_did:            The oai-device-id (from oai-did cookie)
        flow:               The flow identifier (default "__default__")
        geo_info:           Cloudflare geo IP variables to inject into the mock window.

    Returns:
        JSON-stringified token blob
    """
    # The dx blob is XOR-decrypted with the gAAAAAC requirements token (p_token),
    # NOT the enforcement token. See Et(nn) / jt(dx) in the SDK.
    turnstile_token = None
    turnstile = chat_req.get("turnstile", {})
    if turnstile.get("required") and turnstile.get("dx"):
        turnstile_token = execute_turnstile_vm(turnstile["dx"], p_token, geo_info)

    payload = {
        "p": enforcement_token,
        "t": turnstile_token,
        "c": chat_req.get("token"),
        "id": oai_did,
        "flow": flow,
    }
    payload_str = json.dumps(payload, separators=(",", ":"))
    return base64.b64encode(payload_str.encode("utf-8")).decode("utf-8")
