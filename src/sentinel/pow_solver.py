"""
pow_solver.py — Proof-of-work solver ported from the Sentinel SDK.

Implements the FNV-1a hash check and the iterative PoW loop.
Matches the _runCheck / _generateAnswerSync logic from the deobfuscated JS SDK.
"""
import base64
import json
import time


def _encode_config(config: list) -> str:
    """Base64(JSON(config)) — mirrors the JS U() function."""
    payload = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(payload.encode("utf-8")).decode("utf-8")


def _fnv1a_32(s: str) -> str:
    """FNV-1a 32-bit hash — mirrors the JS hash lambda in _runCheck."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822507) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 3266489909) & 0xFFFFFFFF
    h ^= h >> 16
    return f"{h:08x}"


def _run_check(t0_ms: float, seed: str, difficulty: str, config: list, attempt: int):
    """
    Mirrors _runCheck(t, n, e, r, o):
      r[3] = o (attempt index)
      r[9] = Math.round(performance.now() - t)
      c    = U(r) = base64(json(config))
      s    = fnv1a(seed + c)
      return c + "~S" if s[:len(difficulty)] <= difficulty else None

    t0_ms here is the performance.now() value at challenge start (NOT system time),
    so elapsed = current_perf_now - t0_ms accumulates from a realistic starting point.
    """
    config[3] = attempt
    # current_perf_now advances from the simulated page-load value
    current_perf_now = t0_ms + (time.monotonic() * 1000 % 10000)
    config[9] = round(current_perf_now - t0_ms)
    encoded = _encode_config(config)
    digest = _fnv1a_32(seed + encoded)
    if digest[: len(difficulty)] <= difficulty:
        return encoded + "~S"
    return None


def solve_pow(
    seed: str, difficulty: str, config: list,
    max_attempts: int = 500_000,
    page_load_ms: float = 1500.0,
) -> str:
    """
    Solve proof-of-work and return the gAAAAAB-prefixed token.

    page_load_ms: the simulated performance.now() value at the point the SDK
    was initialised (i.e., config[13]). The PoW timer starts there so that
    config[9] = elapsed ms from page-load, not from system boot.
    """
    # The JS SDK records performance.now() when it starts the check loop.
    # We simulate by anchoring to the real monotonic clock offset by page_load_ms.
    solve_start_real = time.monotonic() * 1000     # real ms now
    t0_ms = page_load_ms                           # perf.now at challenge start

    for i in range(max_attempts):
        # Simulate perf.now() advancing: real elapsed since solve_start, added to t0
        real_elapsed = time.monotonic() * 1000 - solve_start_real
        current_perf = t0_ms + real_elapsed
        config[3] = i
        config[9] = round(current_perf - t0_ms)   # = real elapsed, realistic
        encoded = _encode_config(config)
        digest = _fnv1a_32(seed + encoded)
        if digest[:len(difficulty)] <= difficulty:
            return "gAAAAAB" + encoded + "~S"
    raise RuntimeError(f"PoW not solved in {max_attempts} attempts (seed={seed!r})")


def build_requirements_token(config: list, page_load_ms: float = 1500.0) -> str:
    """
    Build the lightweight gAAAAAC requirements token (no PoW iteration).
    Mirrors _generateRequirementsTokenAnswerBlocking():
      config[3] = 1
      config[9] = elapsed_ms (time since page load, not system boot)
    """
    config[3] = 1
    # JS: t[9] = performance.now() - e (time since SDK init, not since page load)
    # We simulate a small elapsed since page load, e.g. 50-200ms for requirements
    import random
    config[9] = page_load_ms + random.uniform(50, 200)
    return "gAAAAAC" + _encode_config(config)
