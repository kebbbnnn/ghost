"""
pow_solver.py — Proof-of-work solver ported from the Sentinel SDK.

Implements the FNV-1a hash check and the iterative PoW loop.
Matches the _runCheck / _generateAnswerSync logic from the deobfuscated JS SDK.
"""
import base64
import json
import time

# ---------------------------------------------------------------------------
# FNV-1a 32-bit hash
# ---------------------------------------------------------------------------

_FNV_OFFSET = 2166136261
_FNV_PRIME  = 16777619

def _fnv1a_32(s: str) -> str:
    """FNV-1a 32-bit hash — mirrors the JS hash lambda in _runCheck."""
    h = _FNV_OFFSET
    for ch in s:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822507) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 3266489909) & 0xFFFFFFFF
    h ^= h >> 16
    return f"{h:08x}"


def _fnv1a_32_continue(s: str, h: int) -> str:
    """
    Continue an FNV-1a hash from an existing state `h`.
    Used to avoid rehashing the static seed prefix on every PoW iteration.
    """
    for ch in s:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822507) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 3266489909) & 0xFFFFFFFF
    h ^= h >> 16
    return f"{h:08x}"


def _fnv1a_state(s: str) -> int:
    """Compute partial FNV-1a hash state (without finalization) for a prefix string."""
    h = _FNV_OFFSET
    for ch in s:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def _encode_config(config: list) -> str:
    """Base64(JSON(config)) — mirrors the JS U() function."""
    payload = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(payload.encode("utf-8")).decode("utf-8")


def solve_pow(
    seed: str, difficulty: str, config: list,
    max_attempts: int = 500_000,
    page_load_ms: float = 1500.0,
) -> str:
    """
    Solve proof-of-work and return the gAAAAAB-prefixed token.

    Optimization: pre-compute the FNV-1a partial hash state for `seed` so that
    each iteration only processes the varying `encoded` suffix rather than
    rehashing the entire (seed + encoded) string from scratch.

    page_load_ms: the simulated performance.now() value at the point the SDK
    was initialised (i.e., config[13]). The PoW timer starts there so that
    config[9] = elapsed ms from page-load, not from system boot.
    """
    diff_len = len(difficulty)

    # Pre-compute the FNV-1a partial hash state for the constant seed string.
    # This eliminates up to 500,000 redundant re-hashes of the same prefix.
    seed_hash_state = _fnv1a_state(seed)

    solve_start_real = time.monotonic() * 1000     # real ms now
    t0_ms = page_load_ms                           # perf.now at challenge start

    for i in range(max_attempts):
        real_elapsed = time.monotonic() * 1000 - solve_start_real
        current_perf = t0_ms + real_elapsed
        config[3] = i
        config[9] = round(current_perf - t0_ms)   # = real elapsed, realistic
        encoded = _encode_config(config)
        # Continue hashing from the pre-computed seed state
        digest = _fnv1a_32_continue(encoded, seed_hash_state)
        if digest[:diff_len] <= difficulty[:diff_len]:
            return "gAAAAAB" + encoded + "~S"
    raise RuntimeError(f"PoW not solved in {max_attempts} attempts (seed={seed!r})")


def build_requirements_token(config: list, page_load_ms: float = 1500.0) -> str:
    """
    Build the lightweight gAAAAAC requirements token (no PoW iteration).
    Mirrors _generateRequirementsTokenAnswerBlocking():
      config[3] = 1
      config[9] = elapsed_ms (time since page load, not system boot)
    """
    import random
    config[3] = 1
    # JS: t[9] = performance.now() - e (time since SDK init, not since page load)
    # We simulate a small elapsed since page load, e.g. 50-200ms for requirements
    config[9] = page_load_ms + random.uniform(50, 200)
    return "gAAAAAC" + _encode_config(config)
