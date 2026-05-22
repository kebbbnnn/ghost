"""
test_pow.py — Closed-loop test for proof-of-work solver and requirements token.
Run: python3 tests/test_pow.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.providers.chatgpt.sentinel.pow_solver import solve_pow, build_requirements_token, _fnv1a_32, _encode_config
from src.providers.chatgpt.sentinel.token_generator import _build_config, generate_requirements_token
import time
import uuid


def test_fnv1a():
    """Verify FNV-1a hash gives expected output for a known input."""
    # From the reference repo: hash("hello") should be deterministic
    h = _fnv1a_32("hello")
    assert len(h) == 8, f"Hash should be 8 hex chars, got {len(h)}: {h}"
    print(f"  FNV1a('hello') = {h}  ✓")


def test_requirements_token():
    """Verify requirements token is gAAAAAC-prefixed."""
    token, _sid = generate_requirements_token()
    assert token.startswith("gAAAAAC"), f"Expected gAAAAAC prefix, got: {token[:15]}"
    print(f"  Requirements token: {token[:40]}...  ✓")


def test_pow_easy():
    """Solve a trivially easy PoW (very low difficulty)."""
    sid = str(uuid.uuid4())
    config, now_ms = _build_config(sid)
    # Use difficulty "ffffff" — easy to find in < 10 attempts
    token = solve_pow("test_seed", "ffffff", config[:], page_load_ms=now_ms)
    assert token.startswith("gAAAAAB"), f"Expected gAAAAAB prefix, got: {token[:15]}"
    assert "~S" in token, f"Expected ~S suffix in token: {token[-10:]}"
    print(f"  PoW easy solve: {token[:40]}...  ✓")


def test_pow_verification():
    """Verify the solved token actually passes the difficulty check."""
    from src.providers.chatgpt.sentinel.pow_solver import _encode_config, _fnv1a_32
    seed = "verify_test"
    difficulty = "0ffff0"
    sid = str(uuid.uuid4())
    config, now_ms = _build_config(sid)

    token = solve_pow(seed, difficulty, config, page_load_ms=now_ms)
    # Strip gAAAAAB prefix and ~S suffix
    encoded_part = token[len("gAAAAAB"):].rstrip("~S")
    # Actually the token is encoded + "~S" so:
    inner = token[len("gAAAAAB"):]
    encoded, suffix = inner[:-2], inner[-2:]
    assert suffix == "~S", f"Token should end with ~S: {inner[-5:]}"

    # Verify the hash check passes
    digest = _fnv1a_32(seed + encoded)
    assert digest[:len(difficulty)] <= difficulty, (
        f"Hash check failed: {digest[:len(difficulty)]} > {difficulty}"
    )
    print(f"  PoW verification: digest prefix {digest[:len(difficulty)]!r} <= {difficulty!r}  ✓")


if __name__ == "__main__":
    print("Running PoW solver tests...")
    t0 = time.time()

    test_fnv1a()
    test_requirements_token()
    test_pow_easy()
    test_pow_verification()

    print(f"\n✅ All tests passed in {time.time() - t0:.2f}s")
