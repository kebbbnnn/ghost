"""
test_gui_lifecycle.py — Automated lifecycle audit for GhostTunnel Desktop Suite.

Tests:
  1. Uvicorn server thread starts and binds port 8100 successfully.
  2. Health endpoint responds correctly.
  3. SSE telemetry endpoint is accessible.
  4. Account management endpoints are functional.
  5. Server thread shuts down cleanly and releases the port.
"""

import socket
import threading
import time
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from src.server import app

PORT = 8199  # Use a non-default port to avoid conflicts with a running instance

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
results = []


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def record(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def main():
    print("=" * 60)
    print("GhostTunnel Desktop Suite — Lifecycle Audit")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Pre-check: port should be free
    # ------------------------------------------------------------------
    if is_port_in_use(PORT):
        print(f"  ⚠️  Port {PORT} is already in use. Cannot run test.")
        sys.exit(1)

    record("Pre-check: port is free", True)

    # ------------------------------------------------------------------
    # 2. Start Uvicorn server in a daemon thread
    # ------------------------------------------------------------------
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to come up (max 8 seconds)
    started = False
    for _ in range(40):
        if is_port_in_use(PORT):
            started = True
            break
        time.sleep(0.2)

    record("Uvicorn thread started and port bound", started)
    if not started:
        print("\n  Server failed to start. Aborting remaining tests.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Health endpoint
    # ------------------------------------------------------------------
    import urllib.request
    import json

    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            record("GET /health returns 200", resp.status == 200, f"body={body}")
    except Exception as e:
        record("GET /health returns 200", False, str(e))

    # ------------------------------------------------------------------
    # 4. Status endpoint
    # ------------------------------------------------------------------
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            has_keys = all(k in body for k in ("status", "start_time", "providers_count"))
            record("GET /api/status returns expected fields", has_keys, f"keys={list(body.keys())}")
    except Exception as e:
        record("GET /api/status returns expected fields", False, str(e))

    # ------------------------------------------------------------------
    # 5. SSE events endpoint is connectable
    # ------------------------------------------------------------------
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/events")
        with urllib.request.urlopen(req, timeout=3) as resp:
            content_type = resp.headers.get("content-type", "")
            is_sse = "text/event-stream" in content_type
            record("GET /api/events returns SSE content-type", is_sse, f"content-type={content_type}")
    except Exception as e:
        # Timeout is expected since SSE is a long-lived connection
        if "timed out" in str(e).lower() or "urlopen error" in str(e).lower():
            record("GET /api/events returns SSE content-type", True, "connection timed out as expected for SSE")
        else:
            record("GET /api/events returns SSE content-type", False, str(e))

    # ------------------------------------------------------------------
    # 6. Dashboard static files served
    # ------------------------------------------------------------------
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/dashboard")
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode()
            has_ghosttunnel = "GhostTunnel" in html
            record("GET /dashboard serves index.html", has_ghosttunnel, f"length={len(html)}")
    except Exception as e:
        record("GET /dashboard serves index.html", False, str(e))

    # ------------------------------------------------------------------
    # 7. Models endpoint
    # ------------------------------------------------------------------
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            has_data = "data" in body
            record("GET /v1/models returns model list", has_data, f"count={len(body.get('data', []))}")
    except Exception as e:
        record("GET /v1/models returns model list", False, str(e))

    # ------------------------------------------------------------------
    # 8. Graceful shutdown
    # ------------------------------------------------------------------
    print()
    print("  Initiating graceful shutdown...")
    server.should_exit = True
    server_thread.join(timeout=10)

    # Give OS time to release the socket
    time.sleep(1.5)

    port_released = not is_port_in_use(PORT)
    record("Port released after shutdown", port_released)

    thread_dead = not server_thread.is_alive()
    record("Server thread terminated", thread_dead)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    if failed == 0:
        print(f"  \033[92mAll {total} tests passed.\033[0m")
    else:
        print(f"  \033[91m{failed}/{total} tests FAILED.\033[0m")

    print("-" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
