"""
config.py — Centralized path configuration for GhostTunnel.

All runtime data (threads, etc.) is stored under DATA_DIR,
which defaults to `./data` relative to the project root. Override with
the GHOSTTUNNEL_DATA_DIR environment variable.
"""
import os

# Project root = two levels up from src/core/config.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.environ.get(
    "GHOSTTUNNEL_DATA_DIR",
    os.path.join(_PROJECT_ROOT, "data"),
)

THREADS_FILE = os.path.join(DATA_DIR, "threads.json")

# Ensure the data directory exists on import
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(THREADS_FILE):
    threads_json = os.environ.get("GHOSTTUNNEL_THREADS_JSON")
    if threads_json:
        try:
            with open(THREADS_FILE, "w", encoding="utf-8") as f:
                f.write(threads_json.strip())
        except Exception as e:
            import sys
            print(f"Failed to bootstrap threads.json from GHOSTTUNNEL_THREADS_JSON: {e}", file=sys.stderr)
