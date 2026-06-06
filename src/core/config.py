"""
config.py — Centralized path configuration for GhostTunnel.

All runtime data (accounts, threads, etc.) is stored under DATA_DIR,
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

ACCOUNTS_FILE = os.path.join(DATA_DIR, "accounts.json")
THREADS_FILE = os.path.join(DATA_DIR, "threads.json")

# Ensure the data directory exists on import
os.makedirs(DATA_DIR, exist_ok=True)

# Bootstrap files from environment variables if not present on disk
if not os.path.exists(ACCOUNTS_FILE):
    accounts_json = os.environ.get("GHOSTTUNNEL_ACCOUNTS_JSON")
    if accounts_json:
        try:
            with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
                f.write(accounts_json.strip())
        except Exception as e:
            import sys
            print(f"Failed to bootstrap accounts.json from GHOSTTUNNEL_ACCOUNTS_JSON: {e}", file=sys.stderr)

if not os.path.exists(THREADS_FILE):
    threads_json = os.environ.get("GHOSTTUNNEL_THREADS_JSON")
    if threads_json:
        try:
            with open(THREADS_FILE, "w", encoding="utf-8") as f:
                f.write(threads_json.strip())
        except Exception as e:
            import sys
            print(f"Failed to bootstrap threads.json from GHOSTTUNNEL_THREADS_JSON: {e}", file=sys.stderr)

