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
