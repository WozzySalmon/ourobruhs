#!/usr/bin/env python3
"""
server_launcher.py — entrypoint for running Ouroboros on a real server (non-Colab).

Usage:
    1. Copy .env.example to .env, fill in your keys.
    2. python3 server_launcher.py

This is just a thin wrapper that loads .env and calls colab_launcher.py.
The name 'colab_launcher' is historical; it now auto-detects its environment.
"""
import os
import pathlib
import sys


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        print(f"[server_launcher] No .env at {path}. Copy .env.example and fill in your keys.")
        sys.exit(1)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        # Don't overwrite env vars already set (e.g. from systemd EnvironmentFile)
        if key not in os.environ:
            os.environ[key] = val


# ── Load .env from project root (optional — systemd EnvironmentFile is preferred)
_here = pathlib.Path(__file__).parent.resolve()
load_dotenv(_here / ".env")

# ── Set data directory defaults (can be overridden in .env)
os.environ.setdefault("OUROBOROS_DRIVE_ROOT", str(_here / "data"))
os.environ.setdefault("OUROBOROS_REPO_DIR",   str(_here))

# ── Hand off to main launcher
import colab_launcher  # noqa: E402, F401
