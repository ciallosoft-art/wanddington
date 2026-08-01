#!/usr/bin/env python3
"""
Set up the Anthropic API auth token for Waddington experiments.

Usage:
    python experiments/setup_auth.py --token sk-ant-...

By default the token is written to Waddington's OWN store at ~/.waddington/agent/auth.json — the same
path the C-arm's LLM client resolves (self-contained; no dependency on feynman). Set
WADDINGTON_REUSE_FEYNMAN=1 to write to ~/.feynman instead, or WADDINGTON_AUTH_PATH to point anywhere.
(For interactive OAuth login across providers, prefer `node frontend/bin/waddington.js setup`.)

Alternatively, set the ANTHROPIC_API_KEY environment variable and this
script will read it automatically:
    export ANTHROPIC_API_KEY=sk-ant-...
    python experiments/setup_auth.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _auth_write_path() -> Path:
    if os.environ.get("WADDINGTON_AUTH_PATH"):
        return Path(os.environ["WADDINGTON_AUTH_PATH"])
    if os.environ.get("WADDINGTON_REUSE_FEYNMAN") == "1":
        return Path.home() / ".feynman" / "agent" / "auth.json"
    return Path.home() / ".waddington" / "agent" / "auth.json"


AUTH_PATH = _auth_write_path()
_DEFAULT_EXPIRY_HOURS = 8


def write_token(token: str, expiry_hours: float = _DEFAULT_EXPIRY_HOURS) -> None:
    AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing or create fresh skeleton
    if AUTH_PATH.exists():
        with open(AUTH_PATH) as f:
            auth = json.load(f)
    else:
        auth = {"anthropic": {}}

    expires_ms = int((time.time() + expiry_hours * 3600) * 1000)
    auth["anthropic"]["access"] = token
    auth["anthropic"]["expires"] = expires_ms

    with open(AUTH_PATH, "w") as f:
        json.dump(auth, f, indent=2)

    print(f"Auth token written to {AUTH_PATH}")
    print(f"  Expires in ~{expiry_hours:.0f} hours ({time.strftime('%Y-%m-%d %H:%M', time.localtime(expires_ms/1000))})")


def check_token() -> bool:
    if not AUTH_PATH.exists():
        return False
    with open(AUTH_PATH) as f:
        auth = json.load(f)
    expires = auth.get("anthropic", {}).get("expires", 0)
    remaining = expires / 1000 - time.time()
    if remaining < 300:
        print(f"[WARN] Token expired or expiring in <5 min ({remaining/3600:.1f}h remaining)")
        return False
    print(f"Token valid ({remaining/3600:.1f}h remaining)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure Anthropic API token for Waddington")
    parser.add_argument("--token", default=None, help="Anthropic API token (sk-ant-...)")
    parser.add_argument("--check", action="store_true", help="Check if current token is valid")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_token() else 1)

    token = args.token or os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        print("ERROR: provide --token or set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    if not token.startswith("sk-ant-"):
        print(f"WARNING: token does not look like an Anthropic key (got: {token[:12]}...)")

    write_token(token)


if __name__ == "__main__":
    main()
