"""Read-only: wait for one account's runner kernel to finish, then print its [ECC] log lines.

Usage: .\\.venv\\Scripts\\python.exe scripts\\watch_runner.py --account kaggle-3 --ref carol/mmculture-qa-new-carol
Works without the local SQLite account rows (token read straight from the OS keyring).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecc.adapters.kaggle import KaggleCLI  # noqa: E402
from ecc.secrets import SecretStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--wait-s", type=int, default=540)
    ap.add_argument("--all-lines", action="store_true")
    a = ap.parse_args()
    s = SecretStore()
    cli = KaggleCLI(a.account, lambda: s.get("kaggle", a.account))
    deadline = time.time() + a.wait_s
    status = "?"
    while time.time() < deadline:
        status, _ = cli.status(a.ref)
        if status.upper() not in ("QUEUED", "RUNNING", "NEW_SCRIPT"):
            break
        time.sleep(15)
    print(f"status={status}")
    for line in cli.logs(a.ref, tail_lines=800):
        t = line.text
        if a.all_lines or "[ECC]" in t or t.startswith(("account", "commit", "repo")):
            print("  ", t[:240])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
