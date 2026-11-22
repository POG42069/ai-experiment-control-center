"""Read-only: wait until each account's ecc-runner leaves QUEUED/RUNNING (max ~6 min), then print status + log tail.

Usage: .\\.venv\\Scripts\\python.exe scripts\\runner_report.py [--tail 15] [--wait-s 360]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecc.service import build_service  # noqa: E402
from ecc.launch.service import build_launch_service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=15)
    ap.add_argument("--wait-s", type=int, default=360)
    a = ap.parse_args()
    svc = build_service()
    launch = build_launch_service(svc)
    accounts = [x.nickname for x in svc.list_accounts() if x.platform == "kaggle"]
    deadline = time.time() + a.wait_s
    pending = set(accounts)
    while pending and time.time() < deadline:
        for nick in list(pending):
            if launch.runner_status(nick) not in ("QUEUED", "RUNNING"):
                pending.discard(nick)
        if pending:
            time.sleep(15)
    ksrc = svc._get_kaggle_source()
    for nick in accounts:
        ref = launch.runner_ref(nick)
        print(f"===== {nick} {ref} status={launch.runner_status(nick)}")
        try:
            for line in ksrc._clients[nick].logs(ref, tail_lines=a.tail):
                print("   ", line.text[:180])
        except Exception as exc:  # noqa: BLE001
            print("    (logs unavailable:", str(exc)[:200], ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
