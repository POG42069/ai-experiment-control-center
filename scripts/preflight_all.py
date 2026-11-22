"""Run a CPU-only Preflight launch (no GPU quota) on every Kaggle account via the audited CLI.

Usage: .\\.venv\\Scripts\\python.exe scripts\\preflight_all.py --repo OWNER/REPO [--branch main] [--accounts kaggle-1,kaggle-2]
Each preflight: plan -> execute (source="claude" audit), prints one summary line per account.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str) -> tuple[int, dict]:
    p = subprocess.run(
        [sys.executable, "-m", "ecc.cli", *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    try:
        data = json.loads(p.stdout) if p.stdout.strip() else {}
    except json.JSONDecodeError:
        data = {"raw": p.stdout[-800:], "stderr": p.stderr[-800:]}
    return p.returncode, data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--accounts", default="kaggle-1,kaggle-2,kaggle-3")
    a = ap.parse_args()
    worst = 0
    for acc in [x.strip() for x in a.accounts.split(",") if x.strip()]:
        rc, plan = _cli("launch", "plan", "--account", acc, "--repo", a.repo, "--branch", a.branch,
                        "--command", "python -c pass", "--accelerator", "none", "--preflight")
        if rc != 0 or "plan_id" not in plan:
            print(f"{acc:9s} PLAN REFUSED rc={rc}: {json.dumps(plan, ensure_ascii=False)[:400]}")
            worst = max(worst, rc or 1)
            continue
        rc, rec = _cli("launch", "execute", "--plan-id", plan["plan_id"])
        print(f"{acc:9s} push rc={rc} ok={rec.get('ok')} version={rec.get('version')} "
              f"kernel={rec.get('kernel_ref')} url={rec.get('url')} msg={str(rec.get('message'))[:200]}")
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
