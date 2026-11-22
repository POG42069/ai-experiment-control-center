"""Read-only check: for each configured Kaggle account, show which Kaggle user owns its kernels.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\verify_accounts.py
Never prints secrets; only nickname, kernel author(s), kernel count and most recent kernel.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecc.service import build_service  # noqa: E402


def main() -> int:
    svc = build_service()
    accounts = [a for a in svc.list_accounts() if a.platform == "kaggle"]
    if not accounts:
        print("No Kaggle accounts configured.")
        return 1
    owners: dict[str, str] = {}
    for acc in accounts:
        kernels = svc.list_kaggle_kernels(nickname=acc.nickname)
        authors = Counter(k.author for k in kernels if getattr(k, "author", None))
        main_author = authors.most_common(1)[0][0] if authors else "?"
        owners[acc.nickname] = main_author
        latest = kernels[0].ref if kernels else "-"
        print(f"{acc.nickname:10s} author={main_author:20s} kernels={len(kernels):3d} latest={latest}")
    dup = [o for o, n in Counter(owners.values()).items() if n > 1 and o != "?"]
    if dup:
        print(f"WARNING: several nicknames map to the same Kaggle user: {dup}")
        return 2
    print("OK: every Kaggle nickname maps to a different Kaggle user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
