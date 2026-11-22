"""Create/refresh the PRIVATE Kaggle dataset `<owner>/ecc-credentials` on each Kaggle account.

Why: Kaggle Secrets are not delivered to notebook versions pushed through the API (HTTP 400),
so the launch runner reads `ecc_credentials.json` from this private dataset instead.

The user runs this script themself:
    .\\.venv\\Scripts\\python.exe scripts\\setup_credentials_dataset.py
- W&B API key: read from Windows Credential Manager (stored earlier with add_account.py --platform wandb).
- GitHub token: read from the clipboard after you press Enter (then the clipboard is cleared).
Values are never printed; the staging file is overwritten and deleted after upload.
The dataset is created PRIVATE (the --public flag is never passed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecc.launch.service import build_launch_service  # noqa: E402
from ecc.redact import redact  # noqa: E402
from ecc.secrets import SecretStore  # noqa: E402
from ecc.service import build_service  # noqa: E402

SLUG = "ecc-credentials"


def _clipboard() -> str:
    ps = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    try:
        p = subprocess.run(ps + ["[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"],
                           capture_output=True, timeout=15)
        return p.stdout.decode("utf-8", "replace").strip()
    finally:
        subprocess.run(ps + ["Set-Clipboard -Value ' '"], capture_output=True, timeout=15)


def _kaggle(nick: str, token: str, args: list[str]) -> subprocess.CompletedProcess:
    cfg = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "ai-ecc" / "kaggle" / nick
    cfg.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("KAGGLE_")}
    env.update(KAGGLE_API_TOKEN=token, KAGGLE_CONFIG_DIR=str(cfg), PYTHONIOENCODING="utf-8")
    assert "--public" not in args and "-u" not in args, "datasets must stay private"
    exe = Path(sys.executable).parent / "kaggle.exe"
    return subprocess.run([str(exe), *args], env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", default="", help="comma list of nicknames (default: all Kaggle accounts)")
    ap.add_argument("--wandb-account", default="", help="wandb nickname in keyring (default: first one)")
    a = ap.parse_args()

    svc = build_service()
    launch = build_launch_service(svc)
    secrets = SecretStore()
    accounts = svc.list_accounts()
    kaggle = [x.nickname for x in accounts if x.platform == "kaggle"]
    if a.accounts:
        kaggle = [n for n in kaggle if n in {s.strip() for s in a.accounts.split(",")}]
    wandb_nicks = [x.nickname for x in accounts if x.platform == "wandb"]
    wnick = a.wandb_account or (wandb_nicks[0] if wandb_nicks else "")
    wandb_key = secrets.get("wandb", wnick) if wnick else None
    if not kaggle or not wandb_key:
        print("Need >=1 Kaggle account and a stored W&B key (add_account.py --platform wandb).")
        return 1

    input("Copy your GitHub fine-grained token (github_pat_...) to the clipboard, then press Enter... ")
    gh_token = _clipboard()
    if not gh_token.startswith(("github_pat_", "ghp_")) or any(c.isspace() for c in gh_token):
        print("Clipboard does not contain a GitHub token (expected github_pat_...). Nothing uploaded.")
        return 1
    print(f"GitHub token read ({len(gh_token)} chars); clipboard cleared. W&B key read from keyring ({wnick}).")

    secret_values = [wandb_key, gh_token]
    rc_all = 0
    for nick in kaggle:
        token = secrets.get("kaggle", nick)
        owner = launch.account_owner(nick)
        if not token or not owner:
            print(f"{nick:9s} SKIP (missing Kaggle token or owner)")
            rc_all = 1
            continue
        ref = f"{owner}/{SLUG}"
        stage = Path(tempfile.mkdtemp(prefix="ecc_cred_"))
        cred_file = stage / "ecc_credentials.json"
        try:
            cred_file.write_text(json.dumps({"WANDB_API_KEY": wandb_key, "GITHUB_TOKEN": gh_token}),
                                 encoding="utf-8")
            (stage / "dataset-metadata.json").write_text(json.dumps({
                "title": SLUG, "id": ref, "licenses": [{"name": "other"}],
            }), encoding="utf-8")
            exists = _kaggle(nick, token, ["datasets", "status", ref]).returncode == 0
            args = (["datasets", "version", "-p", str(stage), "-m", "rotate credentials", "-q"] if exists
                    else ["datasets", "create", "-p", str(stage), "-q"])
            p = _kaggle(nick, token, args)
            out = (p.stdout + p.stderr)
            for v in secret_values + [token]:
                out = out.replace(v, "***")
            state = "versioned" if exists else "created"
            print(f"{nick:9s} {ref:40s} {'OK ' + state if p.returncode == 0 else 'FAILED'} "
                  f"{redact(out).strip()[:200]}")
            rc_all = max(rc_all, 0 if p.returncode == 0 else 1)
        finally:
            if cred_file.exists():
                cred_file.write_text("{}", encoding="utf-8")
            shutil.rmtree(stage, ignore_errors=True)
    print("Done. Datasets are PRIVATE. Kaggle may take ~1 minute to process a new dataset.")
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
