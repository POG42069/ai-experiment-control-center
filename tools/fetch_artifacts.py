"""Download W&B artifacts (latest or a given alias) using the ECC credential store.

Shared by all projects (moved from MMCultureQA's .git/p1_fetch.py, 2026-11-27).
MUST run with the ECC virtualenv (it needs `keyring`) in a real terminal tab:

    C:\\temp_job\\ai-experiment-control-center\\.venv\\Scripts\\python.exe ^
        C:\\temp_job\\ai-experiment-control-center\\tools\\fetch_artifacts.py ^
        --project mmcqa-msa --out C:\\path\\to\\repo\\.git\\phase1_art  name1 name2:v3

Each artifact lands in <out>/<name>/. The W&B cache goes to <out>/../wandb_cache (inside the repo, avoids the
Windows PermissionError of the default cache) and downloads use skip_cache=True.
Prints one line per artifact: "<name> <version> [files...]".
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ECC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECC_ROOT))
from ecc.secrets import SecretStore  # noqa: E402


def _default_entity() -> str:
    """W&B entity from env WANDB_ENTITY, else [wandb].entity of the local (git-ignored) ecc_config.toml."""
    if os.environ.get("WANDB_ENTITY"):
        return os.environ["WANDB_ENTITY"]
    try:
        import tomllib

        with open(ECC_ROOT / "ecc_config.toml", "rb") as fh:
            return str(tomllib.load(fh).get("wandb", {}).get("entity", ""))
    except (OSError, ValueError):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="artifact names, optionally name:alias (default alias: latest)")
    ap.add_argument("--project", required=True, help="W&B project")
    ap.add_argument("--entity", default=None,
                    help="W&B team entity (default: WANDB_ENTITY or [wandb].entity in ecc_config.toml)")
    ap.add_argument("--out", required=True, help="output root directory")
    ap.add_argument("--key-name", default="wandb-main", help="ECC secret nickname of the W&B key")
    args = ap.parse_args()

    entity = args.entity or _default_entity()
    if not entity:
        print("ERROR: no W&B entity (pass --entity or set [wandb].entity in ecc_config.toml)", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_API_KEY"] = SecretStore().get("wandb", args.key_name) or ""
    if not os.environ["WANDB_API_KEY"]:
        print(f"ERROR: no W&B key '{args.key_name}' in the ECC credential store", file=sys.stderr)
        return 2
    os.environ["WANDB_SILENT"] = "true"
    os.environ.setdefault("WANDB_CACHE_DIR", str(out.parent / "wandb_cache"))
    import wandb  # noqa: E402

    api = wandb.Api()
    for spec in args.names:
        name, _, alias = spec.partition(":")
        art = api.artifact(f"{entity}/{args.project}/{name}:{alias or 'latest'}")
        d = Path(art.download(root=str(out / name), skip_cache=True))
        files = sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file())
        print(name, art.version, files, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
