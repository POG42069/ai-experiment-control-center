"""Runtime platform and git metadata detection."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys


def detect_platform() -> str:
    """Detect execution platform: ``kaggle``, ``colab``, or ``local``."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"

    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "colab"

    if "google.colab" in sys.modules:
        return "colab"

    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "colab"
    except Exception:
        pass

    return "local"


def detect_git_info(cwd: str | None = None) -> tuple[str, str]:
    """Return ``(branch, sha)`` for the git repo at ``cwd``, or ``("", "")`` on failure.

    Never raises.
    """
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    sha = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    return branch, sha


def _run_git(args: list[str], cwd: str | None = None) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""
