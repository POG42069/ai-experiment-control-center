"""Wheel builder and Kaggle kernel folder assembler for ECC Launch Control."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from ecc.launch.runner_template import RunnerParams, render_runner
from ecc.redact import redact


class WheelBuildError(RuntimeError):
    """Raised when building the ``experiment_monitor`` wheel fails."""


_WHEEL_CACHE_LOCK = threading.Lock()
_WHEEL_CACHE: dict[tuple[str, str], bytes] = {}


def default_monitor_dir() -> Path:
    """Return the repository's ``monitor/`` directory."""
    return Path(__file__).resolve().parent.parent.parent / "monitor"


def clear_wheel_cache() -> None:
    """Clear the in-memory wheel cache (primarily for tests)."""
    with _WHEEL_CACHE_LOCK:
        _WHEEL_CACHE.clear()


def _compute_monitor_hash(monitor_dir: Path) -> str:
    """Compute a deterministic SHA-256 digest of ``pyproject.toml`` and ``src/**``."""
    digest = hashlib.sha256()

    pyproject = monitor_dir / "pyproject.toml"
    if pyproject.is_file():
        digest.update(b"pyproject.toml\0")
        digest.update(pyproject.read_bytes())

    src_dir = monitor_dir / "src"
    if src_dir.is_dir():
        files = sorted(
            (
                p
                for p in src_dir.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and not p.name.endswith((".pyc", ".pyo"))
            ),
            key=lambda p: p.relative_to(monitor_dir).as_posix(),
        )
        for path in files:
            rel = path.relative_to(monitor_dir).as_posix().encode("utf-8")
            digest.update(rel + b"\0")
            digest.update(path.read_bytes())

    return digest.hexdigest()


def build_wheel(
    monitor_dir: str | Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
    python_exe: str | Path | None = None,
    use_cache: bool = True,
) -> bytes:
    """Build ``experiment_monitor`` wheel via ``python -m pip wheel --no-deps`` and cache by content hash."""
    monitor_path = Path(monitor_dir).resolve()
    if not monitor_path.is_dir():
        raise WheelBuildError(f"Monitor package directory does not exist: {monitor_path}")

    content_hash = _compute_monitor_hash(monitor_path)
    cache_key = (str(monitor_path), content_hash)

    if use_cache:
        with _WHEEL_CACHE_LOCK:
            cached = _WHEEL_CACHE.get(cache_key)
            if cached is not None:
                return cached

    py = str(python_exe) if python_exe is not None else sys.executable
    with tempfile.TemporaryDirectory(prefix="ecc-wheel-") as tmp_str:
        tmp_dir = Path(tmp_str)
        argv = [py, "-m", "pip", "wheel", str(monitor_path), "--no-deps", "-w", str(tmp_dir)]
        try:
            proc = runner(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            raise WheelBuildError(
                f"Failed to invoke pip wheel for {monitor_path}: {redact(str(exc))}"
            ) from exc

        returncode = getattr(proc, "returncode", 0)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise WheelBuildError(
                f"pip wheel failed (exit {returncode}): {redact(detail)}"
            )

        wheels = sorted(tmp_dir.glob("*.whl"))
        if not wheels:
            raise WheelBuildError(
                f"pip wheel succeeded but produced no .whl file in {tmp_dir}"
            )
        wheel_bytes = wheels[-1].read_bytes()

    if use_cache:
        with _WHEEL_CACHE_LOCK:
            _WHEEL_CACHE[cache_key] = wheel_bytes

    return wheel_bytes


_SCRIPT_FOOTER = 'if __name__ == "__main__":\n    raise SystemExit(main())'
_NOTEBOOK_FOOTER = (
    "_ecc_rc = main()\n"
    "if _ecc_rc:\n"
    "    # In a notebook SystemExit would be swallowed and Kaggle would report COMPLETE.\n"
    '    raise RuntimeError(f"ECC runner failed with exit code {_ecc_rc}")'
)


def runner_notebook(runner_source: str) -> dict[str, Any]:
    """Wrap the rendered runner script into a one-cell nbformat-4 notebook that fails loudly."""
    if _SCRIPT_FOOTER not in runner_source:
        raise ValueError("unexpected runner template footer")
    cell_source = runner_source.replace(_SCRIPT_FOOTER, _NOTEBOOK_FOOTER, 1)
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_source,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_folder(
    params: RunnerParams,
    wheel_bytes: bytes,
    dest_dir: str | Path,
) -> Path:
    """Write the runner notebook (``<slug>.ipynb``) and ``kernel-metadata.json`` into ``dest_dir``.

    The runner is shipped as a one-cell notebook (kernel_type "notebook") so it can reuse
    notebooks the user already created on Kaggle (Kaggle may refuse to change a kernel's type).
    ``runner.py`` is also written next to it for review/debugging; only the notebook is pushed.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    wheel_b64 = base64.b64encode(wheel_bytes).decode("ascii")
    runner_source = render_runner(params, wheel_b64)
    (dest / "runner.py").write_text(runner_source, encoding="utf-8", newline="\n")
    notebook_file = f"{params.runner_slug}.ipynb"
    (dest / notebook_file).write_text(
        json.dumps(runner_notebook(runner_source), indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if params.preflight:
        enable_gpu = False
        enable_tpu = False
    else:
        acc = (params.accelerator or "").strip()
        enable_gpu = acc.startswith("Nvidia")
        enable_tpu = acc.startswith("Tpu")

    metadata: dict[str, Any] = {
        "id": f"{params.owner}/{params.runner_slug}",
        "title": params.runner_slug,
        "code_file": notebook_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": enable_tpu,
        "enable_internet": True,
        "dataset_sources": list(params.dataset_sources),
        "kernel_sources": [],
        "model_sources": [],
    }
    (dest / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return dest
