"""Isolated Kaggle kernel push launcher with a dedicated single-command allow-list.

Security rules:
- Separate allow-list containing ONLY ``("kernels", "push")``.
- Rejects all other Kaggle subcommands (``kernels delete``, ``kernels status``,
  ``datasets create``, etc.) and rejects unexpected/disallowed flags (including
  ``--no-run`` tricks).
- Subprocess environment mirrors ``ecc.adapters.kaggle.KaggleCLI``: strips all
  inherited ``KAGGLE_*`` variables, injects ``KAGGLE_API_TOKEN`` via ``env=``
  (never argv), and uses an isolated per-account ``KAGGLE_CONFIG_DIR``.
- All output messages are scrubbed of both the active token and known secret patterns.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ecc.redact import REDACTED, redact

ALLOWED_LAUNCHER_SUBCOMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("kernels", "push"),
    }
)

_ALLOWED_PUSH_FLAGS: frozenset[str] = frozenset(
    {
        "-p",
        "--path",
        "--accelerator",
        "-t",
        "--timeout",
    }
)

_VERSION_RE = re.compile(
    r"Kernel\s+version\s+(\d+)\s+successfully\s+pushed",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(https://www\.kaggle\.com/code/[A-Za-z0-9_./-]+)")
_ERROR_RE = re.compile(r"Kernel\s+push\s+error\s*:\s*(.+)", re.IGNORECASE)


class KaggleLaunchError(RuntimeError):
    """Raised when KaggleLauncher cannot invoke a push (e.g., missing token)."""


@dataclass(frozen=True)
class PushResult:
    ok: bool
    version: int | None
    url: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "url": self.url,
            "message": self.message,
        }


def _default_kaggle_exe() -> str:
    candidate = Path(sys.executable).parent / "kaggle.exe"
    if candidate.is_file():
        return str(candidate)
    return "kaggle"


def _sanitize_text(text: str, token: str | None = None) -> str:
    cleaned = text or ""
    if token:
        cleaned = cleaned.replace(token, REDACTED)
    return redact(cleaned)


def parse_push_output(
    text: str,
    *,
    returncode: int = 0,
    token: str | None = None,
) -> PushResult:
    """Parse ``kaggle kernels push`` output into a redacted ``PushResult``."""
    cleaned = _sanitize_text(text, token).strip()

    if returncode != 0 or _ERROR_RE.search(cleaned):
        msg = cleaned or f"Kaggle push failed (exit code {returncode})"
        return PushResult(ok=False, version=None, url=None, message=msg)

    vm = _VERSION_RE.search(cleaned)
    um = _URL_RE.search(cleaned)
    url = um.group(1).rstrip(".,;)") if um else None

    if vm:
        return PushResult(
            ok=True,
            version=int(vm.group(1)),
            url=url,
            message=cleaned,
        )

    if "successfully pushed" in cleaned.lower():
        return PushResult(ok=True, version=None, url=url, message=cleaned)

    return PushResult(
        ok=False,
        version=None,
        url=url,
        message=cleaned or "Unexpected empty output from kaggle kernels push",
    )


class KaggleLauncher:
    """Dedicated Kaggle launcher allowing ONLY ``kaggle kernels push``."""

    def __init__(
        self,
        nickname: str,
        token_getter: Callable[[], str | None],
        kaggle_exe: str | Path | None = None,
        runner: Callable[..., Any] = subprocess.run,
        timeout_s: float = 120,
    ) -> None:
        self.nickname = nickname
        self._token_getter = token_getter
        self.kaggle_exe = str(kaggle_exe) if kaggle_exe is not None else _default_kaggle_exe()
        self._runner = runner
        self.timeout_s = timeout_s

    def _config_dir(self) -> Path:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            target = Path(local_app_data) / "ai-ecc" / "kaggle" / self.nickname
        else:
            target = Path.home() / ".ai-ecc" / "kaggle" / self.nickname
        try:
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "ai-ecc" / "kaggle" / self.nickname
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def build_env(self, token: str) -> dict[str, str]:
        """Build isolated subprocess environment identical to ``KaggleCLI.build_env``."""
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("KAGGLE_")}
        env["KAGGLE_API_TOKEN"] = token
        env["KAGGLE_CONFIG_DIR"] = str(self._config_dir())
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def _validate_command(args: Sequence[str]) -> None:
        if len(args) < 2:
            raise PermissionError(f"Disallowed Kaggle launcher command: {list(args)!r}")

        cmd_pair = (args[0].strip().lower(), args[1].strip().lower())
        if cmd_pair not in ALLOWED_LAUNCHER_SUBCOMMANDS:
            raise PermissionError(
                f"Disallowed Kaggle launcher subcommand {cmd_pair!r}; "
                f"only {sorted(ALLOWED_LAUNCHER_SUBCOMMANDS)} is permitted"
            )

        idx = 2
        while idx < len(args):
            tok = str(args[idx]).strip()
            if not tok.startswith("-"):
                raise PermissionError(
                    f"Unexpected positional argument in kaggle kernels push: {tok!r}"
                )
            flag_name, has_eq, _val = tok.partition("=")
            if flag_name not in _ALLOWED_PUSH_FLAGS:
                raise PermissionError(
                    f"Disallowed flag for KaggleLauncher: {tok!r}"
                )
            if has_eq:
                idx += 1
            else:
                if idx + 1 >= len(args):
                    raise PermissionError(f"Flag {tok!r} requires a value")
                idx += 2

    def run(self, args: Sequence[str]) -> PushResult:
        """Validate and execute an allow-listed ``kaggle kernels push`` command."""
        arg_list = [str(a) for a in args]
        self._validate_command(arg_list)

        token = self._token_getter()
        if not token:
            raise KaggleLaunchError(
                f"No Kaggle API token available for account '{self.nickname}'"
            )

        if any(token in a for a in arg_list):
            raise PermissionError("Kaggle API token must never be passed in CLI arguments")

        env = self.build_env(token)
        argv = [self.kaggle_exe, *arg_list]

        try:
            proc = self._runner(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PushResult(
                ok=False,
                version=None,
                url=None,
                message=f"Kaggle push timed out after {self.timeout_s}s for account '{self.nickname}'",
            )
        except OSError as exc:
            safe_exc = _sanitize_text(str(exc), token)
            return PushResult(
                ok=False,
                version=None,
                url=None,
                message=f"Failed to invoke Kaggle CLI for account '{self.nickname}': {safe_exc}",
            )

        returncode = getattr(proc, "returncode", 0)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
        return parse_push_output(combined, returncode=returncode, token=token)

    def push(
        self,
        folder: str | Path,
        accelerator: str | None = None,
        timeout_s: int | None = None,
    ) -> PushResult:
        """Push a prepared Kaggle kernel folder."""
        args = ["kernels", "push", "-p", str(folder)]
        if accelerator is not None:
            acc = accelerator.strip()
            if acc and acc.lower() != "none":
                args.extend(["--accelerator", acc])
        if timeout_s is not None:
            args.extend(["-t", str(int(timeout_s))])
        return self.run(args)

    def __repr__(self) -> str:
        return f"KaggleLauncher(nickname={self.nickname!r}, kaggle_exe={self.kaggle_exe!r})"
