"""Read-only multi-account Kaggle CLI adapter and RunSource (ARCHITECTURE §1 & §8).

Security & isolation rules:
- Only read-only subcommands are permitted via an explicit allow-list:
  ``("kernels", "list")``, ``("kernels", "status")``, and ``("kernels", "logs")``
  without ``-f`` / ``--follow``.
- Each account runs with its own ``KAGGLE_API_TOKEN`` injected via subprocess
  ``env=`` (never argv) and an isolated ``KAGGLE_CONFIG_DIR`` under
  ``%LOCALAPPDATA%\\ai-ecc\\kaggle\\<nickname>`` so ``~/.kaggle`` is never read
  or modified.
- All inherited ``KAGGLE_*`` environment variables are stripped before injecting
  the per-account credentials.
- Any stderr/exception text is scrubbed of both known secret patterns (``redact``)
  and the active account token before raising or recording.

Status & heartbeat note:
- The heartbeat staleness rule (``stale_after_s`` / ``offline_after_s``) is
  designed for W&B runs that emit periodic heartbeats.
- Kaggle's ``kernels status`` endpoint reports ``RUNNING`` authoritatively from
  Kaggle's worker state without a per-minute heartbeat timestamp. Therefore, for
  ``RUNNING`` Kaggle kernels, ``KaggleSource`` sets ``last_update = now`` and
  labels ``raw_state = "RUNNING (Kaggle API)"`` so the UI and callers know the
  status came from Kaggle's native API rather than a W&B heartbeat.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from ecc.config import Settings
from ecc.models import LogLine, RunStatus, RunSummary, to_jsonable
from ecc.redact import REDACTED, redact
from ecc.sources.base import classify_log_level
from ecc.status import DEFAULT_OFFLINE_AFTER_S, DEFAULT_STALE_AFTER_S, derive_status
from ecc.store import Store

ALLOWED_SUBCOMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("kernels", "list"),
        ("kernels", "status"),
        ("kernels", "logs"),
    }
)

DEFAULT_KAGGLE_LOOKBACK_HOURS = 72
DEFAULT_MAX_STATUS_PER_ACCOUNT = 15
DEFAULT_CACHE_TTL_S = 30.0

_STATUS_RE = re.compile(
    r'has\s+status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)"?',
    re.IGNORECASE,
)
_BARE_STATUS_RE = re.compile(
    r'^\s*"?(?:KernelWorkerStatus\.)?(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL_REQUESTED|CANCEL_ACKNOWLEDGED|NEW_SCRIPT)"?\s*$',
    re.IGNORECASE,
)
_FAILURE_MSG_RE = re.compile(r"(?im)^\s*Failure\s+message\s*:\s*(.+?)\s*$")
_ISO_PREFIX_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+(?P<rest>.*)$"
)
_TERMINAL_STATUSES = {
    RunStatus.FINISHED,
    RunStatus.FAILED,
    RunStatus.CRASHED,
    RunStatus.KILLED,
}


class KaggleCLIError(RuntimeError):
    """Raised when a read-only Kaggle CLI invocation fails."""


@dataclass
class KaggleKernel:
    """Metadata for a single Kaggle kernel returned by ``kaggle kernels list``."""

    ref: str
    title: str
    author: str
    last_run_time: datetime | None
    url: str
    account: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "title": self.title,
            "author": self.author,
            "last_run_time": to_jsonable(self.last_run_time),
            "url": self.url,
            "account": self.account,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_kaggle_exe() -> str:
    candidate = Path(sys.executable).parent / "kaggle.exe"
    if candidate.is_file():
        return str(candidate)
    return "kaggle"


def _parse_kaggle_datetime(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return _ensure_utc(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # Only treat numeric timestamps >= 1e9 as Unix epoch seconds.
        # Kaggle nbconvert JSON logs often emit relative elapsed seconds (e.g. 1.24).
        val = float(raw)
        if val >= 1_000_000_000:
            try:
                return datetime.fromtimestamp(val, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return _ensure_utc(datetime.fromisoformat(s))
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _sanitize_text(text: str, token: str | None = None) -> str:
    cleaned = text
    if token:
        cleaned = cleaned.replace(token, REDACTED)
    return redact(cleaned)


def parse_status_output(text: str) -> tuple[str, str | None]:
    """Parse stdout of ``kaggle kernels status owner/slug``.

    Handles both ``owner/slug has status "KernelWorkerStatus.RUNNING"`` and
    ``owner/slug has status "RUNNING"``, plus optional ``Failure message: "..."``.
    """
    if not text or not text.strip():
        return "UNKNOWN", None

    status_str = "UNKNOWN"
    m = _STATUS_RE.search(text)
    if m:
        status_str = m.group(1).strip().upper()
    else:
        for line in text.splitlines():
            bm = _BARE_STATUS_RE.match(line)
            if bm:
                status_str = bm.group(1).strip().upper()
                break

    failure_msg: str | None = None
    fm = _FAILURE_MSG_RE.search(text)
    if fm:
        raw_msg = fm.group(1).strip()
        if len(raw_msg) >= 2 and (
            (raw_msg[0] == '"' and raw_msg[-1] == '"')
            or (raw_msg[0] == "'" and raw_msg[-1] == "'")
        ):
            raw_msg = raw_msg[1:-1].strip()
        if raw_msg and raw_msg.lower() != "none":
            failure_msg = redact(raw_msg)

    return status_str, failure_msg


def parse_kaggle_logs(
    text: str,
    tail_lines: int = 500,
    token: str | None = None,
) -> list[LogLine]:
    """Parse ``kaggle kernels logs`` output (plain text or JSON array)."""
    if not text:
        return []

    lines: list[LogLine] = []
    stripped = text.strip()

    parsed_json: Any = None
    if stripped.startswith("["):
        try:
            candidate = json.loads(stripped)
            if isinstance(candidate, list) and all(isinstance(x, dict) for x in candidate):
                parsed_json = candidate
        except json.JSONDecodeError:
            parsed_json = None

    if parsed_json is not None:
        for entry in parsed_json:
            ts = _parse_kaggle_datetime(entry.get("time"))
            data = entry.get("data", "")
            if data is None:
                continue
            data_str = str(data)
            split_lines = data_str.splitlines()
            if not split_lines and data_str:
                split_lines = [data_str]
            for raw_line in split_lines:
                clean = _sanitize_text(raw_line.rstrip("\r\n"), token)
                lines.append(
                    LogLine(
                        ts=ts,
                        stream="kaggle",
                        text=clean,
                        level=classify_log_level(clean),
                    )
                )
    else:
        for raw_line in text.splitlines():
            clean = _sanitize_text(raw_line.rstrip("\r\n"), token)
            ts: datetime | None = None
            body = clean
            im = _ISO_PREFIX_RE.match(clean)
            if im:
                parsed_ts = _parse_kaggle_datetime(im.group("ts"))
                if parsed_ts is not None:
                    ts = parsed_ts
                    body = im.group("rest")
            lines.append(
                LogLine(
                    ts=ts,
                    stream="kaggle",
                    text=body,
                    level=classify_log_level(body),
                )
            )

    if tail_lines <= 0:
        return []
    return lines[-tail_lines:]


def _parse_kernels_csv(csv_text: str, account: str) -> list[KaggleKernel]:
    if not csv_text or not csv_text.strip():
        return []

    non_empty_lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not non_empty_lines:
        return []

    # Kaggle CLI prints plain message like "No kernels found" when empty.
    first_line = non_empty_lines[0]
    if "," not in first_line and "ref" not in first_line.lower():
        return []

    reader = csv.DictReader(io.StringIO("\n".join(non_empty_lines)))
    if not reader.fieldnames:
        return []

    kernels: list[KaggleKernel] = []
    for row in reader:
        norm_row: dict[str, str] = {}
        for k, v in row.items():
            if k is None:
                continue
            norm_row[k.strip().lower()] = (v or "").strip()

        ref = norm_row.get("ref", "")
        if not ref:
            continue
        title = norm_row.get("title") or (ref.split("/")[-1] if "/" in ref else ref)
        author = norm_row.get("author") or (ref.split("/")[0] if "/" in ref else "")
        last_run_raw = norm_row.get("lastruntime") or norm_row.get("last_run_time") or ""
        last_run_time = _parse_kaggle_datetime(last_run_raw)
        kernels.append(
            KaggleKernel(
                ref=ref,
                title=title,
                author=author,
                last_run_time=last_run_time,
                url=f"https://www.kaggle.com/code/{ref}",
                account=account,
            )
        )
    return kernels


class KaggleCLI:
    """Read-only Kaggle CLI subprocess wrapper isolated per account."""

    def __init__(
        self,
        nickname: str,
        token_getter: Callable[[], str | None],
        kaggle_exe: str | Path | None = None,
        timeout_s: float = 20,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.nickname = nickname
        self._token_getter = token_getter
        self.kaggle_exe = str(kaggle_exe) if kaggle_exe is not None else _default_kaggle_exe()
        self.timeout_s = timeout_s
        self._runner = runner

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
        """Build isolated subprocess environment with per-account token and config dir."""
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("KAGGLE_")}
        env["KAGGLE_API_TOKEN"] = token
        env["KAGGLE_CONFIG_DIR"] = str(self._config_dir())
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def _validate_command(args: Sequence[str]) -> None:
        if len(args) < 2:
            raise PermissionError(f"Disallowed Kaggle CLI command: {list(args)!r}")
        cmd_pair = (args[0].strip().lower(), args[1].strip().lower())
        if cmd_pair not in ALLOWED_SUBCOMMANDS:
            raise PermissionError(
                f"Disallowed Kaggle CLI subcommand {cmd_pair!r}; "
                f"only read-only {sorted(ALLOWED_SUBCOMMANDS)} are permitted"
            )
        if cmd_pair == ("kernels", "logs"):
            for flag in args[2:]:
                f_norm = flag.strip()
                if f_norm in {"-f", "--follow"} or f_norm.startswith("--follow="):
                    raise PermissionError(
                        "Live streaming ('-f' / '--follow') is not allowed in read-only MVP adapter"
                    )
                if f_norm.startswith("-") and not f_norm.startswith("--") and "f" in f_norm[1:]:
                    raise PermissionError(
                        "Live streaming ('-f') is not allowed in read-only MVP adapter"
                    )

    def run(self, args: Sequence[str]) -> str:
        """Execute an allow-listed read-only Kaggle CLI command and return stdout."""
        arg_list = [str(a) for a in args]
        self._validate_command(arg_list)

        token = self._token_getter()
        if not token:
            raise KaggleCLIError(
                f"No Kaggle API token available for account '{self.nickname}'"
            )

        # Defense in depth: never allow the token to appear in argv.
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
        except subprocess.TimeoutExpired as exc:
            raise KaggleCLIError(
                f"Kaggle CLI timed out after {self.timeout_s}s for account '{self.nickname}'"
            ) from exc
        except OSError as exc:
            safe_exc = _sanitize_text(str(exc), token)
            raise KaggleCLIError(
                f"Failed to invoke Kaggle CLI for account '{self.nickname}': {safe_exc}"
            ) from exc

        returncode = getattr(proc, "returncode", 0)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""

        if returncode != 0:
            err_detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            safe_err = _sanitize_text(err_detail, token)
            raise KaggleCLIError(
                f"Kaggle CLI failed (exit {returncode}) for account '{self.nickname}': {safe_err}"
            )

        return stdout

    def check(self) -> tuple[bool, str]:
        """Verify credentials using ``kaggle kernels list --mine --page-size 1``."""
        self.run(["kernels", "list", "--mine", "--page-size", "1"])
        return True, f"Kaggle CLI authenticated for account '{self.nickname}'"

    def list_kernels(self, page_size: int = 100) -> list[KaggleKernel]:
        """List kernels owned by this account via ``kaggle kernels list --mine --csv``."""
        output = self.run(
            ["kernels", "list", "--mine", "--csv", "--page-size", str(page_size)]
        )
        return _parse_kernels_csv(output, account=self.nickname)

    def status(self, ref: str) -> tuple[str, str | None]:
        """Fetch kernel status string and optional failure message."""
        output = self.run(["kernels", "status", ref])
        return parse_status_output(output)

    def logs(self, ref: str, tail_lines: int = 500) -> list[LogLine]:
        """Fetch persisted execution logs (one-shot, no ``-f`` streaming)."""
        output = self.run(["kernels", "logs", ref])
        token = self._token_getter()
        return parse_kaggle_logs(output, tail_lines=tail_lines, token=token)

    def __repr__(self) -> str:
        return f"KaggleCLI(nickname={self.nickname!r}, kaggle_exe={self.kaggle_exe!r})"


def _parse_kaggle_run_id(run_id: str) -> tuple[str, str] | None:
    """Parse ``kaggle:<nickname>/<owner>/<slug>`` into ``(nickname, owner/slug)``."""
    if not run_id.startswith("kaggle:"):
        return None
    rest = run_id[len("kaggle:") :]
    if "/" not in rest:
        return None
    nickname, ref = rest.split("/", 1)
    if not nickname or not ref:
        return None
    return nickname, ref


class KaggleSource:
    """Multi-account read-only Kaggle RunSource with concurrency and TTL cache."""

    name: str = "kaggle"

    def __init__(
        self,
        accounts: list[tuple[str, KaggleCLI]],
        store: Store,
        settings: Settings | None = None,
        *,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        max_status_per_account: int = DEFAULT_MAX_STATUS_PER_ACCOUNT,
        max_workers: int = 3,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._accounts: list[tuple[str, KaggleCLI]] = list(accounts)
        self._clients: dict[str, KaggleCLI] = {nick: cli for nick, cli in self._accounts}
        self.store = store
        self.settings = settings
        self._cache_ttl_s = cache_ttl_s
        self._max_status_per_account = max_status_per_account
        self._max_workers = max(1, min(int(max_workers), 3))
        self._now_fn = now_fn
        self._lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._account_errors: dict[str, str] = {}

    @property
    def lookback_hours(self) -> int:
        if self.settings is not None:
            return int(getattr(self.settings, "kaggle_lookback_hours", DEFAULT_KAGGLE_LOOKBACK_HOURS))
        return DEFAULT_KAGGLE_LOOKBACK_HOURS

    def _cache_get(self, key: tuple[Any, ...]) -> tuple[bool, Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False, None
            expires_at, val = entry
            if time.monotonic() >= expires_at:
                self._cache.pop(key, None)
                return False, None
            return True, val

    def _cache_set(self, key: tuple[Any, ...], val: Any) -> None:
        if self._cache_ttl_s <= 0:
            return
        with self._lock:
            self._cache[key] = (time.monotonic() + self._cache_ttl_s, val)

    def _record_account_error(self, nickname: str, exc: Exception) -> None:
        msg = redact(f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._account_errors[nickname] = msg
        try:
            self.store.record_account_check(nickname, ok=False, msg=msg)
        except Exception:
            pass

    def _clear_account_error(self, nickname: str) -> None:
        with self._lock:
            self._account_errors.pop(nickname, None)

    def _fetch_kernels_for_account(self, nickname: str, cli: KaggleCLI) -> list[KaggleKernel]:
        cache_key = ("kernels", nickname)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached)
        try:
            kernels = cli.list_kernels()
            self._clear_account_error(nickname)
            self._cache_set(cache_key, kernels)
            return list(kernels)
        except Exception as exc:
            self._record_account_error(nickname, exc)
            return []

    def list_kernels(self, nickname: str | None = None) -> list[KaggleKernel]:
        """List kernels across configured Kaggle accounts concurrently (max 3 workers)."""
        if nickname is not None:
            cli = self._clients.get(nickname)
            if cli is None:
                return []
            targets = [(nickname, cli)]
        else:
            targets = list(self._accounts)

        if not targets:
            return []

        all_kernels: list[KaggleKernel] = []
        if len(targets) == 1:
            nick, cli = targets[0]
            all_kernels.extend(self._fetch_kernels_for_account(nick, cli))
        else:
            workers = min(self._max_workers, len(targets))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._fetch_kernels_for_account, nick, cli): nick
                    for nick, cli in targets
                }
                for fut in as_completed(futures):
                    all_kernels.extend(fut.result())

        all_kernels.sort(
            key=lambda k: (
                0 if k.last_run_time is not None else 1,
                -_ensure_utc(k.last_run_time).timestamp() if k.last_run_time is not None else 0.0,
                k.ref,
            )
        )
        return all_kernels

    def _fetch_runs_for_account(
        self, nickname: str, cli: KaggleCLI, now: datetime
    ) -> list[RunSummary]:
        cache_key = ("runs", nickname)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached)

        try:
            kernels = cli.list_kernels()
            self._clear_account_error(nickname)
            self._cache_set(("kernels", nickname), kernels)
        except Exception as exc:
            self._record_account_error(nickname, exc)
            return []

        cutoff_s = float(self.lookback_hours) * 3600.0
        recent: list[KaggleKernel] = []
        for k in kernels:
            if k.last_run_time is None:
                continue
            age_s = (now - _ensure_utc(k.last_run_time)).total_seconds()
            if age_s <= cutoff_s:
                recent.append(k)

        recent.sort(
            key=lambda k: -_ensure_utc(k.last_run_time).timestamp()
            if k.last_run_time is not None
            else 0.0
        )
        recent = recent[: self._max_status_per_account]

        stale_after_s = (
            self.settings.stale_after_s if self.settings is not None else DEFAULT_STALE_AFTER_S
        )
        offline_after_s = (
            self.settings.offline_after_s
            if self.settings is not None
            else DEFAULT_OFFLINE_AFTER_S
        )

        runs: list[RunSummary] = []
        for k in recent:
            try:
                kaggle_status, failure_msg = cli.status(k.ref)
            except Exception as exc:
                self._record_account_error(nickname, exc)
                continue

            # Kaggle's native API reports RUNNING authoritatively without a periodic
            # heartbeat (heartbeat staleness is for W&B runs). Use last_update = now
            # and clearly annotate raw_state as coming from the Kaggle API.
            if kaggle_status == "RUNNING":
                last_update: datetime | None = now
                raw_state = "RUNNING (Kaggle API)"
            else:
                last_update = k.last_run_time
                raw_state = kaggle_status

            derived = derive_status(
                kaggle_status,
                last_update,
                now,
                stale_after_s=stale_after_s,
                offline_after_s=offline_after_s,
            )

            cfg: dict[str, Any] = {
                "ecc/platform": "kaggle",
                "ecc/account": nickname,
                "ecc/kaggle_kernel": k.ref,
                "ecc/notebook_url": k.url,
            }
            if failure_msg:
                cfg["kaggle/failure_message"] = failure_msg

            ended_at = k.last_run_time if derived in _TERMINAL_STATUSES else None

            runs.append(
                RunSummary(
                    run_id=f"kaggle:{nickname}/{k.ref}",
                    source="kaggle",
                    project="unassigned",
                    name=k.title or k.ref,
                    status=derived,
                    raw_state=raw_state,
                    platform="kaggle",
                    account=nickname,
                    gpu=None,
                    model_name=None,
                    config=cfg,
                    tags=[],
                    step=None,
                    total_steps=None,
                    epoch=None,
                    total_epochs=None,
                    progress=None,
                    latest={},
                    best_metric_name=None,
                    best_metric=None,
                    best_step=None,
                    best_epoch=None,
                    started_at=k.last_run_time,
                    ended_at=ended_at,
                    last_update=last_update,
                    duration_s=None,
                    eta_s=None,
                    notebook_url=k.url,
                    source_url=k.url,
                    git_branch=None,
                    git_sha=None,
                    dataset_version=None,
                    seed=None,
                    is_demo=False,
                )
            )

        self._cache_set(cache_key, runs)
        return list(runs)

    def _apply_kernel_link_projects(self, runs: list[RunSummary]) -> list[RunSummary]:
        links = {
            link["kernel_ref"]: link["project"]
            for link in self.store.list_kernel_links()
            if link.get("kernel_ref") and link.get("project")
        }
        if not links:
            return runs
        updated: list[RunSummary] = []
        for r in runs:
            ref = r.config.get("ecc/kaggle_kernel")
            proj = links.get(ref) if isinstance(ref, str) else None
            if proj and proj != r.project:
                updated.append(replace(r, project=proj))
            else:
                updated.append(r)
        return updated

    def list_runs(self, project: str | None = None) -> list[RunSummary]:
        if not self._accounts:
            return []

        now = _ensure_utc(self._now_fn())
        raw_runs: list[RunSummary] = []

        if len(self._accounts) == 1:
            nick, cli = self._accounts[0]
            raw_runs.extend(self._fetch_runs_for_account(nick, cli, now))
        else:
            workers = min(self._max_workers, len(self._accounts))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._fetch_runs_for_account, nick, cli, now): nick
                    for nick, cli in self._accounts
                }
                for fut in as_completed(futures):
                    raw_runs.extend(fut.result())

        runs = self._apply_kernel_link_projects(raw_runs)
        if project is not None:
            runs = [r for r in runs if r.project == project]
        return runs

    def get_run(self, run_id: str) -> RunSummary | None:
        parsed = _parse_kaggle_run_id(run_id)
        if parsed is None:
            return None
        nickname, _ = parsed
        if nickname not in self._clients:
            return None
        for r in self.list_runs():
            if r.run_id == run_id:
                return r
        return None

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["step", "epoch"])

    def get_kernel_logs(
        self,
        ref: str,
        nickname: str | None = None,
        tail_lines: int = 500,
    ) -> list[LogLine]:
        """Fetch persisted Kaggle logs for ``ref`` using ``nickname`` (or matching client)."""
        candidates: list[tuple[str, KaggleCLI]] = []
        if nickname and nickname in self._clients:
            candidates.append((nickname, self._clients[nickname]))
        else:
            # Try matching link account first, then fall back to all configured accounts.
            link_acc: str | None = None
            for link in self.store.list_kernel_links():
                if link.get("kernel_ref") == ref and link.get("account") in self._clients:
                    link_acc = str(link["account"])
                    break
            if link_acc is not None:
                candidates.append((link_acc, self._clients[link_acc]))
            for nick, cli in self._accounts:
                if all(n != nick for n, _ in candidates):
                    candidates.append((nick, cli))

        for nick, cli in candidates:
            try:
                return cli.logs(ref, tail_lines=tail_lines)
            except Exception as exc:
                self._record_account_error(nick, exc)
                if nickname is not None:
                    break
        return []

    def get_kernel_status(
        self,
        ref: str,
        nickname: str | None = None,
    ) -> tuple[str, str | None] | None:
        """Fetch native Kaggle status for ``ref`` (cached via list_runs when available)."""
        for r in self.list_runs():
            if r.config.get("ecc/kaggle_kernel") == ref:
                if nickname is None or r.account == nickname:
                    return (
                        r.raw_state or r.status.value,
                        r.config.get("kaggle/failure_message"),
                    )

        cli = self._clients.get(nickname) if nickname else None
        if cli is None and self._accounts:
            cli = self._accounts[0][1]
        if cli is None:
            return None
        try:
            k_status, fail_msg = cli.status(ref)
            label = "RUNNING (Kaggle API)" if k_status == "RUNNING" else k_status
            return label, fail_msg
        except Exception as exc:
            self._record_account_error(cli.nickname, exc)
            return None

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        parsed = _parse_kaggle_run_id(run_id)
        if parsed is None:
            return []
        nickname, ref = parsed
        if nickname not in self._clients:
            return []
        return self.get_kernel_logs(ref=ref, nickname=nickname, tail_lines=tail)

    def list_projects(self) -> list[str]:
        projects: set[str] = set()
        known_accounts = set(self._clients.keys())
        for link in self.store.list_kernel_links():
            if link.get("account") in known_accounts and link.get("project"):
                projects.add(str(link["project"]))
        with self._lock:
            for (kind, _), (_, val) in self._cache.items():
                if kind == "runs" and isinstance(val, list):
                    for r in val:
                        if isinstance(r, RunSummary) and r.project:
                            projects.add(r.project)
        return sorted(projects)

    def health(self) -> tuple[bool, str]:
        if not self._accounts:
            return False, "No Kaggle accounts configured"
        with self._lock:
            errors = dict(self._account_errors)
        if len(errors) >= len(self._accounts):
            summary = "; ".join(f"{k}: {v}" for k, v in sorted(errors.items()))
            return False, redact(summary)
        if errors:
            failed = ", ".join(sorted(errors))
            return True, f"Kaggle source partially healthy ({len(self._accounts) - len(errors)}/{len(self._accounts)} ok; errors: {failed})"
        return True, f"Kaggle source ready ({len(self._accounts)} account(s))"
