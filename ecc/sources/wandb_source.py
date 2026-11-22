"""Read-only W&B RunSource implementation (ARCHITECTURE §3 & §5)."""

from __future__ import annotations

import math
import re
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ecc.config import Settings
from ecc.models import LogLine, RunStatus, RunSummary
from ecc.redact import redact
from ecc.sources.base import classify_log_level
from ecc.status import (
    DEFAULT_OFFLINE_AFTER_S,
    DEFAULT_STALE_AFTER_S,
    compute_eta,
    compute_progress,
    derive_status,
)

_TERMINAL_STATUSES = {
    RunStatus.FINISHED,
    RunStatus.FAILED,
    RunStatus.CRASHED,
    RunStatus.KILLED,
}

_KNOWN_LATEST_KEYS = {
    "train/loss",
    "train/lr",
    "train/grad_norm",
    "train/epoch",
    "train/global_step",
    "eval/loss",
    "eval/accuracy",
    "eval/macro_f1",
    "gpu/mem_alloc_gb",
    "gpu/mem_reserved_gb",
    "gpu/max_mem_alloc_gb",
    "gpu/util_pct",
}

_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"\[(?P<stream>stdout|stderr|kaggle|system)\]\s?(?P<text>.*)$"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)) and float(value) > 0:
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return _ensure_utc(datetime.fromisoformat(s))
        except ValueError:
            return None
    return None


def _safe_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    json_dict = getattr(obj, "_json_dict", None)
    if isinstance(json_dict, dict):
        return dict(json_dict)
    if isinstance(obj, Mapping):
        try:
            return dict(obj)
        except Exception:
            return {}
    try:
        return dict(obj)
    except Exception:
        return {}


def _to_finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(val: Any) -> int | None:
    f = _to_finite_float(val)
    if f is None:
        return None
    return int(f)


def _get_attr_any(obj: Any, *names: str, default: Any = None) -> Any:
    for attr in names:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                return val
    return default


def map_wandb_run(
    run_like: Any,
    now: datetime,
    settings: Settings | None = None,
) -> RunSummary:
    """Pure mapper from a W&B run-like object to canonical RunSummary.

    Handles both snake_case (`created_at`, `heartbeat_at`) and camelCase
    (`createdAt`, `heartbeatAt`) attribute spellings as well as string or
    datetime timestamps.
    """
    now_utc = _ensure_utc(now)
    stale_after_s = settings.stale_after_s if settings is not None else DEFAULT_STALE_AFTER_S
    offline_after_s = (
        settings.offline_after_s if settings is not None else DEFAULT_OFFLINE_AFTER_S
    )

    raw_id = str(_get_attr_any(run_like, "id", default="unknown"))
    entity = str(_get_attr_any(run_like, "entity", default="") or "")
    wb_project = str(_get_attr_any(run_like, "project", default="") or "")
    name = str(_get_attr_any(run_like, "name", default=raw_id) or raw_id)
    raw_state = _get_attr_any(run_like, "state", default=None)
    if raw_state is not None:
        raw_state = str(raw_state)

    config = _safe_mapping(getattr(run_like, "config", None))
    summary = _safe_mapping(getattr(run_like, "summary", None))
    raw_tags = getattr(run_like, "tags", None)
    tags = [str(t) for t in raw_tags] if raw_tags else []
    source_url = _get_attr_any(run_like, "url", default=None)

    started_at = _parse_timestamp(_get_attr_any(run_like, "created_at", "createdAt"))
    heartbeat_at = _parse_timestamp(_get_attr_any(run_like, "heartbeat_at", "heartbeatAt"))
    ecc_heartbeat_at = _parse_timestamp(summary.get("ecc/heartbeat"))

    candidates = [dt for dt in (heartbeat_at, ecc_heartbeat_at) if dt is not None]
    last_update = max(candidates) if candidates else None

    status = derive_status(
        raw_state,
        last_update,
        now_utc,
        stale_after_s=stale_after_s,
        offline_after_s=offline_after_s,
    )

    step = _to_int(summary.get("train/global_step"))
    if step is None:
        step = _to_int(summary.get("_step"))

    total_steps = _to_int(config.get("ecc/total_steps"))
    epoch = _to_finite_float(summary.get("train/epoch"))
    if epoch is None:
        epoch = _to_finite_float(summary.get("epoch"))
    total_epochs = _to_finite_float(config.get("ecc/total_epochs"))

    explicit_progress = _to_finite_float(summary.get("ecc/progress"))
    progress = compute_progress(step, total_steps, explicit=explicit_progress)

    latest: dict[str, float] = {}
    for k, v in summary.items():
        if k in _KNOWN_LATEST_KEYS or (isinstance(k, str) and k.startswith("eval/")):
            fv = _to_finite_float(v)
            if fv is not None:
                latest[str(k)] = fv

    best_metric_name = summary.get("ecc/best_metric_name")
    if best_metric_name is not None:
        best_metric_name = str(best_metric_name)
    best_metric = _to_finite_float(summary.get("ecc/best_metric"))
    best_step = _to_int(summary.get("ecc/best_step"))
    best_epoch = _to_finite_float(summary.get("ecc/best_epoch"))

    is_terminal = status in _TERMINAL_STATUSES
    ended_at = (heartbeat_at or last_update) if is_terminal else None

    duration_s: float | None = None
    if started_at is not None and ended_at is not None:
        duration_s = max(0.0, (ended_at - started_at).total_seconds())
    elif started_at is not None and last_update is not None:
        duration_s = max(0.0, (last_update - started_at).total_seconds())
    else:
        duration_s = _to_finite_float(summary.get("_runtime"))

    eta_s = (
        None
        if is_terminal or status == RunStatus.QUEUED
        else compute_eta(step, total_steps, started_at, last_update)
    )

    project_name = str(config.get("ecc/project") or wb_project or "unknown")
    run_id = f"wandb:{entity}/{wb_project}/{raw_id}"

    seed_raw = config.get("ecc/seed", config.get("seed"))
    seed = _to_int(seed_raw)

    return RunSummary(
        run_id=run_id,
        source="wandb",
        project=project_name,
        name=name,
        status=status,
        raw_state=raw_state,
        platform=str(config["ecc/platform"]) if config.get("ecc/platform") is not None else None,
        account=str(config["ecc/account"]) if config.get("ecc/account") is not None else None,
        gpu=str(config["ecc/gpu_name"]) if config.get("ecc/gpu_name") is not None else None,
        model_name=str(config["ecc/model_name"])
        if config.get("ecc/model_name") is not None
        else None,
        config=config,
        tags=tags,
        step=step,
        total_steps=total_steps,
        epoch=epoch,
        total_epochs=total_epochs,
        progress=progress,
        latest=latest,
        best_metric_name=best_metric_name,
        best_metric=best_metric,
        best_step=best_step,
        best_epoch=best_epoch,
        started_at=started_at,
        ended_at=ended_at,
        last_update=last_update,
        duration_s=duration_s,
        eta_s=eta_s,
        notebook_url=str(config["ecc/notebook_url"])
        if config.get("ecc/notebook_url") is not None
        else None,
        source_url=str(source_url) if source_url is not None else None,
        git_branch=str(config["ecc/git_branch"])
        if config.get("ecc/git_branch") is not None
        else None,
        git_sha=str(config["ecc/git_sha"]) if config.get("ecc/git_sha") is not None else None,
        dataset_version=str(config["ecc/dataset_version"])
        if config.get("ecc/dataset_version") is not None
        else None,
        seed=seed,
        is_demo=False,
    )


def parse_console_log_lines(content: str) -> list[LogLine]:
    """Parse ecc_console.log lines prefixed with `ISO8601 [stdout|stderr] `."""
    lines: list[LogLine] = []
    for raw_line in content.splitlines():
        clean_line = redact(raw_line.rstrip("\r\n"))
        m = _LOG_LINE_RE.match(clean_line)
        if m:
            ts = _parse_timestamp(m.group("ts"))
            stream = m.group("stream")
            text = m.group("text")
        else:
            ts = None
            stream = "stdout"
            text = clean_line
        lines.append(
            LogLine(
                ts=ts,
                stream=stream,
                text=text,
                level=classify_log_level(text),
            )
        )
    return lines


def _default_wandb_api() -> Any:
    import wandb

    return wandb.Api(timeout=20)


def _parse_wandb_run_id(run_id: str) -> tuple[str, str, str] | None:
    if not run_id.startswith("wandb:"):
        return None
    rest = run_id[len("wandb:") :]
    parts = rest.split("/")
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def _fetch_metrics_by_keys(wb_run: Any, keys: list[str]) -> pd.DataFrame:
    """Fetch each requested key (plus step/epoch keys) separately and outer-merge on `_step`.

    W&B's sampled `history(keys=...)` only returns rows where ALL requested keys are
    present. Querying one key per call avoids empty/truncated curves when train, eval,
    and epoch metrics are logged on different rows or omitted.
    """
    fetch_keys = list(dict.fromkeys([*keys, "train/global_step", "train/epoch"]))

    def _fetch_one(key: str) -> tuple[pd.DataFrame | None, Exception | None]:
        try:
            sub = wb_run.history(keys=[key], x_axis="_step", samples=2000, pandas=True)
        except Exception as exc:
            return None, exc
        if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
            return None, None
        if key != "_step" and key not in sub.columns:
            return None, None
        frame = sub.copy()
        if "_step" not in frame.columns:
            frame["_step"] = range(len(frame))
        cols = ["_step"] if key == "_step" else ["_step", key]
        frame = frame[cols].drop_duplicates(subset=["_step"], keep="last")
        return frame, None

    max_workers = min(4, max(1, len(fetch_keys)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = list(executor.map(_fetch_one, fetch_keys))

    frames: list[pd.DataFrame] = []
    errors: list[Exception] = []
    succeeded = 0
    for frame, exc in outcomes:
        if exc is not None:
            errors.append(exc)
        else:
            succeeded += 1
            if frame is not None and not frame.empty:
                frames.append(frame)

    if succeeded == 0 and errors:
        non_key_errors = [e for e in errors if not isinstance(e, KeyError)]
        if non_key_errors:
            raise non_key_errors[0]

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        new_cols = [c for c in frame.columns if c == "_step" or c not in merged.columns]
        merged = pd.merge(merged, frame[new_cols], on="_step", how="outer")

    return merged.sort_values("_step", kind="mergesort").reset_index(drop=True)


def _normalize_metrics_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Derive canonical `step` and `epoch` columns using ffill-then-bfill."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["step", "epoch"])

    result = df.copy()
    has_raw_step = "_step" in result.columns
    if has_raw_step:
        result = result.sort_values("_step", kind="mergesort").reset_index(drop=True)

    if "train/global_step" in result.columns:
        step_series = pd.to_numeric(result["train/global_step"], errors="coerce").ffill().bfill()
        if has_raw_step:
            step_series = step_series.fillna(result["_step"])
        result["step"] = step_series
    elif has_raw_step:
        result["step"] = result["_step"]
    elif "step" not in result.columns:
        result["step"] = range(len(result))

    if "train/epoch" in result.columns:
        epoch_series = pd.to_numeric(result["train/epoch"], errors="coerce").ffill().bfill()
        if "epoch" in result.columns:
            epoch_series = epoch_series.fillna(
                pd.to_numeric(result["epoch"], errors="coerce").ffill().bfill()
            )
        result["epoch"] = epoch_series
    elif "epoch" in result.columns:
        result["epoch"] = pd.to_numeric(result["epoch"], errors="coerce").ffill().bfill()
    else:
        result["epoch"] = pd.Series(float("nan"), index=result.index, dtype="float64")

    # Reorder so step and epoch come first
    front = [c for c in ("step", "epoch") if c in result.columns]
    rest = [c for c in result.columns if c not in front]
    result = result[front + rest]

    if not has_raw_step and "step" in result.columns:
        result = result.sort_values("step", kind="mergesort").reset_index(drop=True)

    return result


class WandbSource:
    """Read-only W&B public API RunSource with in-memory TTL caching."""

    name: str = "wandb"

    def __init__(
        self,
        entity: str,
        projects: list[str],
        api_factory: Callable[[], Any] = _default_wandb_api,
        settings: Settings | None = None,
        cache_ttl_s: float = 10.0,
        max_runs: int = 200,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.entity = entity
        self.projects = list(projects)
        self._api_factory = api_factory
        self._settings = settings
        self._cache_ttl_s = cache_ttl_s
        self._max_runs = max_runs
        self._now_fn = now_fn
        self._api: Any = None
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._last_error: str | None = None

    def _get_api(self) -> Any:
        if self._api is None:
            self._api = self._api_factory()
        return self._api

    def _cache_get(self, key: tuple[Any, ...]) -> tuple[bool, Any]:
        entry = self._cache.get(key)
        if entry is None:
            return False, None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return False, None
        return True, value

    def _cache_set(self, key: tuple[Any, ...], value: Any) -> None:
        if self._cache_ttl_s > 0:
            self._cache[key] = (time.monotonic() + self._cache_ttl_s, value)

    def _record_error(self, exc: Exception) -> None:
        self._last_error = redact(f"{type(exc).__name__}: {exc}")

    def _discover_projects(self) -> list[str]:
        """Return configured projects, or auto-discover all projects of the entity (cached 5 min)."""
        if self.projects:
            return list(self.projects)
        if not self.entity:
            return []
        hit, cached = self._cache_get(("projects",))
        if hit:
            return list(cached)
        try:
            names = sorted(
                str(getattr(p, "name", "")) for p in self._get_api().projects(self.entity)
            )
            names = [n for n in names if n]
        except Exception as exc:
            self._record_error(exc)
            return []
        expires_at = time.monotonic() + max(self._cache_ttl_s, 300.0)
        self._cache[("projects",)] = (expires_at, names)
        return list(names)

    def list_projects(self) -> list[str]:
        return self._discover_projects()

    def list_runs(self, project: str | None = None) -> list[RunSummary]:
        if not self.entity:
            return []
        target_projects = [project] if project else self._discover_projects()
        if not target_projects:
            return []

        cache_key = ("list_runs", project)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached)

        now = _ensure_utc(self._now_fn())
        results: list[RunSummary] = []
        try:
            api = self._get_api()
            for proj in target_projects:
                raw_runs = api.runs(
                    f"{self.entity}/{proj}",
                    order="-created_at",
                    per_page=50,
                )
                for idx, r in enumerate(raw_runs):
                    if idx >= self._max_runs:
                        break
                    results.append(map_wandb_run(r, now, self._settings))
            self._last_error = None
        except Exception as exc:
            self._record_error(exc)
            return []

        self._cache_set(cache_key, results)
        return list(results)

    def _fetch_wandb_run_obj(self, run_id: str) -> Any | None:
        parsed = _parse_wandb_run_id(run_id)
        if parsed is None:
            return None
        entity, project, raw_id = parsed
        api = self._get_api()
        return api.run(f"{entity}/{project}/{raw_id}")

    def get_run(self, run_id: str) -> RunSummary | None:
        if _parse_wandb_run_id(run_id) is None:
            return None

        cache_key = ("get_run", run_id)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return cached

        try:
            wb_run = self._fetch_wandb_run_obj(run_id)
            if wb_run is None:
                return None
            summary = map_wandb_run(wb_run, _ensure_utc(self._now_fn()), self._settings)
            self._last_error = None
            self._cache_set(cache_key, summary)
            return summary
        except Exception as exc:
            self._record_error(exc)
            return None

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame:
        if _parse_wandb_run_id(run_id) is None:
            return pd.DataFrame(columns=["step", "epoch"])

        keys_tuple = tuple(sorted(dict.fromkeys(keys))) if keys is not None else None
        cache_key = ("get_metrics", run_id, keys_tuple)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return cached.copy()

        try:
            wb_run = self._fetch_wandb_run_obj(run_id)
            if wb_run is None:
                return pd.DataFrame(columns=["step", "epoch"])

            df: pd.DataFrame | None = None
            if keys is not None:
                df = _fetch_metrics_by_keys(wb_run, keys)
            else:
                try:
                    df = wb_run.history(samples=2000, pandas=True)
                except Exception:
                    df = None
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    if hasattr(wb_run, "scan_history"):
                        rows = list(wb_run.scan_history())
                        df = pd.DataFrame(rows) if rows else pd.DataFrame()

            result = _normalize_metrics_df(df)

            self._last_error = None
            self._cache_set(cache_key, result)
            return result.copy()
        except Exception as exc:
            self._record_error(exc)
            return pd.DataFrame(columns=["step", "epoch"])

    def _download_log_file(self, wb_run: Any, filename: str, tmpdir: str) -> str | None:
        file_obj = wb_run.file(filename)
        if file_obj is None:
            return None
        downloaded = file_obj.download(root=tmpdir, replace=True)
        path: Path | None = None
        if hasattr(downloaded, "name") and isinstance(downloaded.name, str):
            path = Path(downloaded.name)
            if hasattr(downloaded, "close"):
                try:
                    downloaded.close()
                except Exception:
                    pass
        elif isinstance(downloaded, (str, Path)):
            path = Path(downloaded)
        else:
            candidate = Path(tmpdir) / filename
            if candidate.is_file():
                path = candidate

        if path is None or not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        if _parse_wandb_run_id(run_id) is None:
            return []

        cache_key = ("get_logs", run_id)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached[-tail:]) if tail > 0 else []

        try:
            wb_run = self._fetch_wandb_run_obj(run_id)
            if wb_run is None:
                return []

            content: str | None = None
            with tempfile.TemporaryDirectory(prefix="ecc_wb_log_") as tmpdir:
                for fname in ("ecc_console.log", "output.log"):
                    try:
                        content = self._download_log_file(wb_run, fname, tmpdir)
                    except Exception:
                        content = None
                    if content is not None:
                        break

            if content is None:
                lines = [
                    LogLine(
                        ts=None,
                        stream="system",
                        text=(
                            "No console log uploaded for this run yet "
                            "(needs experiment_monitor capture_console=True)."
                        ),
                        level="INFO",
                    )
                ]
            else:
                lines = parse_console_log_lines(content)

            self._last_error = None
            self._cache_set(cache_key, lines)
            return list(lines[-tail:]) if tail > 0 else []
        except Exception as exc:
            self._record_error(exc)
            return [
                LogLine(
                    ts=None,
                    stream="system",
                    text=f"Failed to fetch W&B logs: {self._last_error}",
                    level="WARNING",
                )
            ]

    def health(self) -> tuple[bool, str]:
        if not self.entity:
            return False, "W&B entity not configured"
        if self._last_error:
            return False, self._last_error
        try:
            self._get_api()
            projects = self._discover_projects()
            if self._last_error:
                return False, self._last_error
            mode = "configured" if self.projects else "auto-discovered"
            return True, (
                f"W&B source ready (entity={self.entity}, {mode} projects={len(projects)})"
            )
        except Exception as exc:
            self._record_error(exc)
            return False, self._last_error or "W&B API unavailable"
