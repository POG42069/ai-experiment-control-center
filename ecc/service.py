"""Read-only high-level service API (ARCHITECTURE §7)."""

from __future__ import annotations

import math
import sqlite3
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from ecc.accounts import AccountManager, AccountStatus
from ecc.adapters.kaggle import KaggleCLI, KaggleKernel, KaggleSource
from ecc.config import Settings, load_config
from ecc.models import LogLine, Project, RunStatus, RunSummary
from ecc.redact import redact
from ecc.secrets import SecretStore
from ecc.sources.base import RunSource, classify_log_level
from ecc.sources.demo import DemoSource
from ecc.sources.wandb_source import WandbSource
from ecc.store import Store

ACTIVE_STATUSES = {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.STALE}
FAILED_STATUSES = {RunStatus.FAILED, RunStatus.CRASHED, RunStatus.KILLED}
STANDARD_CURVE_METRICS = [
    "train/loss",
    "eval/loss",
    "eval/accuracy",
    "eval/macro_f1",
    "train/lr",
    "gpu/mem_alloc_gb",
]


class RunNotFoundError(KeyError):
    """Raised when a requested run_id does not exist in any configured source."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_status_filter(
    status: RunStatus | str | Iterable[RunStatus | str] | None,
) -> set[RunStatus] | None:
    if status is None:
        return None
    items: list[RunStatus | str]
    if isinstance(status, (RunStatus, str)):
        if isinstance(status, str) and "," in status:
            items = [s.strip() for s in status.split(",") if s.strip()]
        else:
            items = [status]
    else:
        items = list(status)

    result: set[RunStatus] = set()
    for item in items:
        if isinstance(item, RunStatus):
            result.add(item)
        else:
            try:
                result.add(RunStatus(str(item).strip().upper()))
            except ValueError as exc:
                raise ValueError(f"Unknown run status: {item!r}") from exc
    return result


def _last_valid_float(df: pd.DataFrame, col: str, fallback: float | None = None) -> float | None:
    if not df.empty and col in df.columns:
        valid = df[col].dropna()
        if not valid.empty:
            val = float(valid.iloc[-1])
            if math.isfinite(val):
                return val
    if fallback is not None and math.isfinite(float(fallback)):
        return float(fallback)
    return None


def _max_valid_float(df: pd.DataFrame, col: str, fallback: float | None = None) -> float | None:
    if not df.empty and col in df.columns:
        valid = df[col].dropna()
        if not valid.empty:
            val = float(valid.max())
            if math.isfinite(val):
                return val
    if fallback is not None and math.isfinite(float(fallback)):
        return float(fallback)
    return None


def _eval_cadence(df: pd.DataFrame, metric: str) -> float | None:
    if df.empty or "step" not in df.columns:
        return None
    col = metric if metric in df.columns else ("eval/loss" if "eval/loss" in df.columns else None)
    if col is None:
        return None
    valid = df.dropna(subset=["step", col])
    if len(valid) < 2:
        return None
    diffs = valid["step"].astype(float).diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    return float(diffs.median())


class Service:
    """Aggregates read-only RunSources with local SQLite metadata."""

    def __init__(
        self,
        sources: list[RunSource],
        store: Store,
        settings: Settings,
        now_fn: Callable[[], datetime] = _utc_now,
        account_manager: AccountManager | None = None,
        secrets: SecretStore | None = None,
    ) -> None:
        self.sources = list(sources)
        self.store = store
        self.settings = settings
        self._now_fn = now_fn
        self._secrets = secrets
        self._account_manager = account_manager

    def _get_account_manager(self) -> AccountManager:
        if self._account_manager is None:
            secrets = self._secrets if self._secrets is not None else SecretStore()
            self._account_manager = AccountManager(
                store=self.store, secrets=secrets, settings=self.settings
            )
        return self._account_manager

    def _get_kaggle_source(self) -> KaggleSource | None:
        for src in self.sources:
            if isinstance(src, KaggleSource):
                return src
        return None

    @staticmethod
    def _attach_kaggle_enrichment(
        run: RunSummary,
        kaggle_status: str,
        kernel_ref: str,
        *,
        account: str | None = None,
        notebook_url: str | None = None,
        failure_msg: str | None = None,
    ) -> RunSummary:
        new_config = dict(run.config or {})
        new_config["kaggle_status"] = kaggle_status
        new_config.setdefault("ecc/kaggle_kernel", kernel_ref)
        if account:
            new_config.setdefault("ecc/kaggle_account", account)
        if failure_msg:
            new_config["kaggle/failure_message"] = failure_msg
        enriched = replace(
            run,
            config=new_config,
            notebook_url=run.notebook_url or notebook_url,
        )
        setattr(enriched, "kaggle_status", kaggle_status)
        return enriched

    def _resolve_linked_kernel(
        self, run: RunSummary, link_by_run_id: dict[str, dict[str, Any]] | None = None
    ) -> tuple[str | None, str | None]:
        ref_val = (run.config or {}).get("ecc/kaggle_kernel")
        ref = str(ref_val).strip() if isinstance(ref_val, str) and ref_val.strip() else None
        acc_val = (run.config or {}).get("ecc/kaggle_account") or run.account
        account = str(acc_val).strip() if isinstance(acc_val, str) and acc_val.strip() else None

        links = link_by_run_id
        if links is None:
            links = {
                str(link["run_id"]): link
                for link in self.store.list_kernel_links()
                if link.get("run_id")
            }

        link = links.get(run.run_id)
        if link is not None:
            if not ref and link.get("kernel_ref"):
                ref = str(link["kernel_ref"])
            if link.get("account"):
                account = str(link["account"])
        return ref, account

    def _run_project_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for link in self.store.list_kernel_links():
            rid = link.get("run_id")
            proj = link.get("project")
            if rid and proj:
                overrides[str(rid)] = str(proj)
        return overrides

    def _apply_project_override(
        self, run: RunSummary, overrides: dict[str, str]
    ) -> RunSummary:
        override = overrides.get(run.run_id)
        if override and override != run.project:
            return replace(run, project=override)
        return run

    def _find_run_and_source(self, run_id: str) -> tuple[RunSummary, RunSource]:
        overrides = self._run_project_overrides()
        for src in self.sources:
            run = src.get_run(run_id)
            if run is not None:
                return self._apply_project_override(run, overrides), src
        raise RunNotFoundError(f"Unknown run_id: {run_id}")

    # ------------------------------------------------------------------
    # Public read-only API (ARCHITECTURE §7)
    # ------------------------------------------------------------------
    def list_projects(self) -> list[Project]:
        by_slug: dict[str, Project] = {}
        for row in self.store.list_projects():
            by_slug[row["slug"]] = Project(
                slug=row["slug"],
                name=row["name"],
                description=row["description"],
                tags=list(row["tags"]),
                notes=row["notes"],
                baseline_run_id=row["baseline_run_id"],
            )

        for src in self.sources:
            for proj in src.list_projects():
                if proj not in by_slug:
                    by_slug[proj] = Project(
                        slug=proj,
                        name=proj,
                        description="",
                        tags=[],
                        notes="",
                        baseline_run_id=None,
                    )

        return [by_slug[slug] for slug in sorted(by_slug)]

    def list_runs(
        self,
        project: str | None = None,
        status: RunStatus | str | Iterable[RunStatus | str] | None = None,
    ) -> list[RunSummary]:
        status_set = _normalize_status_filter(status)
        kernel_links = self.store.list_kernel_links()
        overrides: dict[str, str] = {}
        link_by_run_id: dict[str, dict[str, Any]] = {}
        for link in kernel_links:
            rid = link.get("run_id")
            proj = link.get("project")
            if rid:
                link_by_run_id[str(rid)] = link
                if proj:
                    overrides[str(rid)] = str(proj)

        has_kaggle_source = self._get_kaggle_source() is not None
        fetch_unfiltered = bool(overrides or has_kaggle_source)

        seen: set[str] = set()
        raw_runs: list[RunSummary] = []
        for src in self.sources:
            src_runs = src.list_runs(project=None if fetch_unfiltered else project)
            for run in src_runs:
                if run.run_id in seen:
                    continue
                seen.add(run.run_id)
                raw_runs.append(run)

        # Merge rule: if a non-Kaggle (e.g. W&B) run links to a Kaggle kernel via
        # config `ecc/kaggle_kernel == ref` or store `kernel_links`, hide the standalone
        # Kaggle run and attach `kaggle_status` to the primary run instead.
        kaggle_by_ref: dict[str, RunSummary] = {}
        for r in raw_runs:
            if r.source == "kaggle":
                k_ref = (r.config or {}).get("ecc/kaggle_kernel")
                if isinstance(k_ref, str) and k_ref:
                    kaggle_by_ref[k_ref] = r

        merged_kaggle_ids: set[str] = set()
        merged_runs: list[RunSummary] = []
        for r in raw_runs:
            if r.source == "kaggle":
                continue
            ref, _ = self._resolve_linked_kernel(r, link_by_run_id=link_by_run_id)
            if ref and ref in kaggle_by_ref:
                k_run = kaggle_by_ref[ref]
                merged_kaggle_ids.add(k_run.run_id)
                k_status = k_run.raw_state or k_run.status.value
                r = self._attach_kaggle_enrichment(
                    r,
                    k_status,
                    ref,
                    account=k_run.account,
                    notebook_url=k_run.notebook_url,
                    failure_msg=(k_run.config or {}).get("kaggle/failure_message"),
                )
            merged_runs.append(r)

        for r in raw_runs:
            if r.source == "kaggle" and r.run_id not in merged_kaggle_ids:
                merged_runs.append(r)

        runs: list[RunSummary] = []
        for run in merged_runs:
            adjusted = self._apply_project_override(run, overrides)
            # Re-attach kaggle_status attribute if _apply_project_override used replace().
            if "kaggle_status" in (adjusted.config or {}) and not hasattr(
                adjusted, "kaggle_status"
            ):
                setattr(adjusted, "kaggle_status", adjusted.config["kaggle_status"])
            if project is not None and adjusted.project != project:
                continue
            if status_set is not None and adjusted.status not in status_set:
                continue
            runs.append(adjusted)

        def _sort_key(r: RunSummary) -> tuple[int, float, str]:
            active_rank = 0 if r.status in ACTIVE_STATUSES else 1
            ts = r.last_update or r.started_at
            ts_val = -_ensure_utc(ts).timestamp() if ts is not None else 0.0
            return (active_rank, ts_val, r.run_id)

        runs.sort(key=_sort_key)
        return runs

    def list_active_runs(self, project: str | None = None) -> list[RunSummary]:
        return self.list_runs(project=project, status=ACTIVE_STATUSES)

    def get_run_details(self, run_id: str) -> RunSummary | None:
        try:
            run, _ = self._find_run_and_source(run_id)
        except RunNotFoundError:
            return None

        if run.source != "kaggle":
            kaggle_src = self._get_kaggle_source()
            if kaggle_src is not None:
                ref, account = self._resolve_linked_kernel(run)
                if ref:
                    st = kaggle_src.get_kernel_status(ref, nickname=account)
                    if st is not None:
                        k_status, fail_msg = st
                        run = self._attach_kaggle_enrichment(
                            run,
                            k_status,
                            ref,
                            account=account,
                            failure_msg=fail_msg,
                        )
        return run

    def get_run_metrics(
        self, run_id: str, keys: list[str] | None = None
    ) -> pd.DataFrame:
        _, src = self._find_run_and_source(run_id)
        return src.get_metrics(run_id, keys=keys)

    def get_native_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        """Fetch native platform logs (e.g., Kaggle CLI logs) for a run."""
        run, src = self._find_run_and_source(run_id)
        if run.source == "kaggle":
            return src.get_logs(run_id, tail=tail)

        kaggle_src = self._get_kaggle_source()
        if kaggle_src is None:
            return []
        ref, account = self._resolve_linked_kernel(run)
        if not ref:
            return []
        return kaggle_src.get_kernel_logs(ref=ref, nickname=account, tail_lines=tail)

    def get_run_logs(
        self,
        run_id: str,
        tail: int = 500,
        query: str | None = None,
        level: str | None = None,
        source: str | None = None,
    ) -> list[LogLine]:
        _, src = self._find_run_and_source(run_id)
        fetch_tail = max(tail * 4, 2000) if (query or level) else tail
        source_norm = source.strip().lower() if isinstance(source, str) and source.strip() else None
        if source_norm == "kaggle":
            raw_lines = self.get_native_logs(run_id, tail=fetch_tail)
        elif source_norm in (None, "auto", "default", "wandb"):
            raw_lines = src.get_logs(run_id, tail=fetch_tail)
        else:
            raise ValueError(f"Unsupported log source: {source!r}")

        level_norm = level.strip().upper() if level else None
        if level_norm is not None and level_norm not in {"INFO", "WARNING", "ERROR"}:
            raise ValueError(f"Invalid log level filter: {level!r}")
        query_norm = query.lower() if query else None

        filtered: list[LogLine] = []
        for line in raw_lines:
            derived_level = classify_log_level(line.text)
            normalized_line = (
                line if line.level == derived_level else replace(line, level=derived_level)
            )
            if level_norm is not None and normalized_line.level != level_norm:
                continue
            if query_norm is not None and query_norm not in normalized_line.text.lower():
                continue
            filtered.append(normalized_line)

        if tail <= 0:
            return []
        return filtered[-tail:]

    def compare_runs(
        self,
        run_ids: list[str],
        metric: str = "eval/macro_f1",
        axis: str = "step",
    ) -> dict[str, Any]:
        if not (2 <= len(run_ids) <= 10):
            raise ValueError(
                f"compare_runs requires between 2 and 10 run_ids, got {len(run_ids)}"
            )
        if axis not in ("step", "epoch"):
            raise ValueError(f"compare_runs axis must be 'step' or 'epoch', got {axis!r}")

        runs: list[RunSummary] = []
        metrics_by_id: dict[str, pd.DataFrame] = {}
        for rid in run_ids:
            run, src = self._find_run_and_source(rid)
            runs.append(run)
            metrics_by_id[rid] = src.get_metrics(rid, keys=None)

        # Build comparison table
        table_rows: list[dict[str, Any]] = []
        for r in runs:
            df = metrics_by_id[r.run_id]
            cfg = r.config or {}
            table_rows.append(
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "platform": r.platform,
                    "account": r.account,
                    "gpu": r.gpu,
                    "model": r.model_name or cfg.get("ecc/model_name") or cfg.get("model_name"),
                    "lora_r": cfg.get("lora_r"),
                    "lora_alpha": cfg.get("lora_alpha"),
                    "learning_rate": cfg.get("learning_rate", cfg.get("lr")),
                    "prompt_strategy": cfg.get(
                        "ecc/prompt_strategy", cfg.get("prompt_strategy")
                    ),
                    "seed": r.seed if r.seed is not None else cfg.get("ecc/seed", cfg.get("seed")),
                    "dataset_version": r.dataset_version or cfg.get("ecc/dataset_version"),
                    "best_metric": r.best_metric,
                    "best_step": r.best_step,
                    "best_epoch": r.best_epoch,
                    "duration_s": r.duration_s,
                    "final train/loss": _last_valid_float(
                        df, "train/loss", r.latest.get("train/loss")
                    ),
                    "final eval/loss": _last_valid_float(
                        df, "eval/loss", r.latest.get("eval/loss")
                    ),
                    "eval/accuracy": _last_valid_float(
                        df, "eval/accuracy", r.latest.get("eval/accuracy")
                    ),
                    "eval/macro_f1": _last_valid_float(
                        df, "eval/macro_f1", r.latest.get("eval/macro_f1")
                    ),
                    "max gpu/mem_alloc_gb": _max_valid_float(
                        df,
                        "gpu/mem_alloc_gb",
                        r.latest.get("gpu/max_mem_alloc_gb", r.latest.get("gpu/mem_alloc_gb")),
                    ),
                }
            )
        table_df = pd.DataFrame(table_rows)

        # Build aligned curves (long format: run_id, name, x, value)
        curve_metrics = list(dict.fromkeys([metric, *STANDARD_CURVE_METRICS]))
        curves: dict[str, pd.DataFrame] = {}
        for m in curve_metrics:
            frames: list[pd.DataFrame] = []
            for r in runs:
                df = metrics_by_id[r.run_id]
                if df.empty or axis not in df.columns or m not in df.columns:
                    continue
                sub = df[[axis, m]].dropna()
                if sub.empty:
                    continue
                frames.append(
                    pd.DataFrame(
                        {
                            "run_id": r.run_id,
                            "name": r.name,
                            "x": sub[axis].to_numpy(),
                            "value": sub[m].to_numpy(),
                        }
                    )
                )
            if frames:
                curves[m] = pd.concat(frames, ignore_index=True)
            elif m == metric:
                curves[m] = pd.DataFrame(columns=["run_id", "name", "x", "value"])

        # Comparability warnings (ARCHITECTURE §7)
        warnings: list[str] = []

        # 1. Demo + real mix
        has_demo = any(r.is_demo for r in runs)
        has_real = any(not r.is_demo for r in runs)
        if has_demo and has_real:
            warnings.append(
                "Comparison mixes synthetic demo runs with real runs; metrics are not directly comparable."
            )

        # 2. Differing ecc/eval_signature
        eval_sigs = {
            r.config.get("ecc/eval_signature")
            for r in runs
            if r.config.get("ecc/eval_signature") is not None
        }
        missing_sig = any(r.config.get("ecc/eval_signature") is None for r in runs)
        if len(eval_sigs) > 1 or (len(eval_sigs) == 1 and missing_sig):
            warnings.append(
                f"Runs have differing ecc/eval_signature values ({sorted(str(s) for s in eval_sigs)}); "
                "validation comparability is not guaranteed."
            )

        # 3. Differing ecc/dataset_version
        ds_versions = {
            (r.dataset_version or r.config.get("ecc/dataset_version"))
            for r in runs
            if (r.dataset_version or r.config.get("ecc/dataset_version")) is not None
        }
        missing_ds = any(
            (r.dataset_version or r.config.get("ecc/dataset_version")) is None for r in runs
        )
        if len(ds_versions) > 1 or (len(ds_versions) == 1 and missing_ds):
            warnings.append(
                f"Runs use differing ecc/dataset_version values ({sorted(str(v) for v in ds_versions)})."
            )

        # 4. Differing ecc/dataset_split
        ds_splits = {
            r.config.get("ecc/dataset_split")
            for r in runs
            if r.config.get("ecc/dataset_split") is not None
        }
        missing_split = any(r.config.get("ecc/dataset_split") is None for r in runs)
        if len(ds_splits) > 1 or (len(ds_splits) == 1 and missing_split):
            warnings.append(
                f"Runs use differing ecc/dataset_split values ({sorted(str(v) for v in ds_splits)})."
            )

        # 5. Differing eval cadence
        cadences = {
            c
            for c in (_eval_cadence(metrics_by_id[r.run_id], metric) for r in runs)
            if c is not None
        }
        if len(cadences) > 1:
            warnings.append(
                f"Runs evaluate at different step cadences ({sorted(cadences)} steps)."
            )

        # 6. Requested metric missing in a run
        missing_metric_runs: list[str] = []
        for r in runs:
            df = metrics_by_id[r.run_id]
            in_df = (not df.empty) and (metric in df.columns) and (not df[metric].dropna().empty)
            in_latest = metric in r.latest and math.isfinite(float(r.latest[metric]))
            if not in_df and not in_latest:
                missing_metric_runs.append(r.name)
        if missing_metric_runs:
            warnings.append(
                f"Metric '{metric}' is missing in run(s): {', '.join(missing_metric_runs)}."
            )

        return {
            "runs": [r.to_dict() for r in runs],
            "table": table_df,
            "curves": curves,
            "warnings": warnings,
        }

    def get_failed_runs(
        self,
        project: str | None = None,
        since_hours: float = 72,
    ) -> list[RunSummary]:
        now = _ensure_utc(self._now_fn())
        candidates = self.list_runs(project=project, status=FAILED_STATUSES)
        if since_hours is None:
            return candidates

        cutoff_s = float(since_hours) * 3600.0
        result: list[RunSummary] = []
        for run in candidates:
            ref_ts = run.ended_at or run.last_update or run.started_at
            if ref_ts is None:
                result.append(run)
            else:
                age_s = (now - _ensure_utc(ref_ts)).total_seconds()
                if age_s <= cutoff_s:
                    result.append(run)
        return result

    def _run_metric_score(self, run: RunSummary, metric: str, mode: str) -> float | None:
        candidates: list[float] = []
        if metric in run.latest:
            val = float(run.latest[metric])
            if math.isfinite(val):
                candidates.append(val)
        if run.best_metric_name == metric and run.best_metric is not None:
            val = float(run.best_metric)
            if math.isfinite(val):
                candidates.append(val)
        if not candidates:
            return None
        return max(candidates) if mode == "max" else min(candidates)

    def get_best_experiment(
        self,
        project: str,
        metric: str = "eval/macro_f1",
        mode: str = "max",
    ) -> dict[str, Any]:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        runs = self.list_runs(project=project)
        best_run: RunSummary | None = None
        best_val: float | None = None

        for run in runs:
            score = self._run_metric_score(run, metric, mode)
            if score is None:
                continue
            if best_val is None or (
                (mode == "max" and score > best_val) or (mode == "min" and score < best_val)
            ):
                best_run = run
                best_val = score

        proj_meta = self.store.get_project(project)
        baseline_run: RunSummary | None = None
        baseline_val: float | None = None
        if proj_meta and proj_meta.get("baseline_run_id"):
            baseline_run = self.get_run_details(proj_meta["baseline_run_id"])
            if baseline_run is not None:
                baseline_val = self._run_metric_score(baseline_run, metric, mode)

        delta: float | None = None
        if best_val is not None and baseline_val is not None:
            delta = best_val - baseline_val

        return {
            "project": project,
            "metric": metric,
            "mode": mode,
            "best": best_run.to_dict() if best_run is not None else None,
            "best_value": best_val,
            "baseline": baseline_run.to_dict() if baseline_run is not None else None,
            "baseline_value": baseline_val,
            "delta": delta,
        }

    def list_kaggle_kernels(self, nickname: str | None = None) -> list[KaggleKernel]:
        """List Kaggle kernels across configured Kaggle accounts (or one account)."""
        kaggle_src = self._get_kaggle_source()
        if kaggle_src is None:
            return []
        return kaggle_src.list_kernels(nickname=nickname)

    def link_kernel(
        self,
        kernel_ref: str,
        account: str,
        project: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Associate a Kaggle kernel with an ECC project (and optionally a W&B run_id)."""
        return self.store.link_kernel(
            kernel_ref=kernel_ref,
            account=account,
            project=project,
            run_id=run_id,
        )

    def list_accounts(self) -> list[AccountStatus]:
        """List configured accounts and their non-sensitive status/capabilities."""
        return self._get_account_manager().list()

    def health(self) -> dict[str, Any]:
        sources_health: dict[str, dict[str, Any]] = {}
        overall_ok = True
        for src in self.sources:
            try:
                ok, msg = src.health()
            except Exception as exc:
                ok, msg = False, f"{type(exc).__name__}: {exc}"
            sources_health[src.name] = {"ok": bool(ok), "message": redact(msg)}
            if not ok:
                overall_ok = False

        return {
            "ok": overall_ok,
            "demo_mode": self.settings.demo_mode,
            "db_path": str(self.settings.db_path),
            "sources": sources_health,
        }


def _open_store_with_fallback(db_path: Path) -> Store:
    """Open Store at db_path, falling back to temp dir or in-memory if unwritable."""
    try:
        return Store(db_path)
    except (PermissionError, OSError, sqlite3.OperationalError):
        fallback_path = Path(tempfile.gettempdir()) / "ai-ecc" / "ecc.db"
        try:
            return Store(fallback_path)
        except (PermissionError, OSError, sqlite3.OperationalError):
            return Store(":memory:")


def build_service(config_path: str | Path | None = None) -> Service:
    """Wire DemoSource, WandbSource, and KaggleSource according to ecc_config.toml."""
    settings = load_config(config_path)
    store = _open_store_with_fallback(settings.db_path)
    if store.db_path != str(settings.db_path):
        settings.general.db_path = Path(store.db_path)

    secrets = SecretStore()
    account_manager = AccountManager(store=store, secrets=secrets, settings=settings)

    sources: list[RunSource] = []
    # W&B is enabled once an entity is configured; an empty project list means
    # "auto-discover every project of the entity" (WandbSource._discover_projects).
    has_wandb = bool(settings.wandb.entity)
    if has_wandb:
        wandb_nicks = [
            acc["nickname"]
            for acc in store.list_accounts()
            if acc.get("platform") == "wandb" and acc.get("enabled", True)
        ]

        def _wandb_api_factory(nicks: list[str] = wandb_nicks) -> Any:
            import wandb

            # Prefer the key stored in the OS keyring (scripts/add_account.py --platform wandb);
            # fall back to wandb's own resolution (WANDB_API_KEY env / netrc).
            api_key = next((k for k in (secrets.get("wandb", n) for n in nicks) if k), None)
            if api_key:
                return wandb.Api(api_key=api_key, timeout=20)
            return wandb.Api(timeout=20)

        sources.append(
            WandbSource(
                entity=settings.wandb.entity,
                projects=settings.wandb.projects,
                api_factory=_wandb_api_factory,
                settings=settings,
            )
        )

    kaggle_clients: list[tuple[str, KaggleCLI]] = [
        (
            acc["nickname"],
            KaggleCLI(
                acc["nickname"],
                token_getter=lambda n=acc["nickname"]: secrets.get("kaggle", n),
            ),
        )
        for acc in store.list_accounts()
        if acc.get("platform") == "kaggle" and acc.get("enabled", True)
    ]
    if kaggle_clients:
        sources.append(KaggleSource(kaggle_clients, store=store, settings=settings))

    use_demo = settings.demo_mode == "on" or (settings.demo_mode == "auto" and not has_wandb)
    if use_demo:
        sources.append(
            DemoSource(
                stale_after_s=settings.stale_after_s,
                offline_after_s=settings.offline_after_s,
            )
        )

    return Service(
        sources=sources,
        store=store,
        settings=settings,
        account_manager=account_manager,
        secrets=secrets,
    )
