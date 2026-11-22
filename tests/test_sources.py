"""Tests for DemoSource and WandbSource (zero network calls)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ecc.models import RunStatus
from ecc.sources.demo import DemoSource
from ecc.sources.wandb_source import WandbSource, map_wandb_run, parse_console_log_lines


class TestDemoSource:
    def test_seven_scenarios_and_statuses_at_fixed_now(self, fixed_now: datetime) -> None:
        src = DemoSource(now_fn=lambda: fixed_now, seed=0)
        assert set(src.list_projects()) == {"demo-shared-task", "demo-cv"}
        runs = src.list_runs()
        assert len(runs) >= 7
        assert all(r.is_demo for r in runs)
        assert all(r.source == "demo" for r in runs)
        assert all(r.name.startswith("[DEMO] ") for r in runs)

        by_slug = {r.run_id.split("/")[-1]: r for r in runs}
        assert by_slug["run-a"].status == RunStatus.RUNNING
        assert by_slug["run-a"].account == "kaggle-1"
        assert "LoRA r16/a32" in by_slug["run-a"].name

        assert by_slug["run-b"].status == RunStatus.RUNNING
        assert by_slug["run-b"].account == "kaggle-2"
        assert "LoRA r32/a64" in by_slug["run-b"].name

        assert by_slug["run-c"].status == RunStatus.STALE
        assert by_slug["run-c"].account == "colab-1"
        assert (
            by_slug["run-c"].config["ecc/eval_signature"]
            != by_slug["run-a"].config["ecc/eval_signature"]
        )

        assert by_slug["run-d"].status == RunStatus.FINISHED
        assert by_slug["run-e"].status == RunStatus.FAILED
        assert by_slug["run-f"].status == RunStatus.OFFLINE
        assert by_slug["run-g"].status == RunStatus.QUEUED

        # Check CUDA OOM traceback in failed run logs
        logs_e = src.get_logs(by_slug["run-e"].run_id, tail=200)
        assert any("CUDA out of memory" in line.text and line.level == "ERROR" for line in logs_e)
        assert any("Traceback" in line.text and line.level == "ERROR" for line in logs_e)

    def test_progress_advances_with_now_fn(self, fixed_now: datetime) -> None:
        clock = [fixed_now]
        src = DemoSource(now_fn=lambda: clock[0], seed=0)
        r_before = src.get_run("demo:demo/demo-shared-task/run-a")
        assert r_before is not None
        clock[0] = fixed_now + timedelta(seconds=60)
        r_after = src.get_run("demo:demo/demo-shared-task/run-a")
        assert r_after is not None
        assert r_after.step is not None and r_before.step is not None
        assert r_after.step > r_before.step
        assert r_after.progress is not None and r_before.progress is not None
        assert r_after.progress > r_before.progress

    def test_curves_contain_expected_columns(self, demo_source: DemoSource) -> None:
        df = demo_source.get_metrics("demo:demo/demo-shared-task/run-a")
        for col in (
            "step",
            "epoch",
            "train/loss",
            "eval/loss",
            "eval/accuracy",
            "eval/macro_f1",
            "train/lr",
            "gpu/mem_alloc_gb",
        ):
            assert col in df.columns
        assert not df["train/loss"].dropna().empty
        assert not df["eval/macro_f1"].dropna().empty


class _FakeFile:
    def __init__(self, filename: str, content: str | None) -> None:
        self.filename = filename
        self.content = content

    def download(self, root: str = ".", replace: bool = True) -> Any:
        if self.content is None:
            raise FileNotFoundError(self.filename)
        out = Path(root) / self.filename
        out.write_text(self.content, encoding="utf-8")
        return open(out, "r", encoding="utf-8")


class _FakeWandbRun:
    def __init__(
        self,
        *,
        run_id: str = "abc123",
        name: str = "run-1",
        state: str = "running",
        entity: str = "me",
        project: str = "shared-task",
        config: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        url: str = "https://wandb.ai/me/shared-task/runs/abc123",
        created_at: Any = None,
        heartbeat_at: Any = None,
        use_camel_case: bool = False,
        history_df: pd.DataFrame | None = None,
        scan_rows: list[dict[str, Any]] | None = None,
        files: dict[str, str] | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.state = state
        self.entity = entity
        self.project = project
        self.config = config if config is not None else {}
        self.summary = summary if summary is not None else {}
        self.tags = tags if tags is not None else []
        self.url = url
        if use_camel_case:
            self.createdAt = created_at
            self.heartbeatAt = heartbeat_at
        else:
            self.created_at = created_at
            self.heartbeat_at = heartbeat_at
        self._history_df = history_df
        self._scan_rows = scan_rows
        self._files = files or {}
        self.history_calls: list[dict[str, Any]] = []
        self._history_lock = threading.Lock()

    def history(
        self,
        keys: list[str] | None = None,
        x_axis: str = "_step",
        samples: int = 2000,
        pandas: bool = True,
    ) -> pd.DataFrame:
        with self._history_lock:
            self.history_calls.append(
                {
                    "keys": list(keys) if keys is not None else None,
                    "x_axis": x_axis,
                    "samples": samples,
                    "pandas": pandas,
                }
            )
        if self._history_df is None or self._history_df.empty:
            return pd.DataFrame()
        df = self._history_df.copy()
        if x_axis not in df.columns:
            df[x_axis] = range(len(df))
        if keys is not None:
            if not keys:
                return df[[x_axis]].copy()
            # Model real W&B sampled history: only rows where ALL requested keys are present.
            if any(k not in df.columns for k in keys):
                return pd.DataFrame()
            req_cols = list(dict.fromkeys([x_axis, *keys]))
            mask = df[req_cols].notna().all(axis=1)
            filtered = df.loc[mask, req_cols].reset_index(drop=True)
            return filtered.copy()
        return df.copy()

    def scan_history(self) -> list[dict[str, Any]]:
        return list(self._scan_rows or [])

    def file(self, name: str) -> _FakeFile:
        return _FakeFile(name, self._files.get(name))


class TestMapWandbRun:
    def test_snake_case_datetimes_and_stale_heartbeat(self, fixed_now: datetime) -> None:
        run = _FakeWandbRun(
            run_id="r_snake",
            name="Snake Run",
            state="running",
            created_at=fixed_now - timedelta(minutes=30),
            heartbeat_at=fixed_now - timedelta(seconds=300),  # 5 min -> STALE
            config={
                "ecc/platform": "kaggle",
                "ecc/account": "kaggle-1",
                "ecc/gpu_name": "Tesla T4",
                "ecc/model_name": "Qwen3-VL",
                "ecc/total_steps": 1000,
                "ecc/total_epochs": 3.0,
                "ecc/dataset_version": "v2",
                "ecc/seed": 42,
            },
            summary={
                "train/global_step": 500,
                "train/epoch": 1.5,
                "train/loss": 0.41,
                "eval/macro_f1": 0.805,
                "ecc/best_metric_name": "eval/macro_f1",
                "ecc/best_metric": 0.805,
                "ecc/best_step": 500,
                "ecc/best_epoch": 1.5,
            },
            tags=["lora"],
        )
        summary = map_wandb_run(run, fixed_now)
        assert summary.run_id == "wandb:me/shared-task/r_snake"
        assert summary.status == RunStatus.STALE
        assert summary.step == 500
        assert summary.progress == pytest.approx(0.5)
        assert summary.latest["train/loss"] == pytest.approx(0.41)
        assert summary.latest["eval/macro_f1"] == pytest.approx(0.805)
        assert summary.ended_at is None
        assert summary.eta_s is not None and summary.eta_s > 0
        assert summary.is_demo is False

    def test_camel_case_iso_strings_and_ecc_heartbeat_wins(self, fixed_now: datetime) -> None:
        # heartbeatAt is 10 min old, but summary ecc/heartbeat is 20s old -> RUNNING
        run = _FakeWandbRun(
            run_id="r_camel",
            name="Camel Run",
            state="running",
            use_camel_case=True,
            created_at=(fixed_now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
            heartbeat_at=(fixed_now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
            summary={
                "_step": 250,
                "ecc/heartbeat": (fixed_now - timedelta(seconds=20)).timestamp(),
            },
        )
        summary = map_wandb_run(run, fixed_now)
        assert summary.status == RunStatus.RUNNING
        assert summary.step == 250

    def test_missing_fields_and_crashed_state(self, fixed_now: datetime) -> None:
        minimal = SimpleNamespace(
            id="r_crash",
            name="Crashed Run",
            state="crashed",
            entity="me",
            project="proj",
            config=None,
            summary=None,
            tags=None,
            url=None,
            createdAt=None,
            heartbeatAt=fixed_now - timedelta(minutes=2),
        )
        summary = map_wandb_run(minimal, fixed_now)
        assert summary.status == RunStatus.CRASHED
        assert summary.ended_at == fixed_now - timedelta(minutes=2)
        assert summary.eta_s is None
        assert summary.latest == {}
        assert summary.tags == []


class TestWandbSource:
    def test_list_metrics_logs_and_ttl_cache(self, fixed_now: datetime) -> None:
        log_content = (
            "2026-11-22T11:55:00+00:00 [stdout] Epoch 1/3 | Step 100 | loss=0.91\n"
            "2026-11-22T11:56:00Z [stderr] UserWarning: low batch size\n"
            "2026-11-22T11:57:00+00:00 [stderr] plain stderr status\n"
            "2026-11-22T11:58:00+00:00 [stderr] RuntimeError: CUDA out of memory\n"
        )
        hist_df = pd.DataFrame(
            {
                "train/global_step": [10, 20, 30],
                "train/epoch": [0.1, 0.2, 0.3],
                "train/loss": [1.5, 1.1, 0.8],
            }
        )
        fake_run = _FakeWandbRun(
            run_id="wb1",
            name="WB Run 1",
            state="running",
            created_at=fixed_now - timedelta(minutes=10),
            heartbeat_at=fixed_now - timedelta(seconds=10),
            history_df=hist_df,
            files={"ecc_console.log": log_content},
        )
        calls = {"runs": 0}

        class _FakeApi:
            def runs(self, path: str, order: str = "-created_at", per_page: int = 50) -> list[Any]:
                calls["runs"] += 1
                return [fake_run]

            def run(self, path: str) -> Any:
                return fake_run

        src = WandbSource(
            entity="me",
            projects=["shared-task"],
            api_factory=_FakeApi,
            cache_ttl_s=60.0,
            now_fn=lambda: fixed_now,
        )
        runs1 = src.list_runs()
        runs2 = src.list_runs()
        assert len(runs1) == 1
        assert len(runs2) == 1
        assert calls["runs"] == 1  # Served from TTL cache

        df = src.get_metrics("wandb:me/shared-task/wb1", keys=["train/loss"])
        assert list(df["step"]) == [10, 20, 30]
        assert "epoch" in df.columns

        logs = src.get_logs("wandb:me/shared-task/wb1", tail=50)
        assert len(logs) == 4
        assert logs[0].stream == "stdout" and logs[0].level == "INFO"
        assert logs[0].ts is not None
        assert logs[1].stream == "stderr" and logs[1].level == "WARNING"
        # stderr defaults to INFO unless error/warning pattern matches
        assert logs[2].stream == "stderr" and logs[2].level == "INFO"
        assert logs[3].stream == "stderr" and logs[3].level == "ERROR"

    def test_scan_history_fallback_and_missing_logs_fallback(self, fixed_now: datetime) -> None:
        fake_run = _FakeWandbRun(
            run_id="wb2",
            state="finished",
            heartbeat_at=fixed_now,
            history_df=pd.DataFrame(),
            scan_rows=[{"_step": 1, "train/loss": 1.2}, {"_step": 2, "train/loss": 0.9}],
            files={},  # Neither ecc_console.log nor output.log
        )

        class _FakeApi:
            def runs(self, path: str, **kwargs: Any) -> list[Any]:
                return [fake_run]

            def run(self, path: str) -> Any:
                return fake_run

        src = WandbSource("me", ["shared-task"], api_factory=_FakeApi, now_fn=lambda: fixed_now)
        df = src.get_metrics("wandb:me/shared-task/wb2", keys=None)
        assert list(df["step"]) == [1, 2]

        logs = src.get_logs("wandb:me/shared-task/wb2", tail=10)
        assert len(logs) == 1
        assert logs[0].stream == "system"
        assert "capture_console=True" in logs[0].text

    def test_network_error_caught_and_redacted_in_health(self, fixed_now: datetime) -> None:
        def _boom_api() -> Any:
            raise RuntimeError("401 Unauthorized for key a1b2c3d4e5f60718293a4b5c6d7e8f9012345678")

        src = WandbSource("me", ["shared-task"], api_factory=_boom_api, now_fn=lambda: fixed_now)
        assert src.list_runs() == []
        ok, msg = src.health()
        assert ok is False
        assert "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678" not in msg
        assert "[REDACTED]" in msg

    def test_parse_console_log_lines_unprefixed(self) -> None:
        lines = parse_console_log_lines("plain line\nTraceback (most recent call last):\n")
        assert lines[0].ts is None and lines[0].stream == "stdout" and lines[0].level == "INFO"
        assert lines[1].level == "ERROR"

    def test_get_metrics_disjoint_train_and_eval_rows_and_cache(self, fixed_now: datetime) -> None:
        # Train and eval metrics logged on separate _step rows; eval rows lack train/epoch & global_step
        hist_df = pd.DataFrame(
            {
                "_step": [0, 1, 2, 3],
                "train/global_step": [10.0, float("nan"), 20.0, float("nan")],
                "train/epoch": [0.5, float("nan"), 1.0, float("nan")],
                "train/loss": [1.5, float("nan"), 1.1, float("nan")],
                "eval/macro_f1": [float("nan"), 0.72, float("nan"), 0.81],
            }
        )
        fake_run = _FakeWandbRun(
            run_id="wb_split",
            state="running",
            heartbeat_at=fixed_now,
            history_df=hist_df,
        )

        class _FakeApi:
            def runs(self, path: str, **kwargs: Any) -> list[Any]:
                return [fake_run]

            def run(self, path: str) -> Any:
                return fake_run

        src = WandbSource(
            "me",
            ["shared-task"],
            api_factory=_FakeApi,
            cache_ttl_s=60.0,
            now_fn=lambda: fixed_now,
        )
        df = src.get_metrics("wandb:me/shared-task/wb_split", keys=["train/loss", "eval/macro_f1"])

        assert list(df["_step"]) == [0, 1, 2, 3]
        assert list(df["train/loss"].dropna()) == pytest.approx([1.5, 1.1])
        assert list(df["eval/macro_f1"].dropna()) == pytest.approx([0.72, 0.81])
        assert list(df["step"]) == [10, 10, 20, 20]
        assert list(df["epoch"]) == pytest.approx([0.5, 0.5, 1.0, 1.0])

        # Every per-key history call requested exactly one key with x_axis="_step"
        assert len(fake_run.history_calls) == 4
        assert all(
            call["keys"] is not None and len(call["keys"]) == 1 and call["x_axis"] == "_step"
            for call in fake_run.history_calls
        )

        # Reordered keys should hit the TTL cache keyed by (run_id, tuple(sorted(keys)))
        calls_before = len(fake_run.history_calls)
        df_cached = src.get_metrics(
            "wandb:me/shared-task/wb_split", keys=["eval/macro_f1", "train/loss"]
        )
        assert len(fake_run.history_calls) == calls_before
        assert list(df_cached["eval/macro_f1"].dropna()) == pytest.approx([0.72, 0.81])

        # Missing requested key -> absent column, not an error
        df_missing = src.get_metrics(
            "wandb:me/shared-task/wb_split", keys=["train/loss", "eval/nonexistent"]
        )
        assert "train/loss" in df_missing.columns
        assert "eval/nonexistent" not in df_missing.columns
        assert list(df_missing["train/loss"].dropna()) == pytest.approx([1.5, 1.1])

    def test_get_metrics_without_train_epoch_still_returns_data(self, fixed_now: datetime) -> None:
        hist_df = pd.DataFrame(
            {
                "_step": [0, 1, 2],
                "train/global_step": [10.0, float("nan"), 20.0],
                "train/loss": [1.4, float("nan"), 0.9],
                "eval/macro_f1": [float("nan"), 0.77, float("nan")],
            }
        )
        fake_run = _FakeWandbRun(
            run_id="wb_no_epoch",
            state="finished",
            heartbeat_at=fixed_now,
            history_df=hist_df,
        )

        class _FakeApi:
            def runs(self, path: str, **kwargs: Any) -> list[Any]:
                return [fake_run]

            def run(self, path: str) -> Any:
                return fake_run

        src = WandbSource("me", ["shared-task"], api_factory=_FakeApi, now_fn=lambda: fixed_now)
        df = src.get_metrics(
            "wandb:me/shared-task/wb_no_epoch", keys=["train/loss", "eval/macro_f1"]
        )

        assert not df.empty
        assert list(df["train/loss"].dropna()) == pytest.approx([1.4, 0.9])
        assert list(df["eval/macro_f1"].dropna()) == pytest.approx([0.77])
        assert list(df["step"]) == [10, 10, 20]
        assert "train/epoch" not in df.columns
        assert "epoch" in df.columns
        assert df["epoch"].isna().all()

    def test_get_metrics_step_uses_global_step_ffill_bfill_and_fallback(
        self, fixed_now: datetime
    ) -> None:
        # Eval row at _step=0 precedes first train/global_step (tests bfill),
        # eval row at _step=2 follows train/global_step=100 (tests ffill, not interpolation).
        hist_df = pd.DataFrame(
            {
                "_step": [0, 1, 2, 3],
                "train/global_step": [float("nan"), 100.0, float("nan"), 200.0],
                "train/loss": [float("nan"), 1.2, float("nan"), 0.8],
                "eval/loss": [2.0, float("nan"), 1.4, float("nan")],
            }
        )
        fake_run = _FakeWandbRun(
            run_id="wb_gstep",
            state="running",
            heartbeat_at=fixed_now,
            history_df=hist_df,
        )
        fallback_run = _FakeWandbRun(
            run_id="wb_nostep",
            state="running",
            heartbeat_at=fixed_now,
            history_df=pd.DataFrame({"_step": [5, 10], "train/loss": [1.0, 0.5]}),
        )

        class _FakeApi:
            def runs(self, path: str, **kwargs: Any) -> list[Any]:
                return [fake_run, fallback_run]

            def run(self, path: str) -> Any:
                return fallback_run if path.endswith("/wb_nostep") else fake_run

        src = WandbSource("me", ["shared-task"], api_factory=_FakeApi, now_fn=lambda: fixed_now)

        # With explicit keys
        df_keys = src.get_metrics(
            "wandb:me/shared-task/wb_gstep", keys=["train/loss", "eval/loss"]
        )
        assert list(df_keys["step"]) == [100, 100, 100, 200]
        assert df_keys["epoch"].isna().all()

        # With keys=None (same step/epoch derivation)
        df_all = src.get_metrics("wandb:me/shared-task/wb_gstep", keys=None)
        assert list(df_all["step"]) == [100, 100, 100, 200]
        assert df_all["epoch"].isna().all()

        # Without train/global_step at all -> falls back to _step
        df_fallback = src.get_metrics("wandb:me/shared-task/wb_nostep", keys=["train/loss"])
        assert list(df_fallback["step"]) == [5, 10]
        assert df_fallback["epoch"].isna().all()
