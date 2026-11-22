"""Tests for ecc.service and ecc.cli (via subprocess)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ecc.config import GeneralSettings, Settings, WandbSettings
from ecc.models import LogLine, RunStatus, RunSummary
from ecc.service import RunNotFoundError, Service, build_service
from ecc.sources.demo import DemoSource
from ecc.store import Store


class _SingleRealRunSource:
    """Minimal fake non-demo RunSource to test demo+real comparison warning."""

    name = "real-fake"

    def __init__(self, run: RunSummary, df: pd.DataFrame) -> None:
        self._run = run
        self._df = df

    def list_runs(self, project: str | None = None) -> list[RunSummary]:
        if project is None or project == self._run.project:
            return [self._run]
        return []

    def get_run(self, run_id: str) -> RunSummary | None:
        return self._run if run_id == self._run.run_id else None

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame:
        return self._df.copy()

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        return []

    def list_projects(self) -> list[str]:
        return [self._run.project]

    def health(self) -> tuple[bool, str]:
        return True, "ok"


class TestService:
    def test_list_projects_union_and_kernel_project_override(
        self, demo_service: Service, temp_store: Store
    ) -> None:
        temp_store.upsert_project("custom-local", "Custom Local Project", tags=["local"])
        slugs = [p.slug for p in demo_service.list_projects()]
        assert "custom-local" in slugs
        assert "demo-shared-task" in slugs
        assert "demo-cv" in slugs

        # Override run-a's project via store kernel link
        run_a_id = "demo:demo/demo-shared-task/run-a"
        temp_store.link_kernel("u/nb", "kaggle-1", "custom-local", run_a_id)
        custom_runs = demo_service.list_runs(project="custom-local")
        assert [r.run_id for r in custom_runs] == [run_a_id]

    def test_active_and_failed_runs(self, demo_service: Service) -> None:
        active = demo_service.list_active_runs()
        assert len(active) == 4  # 2 RUNNING + 1 STALE + 1 QUEUED
        assert all(
            r.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.STALE}
            for r in active
        )

        failed = demo_service.get_failed_runs(since_hours=72)
        assert len(failed) == 1
        assert failed[0].status == RunStatus.FAILED

    def test_logs_filtering_query_and_level(self, demo_service: Service) -> None:
        run_e = "demo:demo/demo-shared-task/run-e"
        err_logs = demo_service.get_run_logs(run_e, level="ERROR")
        assert len(err_logs) >= 2
        assert all(line.level == "ERROR" for line in err_logs)

        oom_logs = demo_service.get_run_logs(run_e, query="out of memory")
        assert len(oom_logs) == 1
        assert "CUDA out of memory" in oom_logs[0].text

        with pytest.raises(RunNotFoundError):
            demo_service.get_run_logs("demo:demo/nope/missing")

    def test_compare_runs_warnings_and_validation(
        self, demo_source: DemoSource, temp_store: Store, tmp_path: Path, fixed_now: datetime
    ) -> None:
        run_a = "demo:demo/demo-shared-task/run-a"
        run_b = "demo:demo/demo-shared-task/run-b"
        run_c = "demo:demo/demo-shared-task/run-c"

        settings = Settings(
            general=GeneralSettings(demo_mode="on", db_path=tmp_path / "c.db"),
            wandb=WandbSettings(),
        )
        svc = Service([demo_source], temp_store, settings, now_fn=lambda: fixed_now)

        # ValueError for <2 or >10 runs
        with pytest.raises(ValueError):
            svc.compare_runs([run_a])
        with pytest.raises(ValueError):
            svc.compare_runs([run_a] * 11)

        # Comparable runs A vs B -> no warnings
        res_ab = svc.compare_runs([run_a, run_b], metric="eval/macro_f1", axis="step")
        assert res_ab["warnings"] == []
        table = res_ab["table"]
        assert len(table) == 2
        for col in (
            "name",
            "platform",
            "account",
            "gpu",
            "model",
            "lora_r",
            "lora_alpha",
            "learning_rate",
            "prompt_strategy",
            "seed",
            "dataset_version",
            "best_metric",
            "best_step",
            "best_epoch",
            "duration_s",
            "final train/loss",
            "final eval/loss",
            "eval/accuracy",
            "eval/macro_f1",
            "max gpu/mem_alloc_gb",
        ):
            assert col in table.columns
        curve_f1 = res_ab["curves"]["eval/macro_f1"]
        assert list(curve_f1.columns) == ["run_id", "name", "x", "value"]
        assert set(curve_f1["run_id"]) == {run_a, run_b}

        # Epoch axis also works
        res_epoch = svc.compare_runs([run_a, run_b], metric="eval/macro_f1", axis="epoch")
        assert not res_epoch["curves"]["eval/macro_f1"].empty

        # Run A vs Run C -> ecc/eval_signature mismatch warning
        res_ac = svc.compare_runs([run_a, run_c], metric="eval/macro_f1")
        assert any("eval_signature" in w for w in res_ac["warnings"])

        # Demo + real mix warning
        real_run = RunSummary(
            run_id="wandb:me/demo-shared-task/real1",
            source="wandb",
            project="demo-shared-task",
            name="Real Run 1",
            status=RunStatus.FINISHED,
            raw_state="finished",
            platform="kaggle",
            account="kaggle-1",
            gpu="T4",
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            config={
                "ecc/eval_signature": "v2-vi-90-10-seed42",
                "ecc/dataset_version": "v2",
                "ecc/dataset_split": "train/val=90/10 seed42",
            },
            tags=[],
            step=1000,
            total_steps=1000,
            epoch=3.0,
            total_epochs=3.0,
            progress=1.0,
            latest={"train/loss": 0.4, "eval/macro_f1": 0.85},
            best_metric_name="eval/macro_f1",
            best_metric=0.85,
            best_step=900,
            best_epoch=2.7,
            started_at=fixed_now - timedelta(hours=1),
            ended_at=fixed_now,
            last_update=fixed_now,
            duration_s=3600.0,
            eta_s=None,
            notebook_url=None,
            source_url=None,
            git_branch="main",
            git_sha="1234567",
            dataset_version="v2",
            seed=42,
            is_demo=False,
        )
        real_df = pd.DataFrame(
            {
                "step": [100, 200],
                "epoch": [0.3, 0.6],
                "train/loss": [0.9, 0.4],
                "eval/macro_f1": [0.75, 0.85],
            }
        )
        mixed_svc = Service(
            [demo_source, _SingleRealRunSource(real_run, real_df)],
            temp_store,
            settings,
            now_fn=lambda: fixed_now,
        )
        res_mixed = mixed_svc.compare_runs([run_a, real_run.run_id])
        assert any("mixes synthetic demo runs with real runs" in w for w in res_mixed["warnings"])

    def test_get_best_experiment_with_baseline_delta(
        self, demo_service: Service, temp_store: Store
    ) -> None:
        baseline_id = "demo:demo/demo-shared-task/run-d"
        temp_store.set_baseline("demo-shared-task", baseline_id)
        res = demo_service.get_best_experiment("demo-shared-task", metric="eval/macro_f1", mode="max")
        assert res["best"] is not None
        assert res["baseline"] is not None
        assert res["baseline"]["run_id"] == baseline_id
        assert res["delta"] == pytest.approx(res["best_value"] - res["baseline_value"])

    def test_build_service_demo_modes(self, tmp_path: Path) -> None:
        cfg_off = tmp_path / "off.toml"
        cfg_off.write_text(
            f'[general]\ndemo_mode = "off"\ndb_path = "{(tmp_path / "off.db").as_posix()}"\n',
            encoding="utf-8",
        )
        svc_off = build_service(cfg_off)
        assert svc_off.sources == []


class TestCliSubprocess:
    def _run_cli(self, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ecc.cli", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )

    def test_cli_list_runs_compare_and_unknown_run(self, tmp_path: Path) -> None:
        cfg = tmp_path / "ecc_config.toml"
        db = tmp_path / "cli.db"
        cfg.write_text(
            f'[general]\ndemo_mode = "on"\ndb_path = "{db.as_posix()}"\n',
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["ECC_CONFIG"] = str(cfg)

        # list-runs
        p_runs = self._run_cli(["list-runs"], env)
        assert p_runs.returncode == 0, p_runs.stderr
        runs_data = json.loads(p_runs.stdout)
        assert isinstance(runs_data, list)
        assert len(runs_data) >= 7
        assert all(r["is_demo"] is True for r in runs_data)

        # compare Run A vs Run C
        p_cmp = self._run_cli(
            [
                "compare",
                "--run",
                "demo:demo/demo-shared-task/run-a",
                "demo:demo/demo-shared-task/run-c",
                "--metric",
                "eval/macro_f1",
                "--axis",
                "step",
            ],
            env,
        )
        assert p_cmp.returncode == 0, p_cmp.stderr
        cmp_data = json.loads(p_cmp.stdout)
        assert len(cmp_data["runs"]) == 2
        assert len(cmp_data["table"]) == 2
        assert "eval/macro_f1" in cmp_data["curves"]
        assert any("eval_signature" in w for w in cmp_data["warnings"])

        # get-metrics with downsampling
        p_metrics = self._run_cli(
            [
                "get-metrics",
                "--run",
                "demo:demo/demo-shared-task/run-a",
                "--max-points",
                "10",
            ],
            env,
        )
        assert p_metrics.returncode == 0, p_metrics.stderr
        metrics_data = json.loads(p_metrics.stdout)
        assert 1 <= len(metrics_data) <= 10

        # health + list-projects + best + failed + get-logs
        for cmd in (
            ["list-projects"],
            ["list-active-runs"],
            ["failed"],
            ["best", "--project", "demo-shared-task"],
            ["health"],
            ["get-logs", "--run", "demo:demo/demo-shared-task/run-e", "--level", "ERROR"],
        ):
            proc = self._run_cli(cmd, env)
            assert proc.returncode == 0, f"{cmd} failed: {proc.stderr}"
            json.loads(proc.stdout)

        # Unknown run exits 2 with JSON error
        p_unknown = self._run_cli(["get-run", "--run", "demo:demo/nope/does-not-exist"], env)
        assert p_unknown.returncode == 2
        err_payload = json.loads(p_unknown.stdout)
        assert err_payload["error"] == "run_not_found"
