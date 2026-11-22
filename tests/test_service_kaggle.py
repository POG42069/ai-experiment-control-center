"""Tests for Service + Kaggle integration, merge rule, native logs, and CLI commands."""

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

from ecc.adapters.kaggle import KaggleCLI, KaggleSource
from ecc.config import GeneralSettings, Settings, WandbSettings
from ecc.models import LogLine, RunStatus, RunSummary
from ecc.service import Service
from ecc.store import Store


class _FakeWandbSource:
    name = "wandb"

    def __init__(self, runs: list[RunSummary]) -> None:
        self._runs = {r.run_id: r for r in runs}

    def list_runs(self, project: str | None = None) -> list[RunSummary]:
        vals = list(self._runs.values())
        if project is not None:
            vals = [r for r in vals if r.project == project]
        return vals

    def get_run(self, run_id: str) -> RunSummary | None:
        return self._runs.get(run_id)

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["step", "epoch"])

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        return [LogLine(ts=None, stream="stdout", text="wandb console line", level="INFO")]

    def list_projects(self) -> list[str]:
        return sorted({r.project for r in self._runs.values()})

    def health(self) -> tuple[bool, str]:
        return True, "ok"


def _make_wb_run(
    run_id: str,
    project: str,
    now: datetime,
    kaggle_kernel: str | None = None,
) -> RunSummary:
    cfg: dict[str, Any] = {"ecc/platform": "kaggle", "ecc/account": "kaggle-1"}
    if kaggle_kernel is not None:
        cfg["ecc/kaggle_kernel"] = kaggle_kernel
    return RunSummary(
        run_id=run_id,
        source="wandb",
        project=project,
        name="WB Linked Run",
        status=RunStatus.RUNNING,
        raw_state="running",
        platform="kaggle",
        account="kaggle-1",
        gpu="T4",
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        config=cfg,
        tags=[],
        step=100,
        total_steps=500,
        epoch=0.5,
        total_epochs=2.0,
        progress=0.2,
        latest={"train/loss": 0.65},
        best_metric_name=None,
        best_metric=None,
        best_step=None,
        best_epoch=None,
        started_at=now - timedelta(minutes=20),
        ended_at=None,
        last_update=now - timedelta(seconds=15),
        duration_s=1200.0,
        eta_s=4800.0,
        notebook_url=None,
        source_url="https://wandb.ai/me/shared-task-4/runs/abc1",
        git_branch="exp-1",
        git_sha="deadbee",
        dataset_version="v2",
        seed=42,
        is_demo=False,
    )


class TestServiceKaggleIntegration:
    def test_merge_rule_hides_linked_kaggle_run_and_fetches_native_logs(
        self, temp_store: Store, fixed_now: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        recent_ts = (fixed_now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        csv_out = (
            "ref,title,author,lastRunTime,totalVotes\n"
            f"alice/linked-nb,Linked NB,alice,{recent_ts},0\n"
            f"alice/store-linked-nb,Store Linked NB,alice,{recent_ts},0\n"
            f"alice/standalone-nb,Standalone NB,alice,{recent_ts},0\n"
        )

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            cmd = tuple(argv[1:3])
            if cmd == ("kernels", "list"):
                return subprocess.CompletedProcess(args=argv, returncode=0, stdout=csv_out, stderr="")
            if cmd == ("kernels", "status"):
                ref = argv[3]
                if ref == "alice/standalone-nb":
                    out = f'{ref} has status "QUEUED"'
                else:
                    out = f'{ref} has status "KernelWorkerStatus.RUNNING"'
                return subprocess.CompletedProcess(args=argv, returncode=0, stdout=out, stderr="")
            if cmd == ("kernels", "logs"):
                ref = argv[3]
                return subprocess.CompletedProcess(
                    args=argv,
                    returncode=0,
                    stdout=f"2026-11-22T11:50:00Z native log for {ref}\n",
                    stderr="",
                )
            raise AssertionError(f"Unexpected cmd: {argv}")

        cli = KaggleCLI("kaggle-1", token_getter=lambda: "KGAT_tok_12345678", runner=fake_runner)
        kaggle_src = KaggleSource([("kaggle-1", cli)], store=temp_store, now_fn=lambda: fixed_now)

        wb_config_linked = _make_wb_run(
            "wandb:me/shared-task-4/run1",
            "shared-task-4",
            fixed_now,
            kaggle_kernel="alice/linked-nb",
        )
        wb_store_linked = _make_wb_run(
            "wandb:me/shared-task-4/run2",
            "shared-task-4",
            fixed_now,
            kaggle_kernel=None,
        )
        wb_src = _FakeWandbSource([wb_config_linked, wb_store_linked])

        settings = Settings(
            general=GeneralSettings(demo_mode="off", db_path=tmp_path / "svc.db"),
            wandb=WandbSettings(),
        )
        service = Service(
            sources=[wb_src, kaggle_src],
            store=temp_store,
            settings=settings,
            now_fn=lambda: fixed_now,
        )

        # Link run2 via store kernel_links
        link = service.link_kernel(
            kernel_ref="alice/store-linked-nb",
            account="kaggle-1",
            project="shared-task-4",
            run_id="wandb:me/shared-task-4/run2",
        )
        assert link["kernel_ref"] == "alice/store-linked-nb"

        runs = service.list_runs()
        run_ids = [r.run_id for r in runs]

        # Linked Kaggle runs are hidden; standalone Kaggle run remains visible
        assert "wandb:me/shared-task-4/run1" in run_ids
        assert "wandb:me/shared-task-4/run2" in run_ids
        assert "kaggle:kaggle-1/alice/standalone-nb" in run_ids
        assert "kaggle:kaggle-1/alice/linked-nb" not in run_ids
        assert "kaggle:kaggle-1/alice/store-linked-nb" not in run_ids

        # Enriched W&B runs carry kaggle_status and notebook_url
        by_id = {r.run_id: r for r in runs}
        r1 = by_id["wandb:me/shared-task-4/run1"]
        assert getattr(r1, "kaggle_status", None) == "RUNNING (Kaggle API)"
        assert r1.config["kaggle_status"] == "RUNNING (Kaggle API)"
        assert r1.notebook_url == "https://www.kaggle.com/code/alice/linked-nb"

        r2 = by_id["wandb:me/shared-task-4/run2"]
        assert getattr(r2, "kaggle_status", None) == "RUNNING (Kaggle API)"

        # Default get_run_logs returns W&B logs; source="kaggle" and get_native_logs return Kaggle logs
        wb_logs = service.get_run_logs("wandb:me/shared-task-4/run1")
        assert wb_logs[0].text == "wandb console line"

        k_logs = service.get_run_logs("wandb:me/shared-task-4/run1", source="kaggle")
        assert len(k_logs) == 1
        assert k_logs[0].stream == "kaggle"
        assert "native log for alice/linked-nb" in k_logs[0].text

        native_logs = service.get_native_logs("wandb:me/shared-task-4/run2")
        assert len(native_logs) == 1
        assert "native log for alice/store-linked-nb" in native_logs[0].text

        # list_kaggle_kernels returns all 3 kernels
        kernels = service.list_kaggle_kernels(nickname="kaggle-1")
        assert len(kernels) == 3
        assert service.list_kaggle_kernels(nickname="nonexistent") == []

    def test_cli_accounts_and_kaggle_kernels_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "ecc_config.toml"
        db = tmp_path / "cli_accounts.db"
        cfg.write_text(
            f'[general]\ndemo_mode = "off"\ndb_path = "{db.as_posix()}"\n',
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["ECC_CONFIG"] = str(cfg)

        p_acc = subprocess.run(
            [sys.executable, "-m", "ecc.cli", "accounts"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
        assert p_acc.returncode == 0, p_acc.stderr
        assert json.loads(p_acc.stdout) == []

        p_kern = subprocess.run(
            [sys.executable, "-m", "ecc.cli", "kaggle-kernels"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
        assert p_kern.returncode == 0, p_kern.stderr
        assert json.loads(p_kern.stdout) == []
