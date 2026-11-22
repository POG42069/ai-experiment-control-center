"""Deterministic synthetic DemoSource for offline testing and UI preview."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from ecc.models import LogLine, RunStatus, RunSummary
from ecc.sources.base import classify_log_level
from ecc.status import (
    DEFAULT_OFFLINE_AFTER_S,
    DEFAULT_STALE_AFTER_S,
    compute_eta,
    compute_progress,
    derive_status,
)

DEMO_PROJECTS = ["demo-shared-task", "demo-cv"]
METRIC_COLUMNS = [
    "step",
    "epoch",
    "train/loss",
    "eval/loss",
    "eval/accuracy",
    "eval/macro_f1",
    "train/lr",
    "gpu/mem_alloc_gb",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class _DemoSpec:
    slug: str
    project: str
    name: str
    raw_state: str
    heartbeat_age_s: float | None
    platform: str
    account: str
    gpu: str
    model_name: str
    total_steps: int
    total_epochs: float
    base_step: int
    sec_per_step: float
    eval_every: int
    base_lr: float
    warmup_steps: int
    init_loss: float
    target_loss: float
    peak_f1: float
    peak_acc: float
    mem_gb: float
    config: dict[str, Any]
    tags: list[str]
    notebook_url: str
    git_branch: str
    git_sha: str
    dataset_version: str
    seed: int
    advances: bool = False
    oom_failed: bool = False


def _build_specs() -> list[_DemoSpec]:
    return [
        _DemoSpec(
            slug="run-a",
            project="demo-shared-task",
            name="[DEMO] Qwen3-VL-4B LoRA r16/a32",
            raw_state="running",
            heartbeat_age_s=15.0,
            platform="kaggle",
            account="kaggle-1",
            gpu="NVIDIA Tesla T4",
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            total_steps=1000,
            total_epochs=3.0,
            base_step=620,
            sec_per_step=3.2,
            eval_every=100,
            base_lr=2e-4,
            warmup_steps=100,
            init_loss=2.15,
            target_loss=0.42,
            peak_f1=0.812,
            peak_acc=0.835,
            mem_gb=11.4,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-shared-task",
                "ecc/platform": "kaggle",
                "ecc/account": "kaggle-1",
                "ecc/model_name": "Qwen/Qwen3-VL-4B-Instruct",
                "ecc/prompt_strategy": "vi-default",
                "ecc/dataset_version": "v2",
                "ecc/dataset_split": "train/val=90/10 seed42",
                "ecc/eval_signature": "v2-vi-90-10-seed42",
                "ecc/seed": 42,
                "ecc/total_steps": 1000,
                "ecc/total_epochs": 3.0,
                "ecc/gpu_name": "NVIDIA Tesla T4",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 15.0,
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": 2e-4,
                "batch_size": 8,
            },
            tags=["demo", "qwen3-vl", "lora", "r16"],
            notebook_url="https://www.kaggle.com/code/demo-user/qwen3-vl-lora-r16",
            git_branch="exp/qwen3-lora-r16",
            git_sha="a1b2c3d",
            dataset_version="v2",
            seed=42,
            advances=True,
        ),
        _DemoSpec(
            slug="run-b",
            project="demo-shared-task",
            name="[DEMO] Qwen3-VL-4B LoRA r32/a64",
            raw_state="running",
            heartbeat_age_s=22.0,
            platform="kaggle",
            account="kaggle-2",
            gpu="NVIDIA Tesla P100",
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            total_steps=1000,
            total_epochs=3.0,
            base_step=480,
            sec_per_step=2.9,
            eval_every=100,
            base_lr=1.5e-4,
            warmup_steps=100,
            init_loss=2.12,
            target_loss=0.38,
            peak_f1=0.828,
            peak_acc=0.849,
            mem_gb=13.2,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-shared-task",
                "ecc/platform": "kaggle",
                "ecc/account": "kaggle-2",
                "ecc/model_name": "Qwen/Qwen3-VL-4B-Instruct",
                "ecc/prompt_strategy": "vi-default",
                "ecc/dataset_version": "v2",
                "ecc/dataset_split": "train/val=90/10 seed42",
                "ecc/eval_signature": "v2-vi-90-10-seed42",
                "ecc/seed": 42,
                "ecc/total_steps": 1000,
                "ecc/total_epochs": 3.0,
                "ecc/gpu_name": "NVIDIA Tesla P100",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 16.0,
                "lora_r": 32,
                "lora_alpha": 64,
                "learning_rate": 1.5e-4,
                "batch_size": 8,
            },
            tags=["demo", "qwen3-vl", "lora", "r32"],
            notebook_url="https://www.kaggle.com/code/demo-user2/qwen3-vl-lora-r32",
            git_branch="exp/qwen3-lora-r32",
            git_sha="b2c3d4e",
            dataset_version="v2",
            seed=42,
            advances=True,
        ),
        _DemoSpec(
            slug="run-c",
            project="demo-shared-task",
            name="[DEMO] Qwen3-VL-4B LoRA r16/a32 + EN prompt",
            raw_state="running",
            heartbeat_age_s=300.0,  # 5 minutes old -> STALE
            platform="colab",
            account="colab-1",
            gpu="NVIDIA L4",
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            total_steps=1000,
            total_epochs=3.0,
            base_step=540,
            sec_per_step=2.5,
            eval_every=100,
            base_lr=2e-4,
            warmup_steps=100,
            init_loss=2.18,
            target_loss=0.45,
            peak_f1=0.798,
            peak_acc=0.821,
            mem_gb=12.1,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-shared-task",
                "ecc/platform": "colab",
                "ecc/account": "colab-1",
                "ecc/model_name": "Qwen/Qwen3-VL-4B-Instruct",
                "ecc/prompt_strategy": "en-prompt",
                "ecc/dataset_version": "v2",
                "ecc/dataset_split": "train/val=90/10 seed42",
                "ecc/eval_signature": "v2-en-prompt-90-10-seed42",
                "ecc/seed": 42,
                "ecc/total_steps": 1000,
                "ecc/total_epochs": 3.0,
                "ecc/gpu_name": "NVIDIA L4",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 24.0,
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": 2e-4,
                "batch_size": 8,
            },
            tags=["demo", "qwen3-vl", "en-prompt"],
            notebook_url="https://colab.research.google.com/drive/demo-en-prompt",
            git_branch="exp/qwen3-en-prompt",
            git_sha="c3d4e5f",
            dataset_version="v2",
            seed=42,
            advances=False,
        ),
        _DemoSpec(
            slug="run-d",
            project="demo-shared-task",
            name="[DEMO] Qwen3-VL-4B LoRA r8/a16 baseline",
            raw_state="finished",
            heartbeat_age_s=7200.0,
            platform="kaggle",
            account="kaggle-1",
            gpu="NVIDIA Tesla T4",
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            total_steps=1000,
            total_epochs=3.0,
            base_step=1000,
            sec_per_step=3.0,
            eval_every=100,
            base_lr=2e-4,
            warmup_steps=100,
            init_loss=2.20,
            target_loss=0.49,
            peak_f1=0.785,
            peak_acc=0.808,
            mem_gb=9.8,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-shared-task",
                "ecc/platform": "kaggle",
                "ecc/account": "kaggle-1",
                "ecc/model_name": "Qwen/Qwen3-VL-4B-Instruct",
                "ecc/prompt_strategy": "vi-default",
                "ecc/dataset_version": "v2",
                "ecc/dataset_split": "train/val=90/10 seed42",
                "ecc/eval_signature": "v2-vi-90-10-seed42",
                "ecc/seed": 42,
                "ecc/total_steps": 1000,
                "ecc/total_epochs": 3.0,
                "ecc/gpu_name": "NVIDIA Tesla T4",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 15.0,
                "lora_r": 8,
                "lora_alpha": 16,
                "learning_rate": 2e-4,
                "batch_size": 8,
            },
            tags=["demo", "baseline", "lora"],
            notebook_url="https://www.kaggle.com/code/demo-user/qwen3-vl-baseline",
            git_branch="main",
            git_sha="d4e5f6a",
            dataset_version="v2",
            seed=42,
            advances=False,
        ),
        _DemoSpec(
            slug="run-e",
            project="demo-shared-task",
            name="[DEMO] Qwen3-VL-4B Full FT bs16",
            raw_state="failed",
            heartbeat_age_s=1800.0,
            platform="colab",
            account="colab-2",
            gpu="NVIDIA Tesla T4",
            model_name="Qwen/Qwen3-VL-4B-Instruct",
            total_steps=1000,
            total_epochs=3.0,
            base_step=140,
            sec_per_step=4.1,
            eval_every=100,
            base_lr=5e-5,
            warmup_steps=100,
            init_loss=2.25,
            target_loss=0.60,
            peak_f1=0.620,
            peak_acc=0.655,
            mem_gb=14.9,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-shared-task",
                "ecc/platform": "colab",
                "ecc/account": "colab-2",
                "ecc/model_name": "Qwen/Qwen3-VL-4B-Instruct",
                "ecc/prompt_strategy": "vi-default",
                "ecc/dataset_version": "v2",
                "ecc/dataset_split": "train/val=90/10 seed42",
                "ecc/eval_signature": "v2-vi-90-10-seed42",
                "ecc/seed": 42,
                "ecc/total_steps": 1000,
                "ecc/total_epochs": 3.0,
                "ecc/gpu_name": "NVIDIA Tesla T4",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 15.0,
                "lora_r": None,
                "lora_alpha": None,
                "learning_rate": 5e-5,
                "batch_size": 16,
            },
            tags=["demo", "full-ft", "oom"],
            notebook_url="https://colab.research.google.com/drive/demo-full-ft-oom",
            git_branch="exp/full-ft",
            git_sha="e5f6a7b",
            dataset_version="v2",
            seed=42,
            advances=False,
            oom_failed=True,
        ),
        _DemoSpec(
            slug="run-f",
            project="demo-cv",
            name="[DEMO] ConvNeXt-Large CutMix",
            raw_state="running",
            heartbeat_age_s=1500.0,  # 25 minutes old -> OFFLINE
            platform="kaggle",
            account="kaggle-2",
            gpu="NVIDIA Tesla P100",
            model_name="convnext_large_mlp.clip_laion2b",
            total_steps=800,
            total_epochs=10.0,
            base_step=360,
            sec_per_step=2.2,
            eval_every=80,
            base_lr=1e-4,
            warmup_steps=80,
            init_loss=1.95,
            target_loss=0.31,
            peak_f1=0.875,
            peak_acc=0.889,
            mem_gb=10.6,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-cv",
                "ecc/platform": "kaggle",
                "ecc/account": "kaggle-2",
                "ecc/model_name": "convnext_large_mlp.clip_laion2b",
                "ecc/prompt_strategy": "n/a",
                "ecc/dataset_version": "cv-v1",
                "ecc/dataset_split": "5fold-fold0",
                "ecc/eval_signature": "cv-v1-fold0",
                "ecc/seed": 2026,
                "ecc/total_steps": 800,
                "ecc/total_epochs": 10.0,
                "ecc/gpu_name": "NVIDIA Tesla P100",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 16.0,
                "learning_rate": 1e-4,
                "batch_size": 32,
            },
            tags=["demo", "cv", "convnext"],
            notebook_url="https://www.kaggle.com/code/demo-user2/convnext-cutmix",
            git_branch="exp/cv-convnext",
            git_sha="f6a7b8c",
            dataset_version="cv-v1",
            seed=2026,
            advances=False,
        ),
        _DemoSpec(
            slug="run-g",
            project="demo-cv",
            name="[DEMO] EVA-02-Base Multi-Scale",
            raw_state="QUEUED",
            heartbeat_age_s=None,
            platform="kaggle",
            account="kaggle-1",
            gpu="NVIDIA Tesla T4",
            model_name="eva02_base_patch14_448",
            total_steps=800,
            total_epochs=10.0,
            base_step=0,
            sec_per_step=2.6,
            eval_every=80,
            base_lr=8e-5,
            warmup_steps=80,
            init_loss=1.90,
            target_loss=0.29,
            peak_f1=0.880,
            peak_acc=0.894,
            mem_gb=0.0,
            config={
                "ecc/schema_version": 1,
                "ecc/project": "demo-cv",
                "ecc/platform": "kaggle",
                "ecc/account": "kaggle-1",
                "ecc/model_name": "eva02_base_patch14_448",
                "ecc/prompt_strategy": "n/a",
                "ecc/dataset_version": "cv-v1",
                "ecc/dataset_split": "5fold-fold0",
                "ecc/eval_signature": "cv-v1-fold0",
                "ecc/seed": 2026,
                "ecc/total_steps": 800,
                "ecc/total_epochs": 10.0,
                "ecc/gpu_name": "NVIDIA Tesla T4",
                "ecc/gpu_count": 1,
                "ecc/gpu_mem_total_gb": 15.0,
                "learning_rate": 8e-5,
                "batch_size": 16,
            },
            tags=["demo", "cv", "eva02"],
            notebook_url="https://www.kaggle.com/code/demo-user/eva02-multiscale",
            git_branch="exp/cv-eva02",
            git_sha="0123abc",
            dataset_version="cv-v1",
            seed=2026,
            advances=False,
        ),
    ]


class DemoSource:
    """Deterministic synthetic RunSource for offline development and demos."""

    name: str = "demo"

    def __init__(
        self,
        now_fn: Callable[[], datetime] = _utc_now,
        seed: int = 0,
        stale_after_s: int = DEFAULT_STALE_AFTER_S,
        offline_after_s: int = DEFAULT_OFFLINE_AFTER_S,
    ) -> None:
        self._now_fn = now_fn
        self._seed = seed
        self._stale_after_s = stale_after_s
        self._offline_after_s = offline_after_s
        self._anchor_now = _ensure_utc(now_fn())
        self._specs: dict[str, _DemoSpec] = {}
        self._ordered_ids: list[str] = []
        for spec in _build_specs():
            rid = f"demo:demo/{spec.project}/{spec.slug}"
            self._specs[rid] = spec
            self._ordered_ids.append(rid)

    def _current_step(self, spec: _DemoSpec, now: datetime) -> int:
        if not spec.advances:
            return spec.base_step
        elapsed_since_anchor = max(0.0, (now - self._anchor_now).total_seconds())
        extra = int(elapsed_since_anchor // 5.0)
        return min(spec.total_steps - 1, spec.base_step + extra)

    def _generate_curve(self, run_id: str, spec: _DemoSpec, current_step: int) -> pd.DataFrame:
        if current_step <= 0:
            return pd.DataFrame(columns=METRIC_COLUMNS)

        # Deterministic per-run RNG derived from DemoSource seed + slug
        run_seed = self._seed * 1009 + sum(ord(c) * (i + 1) for i, c in enumerate(spec.slug))
        rng = np.random.default_rng(run_seed)

        step_stride = 10
        steps = list(range(step_stride, current_step + 1, step_stride))
        if not steps or steps[-1] != current_step:
            steps.append(current_step)

        rows: list[dict[str, float]] = []
        for step in steps:
            frac = step / float(spec.total_steps)
            epoch = frac * spec.total_epochs

            # Smooth exponential decay + deterministic jitter for train/loss
            decay = math.exp(-3.2 * frac)
            train_loss = spec.target_loss + (spec.init_loss - spec.target_loss) * decay
            train_loss = max(0.05, train_loss + float(rng.normal(0.0, 0.025)))

            # Cosine schedule with linear warmup
            if step <= spec.warmup_steps:
                lr = spec.base_lr * (step / float(max(1, spec.warmup_steps)))
            else:
                progress_after = (step - spec.warmup_steps) / float(
                    max(1, spec.total_steps - spec.warmup_steps)
                )
                lr = spec.base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress_after)))

            mem_alloc = max(0.5, spec.mem_gb + float(rng.normal(0.0, 0.12)))
            if spec.oom_failed and step >= current_step:
                mem_alloc = spec.config.get("ecc/gpu_mem_total_gb", 15.0)

            eval_loss = float("nan")
            eval_acc = float("nan")
            eval_f1 = float("nan")
            if step % spec.eval_every == 0:
                eval_loss = max(
                    0.08,
                    train_loss + 0.08 + float(rng.normal(0.0, 0.015)),
                )
                gain = 1.0 - math.exp(-3.4 * frac)
                eval_f1 = max(
                    0.1,
                    min(0.99, 0.42 + (spec.peak_f1 - 0.42) * gain + float(rng.normal(0.0, 0.006))),
                )
                eval_acc = max(
                    0.1,
                    min(0.99, 0.45 + (spec.peak_acc - 0.45) * gain + float(rng.normal(0.0, 0.006))),
                )

            rows.append(
                {
                    "step": int(step),
                    "epoch": round(epoch, 4),
                    "train/loss": round(train_loss, 5),
                    "eval/loss": round(eval_loss, 5) if math.isfinite(eval_loss) else float("nan"),
                    "eval/accuracy": round(eval_acc, 5) if math.isfinite(eval_acc) else float("nan"),
                    "eval/macro_f1": round(eval_f1, 5) if math.isfinite(eval_f1) else float("nan"),
                    "train/lr": float(f"{lr:.6g}"),
                    "gpu/mem_alloc_gb": round(mem_alloc, 3),
                }
            )

        return pd.DataFrame(rows, columns=METRIC_COLUMNS)

    def _build_summary(self, run_id: str, spec: _DemoSpec, now: datetime) -> RunSummary:
        current_step = self._current_step(spec, now)
        df = self._generate_curve(run_id, spec, current_step)

        last_update = (
            now - timedelta(seconds=spec.heartbeat_age_s)
            if spec.heartbeat_age_s is not None
            else None
        )
        status = derive_status(
            spec.raw_state,
            last_update,
            now,
            stale_after_s=self._stale_after_s,
            offline_after_s=self._offline_after_s,
        )

        is_terminal = status in {
            RunStatus.FINISHED,
            RunStatus.FAILED,
            RunStatus.CRASHED,
            RunStatus.KILLED,
        }

        if current_step > 0:
            run_elapsed = current_step * spec.sec_per_step
            end_anchor = last_update if last_update is not None else now
            started_at = end_anchor - timedelta(seconds=run_elapsed)
            ended_at = last_update if is_terminal else None
            duration_s = float(run_elapsed)
            epoch = round((current_step / float(spec.total_steps)) * spec.total_epochs, 4)
        else:
            started_at = None
            ended_at = None
            duration_s = None
            epoch = 0.0

        progress = compute_progress(current_step, spec.total_steps)
        eta_s = (
            None
            if is_terminal or status == RunStatus.QUEUED
            else compute_eta(current_step, spec.total_steps, started_at, last_update)
        )

        latest: dict[str, float] = {}
        best_metric_name: str | None = None
        best_metric: float | None = None
        best_step: int | None = None
        best_epoch: float | None = None

        if not df.empty:
            for col in (
                "train/loss",
                "eval/loss",
                "eval/accuracy",
                "eval/macro_f1",
                "train/lr",
                "gpu/mem_alloc_gb",
            ):
                valid = df[col].dropna()
                if not valid.empty:
                    latest[col] = float(valid.iloc[-1])
            latest["gpu/max_mem_alloc_gb"] = float(df["gpu/mem_alloc_gb"].max())

            f1_valid = df.dropna(subset=["eval/macro_f1"])
            if not f1_valid.empty:
                best_idx = f1_valid["eval/macro_f1"].idxmax()
                best_row = f1_valid.loc[best_idx]
                best_metric_name = "eval/macro_f1"
                best_metric = float(best_row["eval/macro_f1"])
                best_step = int(best_row["step"])
                best_epoch = float(best_row["epoch"])

        return RunSummary(
            run_id=run_id,
            source="demo",
            project=spec.project,
            name=spec.name,
            status=status,
            raw_state=spec.raw_state,
            platform=spec.platform,
            account=spec.account,
            gpu=spec.gpu,
            model_name=spec.model_name,
            config=dict(spec.config),
            tags=list(spec.tags),
            step=current_step if current_step > 0 else 0,
            total_steps=spec.total_steps,
            epoch=epoch,
            total_epochs=spec.total_epochs,
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
            notebook_url=spec.notebook_url,
            source_url=spec.notebook_url,
            git_branch=spec.git_branch,
            git_sha=spec.git_sha,
            dataset_version=spec.dataset_version,
            seed=spec.seed,
            is_demo=True,
        )

    def list_projects(self) -> list[str]:
        return list(DEMO_PROJECTS)

    def list_runs(self, project: str | None = None) -> list[RunSummary]:
        now = _ensure_utc(self._now_fn())
        runs: list[RunSummary] = []
        for rid in self._ordered_ids:
            spec = self._specs[rid]
            if project is not None and spec.project != project:
                continue
            runs.append(self._build_summary(rid, spec, now))
        return runs

    def get_run(self, run_id: str) -> RunSummary | None:
        spec = self._specs.get(run_id)
        if spec is None:
            return None
        now = _ensure_utc(self._now_fn())
        return self._build_summary(run_id, spec, now)

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame:
        spec = self._specs.get(run_id)
        if spec is None:
            return pd.DataFrame(columns=METRIC_COLUMNS)
        now = _ensure_utc(self._now_fn())
        current_step = self._current_step(spec, now)
        df = self._generate_curve(run_id, spec, current_step)
        if keys:
            keep = ["step", "epoch"] + [k for k in keys if k in df.columns and k not in ("step", "epoch")]
            return df[keep].copy()
        return df

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        spec = self._specs.get(run_id)
        if spec is None:
            return []

        now = _ensure_utc(self._now_fn())
        current_step = self._current_step(spec, now)
        base_ts = (
            (now - timedelta(seconds=spec.heartbeat_age_s + current_step * spec.sec_per_step))
            if spec.heartbeat_age_s is not None
            else now - timedelta(minutes=2)
        )

        raw_lines: list[tuple[int, str, str]] = [
            (0, "stdout", f"[ecc] Starting {spec.name} on {spec.platform}/{spec.account} ({spec.gpu})"),
            (2, "stdout", f"[ecc] Model={spec.model_name} dataset={spec.dataset_version} seed={spec.seed}"),
        ]

        if spec.raw_state.upper() == "QUEUED":
            raw_lines.append((5, "stdout", "Kernel queued on Kaggle worker pool; waiting for GPU allocation..."))
        else:
            raw_lines.extend(
                [
                    (
                        5,
                        "stderr",
                        "UserWarning: torch.utils.checkpoint: use_reentrant parameter should be passed explicitly.",
                    ),
                    (8, "stdout", f"Loaded train/val split ({spec.config.get('ecc/dataset_split')})"),
                ]
            )
            rng = random.Random(self._seed + len(spec.slug))
            df = self._generate_curve(run_id, spec, current_step)
            # Emit realistic periodic step/eval progress lines
            for _, row in df.iterrows():
                step = int(row["step"])
                if step % 50 == 0 or step == current_step:
                    offset = int(step * spec.sec_per_step)
                    raw_lines.append(
                        (
                            offset,
                            "stdout",
                            f"Epoch {row['epoch']:.2f}/{spec.total_epochs:.1f} | "
                            f"Step {step}/{spec.total_steps} | "
                            f"train/loss={row['train/loss']:.4f} | "
                            f"lr={row['train/lr']:.2e} | "
                            f"vram={row['gpu/mem_alloc_gb']:.2f}GB",
                        )
                    )
                if math.isfinite(row["eval/macro_f1"]):
                    offset = int(step * spec.sec_per_step) + 1
                    raw_lines.append(
                        (
                            offset,
                            "stdout",
                            f"[eval] step={step} eval/loss={row['eval/loss']:.4f} "
                            f"eval/accuracy={row['eval/accuracy']:.4f} "
                            f"eval/macro_f1={row['eval/macro_f1']:.4f}",
                        )
                    )
            if rng.random() >= 0.0 and spec.heartbeat_age_s and spec.heartbeat_age_s >= 300:
                raw_lines.append(
                    (
                        int(current_step * spec.sec_per_step) + 5,
                        "stderr",
                        "WARN: DataLoader worker heartbeat delayed; waiting on I/O.",
                    )
                )

        if spec.oom_failed:
            oom_offset = int(current_step * spec.sec_per_step) + 4
            raw_lines.extend(
                [
                    (oom_offset, "stderr", "Traceback (most recent call last):"),
                    (oom_offset + 1, "stderr", '  File "/kaggle/working/train.py", line 184, in train_step'),
                    (oom_offset + 1, "stderr", "    loss.backward()"),
                    (
                        oom_offset + 2,
                        "stderr",
                        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB "
                        "(GPU 0; 14.75 GiB total capacity; 13.84 GiB already allocated; 412.00 MiB free).",
                    ),
                ]
            )
        elif spec.raw_state == "finished":
            done_offset = int(current_step * spec.sec_per_step) + 5
            raw_lines.append(
                (done_offset, "stdout", "[ecc] Training finished cleanly; best checkpoint saved.")
            )

        lines = [
            LogLine(
                ts=base_ts + timedelta(seconds=offset),
                stream=stream,
                text=text,
                level=classify_log_level(text),
            )
            for offset, stream, text in raw_lines
        ]
        if tail <= 0:
            return []
        return lines[-tail:]

    def health(self) -> tuple[bool, str]:
        return True, f"Demo source active ({len(self._specs)} synthetic runs across {len(DEMO_PROJECTS)} projects)"
