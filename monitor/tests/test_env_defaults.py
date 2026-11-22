"""Tests for experiment_monitor.init() environment-variable defaults and precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import wandb

import experiment_monitor as em
from experiment_monitor import core as em_core


class _FakeConfig(dict):
    def update(self, other: Any = None, **kwargs: Any) -> None:  # type: ignore[override]
        if isinstance(other, dict):
            super().update(other)


class _FakeRun:
    def __init__(
        self,
        *,
        project: str,
        name: str | None,
        config: dict[str, Any],
        tags: list[str] | None,
        run_dir: Path,
    ) -> None:
        self.project = project
        self.name = name
        self.config = _FakeConfig(config)
        self.tags = tags
        self.summary: dict[str, Any] = {}
        self.dir = str(run_dir)
        self.finished_exit_code: int | None = None

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool | None = None) -> None:
        pass

    def finish(self, exit_code: int | None = None) -> None:
        self.finished_exit_code = exit_code


@pytest.fixture
def fake_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "run", None)

    def _fake_init(
        *,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        id: str | None = None,
        resume: str | None = None,
        mode: str | None = None,
    ) -> _FakeRun:
        run = _FakeRun(
            project=project or "",
            name=name,
            config=dict(config or {}),
            tags=tags,
            run_dir=tmp_path,
        )
        calls.append(
            {
                "project": project,
                "entity": entity,
                "name": name,
                "config": dict(config or {}),
                "tags": tags,
                "run": run,
            }
        )
        return run

    monkeypatch.setattr(wandb, "init", _fake_init)
    monkeypatch.setattr(em_core, "detect_git_info", lambda: ("detected-branch", "detected-sha"))

    for var in (
        "ECC_PROJECT",
        "WANDB_PROJECT",
        "ECC_RUN_NAME",
        "ECC_ACCOUNT",
        "ECC_KAGGLE_KERNEL",
        "ECC_NOTEBOOK_URL",
        "ECC_GIT_BRANCH",
        "ECC_GIT_SHA",
        "ECC_LAUNCH_ID",
        "ECC_GIT_REPO",
    ):
        monkeypatch.delenv(var, raising=False)

    return calls


def test_init_raises_value_error_when_no_project_anywhere(
    fake_wandb: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="project"):
        em.init(capture_console=False, gpu_telemetry=False, heartbeat_interval_s=0)
    assert fake_wandb == []


def test_init_uses_ecc_env_defaults_and_overrides_git_info(
    fake_wandb: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ECC_PROJECT", "ecc-proj")
    monkeypatch.setenv("WANDB_PROJECT", "wandb-proj-fallback")
    monkeypatch.setenv("ECC_RUN_NAME", "run-from-env")
    monkeypatch.setenv("ECC_ACCOUNT", "kaggle-1")
    monkeypatch.setenv("ECC_KAGGLE_KERNEL", "alice/ecc-runner")
    monkeypatch.setenv("ECC_NOTEBOOK_URL", "https://www.kaggle.com/code/alice/ecc-runner")
    monkeypatch.setenv("ECC_GIT_BRANCH", "feat/env-branch")
    monkeypatch.setenv("ECC_GIT_SHA", "abcdef1234567890abcdef1234567890abcdef12")
    monkeypatch.setenv("ECC_LAUNCH_ID", "launch-uuid-123")
    monkeypatch.setenv("ECC_GIT_REPO", "alice/vlm-exp")

    with em.init(
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
    ) as mon:
        assert len(fake_wandb) == 1
        call = fake_wandb[0]
        assert call["project"] == "ecc-proj"
        assert call["name"] == "run-from-env"
        cfg = dict(mon.run.config)
        assert cfg["ecc/project"] == "ecc-proj"
        assert cfg["ecc/account"] == "kaggle-1"
        assert cfg["ecc/kaggle_kernel"] == "alice/ecc-runner"
        assert cfg["ecc/notebook_url"] == "https://www.kaggle.com/code/alice/ecc-runner"
        assert cfg["ecc/git_branch"] == "feat/env-branch"
        assert cfg["ecc/git_sha"] == "abcdef1234567890abcdef1234567890abcdef12"
        assert cfg["ecc/launch_id"] == "launch-uuid-123"
        assert cfg["ecc/git_repo"] == "alice/vlm-exp"


def test_init_falls_back_to_wandb_project_when_ecc_project_unset(
    fake_wandb: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "from-wandb-env")

    with em.init(capture_console=False, gpu_telemetry=False, heartbeat_interval_s=0) as mon:
        assert fake_wandb[0]["project"] == "from-wandb-env"
        cfg = dict(mon.run.config)
        assert cfg["ecc/project"] == "from-wandb-env"
        assert cfg["ecc/git_branch"] == "detected-branch"
        assert cfg["ecc/git_sha"] == "detected-sha"
        assert "ecc/launch_id" not in cfg
        assert "ecc/git_repo" not in cfg


def test_explicit_arguments_always_win_over_env(
    fake_wandb: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ECC_PROJECT", "env-proj")
    monkeypatch.setenv("ECC_RUN_NAME", "env-run")
    monkeypatch.setenv("ECC_ACCOUNT", "env-acc")
    monkeypatch.setenv("ECC_KAGGLE_KERNEL", "env/kernel")
    monkeypatch.setenv("ECC_NOTEBOOK_URL", "https://www.kaggle.com/code/env/kernel")
    monkeypatch.setenv("ECC_LAUNCH_ID", "env-launch-id")
    monkeypatch.setenv("ECC_GIT_REPO", "env/repo")

    with em.init(
        project="explicit-proj",
        name="explicit-run",
        account="explicit-acc",
        kaggle_kernel="explicit/kernel",
        notebook_url="https://www.kaggle.com/code/explicit/kernel",
        config={"ecc/launch_id": "explicit-launch-id", "ecc/git_repo": "explicit/repo"},
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
    ) as mon:
        call = fake_wandb[0]
        assert call["project"] == "explicit-proj"
        assert call["name"] == "explicit-run"
        cfg = dict(mon.run.config)
        assert cfg["ecc/project"] == "explicit-proj"
        assert cfg["ecc/account"] == "explicit-acc"
        assert cfg["ecc/kaggle_kernel"] == "explicit/kernel"
        assert cfg["ecc/notebook_url"] == "https://www.kaggle.com/code/explicit/kernel"
        assert cfg["ecc/launch_id"] == "explicit-launch-id"
        assert cfg["ecc/git_repo"] == "explicit/repo"
