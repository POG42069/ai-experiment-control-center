"""Tests for CLI `launch` (plan, execute, status, history) with fakes only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ecc import cli as ecc_cli
from ecc.config import GeneralSettings, LaunchSettings, Settings
from ecc.launch.github import CommitInfo, GitHubReader
from ecc.launch.kaggle_launcher import PushResult
from ecc.launch.service import LaunchService
from ecc.secrets import SecretStore
from ecc.service import Service
from ecc.store import Store


class _InMemoryKeyring:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._data[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._data.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        del self._data[(service, username)]


class _FakeReadService(Service):
    def __init__(
        self,
        store: Store,
        settings: Settings,
        secrets: SecretStore,
        status_by_ref: dict[str, tuple[str, str | None] | None] | None = None,
    ) -> None:
        super().__init__(sources=[], store=store, settings=settings, secrets=secrets)
        self._status_by_ref = dict(status_by_ref or {})

    def get_kernel_status(
        self, ref: str, nickname: str | None = None
    ) -> tuple[str, str | None] | None:
        return self._status_by_ref.get(ref)


class _FakeGitHub(GitHubReader):
    def __init__(self) -> None:
        super().__init__(runner=lambda *a, **kw: None)
        self.calls: list[tuple[str, str]] = []

    def resolve_commit(self, repo: str, ref: str) -> CommitInfo:
        self.calls.append((repo, ref))
        return CommitInfo(
            sha="abcdef0123456789abcdef0123456789abcdef01",
            short_sha="abcdef0",
            message_first_line="Train baseline model",
            author="Bob",
            date="2026-11-22T12:00:00Z",
        )


def _setup_cli_launch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    runner_status: tuple[str, str | None] | None = ("COMPLETE", None),
    push_result: PushResult | None = None,
) -> dict[str, Any]:
    store = Store(tmp_path / "ecc.db")
    store.upsert_account("kaggle-2", "kaggle", username="bob", enabled=True)

    secrets = SecretStore(backend=_InMemoryKeyring())
    secrets.set("kaggle", "kaggle-2", "KGAT_cli_secret_token_987654321")

    settings = Settings(
        general=GeneralSettings(demo_mode="off", db_path=tmp_path / "ecc.db"),
        launch=LaunchSettings(
            enabled=enabled,
            runner_slug="ecc-runner",
            default_accelerator="NvidiaTeslaT4",
            allowed_accelerators=["NvidiaTeslaT4", "TpuV5E8", "none"],
            max_timeout_hours=12,
            wandb_credential_label="WANDB_API_KEY",
            github_credential_label="GITHUB_TOKEN",
            default_wandb_project="shared-task-4",
        ),
    )

    read_svc = _FakeReadService(
        store=store,
        settings=settings,
        secrets=secrets,
        status_by_ref={"bob/ecc-runner": runner_status} if runner_status is not None else {},
    )
    gh = _FakeGitHub()
    clock = [datetime(2026, 11, 22, 14, 0, 0, tzinfo=timezone.utc)]
    push_calls: list[dict[str, Any]] = []

    def fake_launcher_factory(nick: str, token_getter: Any) -> Any:
        class _FakeLauncher:
            def push(
                self,
                folder: str | Path,
                accelerator: str | None = None,
                timeout_s: int | None = None,
            ) -> PushResult:
                push_calls.append(
                    {
                        "nickname": nick,
                        "accelerator": accelerator,
                        "timeout_s": timeout_s,
                        "has_token": bool(token_getter()),
                    }
                )
                return push_result or PushResult(
                    ok=True,
                    version=7,
                    url="https://www.kaggle.com/code/bob/ecc-runner",
                    message="Kernel version 7 successfully pushed.",
                )

        return _FakeLauncher()

    launch_svc = LaunchService(
        store=store,
        secrets=secrets,
        settings=settings,
        read_service=read_svc,
        github=gh,
        launcher_factory=fake_launcher_factory,
        wheel_builder=lambda: b"PK\x03\x04fake-whl",
        now_fn=lambda: clock[0],
    )

    monkeypatch.setattr(ecc_cli, "build_service", lambda config_path=None: read_svc)
    monkeypatch.setattr(ecc_cli, "build_launch_service", lambda rs: launch_svc)

    return {
        "store": store,
        "secrets": secrets,
        "settings": settings,
        "read_svc": read_svc,
        "launch_svc": launch_svc,
        "gh": gh,
        "clock": clock,
        "push_calls": push_calls,
    }


def _invoke_cli(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> tuple[int, Any]:
    rc = ecc_cli.main(argv)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    return rc, data


class TestCliLaunchTwoStep:
    def test_plan_persisted_with_expiry_and_no_secrets_then_execute_happy_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _setup_cli_launch_env(tmp_path, monkeypatch)

        rc_plan, plan_out = _invoke_cli(
            capsys,
            [
                "launch",
                "plan",
                "--account",
                "kaggle-2",
                "--repo",
                "bob/exp-repo",
                "--branch",
                "exp/x",
                "--command",
                "python train.py --config c.yaml",
                "--accelerator",
                "NvidiaTeslaT4",
                "--timeout-hours",
                "9",
                "--monitor-mode",
                "runner",
                "--project",
                "shared-task-4",
                "--run-name",
                "exp-run-1",
                "--requirements",
                "requirements.txt",
            ],
        )
        assert rc_plan == 0
        assert set(plan_out.keys()) == {"plan_id", "summary", "consumes_gpu", "expires_at"}
        assert plan_out["consumes_gpu"] is True
        assert "bob/ecc-runner" in plan_out["summary"]
        assert env["push_calls"] == []

        # Persisted in pending_plans with 10-min expiry and NO secrets in payload
        plan_id = plan_out["plan_id"]
        row = env["store"].get_pending_plan(plan_id)
        assert row is not None
        assert row["executed_at"] is None
        assert "KGAT_cli_secret_token_987654321" not in row["payload_json"]
        created_dt = datetime.fromisoformat(row["created_at"])
        expires_dt = datetime.fromisoformat(plan_out["expires_at"])
        assert expires_dt - created_dt == timedelta(minutes=10)

        # Execute happy path
        rc_exec, exec_out = _invoke_cli(
            capsys,
            ["launch", "execute", "--plan-id", plan_id],
        )
        assert rc_exec == 0
        assert len(env["push_calls"]) == 1
        assert env["push_calls"][0]["nickname"] == "kaggle-2"
        assert env["push_calls"][0]["accelerator"] == "NvidiaTeslaT4"
        assert env["push_calls"][0]["timeout_s"] == 9 * 3600

        # LaunchRecord JSON fields
        assert exec_out["account"] == "kaggle-2"
        assert exec_out["owner"] == "bob"
        assert exec_out["kernel_ref"] == "bob/ecc-runner"
        assert exec_out["url"] == "https://www.kaggle.com/code/bob/ecc-runner"
        assert exec_out["version"] == 7
        assert exec_out["commit_sha"] == "abcdef0123456789abcdef0123456789abcdef01"
        assert exec_out["accelerator"] == "NvidiaTeslaT4"
        assert exec_out["ok"] is True
        assert exec_out["source"] == "claude"
        assert "successfully pushed" in exec_out["message"]

        # Stored in launches with source="claude" and marked executed
        audit = env["store"].get_launch(plan_id)
        assert audit is not None
        assert audit["source"] == "claude"
        assert audit["ok"] is True

        row_after = env["store"].get_pending_plan(plan_id)
        assert row_after is not None
        assert row_after["executed_at"] is not None

        # Replay of already-executed plan -> refused exit 3
        rc_replay, replay_out = _invoke_cli(
            capsys,
            ["launch", "execute", "--plan-id", plan_id],
        )
        assert rc_replay == 3
        assert replay_out["error"] == "launch_refused"
        assert "already been executed" in replay_out["reason"]
        assert len(env["push_calls"]) == 1

    def test_execute_unknown_and_expired_plan_refused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _setup_cli_launch_env(tmp_path, monkeypatch)

        # Unknown plan id -> exit 3 JSON error
        rc_unk, unk_out = _invoke_cli(
            capsys, ["launch", "execute", "--plan-id", "does-not-exist"]
        )
        assert rc_unk == 3
        assert unk_out["error"] == "launch_refused"
        assert "Unknown" in unk_out["reason"]

        # Create a plan, advance clock 11 minutes -> expired -> exit 3 JSON error
        rc_plan, plan_out = _invoke_cli(
            capsys,
            [
                "launch",
                "plan",
                "--account",
                "kaggle-2",
                "--repo",
                "bob/exp-repo",
                "--branch",
                "main",
                "--preflight",
            ],
        )
        assert rc_plan == 0
        assert plan_out["consumes_gpu"] is False

        env["clock"][0] = env["clock"][0] + timedelta(minutes=11)
        rc_exp, exp_out = _invoke_cli(
            capsys, ["launch", "execute", "--plan-id", plan_out["plan_id"]]
        )
        assert rc_exp == 3
        assert exp_out["error"] == "launch_refused"
        assert "expired" in exp_out["reason"].lower()
        assert env["push_calls"] == []

    def test_execute_rechecks_busy_and_failed_push_exits_4(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _setup_cli_launch_env(
            tmp_path,
            monkeypatch,
            push_result=PushResult(
                ok=False,
                version=None,
                url=None,
                message="Kernel push error: quota exceeded",
            ),
        )
        rc_plan, plan_out = _invoke_cli(
            capsys,
            [
                "launch",
                "plan",
                "--account",
                "kaggle-2",
                "--repo",
                "bob/exp-repo",
                "--branch",
                "main",
                "--command",
                "python train.py",
            ],
        )
        assert rc_plan == 0

        # Account becomes RUNNING before execute -> refused (exit 3)
        env["read_svc"]._status_by_ref["bob/ecc-runner"] = ("RUNNING", None)
        rc_busy, busy_out = _invoke_cli(
            capsys, ["launch", "execute", "--plan-id", plan_out["plan_id"]]
        )
        assert rc_busy == 3
        assert "busy" in busy_out["reason"].lower()
        assert env["push_calls"] == []

        # Account returns to COMPLETE, push returns ok=False -> exit 4
        env["read_svc"]._status_by_ref["bob/ecc-runner"] = ("COMPLETE", None)
        rc_fail, fail_out = _invoke_cli(
            capsys, ["launch", "execute", "--plan-id", plan_out["plan_id"]]
        )
        assert rc_fail == 4
        assert fail_out["ok"] is False
        assert fail_out["source"] == "claude"
        assert "quota exceeded" in fail_out["message"]

    def test_status_and_history_commands(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _setup_cli_launch_env(
            tmp_path, monkeypatch, runner_status=("COMPLETE", None)
        )
        rc_st, st_out = _invoke_cli(
            capsys, ["launch", "status", "--account", "kaggle-2"]
        )
        assert rc_st == 0
        assert st_out["account"] == "kaggle-2"
        assert st_out["owner"] == "bob"
        assert st_out["kernel_ref"] == "bob/ecc-runner"
        assert st_out["status"] == "COMPLETE"

        # Empty history initially
        rc_h0, h0 = _invoke_cli(capsys, ["launch", "history", "--limit", "10"])
        assert rc_h0 == 0
        assert h0 == []

        # Plan + execute populates history
        _, plan_out = _invoke_cli(
            capsys,
            [
                "launch",
                "plan",
                "--account",
                "kaggle-2",
                "--repo",
                "bob/exp-repo",
                "--branch",
                "main",
                "--command",
                "python train.py",
            ],
        )
        rc_ex, _ = _invoke_cli(
            capsys, ["launch", "execute", "--plan-id", plan_out["plan_id"]]
        )
        assert rc_ex == 0

        rc_h1, h1 = _invoke_cli(capsys, ["launch", "history", "--limit", "5"])
        assert rc_h1 == 0
        assert len(h1) == 1
        assert h1[0]["id"] == plan_out["plan_id"]
        assert h1[0]["source"] == "claude"

    def test_disabled_launch_refuses_all_launch_subcommands(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _setup_cli_launch_env(tmp_path, monkeypatch, enabled=False)

        for argv in (
            [
                "launch",
                "plan",
                "--account",
                "kaggle-2",
                "--repo",
                "bob/exp-repo",
                "--branch",
                "main",
                "--command",
                "python train.py",
            ],
            ["launch", "execute", "--plan-id", "any-id"],
            ["launch", "status", "--account", "kaggle-2"],
            ["launch", "history"],
        ):
            rc, out = _invoke_cli(capsys, argv)
            assert rc == 3, f"Expected exit 3 for {argv}, got {rc}: {out}"
            assert out["error"] == "launch_refused"
            assert "disabled" in out["reason"].lower()

    def test_subprocess_launch_history_respects_config_switch(self, tmp_path: Path) -> None:
        db = tmp_path / "sub.db"
        cfg_disabled = tmp_path / "disabled.toml"
        cfg_disabled.write_text(
            f'[general]\ndemo_mode = "off"\ndb_path = "{db.as_posix()}"\n[launch]\nenabled = false\n',
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["ECC_CONFIG"] = str(cfg_disabled)
        proc_dis = subprocess.run(
            [sys.executable, "-m", "ecc.cli", "launch", "history"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
        assert proc_dis.returncode == 3
        dis_json = json.loads(proc_dis.stdout)
        assert dis_json["error"] == "launch_refused"

        cfg_enabled = tmp_path / "enabled.toml"
        cfg_enabled.write_text(
            f'[general]\ndemo_mode = "off"\ndb_path = "{db.as_posix()}"\n[launch]\nenabled = true\n',
            encoding="utf-8",
        )
        env["ECC_CONFIG"] = str(cfg_enabled)
        proc_en = subprocess.run(
            [sys.executable, "-m", "ecc.cli", "launch", "history", "--limit", "5"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
        assert proc_en.returncode == 0, proc_en.stderr
        assert json.loads(proc_en.stdout) == []
