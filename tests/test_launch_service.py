"""Tests for LaunchService, [launch] config, Store v1->v2 migration, and ecc.cli isolation."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ecc import cli as ecc_cli
from ecc.adapters.kaggle import KaggleKernel
from ecc.config import ConfigError, LaunchSettings, Settings, load_config
from ecc.launch.github import CommitInfo, GitHubReader
from ecc.launch.kaggle_launcher import KaggleLauncher, PushResult
from ecc.launch.service import LaunchRefused, LaunchRequest, LaunchService
from ecc.secrets import SecretStore
from ecc.store import Store


class _InMemoryKeyring:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._data[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._data.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self._data:
            raise KeyError(username)
        del self._data[(service, username)]


class _FakeReadService:
    def __init__(
        self,
        *,
        kernels: list[KaggleKernel] | None = None,
        status_by_ref: dict[str, tuple[str, str | None] | None] | None = None,
    ) -> None:
        self._kernels = list(kernels or [])
        self._status_by_ref = dict(status_by_ref or {})

    def list_kaggle_kernels(self, nickname: str | None = None) -> list[KaggleKernel]:
        if nickname is None:
            return list(self._kernels)
        return [k for k in self._kernels if k.account == nickname]

    def get_kernel_status(
        self, ref: str, nickname: str | None = None
    ) -> tuple[str, str | None] | None:
        return self._status_by_ref.get(ref)


class _FakeGitHub(GitHubReader):
    def __init__(self, commit: CommitInfo | None = None) -> None:
        super().__init__(runner=lambda *a, **kw: None)
        self._commit = commit or CommitInfo(
            sha="0123456789abcdef0123456789abcdef01234567",
            short_sha="0123456",
            message_first_line="Add baseline training script",
            author="Alice",
            date="2026-11-22T10:00:00Z",
        )
        self.resolved_calls: list[tuple[str, str]] = []

    def resolve_commit(self, repo: str, ref: str) -> CommitInfo:
        self.resolved_calls.append((repo, ref))
        return self._commit


def _make_launch_env(
    tmp_path: Path,
    *,
    enabled: bool = True,
    with_secret: bool = True,
    username: str = "alice",
    runner_status: tuple[str, str | None] | None = ("COMPLETE", None),
    kernels: list[KaggleKernel] | None = None,
    push_result: PushResult | None = None,
    now_ref: list[datetime] | None = None,
) -> dict[str, Any]:
    store = Store(tmp_path / "ecc.db")
    store.upsert_account("kaggle-1", "kaggle", username=username, enabled=True)

    secrets = SecretStore(backend=_InMemoryKeyring())
    if with_secret:
        secrets.set("kaggle", "kaggle-1", "KGAT_fake_secret_token_12345678")

    settings = Settings(
        launch=LaunchSettings(
            enabled=enabled,
            runner_slug="ecc-runner",
            default_accelerator="NvidiaTeslaT4",
            allowed_accelerators=["NvidiaTeslaT4", "TpuV5E8", "none"],
            max_timeout_hours=12,
            wandb_credential_label="WANDB_API_KEY",
            github_credential_label="GITHUB_TOKEN",
            default_wandb_project="shared-task-4",
        )
    )

    read_svc = _FakeReadService(
        kernels=kernels,
        status_by_ref={"alice/ecc-runner": runner_status} if runner_status is not None else {},
    )
    gh = _FakeGitHub()

    pushed_calls: list[dict[str, Any]] = []
    wheel_calls = [0]

    def fake_wheel_builder() -> bytes:
        wheel_calls[0] += 1
        return b"PK\x03\x04fake-wheel-bytes"

    def fake_launcher_factory(nick: str, token_getter: Any) -> Any:
        class _FakeLauncher:
            def push(
                self,
                folder: str | Path,
                accelerator: str | None = None,
                timeout_s: int | None = None,
            ) -> PushResult:
                folder_path = Path(folder)
                meta = json.loads((folder_path / "kernel-metadata.json").read_text(encoding="utf-8"))
                runner_src = (folder_path / "runner.py").read_text(encoding="utf-8")
                pushed_calls.append(
                    {
                        "nickname": nick,
                        "folder": folder_path,
                        "accelerator": accelerator,
                        "timeout_s": timeout_s,
                        "metadata": meta,
                        "runner_len": len(runner_src),
                        "token_present": bool(token_getter()),
                    }
                )
                return push_result or PushResult(
                    ok=True,
                    version=3,
                    url="https://www.kaggle.com/code/alice/ecc-runner",
                    message="Kernel version 3 successfully pushed.",
                )

        return _FakeLauncher()

    clock_box = now_ref if now_ref is not None else [datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)]
    svc = LaunchService(
        store=store,
        secrets=secrets,
        settings=settings,
        read_service=read_svc,
        github=gh,
        launcher_factory=fake_launcher_factory,
        wheel_builder=fake_wheel_builder,
        now_fn=lambda: clock_box[0],
    )
    return {
        "svc": svc,
        "store": store,
        "secrets": secrets,
        "settings": settings,
        "read_svc": read_svc,
        "gh": gh,
        "pushed_calls": pushed_calls,
        "wheel_calls": wheel_calls,
        "clock": clock_box,
    }


def _valid_request(**overrides: Any) -> LaunchRequest:
    base = dict(
        account="kaggle-1",
        repo="alice/vlm-exp",
        branch="main",
        commit_sha=None,
        command="python train.py --epochs 2",
        accelerator="NvidiaTeslaT4",
        timeout_hours=2.0,
        monitor_mode="runner",
        project="shared-task-4",
        run_name="run-01",
        requirements_file="requirements.txt",
        preflight=False,
    )
    base.update(overrides)
    return LaunchRequest(**base)


class TestLaunchConfigAndStoreMigration:
    def test_launch_config_loads_and_still_rejects_secret_keys(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "ecc_config.toml"
        cfg_file.write_text(
            """
[launch]
enabled = true
runner_slug = "my-ecc-runner"
default_accelerator = "TpuV5E8"
allowed_accelerators = ["NvidiaTeslaT4", "TpuV5E8", "none"]
max_timeout_hours = 6
wandb_credential_label = "WANDB_API_KEY"
github_credential_label = "GITHUB_TOKEN"
default_wandb_project = "shared-task-4"
""",
            encoding="utf-8",
        )
        settings = load_config(cfg_file)
        assert settings.launch.enabled is True
        assert settings.launch.runner_slug == "my-ecc-runner"
        assert settings.launch.default_accelerator == "TpuV5E8"
        assert settings.launch.max_timeout_hours == 6
        assert settings.launch.wandb_credential_label == "WANDB_API_KEY"
        assert settings.launch.github_credential_label == "GITHUB_TOKEN"
        assert settings.launch.default_wandb_project == "shared-task-4"

        # Secret-like keys inside [launch] MUST still be rejected
        bad_cfg = tmp_path / "bad_launch.toml"
        bad_cfg.write_text('[launch]\nwandb_secret_name = "WANDB_API_KEY"\n', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad_cfg)

    def test_store_migrates_v1_to_v2_idempotently_and_deletes_account(self, tmp_path: Path) -> None:
        db = tmp_path / "legacy_v1.db"
        conn = sqlite3.connect(db)
        with conn:
            conn.executescript(
                """
                CREATE TABLE projects (
                    slug TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    wandb_entity TEXT, wandb_project TEXT, tags TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '', baseline_run_id TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE run_notes (
                    run_id TEXT PRIMARY KEY, notes TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]', pinned INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE accounts (
                    nickname TEXT PRIMARY KEY,
                    platform TEXT NOT NULL CHECK(platform IN ('kaggle', 'colab', 'wandb')),
                    username TEXT NOT NULL DEFAULT '', secret_ref TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1, last_check_at TEXT,
                    last_check_ok INTEGER, last_check_msg TEXT
                );
                CREATE TABLE kernel_links (
                    kernel_ref TEXT PRIMARY KEY, account TEXT NOT NULL,
                    project TEXT NOT NULL, run_id TEXT
                );
                INSERT INTO accounts (nickname, platform, username, secret_ref, enabled)
                VALUES ('kaggle-old', 'kaggle', 'olduser', 'kaggle:kaggle-old', 1);
                PRAGMA user_version = 1;
                """
            )
        conn.close()

        s1 = Store(db)
        assert s1.user_version() == 3
        assert len(s1.list_accounts()) == 1
        assert s1.list_accounts()[0]["username"] == "olduser"

        # Re-open is idempotent
        s2 = Store(db)
        assert s2.user_version() == 3

        # delete_account works
        assert s2.delete_account("kaggle-old") is True
        assert s2.delete_account("kaggle-old") is False
        assert s2.list_accounts() == []


class TestLaunchServiceGuardsAndHappyPath:
    def test_disabled_refused(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path, enabled=False)
        assert env["svc"].is_enabled() is False
        with pytest.raises(LaunchRefused, match="disabled"):
            env["svc"].plan_launch(_valid_request())

    @pytest.mark.parametrize("busy_status", ["RUNNING", "RUNNING (Kaggle API)", "QUEUED"])
    def test_busy_account_refused(self, tmp_path: Path, busy_status: str) -> None:
        env = _make_launch_env(tmp_path, runner_status=(busy_status, None))
        with pytest.raises(LaunchRefused, match="account busy"):
            env["svc"].plan_launch(_valid_request())

    def test_missing_secret_refused(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path, with_secret=False)
        with pytest.raises(LaunchRefused, match="Missing Kaggle API token"):
            env["svc"].plan_launch(_valid_request())

    def test_timeout_over_max_refused(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path)
        with pytest.raises(LaunchRefused, match="max_timeout_hours"):
            env["svc"].plan_launch(_valid_request(timeout_hours=13.0))

    def test_missing_username_refused_with_script_hint_and_owner_fallback(
        self, tmp_path: Path
    ) -> None:
        env = _make_launch_env(tmp_path, username="", kernels=[])
        with pytest.raises(LaunchRefused, match="scripts/add_account.py --username"):
            env["svc"].plan_launch(_valid_request())

        # Derive most common owner from existing kernels when username not stored
        kernels = [
            KaggleKernel("alice/nb1", "nb1", "alice", None, "https://www.kaggle.com/code/alice/nb1", "kaggle-1"),
            KaggleKernel("alice/nb2", "nb2", "alice", None, "https://www.kaggle.com/code/alice/nb2", "kaggle-1"),
            KaggleKernel("other/nb3", "nb3", "other", None, "https://www.kaggle.com/code/other/nb3", "kaggle-1"),
        ]
        env2 = _make_launch_env(tmp_path / "sub", username="", kernels=kernels)
        assert env2["svc"].account_owner("kaggle-1") == "alice"
        assert env2["svc"].runner_ref("kaggle-1") == "alice/ecc-runner"

    def test_runner_status_not_created_when_missing(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path, runner_status=None)
        assert env["svc"].runner_status("kaggle-1") == "NOT_CREATED"

    @pytest.mark.parametrize(
        "bad_overrides",
        [
            {"command": ""},
            {"command": "echo line1\necho line2"},
            {"command": "x" * 501},
            {"accelerator": "NvidiaTeslaP100"},
            {"repo": "a/b; rm -rf"},
            {"branch": "--upload-pack=x"},
        ],
    )
    def test_invalid_request_fields_refused(
        self, tmp_path: Path, bad_overrides: dict[str, Any]
    ) -> None:
        env = _make_launch_env(tmp_path)
        with pytest.raises(LaunchRefused):
            env["svc"].plan_launch(_valid_request(**bad_overrides))

    def test_unconfirmed_execute_refused(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path)
        plan = env["svc"].plan_launch(_valid_request())
        with pytest.raises(LaunchRefused, match="confirmed_by_user"):
            env["svc"].execute(plan, confirmed_by_user=False)
        assert env["pushed_calls"] == []
        assert env["svc"].list_launches() == []

    def test_stale_plan_refused(self, tmp_path: Path) -> None:
        clock = [datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)]
        env = _make_launch_env(tmp_path, now_ref=clock)
        plan = env["svc"].plan_launch(_valid_request())

        # Advance 11 minutes -> plan is stale (> 10 min)
        clock[0] = clock[0] + timedelta(minutes=11)
        with pytest.raises(LaunchRefused, match="stale"):
            env["svc"].execute(plan, confirmed_by_user=True)
        assert env["pushed_calls"] == []

    def test_execute_rechecks_busy_status_before_push(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path, runner_status=("COMPLETE", None))
        plan = env["svc"].plan_launch(_valid_request())

        # Runner becomes RUNNING between plan and execute
        env["read_svc"]._status_by_ref["alice/ecc-runner"] = ("RUNNING", None)
        with pytest.raises(LaunchRefused, match="account busy"):
            env["svc"].execute(plan, confirmed_by_user=True)
        assert env["pushed_calls"] == []

    def test_happy_path_plans_without_side_effects_executes_audits_and_links_kernel(
        self, tmp_path: Path
    ) -> None:
        env = _make_launch_env(tmp_path)
        svc: LaunchService = env["svc"]

        plan = svc.plan_launch(_valid_request())
        # Plan has NO side effects
        assert env["wheel_calls"][0] == 0
        assert env["pushed_calls"] == []
        assert svc.list_launches() == []
        assert env["store"].list_kernel_links() == []
        assert plan.consumes_gpu_quota is True
        assert plan.short_sha == "0123456"
        assert "alice/ecc-runner" in plan.summary
        assert "alice/vlm-exp@main" in plan.summary
        assert "GPU quota" in plan.summary

        rec = svc.execute(plan, confirmed_by_user=True)
        assert rec.ok is True
        assert rec.version == 3
        assert rec.url == "https://www.kaggle.com/code/alice/ecc-runner"
        assert rec.id == plan.launch_id
        assert rec.account == "kaggle-1"
        assert rec.kernel_ref == "alice/ecc-runner"
        assert rec.commit_sha == "0123456789abcdef0123456789abcdef01234567"

        # Temp folder was cleaned up after push
        pushed_folder: Path = env["pushed_calls"][0]["folder"]
        assert not pushed_folder.exists()
        assert env["pushed_calls"][0]["accelerator"] == "NvidiaTeslaT4"
        assert env["pushed_calls"][0]["timeout_s"] == 7200

        # Audit row persisted and retrievable
        fetched = svc.get_launch(rec.id)
        assert fetched is not None
        assert fetched.id == rec.id
        listed = svc.list_launches(limit=10)
        assert len(listed) == 1
        assert listed[0].id == rec.id

        # Kernel linked to project
        links = env["store"].list_kernel_links(account="kaggle-1")
        assert len(links) == 1
        assert links[0]["kernel_ref"] == "alice/ecc-runner"
        assert links[0]["project"] == "shared-task-4"

    def test_preflight_happy_path_uses_cpu_no_gpu_quota(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path)
        svc: LaunchService = env["svc"]
        plan = svc.plan_launch(
            _valid_request(command="", accelerator="NvidiaTeslaT4", preflight=True)
        )
        assert plan.preflight is True
        assert plan.accelerator == "none"
        assert plan.consumes_gpu_quota is False

        rec = svc.execute(plan, confirmed_by_user=True)
        assert rec.ok is True
        assert rec.preflight is True
        assert env["pushed_calls"][0]["accelerator"] is None
        assert env["pushed_calls"][0]["metadata"]["enable_gpu"] is False
        assert env["pushed_calls"][0]["metadata"]["enable_tpu"] is False


class TestCliLaunchAndSourcePersistence:
    def test_ecc_cli_exposes_two_step_launch_and_dataset_commands(self) -> None:
        parser = ecc_cli._build_parser()
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, ecc_cli.argparse._SubParsersAction)
        )
        choices = set(subparsers_action.choices.keys())
        assert "launch" in choices
        assert "dataset" in choices

        launch_parser = subparsers_action.choices["launch"]
        launch_sub = next(
            a for a in launch_parser._actions if isinstance(a, ecc_cli.argparse._SubParsersAction)
        )
        assert set(launch_sub.choices.keys()) == {"plan", "execute", "status", "history"}

        ds_parser = subparsers_action.choices["dataset"]
        ds_sub = next(
            a for a in ds_parser._actions if isinstance(a, ecc_cli.argparse._SubParsersAction)
        )
        assert set(ds_sub.choices.keys()) == {"upload", "build-monitor-wheel"}

    def test_launch_source_defaults_to_ui_and_persists_claude(self, tmp_path: Path) -> None:
        env = _make_launch_env(tmp_path)
        svc: LaunchService = env["svc"]

        # Default UI source
        plan_ui = svc.plan_launch(_valid_request())
        assert plan_ui.source == "ui"
        rec_ui = svc.execute(plan_ui, confirmed_by_user=True)
        assert rec_ui.source == "ui"
        assert svc.get_launch(rec_ui.id).source == "ui"  # type: ignore[union-attr]

        # Explicit claude source + plan persistence round-trip
        plan_claude = svc.plan_launch(_valid_request(run_name="run-claude"), source="claude")
        assert plan_claude.source == "claude"
        saved = svc.save_plan(plan_claude)
        assert saved["plan_id"] == plan_claude.launch_id

        rec_claude = svc.execute_saved_plan(
            plan_claude.launch_id, confirmed_by_user=True, source="claude"
        )
        assert rec_claude.source == "claude"
        assert svc.get_launch(rec_claude.id).source == "claude"  # type: ignore[union-attr]

    def test_store_migrates_v2_to_v3_preserving_launches_with_ui_default(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "legacy_v2.db"
        conn = sqlite3.connect(db)
        with conn:
            conn.executescript(
                """
                CREATE TABLE projects (
                    slug TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    wandb_entity TEXT, wandb_project TEXT, tags TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '', baseline_run_id TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE run_notes (
                    run_id TEXT PRIMARY KEY, notes TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]', pinned INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE accounts (
                    nickname TEXT PRIMARY KEY,
                    platform TEXT NOT NULL CHECK(platform IN ('kaggle', 'colab', 'wandb')),
                    username TEXT NOT NULL DEFAULT '', secret_ref TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1, last_check_at TEXT,
                    last_check_ok INTEGER, last_check_msg TEXT
                );
                CREATE TABLE kernel_links (
                    kernel_ref TEXT PRIMARY KEY, account TEXT NOT NULL,
                    project TEXT NOT NULL, run_id TEXT
                );
                CREATE TABLE launches (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, account TEXT NOT NULL,
                    kernel_ref TEXT NOT NULL, repo TEXT NOT NULL, branch TEXT NOT NULL,
                    commit_sha TEXT NOT NULL, command TEXT NOT NULL, accelerator TEXT NOT NULL,
                    timeout_s INTEGER NOT NULL, monitor_mode TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT '', run_name TEXT NOT NULL DEFAULT '',
                    preflight INTEGER NOT NULL DEFAULT 0, ok INTEGER NOT NULL DEFAULT 0,
                    version INTEGER, url TEXT, message TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO launches (
                    id, created_at, account, kernel_ref, repo, branch, commit_sha,
                    command, accelerator, timeout_s, monitor_mode, ok
                ) VALUES (
                    'legacy-1', '2026-11-20T00:00:00+00:00', 'kaggle-1', 'alice/ecc-runner',
                    'alice/repo', 'main', '0123456789abcdef0123456789abcdef01234567',
                    'python train.py', 'NvidiaTeslaT4', 3600, 'runner', 1
                );
                PRAGMA user_version = 2;
                """
            )
        conn.close()

        store = Store(db)
        assert store.user_version() == 3
        legacy = store.get_launch("legacy-1")
        assert legacy is not None
        assert legacy["source"] == "ui"

        # Re-open is idempotent
        store2 = Store(db)
        assert store2.user_version() == 3
        assert store2.get_pending_plan("missing") is None
        assert store2.list_dataset_uploads() == []
