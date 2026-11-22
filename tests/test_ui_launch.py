"""Streamlit AppTest and unit tests for the Launch page and launch wiring."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from app.ui import state as ui_state
from ecc.config import LaunchSettings, Settings
from ecc.launch.github import BranchInfo, CommitInfo, GitHubError, GitHubReader, RepoInfo
from ecc.launch.service import (
    BUSY_STATUSES,
    LaunchPlan,
    LaunchRecord,
    LaunchRefused,
    LaunchRequest,
    LaunchService,
    build_launch_service,
)
from ecc.secrets import SecretStore
from ecc.service import Service
from ecc.store import Store

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
LAUNCH_PAGE = APP_DIR / "pages" / "launch.py"


class _FakeGitHubReader(GitHubReader):
    def __init__(
        self,
        *,
        repos: list[RepoInfo] | None = None,
        branches: dict[str, list[BranchInfo]] | None = None,
        commits: dict[tuple[str, str], list[CommitInfo]] | None = None,
        error: GitHubError | None = None,
    ) -> None:
        super().__init__(runner=lambda *a, **kw: None)
        self._repos = list(
            repos
            if repos is not None
            else [
                RepoInfo(
                    name_with_owner="alice/vision-exp",
                    is_private=True,
                    default_branch="main",
                    updated_at="2026-11-22T09:00:00Z",
                ),
            ]
        )
        self._branches = (
            branches
            if branches is not None
            else {
                "alice/vision-exp": [
                    BranchInfo(name="main", sha="abcdef1234567890abcdef1234567890abcdef12"),
                    BranchInfo(name="exp/lora", sha="11223344556677889900aabbccddeeff00112233"),
                ]
            }
        )
        self._commits = (
            commits
            if commits is not None
            else {
                ("alice/vision-exp", "main"): [
                    CommitInfo(
                        sha="abcdef1234567890abcdef1234567890abcdef12",
                        short_sha="abcdef1",
                        message_first_line="Add Qwen3-VL LoRA trainer",
                        author="Alice",
                        date="2026-11-22T09:00:00Z",
                    ),
                    CommitInfo(
                        sha="99887766554433221100ffeeddccbbaa99887766",
                        short_sha="9988776",
                        message_first_line="Initial commit",
                        author="Alice",
                        date="2026-11-21T08:00:00Z",
                    ),
                ],
                ("alice/vision-exp", "exp/lora"): [
                    CommitInfo(
                        sha="11223344556677889900aabbccddeeff00112233",
                        short_sha="1122334",
                        message_first_line="Tune rank 16",
                        author="Alice",
                        date="2026-11-22T10:30:00Z",
                    ),
                ],
            }
        )
        self._error = error

    def list_repos(self, owner: str | None = None) -> list[RepoInfo]:
        if self._error is not None:
            raise self._error
        return list(self._repos)

    def list_branches(self, repo: str) -> list[BranchInfo]:
        if self._error is not None:
            raise self._error
        return list(self._branches.get(repo, []))

    def list_commits(self, repo: str, branch: str, n: int = 15) -> list[CommitInfo]:
        if self._error is not None:
            raise self._error
        return list(self._commits.get((repo, branch), []))[:n]

    def resolve_commit(self, repo: str, ref: str) -> CommitInfo:
        if self._error is not None:
            raise self._error
        for (r, _b), clist in self._commits.items():
            if r != repo:
                continue
            for c in clist:
                if c.sha == ref or c.short_sha == ref:
                    return c
        if (repo, ref) in self._commits and self._commits[(repo, ref)]:
            return self._commits[(repo, ref)][0]
        raise GitHubError(f"Unknown commit {ref!r} in {repo!r}")


class _FakeLaunchService:
    """In-memory fake LaunchService for deterministic AppTest UI testing."""

    def __init__(
        self,
        store: Store,
        *,
        enabled: bool = True,
        runner_status: str = "COMPLETE",
        owner_by_account: dict[str, str] | None = None,
        github: _FakeGitHubReader | None = None,
        history: list[LaunchRecord] | None = None,
        execute_ok: bool = True,
        execute_message: str = "Kernel version 7 successfully pushed.",
        execute_version: int | None = 7,
    ) -> None:
        self.store = store
        self.settings = Settings(
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
        self.github = github or _FakeGitHubReader()
        self._runner_status = runner_status
        self._owner_by_account = dict(owner_by_account or {"kaggle-1": "alice"})
        self._history: list[LaunchRecord] = list(history or [])
        self._execute_ok = execute_ok
        self._execute_message = execute_message
        self._execute_version = execute_version
        self.plan_calls: list[LaunchRequest] = []
        self.execute_calls: list[tuple[LaunchPlan, bool]] = []

    def is_enabled(self) -> bool:
        return bool(self.settings.launch.enabled)

    def account_owner(self, nickname: str) -> str | None:
        return self._owner_by_account.get(nickname)

    def runner_ref(self, nickname: str) -> str:
        owner = self.account_owner(nickname)
        if not owner:
            raise LaunchRefused(f"Kaggle username is not set for '{nickname}'")
        return f"{owner}/{self.settings.launch.runner_slug}"

    def runner_status(self, nickname: str) -> str:
        self.runner_ref(nickname)
        return self._runner_status

    def plan_launch(self, request: LaunchRequest) -> LaunchPlan:
        self.plan_calls.append(request)
        if not self.is_enabled():
            raise LaunchRefused("Launch control is disabled.")
        if self._runner_status in BUSY_STATUSES:
            raise LaunchRefused(f"account busy: runner has status {self._runner_status}")
        cmd = request.command.strip()
        if request.preflight and not cmd:
            cmd = "preflight"
        if not cmd:
            raise LaunchRefused("Command must be non-empty")

        owner = self.account_owner(request.account) or "alice"
        commit = self.github.resolve_commit(
            request.repo, request.commit_sha or request.branch
        )
        eff_acc = "none" if request.preflight else request.accelerator
        consumes_gpu = (not request.preflight) and (eff_acc.lower() != "none")
        kernel_ref = f"{owner}/{self.settings.launch.runner_slug}"
        kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
        timeout_s = int(round(request.timeout_hours * 3600))
        summary = (
            f"Account      : {request.account} (owner: {owner})\n"
            f"Kernel       : {kernel_url}\n"
            f"Source       : {request.repo}@{request.branch} ({commit.short_sha})\n"
            f"Command      : {cmd}\n"
            f"Accelerator  : {eff_acc}\n"
            f"GPU quota    : {'YES' if consumes_gpu else 'NO'}"
        )
        return LaunchPlan(
            launch_id="plan-1234",
            created_at=datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc),
            account=request.account,
            owner=owner,
            runner_slug=self.settings.launch.runner_slug,
            kernel_ref=kernel_ref,
            kernel_url=kernel_url,
            repo=request.repo,
            branch=request.branch,
            commit_sha=commit.sha,
            short_sha=commit.short_sha,
            commit_message=commit.message_first_line,
            command=cmd,
            accelerator=eff_acc,
            timeout_s=timeout_s,
            monitor_mode=request.monitor_mode,
            project=request.project,
            run_name=request.run_name,
            requirements_file=request.requirements_file,
            preflight=request.preflight,
            consumes_gpu_quota=consumes_gpu,
            summary=summary,
            request=request,
        )

    def execute(self, plan: LaunchPlan, confirmed_by_user: bool) -> LaunchRecord:
        self.execute_calls.append((plan, confirmed_by_user))
        if confirmed_by_user is not True:
            raise LaunchRefused("confirmed_by_user must be True")
        rec = LaunchRecord(
            id=plan.launch_id,
            created_at="2026-11-22T12:01:00+00:00",
            account=plan.account,
            kernel_ref=plan.kernel_ref,
            repo=plan.repo,
            branch=plan.branch,
            commit_sha=plan.commit_sha,
            command=plan.command,
            accelerator=plan.accelerator,
            timeout_s=plan.timeout_s,
            monitor_mode=plan.monitor_mode,
            project=plan.project,
            run_name=plan.run_name,
            preflight=plan.preflight,
            ok=self._execute_ok,
            version=self._execute_version if self._execute_ok else None,
            url=plan.kernel_url,
            message=self._execute_message,
        )
        self._history.insert(0, rec)
        return rec

    def list_launches(self, limit: int = 50) -> list[LaunchRecord]:
        return list(self._history[:limit])

    def get_launch(self, launch_id: str) -> LaunchRecord | None:
        for rec in self._history:
            if rec.id == launch_id:
                return rec
        return None


def _write_temp_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    launch_enabled: bool,
) -> tuple[Path, Path]:
    db_path = tmp_path / "launch_ui_test.db"
    cfg_path = tmp_path / "ecc_config.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[general]",
                "refresh_seconds = 15",
                "stale_after_s = 180",
                "offline_after_s = 900",
                'demo_mode = "on"',
                f'db_path = "{db_path.as_posix()}"',
                "",
                "[wandb]",
                'entity = ""',
                "projects = []",
                "",
                "[launch]",
                f"enabled = {'true' if launch_enabled else 'false'}",
                'default_wandb_project = "shared-task-4"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ECC_CONFIG", str(cfg_path))
    ui_state.clear_service_cache()
    return cfg_path, db_path


def test_build_launch_service_shares_store_settings_and_secrets(tmp_path: Path) -> None:
    store = Store(tmp_path / "ecc.db")
    settings = Settings(launch=LaunchSettings(enabled=True, runner_slug="ecc-runner"))
    secrets = SecretStore()
    read_svc = Service(sources=[], store=store, settings=settings, secrets=secrets)
    fake_gh = _FakeGitHubReader()

    launch_svc = build_launch_service(read_svc, github=fake_gh)

    assert isinstance(launch_svc, LaunchService)
    assert launch_svc.store is store
    assert launch_svc.settings is settings
    assert launch_svc.secrets is secrets
    assert launch_svc.read_service is read_svc
    assert launch_svc.github is fake_gh


def test_launch_page_disabled_shows_info_and_hides_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_temp_config(tmp_path, monkeypatch, launch_enabled=False)
    try:
        at = AppTest.from_file(str(LAUNCH_PAGE))
        at.run(timeout=30)

        assert not at.exception
        info_text = "\n".join(i.value for i in at.info)
        assert "enabled = true" in info_text
        assert "ecc_config.toml" in info_text
        assert "GPU quota" in info_text
        assert "Claude never launches" in info_text

        button_labels = [b.label for b in at.button]
        assert not any("Launch training" in lbl for lbl in button_labels)
        assert not any("Preflight" in lbl for lbl in button_labels)
    finally:
        ui_state.clear_service_cache()


def test_launch_page_confirm_and_execute_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, db_path = _write_temp_config(tmp_path, monkeypatch, launch_enabled=True)
    store = Store(db_path)
    store.upsert_account("kaggle-1", "kaggle", username="alice", enabled=True)

    fake_launch = _FakeLaunchService(store=store, enabled=True, runner_status="COMPLETE")
    monkeypatch.setattr(ui_state, "get_launch_service", lambda: fake_launch)

    try:
        at = AppTest.from_file(str(LAUNCH_PAGE))
        at.run(timeout=30)
        assert not at.exception

        # Select account, repo, branch, commit, and enter training command
        at.selectbox(key="launch_account_select").select("kaggle-1").run(timeout=30)
        at.selectbox(key="launch_repo_select").select("alice/vision-exp").run(timeout=30)
        at.selectbox(key="launch_branch_select").select("main").run(timeout=30)
        at.selectbox(key="launch_commit_select").select(
            "abcdef1234567890abcdef1234567890abcdef12"
        ).run(timeout=30)
        at.text_input(key="launch_command_input").input(
            "python train.py --config configs/qwen3vl_r16.yaml"
        ).run(timeout=30)

        # Click Launch training -> opens confirmation panel
        at.button(key="launch_train_btn").click().run(timeout=30)
        assert not at.exception
        assert len(fake_launch.plan_calls) == 1
        assert len(fake_launch.execute_calls) == 0

        md_text = "\n".join(m.value for m in at.markdown)
        assert "kaggle-1" in md_text
        assert "alice" in md_text
        assert "abcdef1" in md_text
        assert "Consumes GPU quota" in md_text

        # Confirm button must be disabled before ticking confirmation checkbox
        confirm_btn = at.button(key="launch_confirm_push_btn")
        assert confirm_btn.disabled is True

        # Tick confirmation checkbox -> Confirm & push becomes enabled
        at.checkbox(key="launch_confirm_checkbox").check().run(timeout=30)
        confirm_btn = at.button(key="launch_confirm_push_btn")
        assert confirm_btn.disabled is False

        # Click Confirm & push -> execute called exactly once with confirmed_by_user=True
        confirm_btn.click().run(timeout=30)
        assert not at.exception
        assert len(fake_launch.execute_calls) == 1
        called_plan, confirmed_flag = fake_launch.execute_calls[0]
        assert confirmed_flag is True
        assert called_plan.account == "kaggle-1"
        assert called_plan.short_sha == "abcdef1"
        assert called_plan.preflight is False

        # Success message shows version + kernel link + View on Overview, pending plan cleared
        success_text = "\n".join(s.value for s in at.success)
        assert "v7" in success_text
        assert "https://www.kaggle.com/code/alice/ecc-runner" in success_text
        assert "View on Overview" in success_text
        assert "launch_pending_plan" not in at.session_state
    finally:
        ui_state.clear_service_cache()


def test_launch_page_busy_runner_disables_launch_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, db_path = _write_temp_config(tmp_path, monkeypatch, launch_enabled=True)
    store = Store(db_path)
    store.upsert_account("kaggle-1", "kaggle", username="alice", enabled=True)

    fake_launch = _FakeLaunchService(store=store, enabled=True, runner_status="RUNNING")
    monkeypatch.setattr(ui_state, "get_launch_service", lambda: fake_launch)

    try:
        at = AppTest.from_file(str(LAUNCH_PAGE))
        at.run(timeout=30)

        assert not at.exception
        warning_text = "\n".join(w.value for w in at.warning)
        assert "account busy" in warning_text.lower()
        assert "wait or stop it on kaggle" in warning_text.lower()
        assert "https://www.kaggle.com/code/alice/ecc-runner" in warning_text

        assert at.button(key="launch_preflight_btn").disabled is True
        assert at.button(key="launch_train_btn").disabled is True
    finally:
        ui_state.clear_service_cache()


def test_launch_page_renders_history_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, db_path = _write_temp_config(tmp_path, monkeypatch, launch_enabled=True)
    store = Store(db_path)
    store.upsert_account("kaggle-1", "kaggle", username="alice", enabled=True)

    long_cmd = "python train.py --config configs/qwen3vl_r16.yaml --epochs 10 --lr 2e-4 --batch-size 16"
    fake_records = [
        LaunchRecord(
            id="rec-1",
            created_at="2026-11-22T10:15:00+00:00",
            account="kaggle-1",
            kernel_ref="alice/ecc-runner",
            repo="alice/vision-exp",
            branch="main",
            commit_sha="abcdef1234567890abcdef1234567890abcdef12",
            command=long_cmd,
            accelerator="NvidiaTeslaT4",
            timeout_s=7200,
            monitor_mode="runner",
            project="shared-task-4",
            run_name="vision-exp-main-abcdef1",
            preflight=False,
            ok=True,
            version=4,
            url="https://www.kaggle.com/code/alice/ecc-runner",
            message="ok",
        ),
        LaunchRecord(
            id="rec-2",
            created_at="2026-11-22T09:00:00+00:00",
            account="kaggle-1",
            kernel_ref="alice/ecc-runner",
            repo="alice/vision-exp",
            branch="exp/lora",
            commit_sha="11223344556677889900aabbccddeeff00112233",
            command="preflight",
            accelerator="none",
            timeout_s=3600,
            monitor_mode="runner",
            project="shared-task-4",
            run_name="vision-exp-exp-lora-1122334",
            preflight=True,
            ok=False,
            version=None,
            url="https://www.kaggle.com/code/alice/ecc-runner",
            message="failed",
        ),
    ]

    fake_launch = _FakeLaunchService(
        store=store,
        enabled=True,
        runner_status="COMPLETE",
        history=fake_records,
    )
    monkeypatch.setattr(ui_state, "get_launch_service", lambda: fake_launch)

    try:
        at = AppTest.from_file(str(LAUNCH_PAGE))
        at.run(timeout=30)

        assert not at.exception
        assert len(at.dataframe) >= 1
        df = at.dataframe[0].value
        assert len(df) == 2
        expected_cols = {
            "Time",
            "Account",
            "Repo@Branch",
            "Short SHA",
            "Command",
            "Accelerator",
            "Preflight",
            "OK",
            "Version",
            "Link",
        }
        assert expected_cols.issubset(set(df.columns))
        assert df.iloc[0]["Account"] == "kaggle-1"
        assert df.iloc[0]["Repo@Branch"] == "alice/vision-exp@main"
        assert df.iloc[0]["Short SHA"] == "abcdef1"
        assert df.iloc[0]["Command"].endswith("...")
        assert df.iloc[0]["Preflight"] == "No"
        assert df.iloc[0]["OK"] == "Yes"
        assert df.iloc[0]["Version"] == "v4"
        assert df.iloc[1]["Preflight"] == "Yes"
        assert df.iloc[1]["OK"] == "No"
    finally:
        ui_state.clear_service_cache()


def test_launch_page_github_error_shows_auth_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, db_path = _write_temp_config(tmp_path, monkeypatch, launch_enabled=True)
    store = Store(db_path)
    store.upsert_account("kaggle-1", "kaggle", username="alice", enabled=True)

    failing_gh = _FakeGitHubReader(error=GitHubError("gh CLI failed: not logged in"))
    fake_launch = _FakeLaunchService(
        store=store, enabled=True, runner_status="NOT_CREATED", github=failing_gh
    )
    monkeypatch.setattr(ui_state, "get_launch_service", lambda: fake_launch)

    try:
        at = AppTest.from_file(str(LAUNCH_PAGE))
        at.run(timeout=30)

        assert not at.exception
        err_text = "\n".join(e.value for e in at.error)
        assert "gh auth status" in err_text
    finally:
        ui_state.clear_service_cache()


def test_launch_page_preflight_and_error_and_refused_flows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, db_path = _write_temp_config(tmp_path, monkeypatch, launch_enabled=True)
    store = Store(db_path)
    store.upsert_account("kaggle-1", "kaggle", username="alice", enabled=True)

    fake_launch = _FakeLaunchService(
        store=store,
        enabled=True,
        runner_status="NOT_CREATED",
        execute_ok=False,
        execute_message="Kaggle push failed (redacted)",
    )
    monkeypatch.setattr(ui_state, "get_launch_service", lambda: fake_launch)

    try:
        at = AppTest.from_file(str(LAUNCH_PAGE))
        at.run(timeout=30)
        assert not at.exception

        # 1. Launch training with empty command -> LaunchRefused -> st.warning, no confirm panel
        at.button(key="launch_train_btn").click().run(timeout=30)
        assert not at.exception
        warn_text = "\n".join(w.value for w in at.warning)
        assert "non-empty" in warn_text.lower()
        assert "launch_pending_plan" not in at.session_state

        # 2. Preflight with empty command -> builds preflight=True, accelerator="none", NO GPU quota
        at.button(key="launch_preflight_btn").click().run(timeout=30)
        assert not at.exception
        assert fake_launch.plan_calls[-1].preflight is True
        assert fake_launch.plan_calls[-1].accelerator == "none"
        md_text = "\n".join(m.value for m in at.markdown)
        assert "NO" in md_text
        assert at.button(key="launch_confirm_push_btn").disabled is True

        # 3. Confirm & push when execute returns ok=False -> st.error shown, pending plan cleared
        at.checkbox(key="launch_confirm_checkbox").check().run(timeout=30)
        at.button(key="launch_confirm_push_btn").click().run(timeout=30)
        assert not at.exception
        assert len(fake_launch.execute_calls) == 1
        assert fake_launch.execute_calls[0][1] is True
        err_text = "\n".join(e.value for e in at.error)
        assert "Kaggle push failed (redacted)" in err_text
        assert "launch_pending_plan" not in at.session_state
    finally:
        ui_state.clear_service_cache()


def test_launch_page_has_no_direct_subprocess_or_network_imports() -> None:
    forbidden = {
        "subprocess",
        "wandb",
        "kaggle",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
    }
    tree = ast.parse(LAUNCH_PAGE.read_text(encoding="utf-8"), filename=str(LAUNCH_PAGE))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden
