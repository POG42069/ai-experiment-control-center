"""High-level LaunchService: plan validation, human-confirmed execution, and audit log."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ecc.adapters.kaggle import KaggleCLIError
from ecc.config import Settings
from ecc.launch.builder import build_kernel_folder, build_wheel, default_monitor_dir
from ecc.launch.github import (
    GitHubError,
    GitHubReader,
    validate_branch,
    validate_ref,
    validate_repo,
)
from ecc.launch.kaggle_launcher import KaggleLauncher
from ecc.launch.runner_template import RunnerParams
from ecc.models import to_jsonable
from ecc.redact import redact
from ecc.secrets import SecretStore
from ecc.store import Store

BUSY_STATUSES: frozenset[str] = frozenset({"QUEUED", "RUNNING"})
PLAN_MAX_AGE_SECONDS = 600.0  # 10 minutes
MAX_COMMAND_LENGTH = 500


class LaunchRefused(RuntimeError):
    """Raised when a launch plan or execution violates safety or validation rules."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_runner_status(raw: str | None) -> str:
    if not raw:
        return "UNKNOWN"
    s = raw.strip().upper()
    if s.startswith("RUNNING"):
        return "RUNNING"
    if s.startswith("KERNELWORKERSTATUS."):
        s = s.split(".", 1)[1]
    return s


@dataclass
class LaunchRequest:
    account: str
    repo: str
    branch: str
    commit_sha: str | None = None
    command: str = ""
    accelerator: str = "NvidiaTeslaT4"
    timeout_hours: float = 1.0
    monitor_mode: str = "runner"
    project: str = ""
    run_name: str = ""
    requirements_file: str = ""
    preflight: bool = False
    source: str = "ui"


@dataclass
class LaunchPlan:
    launch_id: str
    created_at: datetime
    account: str
    owner: str
    runner_slug: str
    kernel_ref: str
    kernel_url: str
    repo: str
    branch: str
    commit_sha: str
    short_sha: str
    commit_message: str
    command: str
    accelerator: str
    timeout_s: int
    monitor_mode: str
    project: str
    run_name: str
    requirements_file: str
    preflight: bool
    consumes_gpu_quota: bool
    summary: str
    request: LaunchRequest
    source: str = "ui"

    def to_dict(self) -> dict[str, Any]:
        return {
            "launch_id": self.launch_id,
            "created_at": to_jsonable(self.created_at),
            "account": self.account,
            "owner": self.owner,
            "runner_slug": self.runner_slug,
            "kernel_ref": self.kernel_ref,
            "kernel_url": self.kernel_url,
            "repo": self.repo,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "short_sha": self.short_sha,
            "commit_message": self.commit_message,
            "command": self.command,
            "accelerator": self.accelerator,
            "timeout_s": self.timeout_s,
            "monitor_mode": self.monitor_mode,
            "project": self.project,
            "run_name": self.run_name,
            "requirements_file": self.requirements_file,
            "preflight": self.preflight,
            "consumes_gpu_quota": self.consumes_gpu_quota,
            "summary": self.summary,
            "source": self.source,
        }


@dataclass
class LaunchRecord:
    id: str
    created_at: str
    account: str
    kernel_ref: str
    repo: str
    branch: str
    commit_sha: str
    command: str
    accelerator: str
    timeout_s: int
    monitor_mode: str
    project: str
    run_name: str
    preflight: bool
    ok: bool
    version: int | None
    url: str | None
    message: str
    source: str = "ui"

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> LaunchRecord:
        return cls(
            id=str(row["id"]),
            created_at=str(row["created_at"]),
            account=str(row["account"]),
            kernel_ref=str(row["kernel_ref"]),
            repo=str(row["repo"]),
            branch=str(row["branch"]),
            commit_sha=str(row["commit_sha"]),
            command=str(row["command"]),
            accelerator=str(row["accelerator"]),
            timeout_s=int(row["timeout_s"]),
            monitor_mode=str(row["monitor_mode"]),
            project=str(row.get("project") or ""),
            run_name=str(row.get("run_name") or ""),
            preflight=bool(row["preflight"]),
            ok=bool(row["ok"]),
            version=row.get("version"),
            url=row.get("url"),
            message=str(row.get("message") or ""),
            source=str(row.get("source") or "ui"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "account": self.account,
            "kernel_ref": self.kernel_ref,
            "repo": self.repo,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "command": self.command,
            "accelerator": self.accelerator,
            "timeout_s": self.timeout_s,
            "monitor_mode": self.monitor_mode,
            "project": self.project,
            "run_name": self.run_name,
            "preflight": self.preflight,
            "ok": self.ok,
            "version": self.version,
            "url": self.url,
            "message": self.message,
            "source": self.source,
        }


class LaunchService:
    """Plan, validate, execute (with explicit human confirmation), and audit Kaggle launches."""

    def __init__(
        self,
        store: Store,
        secrets: SecretStore,
        settings: Settings,
        read_service: Any,
        github: GitHubReader,
        launcher_factory: Callable[[str, Callable[[], str | None]], KaggleLauncher] | None = None,
        wheel_builder: Callable[[], bytes] | None = None,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.store = store
        self.secrets = secrets
        self.settings = settings
        self.read_service = read_service
        self.github = github
        self._launcher_factory = launcher_factory or (
            lambda nick, getter: KaggleLauncher(nick, token_getter=getter)
        )
        self._wheel_builder = wheel_builder or (lambda: build_wheel(default_monitor_dir()))
        self._now_fn = now_fn

    def is_enabled(self) -> bool:
        return bool(self.settings.launch.enabled)

    def _find_account(self, nickname: str) -> dict[str, Any] | None:
        for acc in self.store.list_accounts():
            if acc.get("nickname") == nickname:
                return acc
        return None

    def account_owner(self, nickname: str) -> str | None:
        """Resolve the Kaggle owner username for ``nickname``."""
        acc = self._find_account(nickname)
        if acc is not None:
            stored_user = str(acc.get("username") or "").strip()
            if stored_user:
                return stored_user

        if self.read_service is not None and hasattr(self.read_service, "list_kaggle_kernels"):
            try:
                kernels = self.read_service.list_kaggle_kernels(nickname)
            except Exception:
                kernels = []
            owners: list[str] = []
            for k in kernels:
                ref = getattr(k, "ref", "") or ""
                if "/" in ref:
                    owner = ref.split("/", 1)[0].strip()
                    if owner:
                        owners.append(owner)
            if owners:
                return Counter(owners).most_common(1)[0][0]
        return None

    def runner_ref(self, nickname: str) -> str:
        """Return ``<owner>/<runner_slug>`` for ``nickname`` or raise ``LaunchRefused``."""
        owner = self.account_owner(nickname)
        if not owner:
            raise LaunchRefused(
                f"Kaggle username is not set for account '{nickname}'; "
                "set username via scripts/add_account.py --username"
            )
        return f"{owner}/{self.runner_slug_for(owner, nickname)}"

    def credentials_dataset_sources(self, owner: str, nickname: str) -> tuple[str, ...]:
        """``("<owner>/ecc-credentials",)`` from ``[launch].credentials_dataset``, or ``()`` if unset."""
        template = (getattr(self.settings.launch, "credentials_dataset", "") or "").strip()
        if not template:
            return ()
        try:
            ref = template.format(owner=owner, account=nickname).strip().lower()
        except (KeyError, IndexError, ValueError) as exc:
            raise LaunchRefused(f"invalid [launch].credentials_dataset template {template!r}") from exc
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9-]{2,49}", ref):
            raise LaunchRefused(f"credentials dataset ref {ref!r} is not a valid Kaggle dataset ref")
        return (ref,)

    def runner_slug_for(self, owner: str, nickname: str) -> str:
        """Expand ``[launch].runner_slug``; supports ``{owner}`` / ``{account}`` placeholders.

        Example: ``"mmculture-qa-new-{owner}"`` -> ``"mmculture-qa-new-alice"``.
        """
        template = self.settings.launch.runner_slug
        try:
            slug = template.format(owner=owner, account=nickname)
        except (KeyError, IndexError, ValueError) as exc:
            raise LaunchRefused(
                f"invalid [launch].runner_slug template {template!r}: only {{owner}} and {{account}} are allowed"
            ) from exc
        slug = slug.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{3,49}", slug):
            raise LaunchRefused(f"runner slug {slug!r} is not a valid Kaggle notebook slug")
        return slug

    def runner_status(self, nickname: str) -> str:
        """Check current status of the account's fixed runner kernel via the read adapter.

        Returns ``"NOT_CREATED"`` when the runner kernel does not exist yet.
        """
        ref = self.runner_ref(nickname)
        if self.read_service is None:
            return "NOT_CREATED"

        get_ksrc = getattr(self.read_service, "_get_kaggle_source", None)
        kaggle_src = get_ksrc() if callable(get_ksrc) else None

        if kaggle_src is not None:
            clients = getattr(kaggle_src, "_clients", None)
            if isinstance(clients, dict) and nickname in clients:
                cli = clients[nickname]
                try:
                    status_str, _ = cli.status(ref)
                    return _normalize_runner_status(status_str)
                except KaggleCLIError as exc:
                    msg_lower = str(exc).lower()
                    if (
                        "404" in msg_lower
                        or "not found" in msg_lower
                        or "does not exist" in msg_lower
                        or "no kernel" in msg_lower
                    ):
                        return "NOT_CREATED"
                    # Kaggle answers 403 "Permission 'kernels.get' was denied" for a kernel that does
                    # not exist. The runner belongs to this very account, so confirm via our own kernel
                    # list: if the ref is absent there, it simply has not been created yet.
                    if "was denied" in msg_lower or "403" in msg_lower:
                        try:
                            own_refs = {k.ref.lower() for k in cli.list_kernels()}
                        except Exception:
                            own_refs = None
                        if own_refs is not None and ref.lower() not in own_refs:
                            return "NOT_CREATED"
                    raise LaunchRefused(
                        f"Failed to check runner status for '{nickname}': {redact(str(exc))}"
                    ) from exc

            st = kaggle_src.get_kernel_status(ref, nickname=nickname)
            if st is None:
                return "NOT_CREATED"
            return _normalize_runner_status(st[0])

        if hasattr(self.read_service, "get_kernel_status"):
            st = self.read_service.get_kernel_status(ref, nickname=nickname)
            if st is None:
                return "NOT_CREATED"
            raw = st[0] if isinstance(st, tuple) else str(st)
            return _normalize_runner_status(raw)

        return "NOT_CREATED"

    def plan_launch(
        self,
        request: LaunchRequest,
        *,
        source: str | None = None,
    ) -> LaunchPlan:
        """Validate a launch request and produce a side-effect-free ``LaunchPlan``."""
        effective_source = (source if source is not None else request.source or "ui").strip() or "ui"
        if request.source != effective_source:
            request = replace(request, source=effective_source)
        if not self.is_enabled():
            raise LaunchRefused(
                "Launch control is disabled. Set [launch] enabled = true in ecc_config.toml."
            )

        acc = self._find_account(request.account)
        if acc is None:
            raise LaunchRefused(f"Unknown account '{request.account}'")
        if acc.get("platform") != "kaggle":
            raise LaunchRefused(
                f"Account '{request.account}' is platform '{acc.get('platform')}'; only Kaggle accounts can be launched"
            )
        if not acc.get("enabled", True):
            raise LaunchRefused(f"Account '{request.account}' is disabled")

        if not self.secrets.has("kaggle", request.account):
            raise LaunchRefused(
                f"Missing Kaggle API token for account '{request.account}'"
            )

        owner = self.account_owner(request.account)
        if not owner:
            raise LaunchRefused(
                f"Kaggle username is unknown for account '{request.account}'; "
                "set username via scripts/add_account.py --username"
            )

        if request.monitor_mode not in {"runner", "script"}:
            raise LaunchRefused(
                f"monitor_mode must be 'runner' or 'script', got {request.monitor_mode!r}"
            )

        # Timeout validation
        if (
            isinstance(request.timeout_hours, bool)
            or not isinstance(request.timeout_hours, (int, float))
            or request.timeout_hours <= 0
        ):
            raise LaunchRefused(f"timeout_hours must be > 0, got {request.timeout_hours!r}")
        if request.timeout_hours > self.settings.launch.max_timeout_hours:
            raise LaunchRefused(
                f"timeout_hours ({request.timeout_hours}) exceeds max_timeout_hours "
                f"({self.settings.launch.max_timeout_hours})"
            )
        timeout_s = int(round(float(request.timeout_hours) * 3600))

        # Accelerator validation (preflight always runs on CPU / "none")
        if request.preflight:
            effective_accelerator = "none"
        else:
            effective_accelerator = (request.accelerator or "").strip()
            if effective_accelerator not in self.settings.launch.allowed_accelerators:
                raise LaunchRefused(
                    f"Accelerator {effective_accelerator!r} is not in allowed_accelerators "
                    f"{self.settings.launch.allowed_accelerators!r}"
                )

        # Command validation
        raw_cmd = request.command if isinstance(request.command, str) else ""
        if "\n" in raw_cmd or "\r" in raw_cmd:
            raise LaunchRefused("Command must be a single line (no newline characters)")
        cmd = raw_cmd.strip()
        if request.preflight and not cmd:
            cmd = "preflight"
        if not cmd:
            raise LaunchRefused("Command must be non-empty")
        if len(cmd) > MAX_COMMAND_LENGTH:
            raise LaunchRefused(
                f"Command exceeds maximum length of {MAX_COMMAND_LENGTH} characters ({len(cmd)})"
            )

        # Repo / branch / commit validation & resolution
        try:
            repo = validate_repo(request.repo)
            branch = validate_branch(request.branch)
            target_ref = validate_ref(request.commit_sha) if request.commit_sha else branch
        except ValueError as exc:
            raise LaunchRefused(str(exc)) from exc

        try:
            commit_info = self.github.resolve_commit(repo, target_ref)
        except GitHubError as exc:
            raise LaunchRefused(f"Failed to resolve GitHub commit: {redact(str(exc))}") from exc

        # Busy check via read adapter
        status = self.runner_status(request.account)
        if status in BUSY_STATUSES:
            raise LaunchRefused(
                f"account busy: runner kernel '{owner}/{self.runner_slug_for(owner, request.account)}' "
                f"has status {status}"
            )

        project = (request.project or self.settings.launch.default_wandb_project or "").strip()
        if not request.preflight and request.monitor_mode == "runner" and not project:
            raise LaunchRefused(
                "A W&B project is required when monitor_mode='runner' (set project or [launch].default_wandb_project)"
            )

        runner_slug = self.runner_slug_for(owner, request.account)
        kernel_ref = f"{owner}/{runner_slug}"
        kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
        consumes_gpu_quota = (not request.preflight) and (
            effective_accelerator.lower() != "none"
        )
        now = _ensure_utc(self._now_fn())
        launch_id = uuid.uuid4().hex
        run_name = (request.run_name or "").strip()
        req_file = (request.requirements_file or "").strip()

        timeout_h_str = f"{request.timeout_hours:g}h ({timeout_s}s)"
        quota_label = "YES (consumes GPU/TPU quota)" if consumes_gpu_quota else "NO (CPU only)"
        mode_label = "preflight" if request.preflight else request.monitor_mode
        summary = "\n".join(
            [
                f"Account      : {request.account} (owner: {owner})",
                f"Kernel       : {kernel_url}",
                f"Source       : {repo}@{branch} ({commit_info.short_sha} - {commit_info.message_first_line})",
                f"Command      : {cmd}",
                f"Accelerator  : {effective_accelerator}",
                f"Timeout      : {timeout_h_str}",
                f"Monitor mode : {mode_label}",
                f"Project      : {project or '(none)'}",
                f"GPU quota    : {quota_label}",
            ]
        )

        return LaunchPlan(
            launch_id=launch_id,
            created_at=now,
            account=request.account,
            owner=owner,
            runner_slug=runner_slug,
            kernel_ref=kernel_ref,
            kernel_url=kernel_url,
            repo=repo,
            branch=branch,
            commit_sha=commit_info.sha,
            short_sha=commit_info.short_sha,
            commit_message=commit_info.message_first_line,
            command=cmd,
            accelerator=effective_accelerator,
            timeout_s=timeout_s,
            monitor_mode=request.monitor_mode,
            project=project,
            run_name=run_name,
            requirements_file=req_file,
            preflight=bool(request.preflight),
            consumes_gpu_quota=consumes_gpu_quota,
            summary=summary,
            request=request,
            source=effective_source,
        )

    def save_plan(self, plan: LaunchPlan) -> dict[str, Any]:
        """Persist a validated ``LaunchPlan`` in SQLite ``pending_plans`` (no secrets)."""
        created_utc = _ensure_utc(plan.created_at)
        expires_utc = created_utc + timedelta(seconds=PLAN_MAX_AGE_SECONDS)
        payload: dict[str, Any] = {
            "launch_id": plan.launch_id,
            "created_at": created_utc.isoformat(),
            "account": plan.account,
            "owner": plan.owner,
            "runner_slug": plan.runner_slug,
            "kernel_ref": plan.kernel_ref,
            "kernel_url": plan.kernel_url,
            "repo": plan.repo,
            "branch": plan.branch,
            "commit_sha": plan.commit_sha,
            "short_sha": plan.short_sha,
            "commit_message": plan.commit_message,
            "command": plan.command,
            "accelerator": plan.accelerator,
            "timeout_s": plan.timeout_s,
            "timeout_hours": float(plan.request.timeout_hours),
            "monitor_mode": plan.monitor_mode,
            "project": plan.project,
            "run_name": plan.run_name,
            "requirements_file": plan.requirements_file,
            "preflight": bool(plan.preflight),
            "consumes_gpu_quota": bool(plan.consumes_gpu_quota),
            "summary": plan.summary,
            "source": plan.source,
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        self.store.save_pending_plan(
            plan_id=plan.launch_id,
            created_at=created_utc.isoformat(),
            payload_json=payload_json,
        )
        return {
            "plan_id": plan.launch_id,
            "created_at": created_utc.isoformat(),
            "expires_at": expires_utc.isoformat(),
        }

    def load_plan(self, plan_id: str) -> LaunchPlan:
        """Load a pending ``LaunchPlan`` by ID, enforcing existence, non-reuse, and TTL."""
        row = self.store.get_pending_plan(plan_id)
        if row is None:
            raise LaunchRefused(f"Unknown launch plan_id: '{plan_id}'")
        if row.get("executed_at"):
            raise LaunchRefused(
                f"Launch plan '{plan_id}' has already been executed at {row['executed_at']}"
            )
        try:
            payload = json.loads(row["payload_json"])
            created_at = _ensure_utc(datetime.fromisoformat(str(payload["created_at"])))
        except Exception as exc:
            raise LaunchRefused(f"Corrupt pending launch plan '{plan_id}': {exc}") from exc

        now = _ensure_utc(self._now_fn())
        age_s = (now - created_at).total_seconds()
        if age_s > PLAN_MAX_AGE_SECONDS:
            raise LaunchRefused(
                f"Launch plan '{plan_id}' is expired ({int(age_s)}s old; "
                f"max {int(PLAN_MAX_AGE_SECONDS)}s). Please re-plan before launching."
            )
        if age_s < -60.0:
            raise LaunchRefused("Launch plan timestamp is in the future; please re-plan.")

        source = str(payload.get("source") or "ui")
        timeout_s = int(payload["timeout_s"])
        timeout_hours = float(payload.get("timeout_hours", timeout_s / 3600.0))
        req = LaunchRequest(
            account=str(payload["account"]),
            repo=str(payload["repo"]),
            branch=str(payload["branch"]),
            commit_sha=str(payload["commit_sha"]),
            command=str(payload["command"]),
            accelerator=str(payload["accelerator"]),
            timeout_hours=timeout_hours,
            monitor_mode=str(payload["monitor_mode"]),
            project=str(payload.get("project") or ""),
            run_name=str(payload.get("run_name") or ""),
            requirements_file=str(payload.get("requirements_file") or ""),
            preflight=bool(payload.get("preflight", False)),
            source=source,
        )
        return LaunchPlan(
            launch_id=str(payload["launch_id"]),
            created_at=created_at,
            account=str(payload["account"]),
            owner=str(payload["owner"]),
            runner_slug=str(payload["runner_slug"]),
            kernel_ref=str(payload["kernel_ref"]),
            kernel_url=str(payload["kernel_url"]),
            repo=str(payload["repo"]),
            branch=str(payload["branch"]),
            commit_sha=str(payload["commit_sha"]),
            short_sha=str(payload["short_sha"]),
            commit_message=str(payload.get("commit_message") or ""),
            command=str(payload["command"]),
            accelerator=str(payload["accelerator"]),
            timeout_s=timeout_s,
            monitor_mode=str(payload["monitor_mode"]),
            project=str(payload.get("project") or ""),
            run_name=str(payload.get("run_name") or ""),
            requirements_file=str(payload.get("requirements_file") or ""),
            preflight=bool(payload.get("preflight", False)),
            consumes_gpu_quota=bool(payload.get("consumes_gpu_quota", False)),
            summary=str(payload.get("summary") or ""),
            request=req,
            source=source,
        )

    def mark_plan_executed(self, plan_id: str) -> bool:
        """Mark a persisted pending plan as executed."""
        now = _ensure_utc(self._now_fn())
        return self.store.mark_pending_plan_executed(plan_id, executed_at=now.isoformat())

    def execute_saved_plan(
        self,
        plan_id: str,
        *,
        confirmed_by_user: bool,
        source: str | None = None,
    ) -> LaunchRecord:
        """Load a persisted plan, execute it, and mark it executed."""
        if confirmed_by_user is not True:
            raise LaunchRefused(
                "Launch execution refused: confirmed_by_user must be explicitly True"
            )
        if not self.is_enabled():
            raise LaunchRefused(
                "Launch control is disabled. Set [launch] enabled = true in ecc_config.toml."
            )
        plan = self.load_plan(plan_id)
        record = self.execute(
            plan,
            confirmed_by_user=confirmed_by_user,
            source=source or plan.source,
        )
        self.mark_plan_executed(plan_id)
        return record

    def execute(
        self,
        plan: LaunchPlan,
        confirmed_by_user: bool,
        *,
        source: str | None = None,
    ) -> LaunchRecord:
        """Execute a validated ``LaunchPlan`` after explicit human confirmation."""
        if confirmed_by_user is not True:
            raise LaunchRefused(
                "Launch execution refused: confirmed_by_user must be explicitly True"
            )
        if not self.is_enabled():
            raise LaunchRefused(
                "Launch control is disabled. Set [launch] enabled = true in ecc_config.toml."
            )

        now = _ensure_utc(self._now_fn())
        age_s = (now - _ensure_utc(plan.created_at)).total_seconds()
        if age_s > PLAN_MAX_AGE_SECONDS:
            raise LaunchRefused(
                f"Launch plan is stale ({int(age_s)}s old; max {int(PLAN_MAX_AGE_SECONDS)}s). "
                "Please re-plan before launching."
            )
        if age_s < -60.0:
            raise LaunchRefused("Launch plan timestamp is in the future; please re-plan.")

        # Re-check runner status right before pushing
        status = self.runner_status(plan.account)
        if status in BUSY_STATUSES:
            raise LaunchRefused(
                f"account busy: runner kernel '{plan.kernel_ref}' has status {status}"
            )

        wheel_bytes = self._wheel_builder()
        runner_params = RunnerParams(
            launch_id=plan.launch_id,
            account=plan.account,
            owner=plan.owner,
            runner_slug=plan.runner_slug,
            repo=plan.repo,
            branch=plan.branch,
            commit_sha=plan.commit_sha,
            command=plan.command,
            monitor_mode=plan.monitor_mode,
            project=plan.project,
            run_name=plan.run_name,
            requirements_file=plan.requirements_file,
            preflight=plan.preflight,
            wandb_credential_label=self.settings.launch.wandb_credential_label,
            github_credential_label=self.settings.launch.github_credential_label,
            accelerator=plan.accelerator,
            dataset_sources=self.credentials_dataset_sources(plan.owner, plan.account),
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="ecc-launch-"))
        try:
            build_kernel_folder(runner_params, wheel_bytes, tmp_dir)
            launcher = self._launcher_factory(
                plan.account,
                lambda n=plan.account: self.secrets.get("kaggle", n),
            )
            push_acc = None if (plan.preflight or plan.accelerator.lower() == "none") else plan.accelerator
            push_res = launcher.push(
                tmp_dir,
                accelerator=push_acc,
                timeout_s=plan.timeout_s,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        effective_source = (source if source is not None else plan.source or "ui").strip() or "ui"
        row = self.store.record_launch(
            launch_id=plan.launch_id,
            created_at=now.isoformat(),
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
            ok=push_res.ok,
            version=push_res.version,
            url=push_res.url,
            message=redact(push_res.message),
            source=effective_source,
        )

        if push_res.ok:
            self.store.link_kernel(
                plan.kernel_ref,
                plan.account,
                plan.project or "unassigned",
            )

        return LaunchRecord.from_row(row)

    def list_launches(self, limit: int = 50) -> list[LaunchRecord]:
        return [LaunchRecord.from_row(r) for r in self.store.list_launches(limit=limit)]

    def get_launch(self, launch_id: str) -> LaunchRecord | None:
        row = self.store.get_launch(launch_id)
        return LaunchRecord.from_row(row) if row is not None else None


def build_launch_service(
    read_service: Any,
    *,
    github: GitHubReader | None = None,
) -> LaunchService:
    """Build a ``LaunchService`` sharing store, settings, and secrets with ``read_service``."""
    secrets = getattr(read_service, "_secrets", None)
    if secrets is None:
        secrets = SecretStore()
    return LaunchService(
        store=read_service.store,
        secrets=secrets,
        settings=read_service.settings,
        read_service=read_service,
        github=github if github is not None else GitHubReader(),
    )
