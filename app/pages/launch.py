"""Launch page — human-confirmed Kaggle training & preflight launches + audit history."""

from __future__ import annotations

import html
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from app.ui import state as ui_state
from app.ui.components import (
    MISSING,
    TEXT_MUTED,
    ensure_standalone_chrome,
    fmt_timestamp,
    render_empty_state,
)
from ecc.launch.github import BranchInfo, CommitInfo, GitHubError, RepoInfo
from ecc.launch.service import (
    BUSY_STATUSES,
    LaunchPlan,
    LaunchRecord,
    LaunchRefused,
    LaunchRequest,
    LaunchService,
)

CUSTOM_REPO_OPTION = "Other (enter owner/repo)..."

PENDING_PLAN_KEY = "launch_pending_plan"
CONFIRM_CHECKBOX_KEY = "launch_confirm_checkbox"
LAST_RESULT_KEY = "launch_last_result"
RUNNER_STATUS_CACHE_KEY = "launch_runner_status_cache"
AUTO_RUN_NAME_KEY = "launch_auto_run_name"

ACCELERATOR_LABELS: dict[str, str] = {
    "NvidiaTeslaT4": "GPU T4 ×2",
    "TpuV5E8": "TPU v5e-8",
    "none": "CPU only",
}

MONITOR_MODE_LABELS: dict[str, str] = {
    "runner": "runner — dashboard captures logs even if the script is not instrumented",
    "script": "script — the script calls experiment_monitor itself",
}

RUNNER_STATUS_COLORS: dict[str, str] = {
    "RUNNING": "#3FB950",
    "QUEUED": "#58A6FF",
    "COMPLETE": "#8B949E",
    "FINISHED": "#8B949E",
    "ERROR": "#F85149",
    "FAILED": "#F85149",
    "CANCELLED": "#A371F7",
    "NOT_CREATED": "#6E7681",
    "UNKNOWN": "#D29922",
}


def _format_accelerator(acc: str) -> str:
    return ACCELERATOR_LABELS.get(acc, acc)


def _runner_status_badge_html(status: str) -> str:
    label = (status or "UNKNOWN").strip().upper()
    color = RUNNER_STATUS_COLORS.get(label, "#6E7681")
    safe_label = html.escape(label)
    return (
        f'<span class="ecc-badge" style="color:{color};'
        f'border-color:{color}66;background:{color}1A;">'
        f'<span class="ecc-badge-dot" style="background:{color};"></span>'
        f"{safe_label}</span>"
    )


def _truncate_command(cmd: str, max_len: int = 60) -> str:
    cleaned = (cmd or "").strip()
    if not cleaned:
        return MISSING
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _format_record_time(raw_iso: str) -> str:
    txt = (raw_iso or "").strip()
    if not txt:
        return MISSING
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return fmt_timestamp(dt)
    except ValueError:
        return txt


def _default_run_name(repo: str, branch: str, short_sha: str) -> str:
    repo_part = repo.split("/")[-1].strip() if repo and "/" in repo else (repo or "run").strip()
    branch_part = (branch or "main").strip().replace("/", "-")
    sha_part = (short_sha or "head").strip()[:7]
    return f"{repo_part}-{branch_part}-{sha_part}"


def _list_kaggle_accounts(launch_service: LaunchService, read_service: Any) -> list[dict[str, Any]]:
    store = getattr(launch_service, "store", None) or getattr(read_service, "store", None)
    if store is None or not hasattr(store, "list_accounts"):
        return []
    return [
        acc
        for acc in store.list_accounts()
        if acc.get("platform") == "kaggle" and acc.get("enabled", True)
    ]


def _render_disabled_panel() -> None:
    st.info(
        "**Launch control is currently disabled.**\n\n"
        "- **How to enable:** add `[launch]` with `enabled = true` to `ecc_config.toml` "
        "and restart the dashboard.\n"
        "- **What it does:** pushes a fixed Kaggle runner kernel (`<owner>/ecc-runner`) "
        "to clone a pinned GitHub commit and execute your training command on the chosen Kaggle account.\n"
        "- **Quota & safety:** GPU/TPU launches consume GPU quota on the chosen Kaggle account "
        "(CPU preflight does not). **Claude never launches runs** — only an explicit human "
        "confirmation in this UI can push a launch."
    )


def _render_first_time_help(launch_service: LaunchService) -> None:
    launch_cfg = launch_service.settings.launch
    wandb_label = launch_cfg.wandb_credential_label or "WANDB_API_KEY"
    github_label = launch_cfg.github_credential_label or "GITHUB_TOKEN"
    runner_slug = launch_cfg.runner_slug or "ecc-runner"

    with st.expander("First-time setup & Kaggle runner help", expanded=False):
        st.markdown(
            f"""
1. **Attach secrets per Kaggle account** (once per `<owner>/{runner_slug}` kernel):
   - Run **Preflight (CPU, no GPU quota)** once so Kaggle creates `<owner>/{runner_slug}` (or open your existing runner kernel page).
   - Open the kernel page on Kaggle → **Add-ons → Secrets** and attach:
     - `{wandb_label}` — required to stream training metrics and console logs to W&B.
     - `{github_label}` — only needed for **private GitHub repositories**; use a fine-grained read-only token with **"Contents: Read"** permission only.
   - Run **Preflight (CPU, no GPU quota)** again to verify secrets, git clone, wheel install, and `torch` import without spending GPU quota.
2. **Phone verification required:** Each Kaggle account must be **phone-verified** on Kaggle before it can use GPU/TPU accelerators or Internet access.
3. **Stopping a running job:** Kaggle provides no official API to cancel a running kernel. To stop a run, open the runner kernel page on Kaggle and stop/cancel it directly there.
            """
        )


def _render_last_result() -> None:
    last = st.session_state.get(LAST_RESULT_KEY)
    if not isinstance(last, dict):
        return
    kind = last.get("kind")
    if kind == "success":
        version = last.get("version")
        version_txt = f"v{version}" if version is not None else "pushed"
        kernel_ref = last.get("kernel_ref") or ""
        account = last.get("account") or ""
        kernel_url = last.get("url") or ""
        parts = [f"Launched `{kernel_ref}` ({version_txt}) on `{account}`."]
        if kernel_url:
            parts.append(f"[Open kernel on Kaggle ↗]({kernel_url})")
        parts.append("[View on Overview](/overview)")
        st.success(" · ".join(parts))
        if ui_state.in_main_entrypoint():
            st.page_link("pages/overview.py", label="View on Overview", icon="📊")
    elif kind == "error":
        st.error(str(last.get("message") or "Launch failed."))
    elif kind == "warning":
        st.warning(str(last.get("message") or "Launch refused."))


def _render_confirmation_panel(launch_service: LaunchService) -> None:
    plan = st.session_state.get(PENDING_PLAN_KEY)
    if not isinstance(plan, LaunchPlan):
        return

    with st.container(border=True):
        title = "⚠️ Confirm Preflight Launch" if plan.preflight else "⚠️ Confirm Training Launch"
        st.markdown(f"### {title}")

        quota_str = "YES" if plan.consumes_gpu_quota else "NO"
        quota_detail = (
            "YES — consumes GPU/TPU quota on this Kaggle account"
            if plan.consumes_gpu_quota
            else "NO — CPU only (no GPU quota)"
        )
        mode_label = "preflight" if plan.preflight else plan.monitor_mode
        timeout_h = plan.timeout_s / 3600.0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"- **Account:** `{plan.account}` (owner: `{plan.owner}`)\n"
                f"- **Runner kernel:** [`{plan.kernel_ref}`]({plan.kernel_url})\n"
                f"- **Repository:** `{plan.repo}` @ `{plan.branch}`\n"
                f"- **Commit SHA:** `{plan.short_sha}` (`{plan.commit_sha}`)\n"
                f"- **Commit message:** {plan.commit_message or MISSING}\n"
                f"- **Command:** `{plan.command}`"
            )
        with c2:
            st.markdown(
                f"- **Accelerator:** `{_format_accelerator(plan.accelerator)}` (`{plan.accelerator}`)\n"
                f"- **Consumes GPU quota:** **{quota_str}** ({quota_detail})\n"
                f"- **Timeout:** `{timeout_h:g}h` ({plan.timeout_s}s)\n"
                f"- **Monitor mode:** `{mode_label}`\n"
                f"- **W&B project:** `{plan.project or '(none)'}`\n"
                f"- **Run name:** `{plan.run_name or '(auto)'}`\n"
                f"- **Requirements file:** `{plan.requirements_file or '(none)'}`"
            )

        if plan.summary:
            st.code(plan.summary, language="text")

        confirmed = st.checkbox(
            f"I confirm launching this on {plan.account} ({plan.owner})",
            key=CONFIRM_CHECKBOX_KEY,
        )

        btn_confirm, btn_cancel = st.columns([1.4, 1.0])
        with btn_confirm:
            push_clicked = st.button(
                "Confirm & push",
                type="primary",
                disabled=not confirmed,
                key="launch_confirm_push_btn",
                width="stretch",
            )
        with btn_cancel:
            cancel_clicked = st.button(
                "Cancel",
                key="launch_cancel_plan_btn",
                width="stretch",
            )

        if cancel_clicked:
            st.session_state.pop(PENDING_PLAN_KEY, None)
            st.rerun()

        if push_clicked:
            st.session_state.pop(PENDING_PLAN_KEY, None)
            status_cache = st.session_state.get(RUNNER_STATUS_CACHE_KEY)
            if isinstance(status_cache, dict):
                status_cache.pop(plan.account, None)
            try:
                record = launch_service.execute(plan, confirmed_by_user=True)
            except (LaunchRefused, ValueError) as exc:
                st.session_state[LAST_RESULT_KEY] = {
                    "kind": "warning",
                    "message": str(exc),
                }
            else:
                if record.ok:
                    st.session_state[LAST_RESULT_KEY] = {
                        "kind": "success",
                        "account": record.account,
                        "kernel_ref": record.kernel_ref,
                        "version": record.version,
                        "url": record.url or plan.kernel_url,
                        "message": record.message,
                    }
                else:
                    st.session_state[LAST_RESULT_KEY] = {
                        "kind": "error",
                        "message": record.message or "Kaggle kernel push failed.",
                    }
            st.rerun()


def _render_history_section(launch_service: LaunchService) -> None:
    st.divider()
    st.markdown("### Launch History")
    try:
        records: list[LaunchRecord] = launch_service.list_launches(limit=50)
    except Exception as exc:
        st.error(f"Failed to load launch history: {exc}")
        return

    if not records:
        render_empty_state(
            "No launches yet",
            "Preflight checks and training launches pushed from this dashboard will appear here.",
        )
        return

    rows: list[dict[str, Any]] = []
    for rec in records:
        short_sha = rec.commit_sha[:7] if rec.commit_sha else MISSING
        kernel_link = rec.url or (
            f"https://www.kaggle.com/code/{rec.kernel_ref}" if rec.kernel_ref else None
        )
        rows.append(
            {
                "Time": _format_record_time(rec.created_at),
                "Account": rec.account,
                "Repo@Branch": f"{rec.repo}@{rec.branch}",
                "Short SHA": short_sha,
                "Command": _truncate_command(rec.command),
                "Accelerator": _format_accelerator(rec.accelerator),
                "Preflight": "Yes" if rec.preflight else "No",
                "OK": "Yes" if rec.ok else "No",
                "Version": f"v{rec.version}" if rec.version is not None else MISSING,
                "Link": kernel_link,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Link", display_text="Open ↗"),
        },
    )


def render_page() -> None:
    service = ui_state.get_service()
    ensure_standalone_chrome(service)
    launch_service = ui_state.get_launch_service()
    ui_state.mark_refreshed()

    st.markdown("## 🚀 Launch Training")

    if not launch_service.is_enabled():
        _render_disabled_panel()
        return

    _render_last_result()

    # -----------------------------------------------------------------------
    # Step 1: Account & runner status
    # -----------------------------------------------------------------------
    st.markdown("### Step 1 · Kaggle Account")
    kaggle_accounts = _list_kaggle_accounts(launch_service, service)

    if not kaggle_accounts:
        render_empty_state(
            "No Kaggle accounts configured",
            "Add a Kaggle account on the Connected Accounts page (or via scripts\\add_account.py) before launching.",
        )
        _render_first_time_help(launch_service)
        _render_history_section(launch_service)
        return

    account_nicknames = [str(acc["nickname"]) for acc in kaggle_accounts]
    owners_by_nick: dict[str, str | None] = {}
    for nick in account_nicknames:
        try:
            owners_by_nick[nick] = launch_service.account_owner(nick)
        except Exception:
            owners_by_nick[nick] = None

    def _fmt_account(nick: str) -> str:
        owner = owners_by_nick.get(nick)
        return f"{nick} ({owner})" if owner else f"{nick} (owner unknown)"

    acc_col, ref_col, refresh_col = st.columns([2.0, 2.2, 1.0])
    with acc_col:
        selected_account = st.selectbox(
            "Kaggle account",
            options=account_nicknames,
            format_func=_fmt_account,
            key="launch_account_select",
        )

    status_cache: dict[str, str] = st.session_state.setdefault(RUNNER_STATUS_CACHE_KEY, {})

    with refresh_col:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button(
            "Refresh status",
            key="launch_refresh_status_btn",
            width="stretch",
        )
        if refresh_clicked:
            status_cache.pop(selected_account, None)

    kernel_ref: str | None = None
    kernel_url: str | None = None
    owner_missing = False
    runner_status_str = "UNKNOWN"
    status_warning: str | None = None

    try:
        kernel_ref = launch_service.runner_ref(selected_account)
        kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
    except LaunchRefused as exc:
        owner_missing = True
        status_warning = str(exc)

    if not owner_missing:
        if selected_account not in status_cache:
            try:
                status_cache[selected_account] = launch_service.runner_status(selected_account)
            except (LaunchRefused, Exception) as exc:
                status_cache[selected_account] = "ERROR"
                status_warning = str(exc)
        runner_status_str = status_cache.get(selected_account, "UNKNOWN")

    with ref_col:
        badge_html = _runner_status_badge_html(runner_status_str)
        if kernel_ref and kernel_url:
            st.markdown(
                f'<div style="font-size:0.82rem;color:{TEXT_MUTED};margin-bottom:0.2rem;">Runner kernel</div>'
                f'<div><a href="{html.escape(kernel_url)}" target="_blank" rel="noopener noreferrer" '
                f'class="ecc-mono" style="margin-right:0.5rem;">{html.escape(kernel_ref)} ↗</a>'
                f"{badge_html}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-size:0.82rem;color:{TEXT_MUTED};margin-bottom:0.2rem;">Runner kernel</div>'
                f"<div>{MISSING} {badge_html}</div>",
                unsafe_allow_html=True,
            )

    if status_warning:
        st.warning(status_warning)

    is_busy = runner_status_str.strip().upper() in BUSY_STATUSES
    if is_busy:
        link_md = f" — [Open kernel on Kaggle ↗]({kernel_url})" if kernel_url else ""
        st.warning(
            f"**Account busy** ({runner_status_str}) — account busy — wait or stop it on Kaggle{link_md}"
        )

    # -----------------------------------------------------------------------
    # Step 2: Code (GitHub repo, branch, commit)
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### Step 2 · GitHub Code")
    github = launch_service.github

    repos: list[RepoInfo] = []
    try:
        repos = github.list_repos()
    except GitHubError as exc:
        st.error(
            f"GitHub error listing repositories: {exc}\n\n"
            "Hint: run `gh auth status` in your terminal to verify GitHub CLI authentication."
        )
    except Exception as exc:
        st.error(f"Unexpected error listing repositories: {exc}\n\nHint: run `gh auth status`.")

    repo_by_name: dict[str, RepoInfo] = {r.name_with_owner: r for r in repos}
    selected_repo = ""

    code_c1, code_c2 = st.columns([1.4, 1.0])
    with code_c1:
        if repos:
            repo_options = [r.name_with_owner for r in repos] + [CUSTOM_REPO_OPTION]
            picked_repo = st.selectbox(
                "Repository",
                options=repo_options,
                key="launch_repo_select",
            )
            if picked_repo == CUSTOM_REPO_OPTION:
                selected_repo = st.text_input(
                    "Repository (owner/repo)",
                    placeholder="owner/repo",
                    key="launch_repo_manual",
                ).strip()
            else:
                selected_repo = picked_repo
        else:
            selected_repo = st.text_input(
                "Repository (owner/repo)",
                placeholder="owner/repo",
                key="launch_repo_manual",
            ).strip()

    branches: list[BranchInfo] = []
    if selected_repo:
        try:
            branches = github.list_branches(selected_repo)
        except GitHubError as exc:
            st.error(
                f"GitHub error loading branches for `{selected_repo}`: {exc}\n\n"
                "Hint: run `gh auth status` in your terminal."
            )
        except ValueError as exc:
            st.warning(str(exc))

    selected_branch = ""
    with code_c2:
        if branches:
            branch_names = [b.name for b in branches]
            if st.session_state.get("launch_branch_select") not in branch_names:
                st.session_state.pop("launch_branch_select", None)
            repo_meta = repo_by_name.get(selected_repo)
            default_branch_idx = (
                branch_names.index(repo_meta.default_branch)
                if repo_meta and repo_meta.default_branch in branch_names
                else 0
            )
            selected_branch = st.selectbox(
                "Branch",
                options=branch_names,
                index=default_branch_idx,
                key="launch_branch_select",
            )
        else:
            selected_branch = st.text_input(
                "Branch",
                value="main",
                placeholder="main",
                key="launch_branch_manual",
            ).strip()

    commits: list[CommitInfo] = []
    if selected_repo and selected_branch:
        try:
            commits = github.list_commits(selected_repo, selected_branch, n=15)
        except GitHubError as exc:
            st.error(
                f"GitHub error loading commits for `{selected_repo}@{selected_branch}`: {exc}\n\n"
                "Hint: run `gh auth status` in your terminal."
            )
        except ValueError as exc:
            st.warning(str(exc))

    selected_commit_sha = ""
    selected_short_sha = "head"
    if commits:
        commit_by_sha = {c.sha: c for c in commits}
        commit_shas = [c.sha for c in commits]
        if st.session_state.get("launch_commit_select") not in commit_shas:
            st.session_state.pop("launch_commit_select", None)

        def _fmt_commit(sha: str) -> str:
            c = commit_by_sha[sha]
            msg = c.message_first_line or "(no message)"
            dt = c.date or MISSING
            return f"{c.short_sha} — {msg} — {dt}"

        selected_commit_sha = st.selectbox(
            "Commit (latest 15; default = HEAD)",
            options=commit_shas,
            index=0,
            format_func=_fmt_commit,
            key="launch_commit_select",
        )
        selected_short_sha = commit_by_sha[selected_commit_sha].short_sha
    else:
        selected_commit_sha = st.text_input(
            "Commit SHA (optional, defaults to branch HEAD)",
            value="",
            placeholder="Leave blank to resolve branch HEAD",
            key="launch_commit_manual",
        ).strip()
        if selected_commit_sha:
            selected_short_sha = selected_commit_sha[:7]

    # -----------------------------------------------------------------------
    # Step 3: Run configuration
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### Step 3 · Run Configuration")

    launch_cfg = launch_service.settings.launch

    command = st.text_input(
        "Training command",
        placeholder="python train.py --config configs/qwen3vl_r16.yaml",
        key="launch_command_input",
    )

    req_c1, req_c2 = st.columns([1.0, 2.0])
    with req_c1:
        use_req_file = st.checkbox(
            "Install requirements file",
            value=False,
            key="launch_use_requirements",
        )
    with req_c2:
        req_file_raw = st.text_input(
            "Requirements file",
            value="requirements.txt",
            disabled=not use_req_file,
            key="launch_requirements_file",
        )
    requirements_file = req_file_raw.strip() if use_req_file else ""

    monitor_mode = st.radio(
        "Monitor mode",
        options=["runner", "script"],
        format_func=lambda m: MONITOR_MODE_LABELS.get(m, m),
        key="launch_monitor_mode",
    )

    auto_run_name = _default_run_name(selected_repo, selected_branch, selected_short_sha)
    prev_auto = st.session_state.get(AUTO_RUN_NAME_KEY)
    if (
        "launch_run_name" not in st.session_state
        or st.session_state.get("launch_run_name") == prev_auto
    ):
        st.session_state["launch_run_name"] = auto_run_name
    st.session_state[AUTO_RUN_NAME_KEY] = auto_run_name

    default_project = (
        launch_cfg.default_wandb_project or ui_state.get_selected_project() or ""
    )

    run_c1, run_c2 = st.columns(2)
    with run_c1:
        wandb_project = st.text_input(
            "W&B project",
            value=default_project,
            placeholder="e.g. shared-task-4",
            key="launch_wandb_project",
        )
    with run_c2:
        run_name = st.text_input(
            "Run name",
            key="launch_run_name",
        )

    allowed_accs = list(launch_cfg.allowed_accelerators) or ["NvidiaTeslaT4", "TpuV5E8", "none"]
    default_acc = launch_cfg.default_accelerator
    default_acc_idx = allowed_accs.index(default_acc) if default_acc in allowed_accs else 0
    max_hours = max(1, int(launch_cfg.max_timeout_hours))

    hw_c1, hw_c2 = st.columns(2)
    with hw_c1:
        accelerator = st.selectbox(
            "Accelerator",
            options=allowed_accs,
            index=default_acc_idx,
            format_func=_format_accelerator,
            key="launch_accelerator_select",
        )
    with hw_c2:
        timeout_hours = st.number_input(
            "Timeout (hours)",
            min_value=1,
            max_value=max_hours,
            value=1,
            step=1,
            key="launch_timeout_hours",
        )

    # -----------------------------------------------------------------------
    # Actions: Preflight & Launch training
    # -----------------------------------------------------------------------
    launch_disabled = is_busy or owner_missing or not selected_account
    act_c1, act_c2 = st.columns(2)
    with act_c1:
        preflight_clicked = st.button(
            "🧪 Preflight (CPU, no GPU quota)",
            disabled=launch_disabled,
            key="launch_preflight_btn",
            width="stretch",
        )
    with act_c2:
        launch_clicked = st.button(
            "🚀 Launch training",
            type="primary",
            disabled=launch_disabled,
            key="launch_train_btn",
            width="stretch",
        )

    if preflight_clicked or launch_clicked:
        is_preflight = bool(preflight_clicked)
        req = LaunchRequest(
            account=selected_account,
            repo=selected_repo,
            branch=selected_branch,
            commit_sha=selected_commit_sha or None,
            command=command,
            accelerator="none" if is_preflight else accelerator,
            timeout_hours=float(timeout_hours),
            monitor_mode=monitor_mode,
            project=wandb_project,
            run_name=run_name,
            requirements_file=requirements_file,
            preflight=is_preflight,
        )
        st.session_state.pop(LAST_RESULT_KEY, None)
        try:
            plan = launch_service.plan_launch(req)
        except (LaunchRefused, ValueError) as exc:
            st.session_state.pop(PENDING_PLAN_KEY, None)
            st.warning(str(exc))
        else:
            st.session_state[PENDING_PLAN_KEY] = plan
            st.session_state[CONFIRM_CHECKBOX_KEY] = False

    _render_confirmation_panel(launch_service)
    _render_first_time_help(launch_service)
    _render_history_section(launch_service)


render_page()
