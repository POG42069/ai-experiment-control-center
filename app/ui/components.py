"""Reusable dark-mode UI components, formatting helpers, and sidebar chrome."""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import streamlit as st

from app.ui.state import (
    MAX_REFRESH_SECONDS,
    MIN_REFRESH_SECONDS,
    default_refresh_interval,
    in_main_entrypoint,
    init_ui_state,
)
from ecc.models import LogLine, RunStatus, RunSummary
from ecc.service import Service

BG_DARK = "#0E1117"
BG_SECONDARY = "#161B22"
PRIMARY = "#4F8BF9"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED = "#8B949E"
BORDER_SUBTLE = "#30363D"

MISSING = "—"

DEMO_BANNER_TEXT = "DEMO MODE — synthetic data, not real runs"

STATUS_COLORS: dict[RunStatus, str] = {
    RunStatus.RUNNING: "#3FB950",
    RunStatus.QUEUED: "#58A6FF",
    RunStatus.STALE: "#D29922",
    RunStatus.OFFLINE: "#DB6D28",
    RunStatus.FINISHED: "#8B949E",
    RunStatus.FAILED: "#F85149",
    RunStatus.CRASHED: "#F85149",
    RunStatus.KILLED: "#A371F7",
    RunStatus.UNKNOWN: "#6E7681",
}

PLATFORM_META: dict[str, tuple[str, str]] = {
    "kaggle": ("🔷", "Kaggle"),
    "colab": ("🟠", "Colab"),
    "wandb": ("🟡", "W&B"),
    "local": ("💻", "Local"),
    "other": ("⚙️", "Other"),
}


# ---------------------------------------------------------------------------
# Pure formatting helpers (missing values always render as "—", never 0)
# ---------------------------------------------------------------------------
def _is_valid_number(val: Any) -> bool:
    if val is None or isinstance(val, bool):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def fmt_float(val: Any, decimals: int = 4) -> str:
    if not _is_valid_number(val):
        return MISSING
    return f"{float(val):.{decimals}f}"


def fmt_lr(val: Any) -> str:
    if not _is_valid_number(val):
        return MISSING
    return f"{float(val):.2e}"


def fmt_int(val: Any) -> str:
    if not _is_valid_number(val):
        return MISSING
    return f"{int(round(float(val))):,}"


def fmt_step_progress(step: int | None, total_steps: int | None) -> str:
    if step is None and total_steps is None:
        return MISSING
    s_txt = fmt_int(step)
    t_txt = fmt_int(total_steps)
    return f"{s_txt} / {t_txt}"


def fmt_epoch_progress(epoch: float | None, total_epochs: float | None) -> str:
    if epoch is None and total_epochs is None:
        return MISSING
    e_txt = fmt_float(epoch, decimals=2)
    t_txt = fmt_float(total_epochs, decimals=1)
    return f"{e_txt} / {t_txt}"


def fmt_duration(seconds: float | None) -> str:
    if not _is_valid_number(seconds):
        return MISSING
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_relative_time(dt: datetime | None, now: datetime | None = None) -> str:
    if dt is None:
        return f"updated {MISSING}"
    ref_now = now or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    target = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    delta_s = int(max(0.0, (ref_now - target).total_seconds()))
    if delta_s < 60:
        return f"updated {delta_s}s ago"
    if delta_s < 3600:
        return f"updated {delta_s // 60}m ago"
    if delta_s < 86400:
        return f"updated {delta_s // 3600}h ago"
    return f"updated {delta_s // 86400}d ago"


def fmt_timestamp(dt: datetime | None) -> str:
    if dt is None:
        return MISSING
    target = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return target.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_platform(platform: str | None, account: str | None = None) -> str:
    if not platform and not account:
        return MISSING
    key = (platform or "").strip().lower()
    icon, label = PLATFORM_META.get(key, ("🔹", platform or MISSING))
    if account:
        return f"{icon} {label} · {account}"
    return f"{icon} {label}"


def fmt_text(val: Any) -> str:
    if val is None:
        return MISSING
    txt = str(val).strip()
    return txt if txt else MISSING


def status_color(status: RunStatus | str) -> str:
    if isinstance(status, RunStatus):
        return STATUS_COLORS.get(status, "#6E7681")
    try:
        return STATUS_COLORS.get(RunStatus(str(status).upper()), "#6E7681")
    except ValueError:
        return "#6E7681"


def status_badge_html(status: RunStatus | str) -> str:
    label = status.value if isinstance(status, RunStatus) else str(status).upper()
    color = status_color(status)
    safe_label = html.escape(label)
    return (
        f'<span class="ecc-badge" style="color:{color};'
        f'border-color:{color}66;background:{color}1A;">'
        f'<span class="ecc-badge-dot" style="background:{color};"></span>'
        f"{safe_label}</span>"
    )


# ---------------------------------------------------------------------------
# Global dark styling & layout chrome
# ---------------------------------------------------------------------------
def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        .ecc-mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }}
        .ecc-demo-banner {{
            background: rgba(210, 153, 34, 0.16);
            border: 1px solid #D29922;
            border-left: 4px solid #D29922;
            color: #F2CC60;
            padding: 0.55rem 0.9rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.88rem;
            margin-bottom: 0.9rem;
            letter-spacing: 0.02em;
        }}
        .ecc-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.14rem 0.55rem;
            border-radius: 999px;
            border: 1px solid;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            white-space: nowrap;
        }}
        .ecc-badge-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }}
        .ecc-card-title {{
            font-weight: 600;
            font-size: 0.98rem;
            color: {TEXT_PRIMARY};
            margin: 0.25rem 0 0.2rem 0;
            word-break: break-word;
        }}
        .ecc-card-meta {{
            font-size: 0.8rem;
            color: {TEXT_MUTED};
            margin-bottom: 0.45rem;
        }}
        .ecc-metric-grid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.35rem;
            margin: 0.45rem 0;
            padding: 0.45rem 0.55rem;
            background: {BG_DARK};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 6px;
        }}
        .ecc-metric-cell {{
            display: flex;
            flex-direction: column;
        }}
        .ecc-metric-label {{
            font-size: 0.68rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .ecc-metric-val {{
            font-size: 0.82rem;
            color: {TEXT_PRIMARY};
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-weight: 600;
        }}
        .ecc-card-footer {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem;
            font-size: 0.76rem;
            color: {TEXT_MUTED};
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            margin: 0.3rem 0 0.5rem 0;
        }}
        .ecc-kpi-box {{
            background: {BG_SECONDARY};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }}
        .ecc-kpi-label {{
            font-size: 0.74rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .ecc-kpi-value {{
            font-size: 1.55rem;
            font-weight: 700;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            margin-top: 0.15rem;
        }}
        .ecc-empty {{
            background: {BG_SECONDARY};
            border: 1px dashed {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 1.4rem 1.2rem;
            text-align: center;
            color: {TEXT_MUTED};
            margin: 0.6rem 0;
        }}
        .ecc-empty-title {{
            font-size: 0.98rem;
            font-weight: 600;
            color: {TEXT_PRIMARY};
            margin-bottom: 0.25rem;
        }}
        .ecc-health-row {{
            display: flex;
            align-items: flex-start;
            gap: 0.45rem;
            font-size: 0.78rem;
            margin: 0.3rem 0;
            color: {TEXT_PRIMARY};
        }}
        .ecc-health-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-top: 0.32rem;
            flex-shrink: 0;
        }}
        .ecc-log-box {{
            background: #0B0E14;
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 6px;
            padding: 0.65rem 0.8rem;
            max-height: 460px;
            overflow-y: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.78rem;
            line-height: 1.45;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .ecc-log-line {{
            padding: 0.08rem 0.25rem;
            border-radius: 3px;
        }}
        .ecc-log-error {{
            color: #FF7B72;
            background: rgba(248, 81, 73, 0.13);
        }}
        .ecc-log-warning {{
            color: #F2CC60;
            background: rgba(210, 153, 34, 0.11);
        }}
        .ecc-log-info {{
            color: {TEXT_PRIMARY};
        }}
        .ecc-log-meta {{
            color: {TEXT_MUTED};
        }}
        .ecc-ext-link {{
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            border: 1px solid {BORDER_SUBTLE};
            background: {BG_DARK};
            color: {PRIMARY} !important;
            text-decoration: none !important;
            font-size: 0.82rem;
            font-weight: 500;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
        }}
        .ecc-ext-link:hover {{
            border-color: {PRIMARY};
            background: rgba(79, 139, 249, 0.1);
        }}
        .ecc-ext-disabled {{
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            border: 1px solid {BORDER_SUBTLE};
            background: {BG_DARK};
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_demo_banner(runs: Iterable[RunSummary] | None = None, *, force: bool = False) -> bool:
    """Render persistent amber DEMO MODE banner whenever any displayed run is synthetic."""
    has_demo = force or (runs is not None and any(r.is_demo for r in runs))
    if has_demo:
        st.markdown(
            f'<div class="ecc-demo-banner">⚠️ {html.escape(DEMO_BANNER_TEXT)}</div>',
            unsafe_allow_html=True,
        )
    return has_demo


def render_kpi_card(label: str, value: Any, accent: str = TEXT_PRIMARY) -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    st.markdown(
        f"""
        <div class="ecc-kpi-box">
            <div class="ecc-kpi-label">{safe_label}</div>
            <div class="ecc-kpi-value" style="color:{accent};">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str, hint: str | None = None) -> None:
    hint_html = (
        f'<div style="margin-top:0.4rem;font-size:0.78rem;color:{PRIMARY};">'
        f"{html.escape(hint)}</div>"
        if hint
        else ""
    )
    st.markdown(
        f"""
        <div class="ecc-empty">
            <div class="ecc-empty-title">{html.escape(title)}</div>
            <div style="font-size:0.84rem;">{html.escape(description)}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(service: Service) -> None:
    """Render shared sidebar: project selector, auto-refresh controls, source health."""
    init_ui_state(service)
    projects = service.list_projects()
    project_options = ["All", *[p.slug for p in projects]]

    if st.session_state.get("ecc_selected_project") not in project_options:
        st.session_state["ecc_selected_project"] = "All"

    with st.sidebar:
        st.markdown("### 🧪 AI Experiment Control Center")
        st.selectbox(
            "Project",
            options=project_options,
            key="ecc_selected_project",
            help="Filter Overview and defaults by project.",
        )

        st.divider()
        st.markdown("**Auto-refresh**")
        st.toggle("Enable auto-refresh", key="ecc_auto_refresh")
        st.slider(
            "Interval (seconds)",
            min_value=MIN_REFRESH_SECONDS,
            max_value=MAX_REFRESH_SECONDS,
            step=5,
            key="ecc_refresh_interval",
            disabled=not st.session_state.get("ecc_auto_refresh", True),
            help=f"Config default: {default_refresh_interval(service)}s",
        )
        last_ref = st.session_state.get("ecc_last_refreshed", MISSING)
        st.markdown(
            f'<div class="ecc-mono" style="font-size:0.76rem;color:{TEXT_MUTED};">'
            f"Last refreshed at {html.escape(str(last_ref))}</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**Source health**")
        health = service.health()
        sources = health.get("sources", {})
        if not sources:
            st.caption("No sources configured.")
        else:
            for src_name, info in sources.items():
                ok = bool(info.get("ok", False))
                msg = str(info.get("message", ""))
                dot_color = "#3FB950" if ok else "#F85149"
                st.markdown(
                    f"""
                    <div class="ecc-health-row">
                        <span class="ecc-health-dot" style="background:{dot_color};"></span>
                        <div>
                            <strong class="ecc-mono">{html.escape(str(src_name))}</strong>
                            <div style="color:{TEXT_MUTED};font-size:0.73rem;">{html.escape(msg)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def ensure_standalone_chrome(service: Service) -> None:
    """When a page file is executed directly (e.g. AppTest), configure page + sidebar."""
    if in_main_entrypoint():
        return
    try:
        st.set_page_config(
            page_title="AI Experiment Control Center",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass
    inject_global_css()
    render_sidebar(service)


def render_run_card(
    run: RunSummary,
    *,
    on_details: Callable[[str], None] | None = None,
    key_prefix: str = "card",
) -> None:
    """Render a compact dark run summary card for an active run."""
    with st.container(border=True):
        badge_html = status_badge_html(run.status)
        kaggle_st = (run.config or {}).get("kaggle_status")
        extra_badge = ""
        if kaggle_st:
            extra_badge = (
                f' <span class="ecc-mono" style="font-size:0.72rem;color:{TEXT_MUTED};">'
                f"Kaggle: {html.escape(str(kaggle_st))}</span>"
            )

        st.markdown(
            f"<div>{badge_html}{extra_badge}</div>"
            f'<div class="ecc-card-title">{html.escape(run.name)}</div>'
            f'<div class="ecc-card-meta">'
            f"<strong>{html.escape(fmt_text(run.model_name))}</strong> · "
            f"{html.escape(fmt_platform(run.platform, run.account))} · "
            f'<span class="ecc-mono">{html.escape(fmt_text(run.gpu))}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        step_str = fmt_step_progress(run.step, run.total_steps)
        epoch_str = fmt_epoch_progress(run.epoch, run.total_epochs)
        pct_str = (
            f"{run.progress * 100.0:.1f}%"
            if _is_valid_number(run.progress)
            else MISSING
        )
        st.markdown(
            f'<div class="ecc-mono" style="display:flex;justify-content:space-between;'
            f'font-size:0.76rem;color:{TEXT_MUTED};margin-bottom:0.2rem;">'
            f"<span>Step: {html.escape(step_str)}</span>"
            f"<span>Epoch: {html.escape(epoch_str)}</span>"
            f"<span>{html.escape(pct_str)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if _is_valid_number(run.progress):
            st.progress(max(0.0, min(1.0, float(run.progress))))
        else:
            st.progress(0.0)

        latest = run.latest or {}
        t_loss = fmt_float(latest.get("train/loss"), 4)
        e_loss = fmt_float(latest.get("eval/loss"), 4)
        e_acc = fmt_float(latest.get("eval/accuracy"), 4)
        e_f1 = fmt_float(latest.get("eval/macro_f1"), 4)
        lr_val = fmt_lr(latest.get("train/lr"))

        st.markdown(
            f"""
            <div class="ecc-metric-grid">
                <div class="ecc-metric-cell">
                    <span class="ecc-metric-label">train/loss</span>
                    <span class="ecc-metric-val">{html.escape(t_loss)}</span>
                </div>
                <div class="ecc-metric-cell">
                    <span class="ecc-metric-label">eval/loss</span>
                    <span class="ecc-metric-val">{html.escape(e_loss)}</span>
                </div>
                <div class="ecc-metric-cell">
                    <span class="ecc-metric-label">eval/acc</span>
                    <span class="ecc-metric-val">{html.escape(e_acc)}</span>
                </div>
                <div class="ecc-metric-cell">
                    <span class="ecc-metric-label">eval/f1</span>
                    <span class="ecc-metric-val">{html.escape(e_f1)}</span>
                </div>
                <div class="ecc-metric-cell">
                    <span class="ecc-metric-label">lr</span>
                    <span class="ecc-metric-val">{html.escape(lr_val)}</span>
                </div>
            </div>
            <div class="ecc-card-footer">
                <span>⏱ {html.escape(fmt_duration(run.duration_s))}</span>
                <span>ETA: {html.escape(fmt_duration(run.eta_s))}</span>
                <span>{html.escape(fmt_relative_time(run.last_update))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_det, col_nb, col_wb = st.columns([1.1, 1.0, 1.0])
        with col_det:
            if st.button(
                "Details",
                key=f"{key_prefix}_details_{run.run_id}",
                width="stretch",
            ):
                if on_details is not None:
                    on_details(run.run_id)
        with col_nb:
            if run.notebook_url:
                st.markdown(
                    f'<a class="ecc-ext-link" href="{html.escape(run.notebook_url)}" '
                    f'target="_blank" rel="noopener noreferrer">Notebook ↗</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="ecc-ext-disabled">Notebook {MISSING}</span>',
                    unsafe_allow_html=True,
                )
        with col_wb:
            if run.source_url:
                st.markdown(
                    f'<a class="ecc-ext-link" href="{html.escape(run.source_url)}" '
                    f'target="_blank" rel="noopener noreferrer">W&amp;B ↗</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="ecc-ext-disabled">W&amp;B {MISSING}</span>',
                    unsafe_allow_html=True,
                )


def render_log_viewer(lines: list[LogLine], *, newest_first: bool = True) -> None:
    """Render formatted, color-coded console log lines in a fixed-height scrollable container."""
    ordered = list(reversed(lines)) if newest_first else list(lines)
    html_rows: list[str] = []
    for line in ordered:
        lvl = (line.level or "INFO").upper()
        if lvl == "ERROR":
            cls = "ecc-log-error"
        elif lvl == "WARNING":
            cls = "ecc-log-warning"
        else:
            cls = "ecc-log-info"
        ts_str = (
            line.ts.astimezone(timezone.utc).strftime("%H:%M:%S")
            if line.ts is not None
            else "--:--:--"
        )
        prefix = html.escape(f"{ts_str} [{line.stream}] [{lvl}] ")
        body = html.escape(line.text)
        html_rows.append(
            f'<div class="ecc-log-line {cls}">'
            f'<span class="ecc-log-meta">{prefix}</span>{body}</div>'
        )
    st.markdown(
        f'<div class="ecc-log-box">{"".join(html_rows)}</div>',
        unsafe_allow_html=True,
    )
