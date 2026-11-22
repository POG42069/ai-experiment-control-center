"""Overview page — KPIs, active run cards, and filterable/selectable runs table."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from app.ui.components import (
    MISSING,
    STATUS_COLORS,
    ensure_standalone_chrome,
    fmt_duration,
    fmt_epoch_progress,
    fmt_lr,
    fmt_platform,
    fmt_relative_time,
    fmt_step_progress,
    fmt_text,
    render_demo_banner,
    render_empty_state,
    render_kpi_card,
    render_run_card,
)
from app.ui.state import (
    get_refresh_settings,
    get_selected_project,
    get_service,
    mark_refreshed,
    navigate_to_compare,
    navigate_to_run_detail,
)
from ecc.models import RunStatus, RunSummary
from ecc.service import Service

CARD_ACTIVE_STATUSES = {
    RunStatus.QUEUED,
    RunStatus.RUNNING,
    RunStatus.STALE,
    RunStatus.OFFLINE,
}


def _is_within_hours(dt: datetime | None, now: datetime, hours: float) -> bool:
    if dt is None:
        return False
    target = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return (now - target).total_seconds() <= hours * 3600.0


def _build_runs_table(runs: list[RunSummary]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in runs:
        latest = r.latest or {}
        rows.append(
            {
                "Status": r.status.value,
                "Name": r.name,
                "Project": r.project,
                "Model": fmt_text(r.model_name),
                "Platform / Account": fmt_platform(r.platform, r.account),
                "GPU": fmt_text(r.gpu),
                "Step": fmt_step_progress(r.step, r.total_steps),
                "Epoch": fmt_epoch_progress(r.epoch, r.total_epochs),
                "Progress": round(float(r.progress) * 100.0, 1)
                if r.progress is not None
                else None,
                "train/loss": latest.get("train/loss"),
                "eval/loss": latest.get("eval/loss"),
                "eval/accuracy": latest.get("eval/accuracy"),
                "eval/macro_f1": latest.get("eval/macro_f1"),
                "LR": fmt_lr(latest.get("train/lr")),
                "Duration": fmt_duration(r.duration_s),
                "Updated": fmt_relative_time(r.last_update),
                "Tags": ", ".join(r.tags) if r.tags else MISSING,
            }
        )
    return pd.DataFrame(rows)


def _render_data_section(
    service: Service,
    selected_project: str | None,
    status_filter: list[str],
    search_query: str,
) -> None:
    runs = service.list_runs(project=selected_project)
    mark_refreshed()
    render_demo_banner(runs)

    now = datetime.now(timezone.utc)
    running_cnt = sum(1 for r in runs if r.status == RunStatus.RUNNING)
    stale_offline_cnt = sum(
        1 for r in runs if r.status in {RunStatus.STALE, RunStatus.OFFLINE}
    )
    queued_cnt = sum(1 for r in runs if r.status == RunStatus.QUEUED)
    finished_24h_cnt = sum(
        1
        for r in runs
        if r.status == RunStatus.FINISHED
        and _is_within_hours(r.ended_at or r.last_update or r.started_at, now, 24.0)
    )
    failed_24h_cnt = len(
        service.get_failed_runs(project=selected_project, since_hours=24.0)
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Running", running_cnt, STATUS_COLORS[RunStatus.RUNNING])
    with k2:
        render_kpi_card(
            "Stale / Offline", stale_offline_cnt, STATUS_COLORS[RunStatus.STALE]
        )
    with k3:
        render_kpi_card("Queued", queued_cnt, STATUS_COLORS[RunStatus.QUEUED])
    with k4:
        render_kpi_card(
            "Finished (24h)", finished_24h_cnt, STATUS_COLORS[RunStatus.FINISHED]
        )
    with k5:
        render_kpi_card(
            "Failed (24h)", failed_24h_cnt, STATUS_COLORS[RunStatus.FAILED]
        )

    st.markdown("### Active Runs")
    active_runs = [r for r in runs if r.status in CARD_ACTIVE_STATUSES]
    if not active_runs:
        render_empty_state(
            "No active runs",
            "There are currently no QUEUED, RUNNING, STALE, or OFFLINE runs matching the selected project.",
            hint="Switch project filter to 'All' or start a monitored training run.",
        )
    else:
        for idx in range(0, len(active_runs), 2):
            cols = st.columns(2)
            for offset in range(2):
                if idx + offset < len(active_runs):
                    with cols[offset]:
                        render_run_card(
                            active_runs[idx + offset],
                            on_details=navigate_to_run_detail,
                            key_prefix="overview_card",
                        )

    st.markdown("### All Runs")
    filtered_runs = runs
    if status_filter:
        allowed = set(status_filter)
        filtered_runs = [r for r in filtered_runs if r.status.value in allowed]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_runs = [
            r
            for r in filtered_runs
            if q in r.name.lower()
            or q in r.run_id.lower()
            or (r.model_name and q in r.model_name.lower())
            or any(q in tag.lower() for tag in r.tags)
        ]

    if not filtered_runs:
        render_empty_state(
            "No runs match filters",
            "Try clearing the status filter or search query.",
        )
        return

    table_df = _build_runs_table(filtered_runs)
    selection_state = st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        placeholder=MISSING,
        key="overview_runs_table",
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
            "train/loss": st.column_config.NumberColumn("train/loss", format="%.4f"),
            "eval/loss": st.column_config.NumberColumn("eval/loss", format="%.4f"),
            "eval/accuracy": st.column_config.NumberColumn(
                "eval/accuracy", format="%.4f"
            ),
            "eval/macro_f1": st.column_config.NumberColumn(
                "eval/macro_f1", format="%.4f"
            ),
        },
    )

    selected_indices: list[int] = []
    if selection_state is not None and hasattr(selection_state, "selection"):
        selected_indices = list(selection_state.selection.get("rows", []))
    selected_run_ids = [
        filtered_runs[i].run_id
        for i in selected_indices
        if 0 <= i < len(filtered_runs)
    ]

    act_col1, act_col2, act_col3 = st.columns([1.4, 1.1, 2.5])
    with act_col1:
        can_compare = 2 <= len(selected_run_ids) <= 10
        compare_label = (
            f"Compare selected ({len(selected_run_ids)})"
            if selected_run_ids
            else "Compare selected (2–10)"
        )
        if st.button(
            compare_label,
            type="primary",
            disabled=not can_compare,
            key="overview_compare_btn",
            width="stretch",
        ):
            navigate_to_compare(selected_run_ids)
    with act_col2:
        if st.button(
            "Open selected run",
            disabled=len(selected_run_ids) != 1,
            key="overview_open_detail_btn",
            width="stretch",
        ):
            navigate_to_run_detail(selected_run_ids[0])
    with act_col3:
        if len(selected_run_ids) == 1:
            st.caption("Select 1 more run to compare, or click 'Open selected run'.")
        elif len(selected_run_ids) > 10:
            st.warning("Select at most 10 runs to compare.")
        else:
            st.caption(
                f"Showing {len(filtered_runs)} of {len(runs)} runs. Select 2–10 rows to compare."
            )


def render_page() -> None:
    service = get_service()
    ensure_standalone_chrome(service)

    selected_project = get_selected_project()
    auto_refresh, interval_s = get_refresh_settings(service)

    header_col, refresh_col = st.columns([4, 1])
    with header_col:
        proj_suffix = f" · `{selected_project}`" if selected_project else ""
        st.markdown(f"## 📊 Experiment Overview{proj_suffix}")
    with refresh_col:
        if st.button("↻ Refresh now", key="overview_manual_refresh", width="stretch"):
            st.rerun()

    filter_col1, filter_col2 = st.columns([2, 2])
    with filter_col1:
        status_options = [s.value for s in RunStatus]
        status_filter = st.multiselect(
            "Filter by status",
            options=status_options,
            default=[],
            placeholder="All statuses",
            key="overview_status_filter",
        )
    with filter_col2:
        search_query = st.text_input(
            "Search runs (name, id, model, tags)",
            value="",
            placeholder="e.g. qwen, lora, baseline...",
            key="overview_search_query",
        )

    run_every = interval_s if auto_refresh else None

    @st.fragment(run_every=run_every)
    def _live_overview_fragment() -> None:
        _render_data_section(service, selected_project, status_filter, search_query)

    _live_overview_fragment()


render_page()
