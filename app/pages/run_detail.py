"""Run Detail page — metadata, editable notes, metric charts, and structured/console/Kaggle logs."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from app.ui.charts import (
    build_eval_metrics_chart,
    build_gpu_memory_chart,
    build_loss_chart,
    build_lr_chart,
)
from app.ui.components import (
    MISSING,
    TEXT_MUTED,
    ensure_standalone_chrome,
    fmt_duration,
    fmt_epoch_progress,
    fmt_float,
    fmt_lr,
    fmt_platform,
    fmt_relative_time,
    fmt_step_progress,
    fmt_text,
    fmt_timestamp,
    render_demo_banner,
    render_empty_state,
    render_log_viewer,
    status_badge_html,
)
from app.ui.state import (
    get_selected_project,
    get_service,
    mark_refreshed,
    navigate_to_compare,
)
from ecc.models import LogLine, RunSummary
from ecc.service import Service

_EXCEPTION_RE = re.compile(
    r"(CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|"
    r"RuntimeError|ValueError|KeyError|AssertionError|OOM|Exception:)",
    re.IGNORECASE,
)


def _detect_critical_log_lines(lines: list[LogLine]) -> list[str]:
    hits: list[str] = []
    for line in lines:
        if _EXCEPTION_RE.search(line.text):
            hits.append(line.text)
    return hits


def _has_linked_kaggle_kernel(service: Service, run: RunSummary) -> bool:
    if run.source == "kaggle":
        return True
    if (run.config or {}).get("ecc/kaggle_kernel"):
        return True
    for link in service.store.list_kernel_links():
        if link.get("run_id") == run.run_id:
            return True
    return False


def _render_metadata_panel(service: Service, run: RunSummary) -> None:
    cfg = run.config or {}
    meta_rows = [
        ("Project", fmt_text(run.project)),
        ("Platform / Account", fmt_platform(run.platform, run.account)),
        ("GPU", fmt_text(run.gpu)),
        ("Model", fmt_text(run.model_name)),
        ("Prompt strategy", fmt_text(cfg.get("ecc/prompt_strategy"))),
        (
            "Git branch / SHA",
            f"{fmt_text(run.git_branch)} @ {fmt_text(run.git_sha)}"
            if (run.git_branch or run.git_sha)
            else MISSING,
        ),
        (
            "Dataset version / split",
            f"{fmt_text(run.dataset_version)} · {fmt_text(cfg.get('ecc/dataset_split'))}"
            if (run.dataset_version or cfg.get("ecc/dataset_split"))
            else MISSING,
        ),
        ("Seed", str(run.seed) if run.seed is not None else MISSING),
        ("Step / Total", fmt_step_progress(run.step, run.total_steps)),
        ("Epoch / Total", fmt_epoch_progress(run.epoch, run.total_epochs)),
        ("Started", fmt_timestamp(run.started_at)),
        ("Ended", fmt_timestamp(run.ended_at)),
        ("Duration", fmt_duration(run.duration_s)),
        ("ETA", fmt_duration(run.eta_s)),
        ("Last update", fmt_relative_time(run.last_update)),
        ("Eval signature", fmt_text(cfg.get("ecc/eval_signature"))),
    ]

    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("#### Run Metadata")
        meta_df = pd.DataFrame(meta_rows, columns=["Field", "Value"])
        st.dataframe(meta_df, width="stretch", hide_index=True, height=340)

        st.markdown("#### Checkpoints")
        checkpoints = cfg.get("ecc/checkpoints") or getattr(run, "checkpoints", None)
        if isinstance(checkpoints, list) and checkpoints:
            st.dataframe(
                pd.DataFrame(checkpoints),
                width="stretch",
                hide_index=True,
                placeholder=MISSING,
            )
        else:
            st.caption(f"{MISSING} (no checkpoints recorded in run summary)")

    with c_right:
        st.markdown("#### Notes & Local Tags")
        existing_note = service.store.get_run_note(run.run_id) or {}
        source_tags = ", ".join(run.tags) if run.tags else MISSING
        st.caption(f"Source tags: {source_tags}")

        note_text_default = str(existing_note.get("notes", ""))
        note_tags_default = ", ".join(existing_note.get("tags", []))
        note_pinned_default = bool(existing_note.get("pinned", False))

        with st.form(f"run_note_form_{run.run_id}"):
            notes_val = st.text_area(
                "Notes",
                value=note_text_default,
                height=140,
                placeholder="Observations, failure diagnosis, next steps...",
            )
            tags_val = st.text_input(
                "Local tags (comma-separated)",
                value=note_tags_default,
                placeholder="e.g. candidate, oom-checked",
            )
            pinned_val = st.checkbox("Pin run", value=note_pinned_default)
            submitted = st.form_submit_button("Save notes & tags", type="primary")
            if submitted:
                parsed_tags = [t.strip() for t in tags_val.split(",") if t.strip()]
                service.store.set_run_note(
                    run.run_id,
                    notes=notes_val,
                    tags=parsed_tags,
                    pinned=pinned_val,
                )
                st.success("Saved local notes & tags.")

        with st.expander("Full Configuration (JSON)", expanded=False):
            st.json(cfg)


def _render_console_tab(
    lines: list[LogLine],
    *,
    tab_key: str,
    empty_explanation: str,
) -> None:
    if not lines:
        st.info(empty_explanation)
        return

    # Backend unavailable / informational system notice (never invent log lines)
    if len(lines) == 1 and lines[0].stream == "system":
        st.info(lines[0].text)
        return

    critical_hits = _detect_critical_log_lines(lines)
    if critical_hits:
        preview = "\n".join(f"- `{hit[:240]}`" for hit in critical_hits[-8:])
        st.error(
            f"**Detected {len(critical_hits)} exception / OOM log line(s):**\n\n{preview}"
        )

    newest_first = st.toggle(
        "Newest first",
        value=True,
        key=f"{tab_key}_newest_first",
        help="Show most recent lines at the top so latest output is always visible.",
    )
    render_log_viewer(lines, newest_first=newest_first)


def render_page() -> None:
    service = get_service()
    ensure_standalone_chrome(service)

    selected_project = get_selected_project()
    all_runs = service.list_runs(project=selected_project)
    if not all_runs and selected_project is not None:
        all_runs = service.list_runs(project=None)
    mark_refreshed()

    st.markdown("## 🔍 Run Detail")

    query_run_id = st.query_params.get("run")
    if isinstance(query_run_id, list):
        query_run_id = query_run_id[0] if query_run_id else None

    candidate_ids = [r.run_id for r in all_runs]
    if query_run_id and query_run_id not in candidate_ids:
        direct = service.get_run_details(query_run_id)
        if direct is not None:
            all_runs = [direct, *all_runs]
            candidate_ids = [r.run_id for r in all_runs]

    if not all_runs:
        render_empty_state(
            "No runs available",
            "Configure W&B / Kaggle or enable demo mode to inspect runs.",
        )
        return

    initial_id = (
        query_run_id
        or st.session_state.get("ecc_active_run_id")
        or candidate_ids[0]
    )
    if initial_id not in candidate_ids:
        initial_id = candidate_ids[0]

    label_by_id = {
        r.run_id: f"[{r.status.value}] {r.name} ({r.project})" for r in all_runs
    }
    chosen_id = st.selectbox(
        "Select run",
        options=candidate_ids,
        index=candidate_ids.index(initial_id),
        format_func=lambda rid: label_by_id.get(rid, rid),
        key="run_detail_selector",
    )
    st.session_state["ecc_active_run_id"] = chosen_id
    if st.query_params.get("run") != chosen_id:
        st.query_params["run"] = chosen_id

    run = service.get_run_details(chosen_id)
    if run is None:
        st.warning(f"Run `{chosen_id}` could not be loaded from active sources.")
        return

    render_demo_banner([run])

    # Header
    h_left, h_right = st.columns([3.5, 1.5])
    with h_left:
        kaggle_st = (run.config or {}).get("kaggle_status")
        k_html = (
            f' · <span class="ecc-mono" style="color:{TEXT_MUTED};font-size:0.82rem;">'
            f"Kaggle status: {html.escape(str(kaggle_st))}</span>"
            if kaggle_st
            else ""
        )
        raw_st = (
            f' · <span class="ecc-mono" style="color:{TEXT_MUTED};font-size:0.82rem;">'
            f"raw: {html.escape(str(run.raw_state))}</span>"
            if run.raw_state
            else ""
        )
        st.markdown(
            f"<div>{status_badge_html(run.status)}{raw_st}{k_html}</div>"
            f'<h3 style="margin:0.3rem 0 0.15rem 0;">{html.escape(run.name)}</h3>'
            f'<div class="ecc-mono" style="font-size:0.8rem;color:{TEXT_MUTED};">'
            f"ID: {html.escape(run.run_id)} · Source: {html.escape(run.source)}</div>",
            unsafe_allow_html=True,
        )
    with h_right:
        b1, b2 = st.columns(2)
        with b1:
            if run.notebook_url:
                st.link_button("Notebook ↗", run.notebook_url, width="stretch")
            else:
                st.button("Notebook —", disabled=True, width="stretch")
        with b2:
            if run.source_url:
                st.link_button("W&B ↗", run.source_url, width="stretch")
            else:
                st.button("W&B —", disabled=True, width="stretch")
        peer_ids = [r.run_id for r in all_runs if r.project == run.project][:3]
        if run.run_id not in peer_ids:
            peer_ids = [run.run_id, *peer_ids[:2]]
        if len(peer_ids) >= 2 and st.button(
            "Compare with peers", key="run_detail_compare_btn", width="stretch"
        ):
            navigate_to_compare(peer_ids)

    st.divider()
    _render_metadata_panel(service, run)

    st.divider()
    st.markdown("### Metric Curves")
    axis = st.radio(
        "X-axis",
        options=["step", "epoch"],
        horizontal=True,
        key="run_detail_axis",
    )
    metrics_df = service.get_run_metrics(run.run_id)
    total_mem = (run.config or {}).get("ecc/gpu_mem_total_gb")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.plotly_chart(
            build_loss_chart(metrics_df, axis=axis),
            width="stretch",
            theme=None,
        )
    with row1_c2:
        st.plotly_chart(
            build_eval_metrics_chart(metrics_df, axis=axis),
            width="stretch",
            theme=None,
        )

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.plotly_chart(
            build_lr_chart(metrics_df, axis=axis),
            width="stretch",
            theme=None,
        )
    with row2_c2:
        st.plotly_chart(
            build_gpu_memory_chart(
                metrics_df,
                axis=axis,
                total_mem_gb=float(total_mem) if total_mem is not None else None,
            ),
            width="stretch",
            theme=None,
        )

    st.divider()
    st.markdown("### Logs")
    has_kaggle = _has_linked_kaggle_kernel(service, run)
    tab_names = ["Structured metrics (latest)", "Console (stdout/stderr)"]
    if has_kaggle:
        tab_names.append("Kaggle native log")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        latest = run.latest or {}
        best_col1, best_col2, best_col3, best_col4 = st.columns(4)
        best_col1.metric("Best metric", fmt_text(run.best_metric_name))
        best_col2.metric("Best value", fmt_float(run.best_metric, 4))
        best_col3.metric(
            "Best step",
            str(run.best_step) if run.best_step is not None else MISSING,
        )
        best_col4.metric("Best epoch", fmt_float(run.best_epoch, 2))

        if latest:
            latest_rows = [
                {
                    "Metric": k,
                    "Latest value": fmt_lr(v) if k.endswith("/lr") else fmt_float(v, 5),
                }
                for k, v in sorted(latest.items())
            ]
            st.dataframe(
                pd.DataFrame(latest_rows),
                width="stretch",
                hide_index=True,
                placeholder=MISSING,
            )
        else:
            st.info(
                "No structured metrics recorded yet for this run. "
                "(For Kaggle-only runs without W&B/experiment_monitor, structured metrics are not available.)"
            )

        if not metrics_df.empty:
            st.caption("Recent logged rows")
            st.dataframe(
                metrics_df.tail(10),
                width="stretch",
                hide_index=True,
                placeholder=MISSING,
            )

    with tabs[1]:
        f1, f2, f3, f4 = st.columns([2.2, 1.0, 1.0, 0.9])
        with f1:
            log_query = st.text_input(
                "Search logs",
                value="",
                placeholder="Filter by substring (e.g. OOM, eval, loss)...",
                key="run_detail_log_query",
            )
        with f2:
            log_level = st.selectbox(
                "Level",
                options=["ALL", "INFO", "WARNING", "ERROR"],
                key="run_detail_log_level",
            )
        with f3:
            log_stream = st.selectbox(
                "Stream",
                options=["ALL", "stdout", "stderr", "system", "kaggle"],
                key="run_detail_log_stream",
            )
        with f4:
            log_tail = st.selectbox(
                "Tail N",
                options=[200, 500, 2000],
                index=1,
                key="run_detail_log_tail",
            )

        console_lines = service.get_run_logs(
            run.run_id,
            tail=int(log_tail),
            query=log_query.strip() or None,
            level=None if log_level == "ALL" else log_level,
        )
        if log_stream != "ALL":
            console_lines = [ln for ln in console_lines if ln.stream == log_stream]

        _render_console_tab(
            console_lines,
            tab_key="console_log",
            empty_explanation=(
                "No console log lines available for this filter/run. "
                "Live console capture requires `experiment_monitor` with `capture_console=True`."
            ),
        )

    if has_kaggle:
        with tabs[2]:
            k_tail = st.selectbox(
                "Tail N (Kaggle native)",
                options=[200, 500, 2000],
                index=1,
                key="run_detail_kaggle_tail",
            )
            kaggle_lines = service.get_native_logs(run.run_id, tail=int(k_tail))
            _render_console_tab(
                kaggle_lines,
                tab_key="kaggle_native_log",
                empty_explanation=(
                    "Kaggle native logs are currently unavailable for this linked kernel "
                    "(verify Kaggle account credentials and that the kernel has produced persisted logs)."
                ),
            )


render_page()
