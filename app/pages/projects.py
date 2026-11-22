"""Projects page — project list, create/edit metadata, baseline selection, and run history."""

from __future__ import annotations

import re
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
    fmt_float,
    fmt_platform,
    fmt_text,
    fmt_timestamp,
    render_demo_banner,
    render_empty_state,
    render_kpi_card,
)
from app.ui.state import (
    get_selected_project,
    get_service,
    mark_refreshed,
    navigate_to_run_detail,
)
from ecc.models import RunStatus, RunSummary

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_HISTORY_STATUSES = {
    RunStatus.FINISHED,
    RunStatus.FAILED,
    RunStatus.CRASHED,
    RunStatus.KILLED,
}


def _start_ts_key(run: RunSummary) -> float:
    ts = run.started_at or run.last_update
    if ts is None:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def _format_delta(delta: float | None, mode: str) -> tuple[str, str]:
    if delta is None:
        return MISSING, "#8B949E"
    sign = "+" if delta >= 0 else ""
    good = (delta >= 0) if mode == "max" else (delta <= 0)
    color = "#3FB950" if good else "#F85149"
    return f"{sign}{delta:.4f}", color


def render_page() -> None:
    service = get_service()
    ensure_standalone_chrome(service)

    projects = service.list_projects()
    all_runs = service.list_runs(project=None)
    mark_refreshed()
    render_demo_banner(all_runs)

    st.markdown("## 📁 Projects")

    # Summary table of all projects
    runs_by_project: dict[str, list[RunSummary]] = {}
    for r in all_runs:
        runs_by_project.setdefault(r.project, []).append(r)

    summary_rows: list[dict[str, Any]] = []
    for proj in projects:
        p_runs = runs_by_project.get(proj.slug, [])
        best_info = service.get_best_experiment(
            proj.slug, metric="eval/macro_f1", mode="max"
        )
        best_run = best_info.get("best")
        best_val = best_info.get("best_value")
        base_run = best_info.get("baseline")
        base_val = best_info.get("baseline_value")
        delta = best_info.get("delta")

        best_txt = (
            f"{best_run['name']} ({fmt_float(best_val, 4)})"
            if best_run is not None
            else MISSING
        )
        base_txt = (
            f"{base_run['name']} ({fmt_float(base_val, 4)})"
            if base_run is not None
            else (proj.baseline_run_id or MISSING)
        )
        delta_txt, _ = _format_delta(delta, "max")

        summary_rows.append(
            {
                "Slug": proj.slug,
                "Name": proj.name,
                "Total": len(p_runs),
                "Running": sum(1 for r in p_runs if r.status == RunStatus.RUNNING),
                "Stale/Offline": sum(
                    1
                    for r in p_runs
                    if r.status in {RunStatus.STALE, RunStatus.OFFLINE}
                ),
                "Queued": sum(1 for r in p_runs if r.status == RunStatus.QUEUED),
                "Finished": sum(1 for r in p_runs if r.status == RunStatus.FINISHED),
                "Failed": sum(
                    1
                    for r in p_runs
                    if r.status
                    in {RunStatus.FAILED, RunStatus.CRASHED, RunStatus.KILLED}
                ),
                "Best run (eval/macro_f1)": best_txt,
                "Baseline": base_txt,
                "Delta": delta_txt,
            }
        )

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
    else:
        render_empty_state(
            "No projects yet",
            "Create a project below or configure W&B projects in ecc_config.toml.",
        )

    # Create / Edit Project form
    with st.expander("Create or Edit Project", expanded=not bool(projects)):
        existing_slugs = [p.slug for p in projects]
        mode_choice = st.radio(
            "Action",
            options=["Create new project", "Edit existing project"],
            horizontal=True,
            key="projects_form_mode",
            disabled=not bool(existing_slugs),
        )
        edit_meta: dict[str, Any] = {}
        edit_slug = ""
        if mode_choice == "Edit existing project" and existing_slugs:
            edit_slug = st.selectbox(
                "Project to edit",
                options=existing_slugs,
                key="projects_edit_select",
            )
            edit_meta = service.store.get_project(edit_slug) or {}
            if not edit_meta:
                proj_obj = next((p for p in projects if p.slug == edit_slug), None)
                if proj_obj is not None:
                    edit_meta = {
                        "slug": proj_obj.slug,
                        "name": proj_obj.name,
                        "description": proj_obj.description,
                        "tags": proj_obj.tags,
                        "notes": proj_obj.notes,
                    }

        with st.form("project_upsert_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                slug_in = st.text_input(
                    "Slug",
                    value=str(edit_meta.get("slug", edit_slug)),
                    placeholder="e.g. shared-task-4",
                    disabled=(mode_choice == "Edit existing project"),
                )
                wb_entity_in = st.text_input(
                    "W&B entity (optional)",
                    value=str(edit_meta.get("wandb_entity") or ""),
                    placeholder="my-wandb-entity",
                )
                tags_in = st.text_input(
                    "Tags (comma-separated)",
                    value=", ".join(edit_meta.get("tags") or []),
                    placeholder="vlm, lora, competition",
                )
            with fc2:
                name_in = st.text_input(
                    "Display name",
                    value=str(edit_meta.get("name", "")),
                    placeholder="e.g. Shared Task 4 — VLM Fine-Tuning",
                )
                wb_proj_in = st.text_input(
                    "W&B project (optional)",
                    value=str(edit_meta.get("wandb_project") or ""),
                    placeholder="shared-task-4",
                )
                desc_in = st.text_input(
                    "Description",
                    value=str(edit_meta.get("description", "")),
                    placeholder="Short goal or dataset description",
                )
            notes_in = st.text_area(
                "Notes",
                value=str(edit_meta.get("notes", "")),
                height=90,
                placeholder="Experiment hypotheses, dataset notes, evaluation protocol...",
            )
            save_proj = st.form_submit_button("Save project", type="primary")
            if save_proj:
                final_slug = (edit_slug if mode_choice == "Edit existing project" else slug_in).strip().lower()
                final_name = name_in.strip() or final_slug
                if not final_slug or not _SLUG_RE.match(final_slug):
                    st.error(
                        "Invalid slug: use 1–64 lowercase alphanumeric, '-', or '_' characters starting with alphanumeric."
                    )
                else:
                    parsed_tags = [t.strip() for t in tags_in.split(",") if t.strip()]
                    service.store.upsert_project(
                        slug=final_slug,
                        name=final_name,
                        description=desc_in.strip(),
                        wandb_entity=wb_entity_in.strip() or None,
                        wandb_project=wb_proj_in.strip() or None,
                        tags=parsed_tags,
                        notes=notes_in.strip(),
                    )
                    st.success(f"Saved project `{final_slug}`.")
                    st.rerun()

    if not projects:
        return

    st.divider()
    st.markdown("### Project Workspace")
    sidebar_proj = get_selected_project()
    slug_list = [p.slug for p in projects]
    default_idx = slug_list.index(sidebar_proj) if sidebar_proj in slug_list else 0
    active_slug = st.selectbox(
        "Inspect project",
        options=slug_list,
        index=default_idx,
        key="projects_active_slug",
    )
    active_proj = next(p for p in projects if p.slug == active_slug)
    proj_runs = runs_by_project.get(active_slug, [])

    if active_proj.description:
        st.caption(active_proj.description)
    if active_proj.notes:
        st.markdown(f"**Notes:** {active_proj.notes}")

    # Baseline & Best-vs-Baseline Delta
    b_col1, b_col2, b_col3 = st.columns([2.2, 1.2, 1.0])
    with b_col1:
        baseline_options = ["(None)", *[r.run_id for r in proj_runs]]
        curr_base = (
            active_proj.baseline_run_id
            if active_proj.baseline_run_id in baseline_options
            else "(None)"
        )
        run_label = {r.run_id: f"[{r.status.value}] {r.name}" for r in proj_runs}
        chosen_baseline = st.selectbox(
            "Baseline run",
            options=baseline_options,
            index=baseline_options.index(curr_base),
            format_func=lambda x: x if x == "(None)" else run_label.get(x, x),
            key=f"projects_baseline_{active_slug}",
        )
        if st.button("Set baseline", key=f"projects_save_baseline_{active_slug}"):
            service.store.set_baseline(
                active_slug,
                None if chosen_baseline == "(None)" else chosen_baseline,
            )
            st.success("Baseline updated.")
            st.rerun()
    with b_col2:
        delta_metric = st.selectbox(
            "Comparison metric",
            options=["eval/macro_f1", "eval/accuracy", "eval/loss", "train/loss"],
            key=f"projects_delta_metric_{active_slug}",
        )
    with b_col3:
        default_mode_idx = 1 if delta_metric.endswith("loss") else 0
        delta_mode = st.radio(
            "Direction",
            options=["max", "min"],
            index=default_mode_idx,
            horizontal=True,
            key=f"projects_delta_mode_{active_slug}",
        )

    best_exp = service.get_best_experiment(
        active_slug, metric=delta_metric, mode=delta_mode
    )
    best_obj = best_exp.get("best")
    base_obj = best_exp.get("baseline")
    delta_txt, delta_color = _format_delta(best_exp.get("delta"), delta_mode)

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card(
            f"Best ({delta_metric})",
            fmt_float(best_exp.get("best_value"), 4),
            STATUS_COLORS[RunStatus.RUNNING],
        )
        st.caption(best_obj["name"] if best_obj else MISSING)
    with k2:
        render_kpi_card(
            f"Baseline ({delta_metric})",
            fmt_float(best_exp.get("baseline_value"), 4),
            STATUS_COLORS[RunStatus.QUEUED],
        )
        st.caption(base_obj["name"] if base_obj else "No baseline set")
    with k3:
        render_kpi_card("Best vs Baseline Delta", delta_txt, delta_color)
        st.caption(f"Mode: {delta_mode}")

    # Per-project runs table
    st.markdown("#### Project Runs")
    if not proj_runs:
        render_empty_state(
            "No runs in this project",
            f"No experiments have been recorded under `{active_slug}` yet.",
        )
        return

    proj_table_rows = [
        {
            "Status": r.status.value,
            "Name": r.name,
            "Model": fmt_text(r.model_name),
            "Platform / Account": fmt_platform(r.platform, r.account),
            "GPU": fmt_text(r.gpu),
            "eval/macro_f1": (r.latest or {}).get("eval/macro_f1"),
            "eval/loss": (r.latest or {}).get("eval/loss"),
            "train/loss": (r.latest or {}).get("train/loss"),
            "Duration": fmt_duration(r.duration_s),
            "Started": fmt_timestamp(r.started_at),
        }
        for r in proj_runs
    ]
    st.dataframe(
        pd.DataFrame(proj_table_rows),
        width="stretch",
        hide_index=True,
        placeholder=MISSING,
        column_config={
            "eval/macro_f1": st.column_config.NumberColumn(format="%.4f"),
            "eval/loss": st.column_config.NumberColumn(format="%.4f"),
            "train/loss": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    quick_open = st.selectbox(
        "Jump to run detail",
        options=[r.run_id for r in proj_runs],
        format_func=lambda rid: run_label.get(rid, rid),
        key=f"projects_jump_run_{active_slug}",
    )
    if st.button("Open run detail", key=f"projects_jump_btn_{active_slug}"):
        navigate_to_run_detail(quick_open)

    # History (finished/failed sorted by start time descending)
    st.markdown("#### Run History (Finished & Failed)")
    history_runs = sorted(
        [r for r in proj_runs if r.status in _HISTORY_STATUSES],
        key=_start_ts_key,
        reverse=True,
    )
    if not history_runs:
        st.caption("No finished or failed runs yet in this project.")
    else:
        hist_rows = [
            {
                "Started": fmt_timestamp(r.started_at),
                "Ended": fmt_timestamp(r.ended_at),
                "Status": r.status.value,
                "Name": r.name,
                "Best metric": fmt_float(r.best_metric, 4),
                "eval/macro_f1": fmt_float((r.latest or {}).get("eval/macro_f1"), 4),
                "Duration": fmt_duration(r.duration_s),
            }
            for r in history_runs
        ]
        st.dataframe(pd.DataFrame(hist_rows), width="stretch", hide_index=True)


render_page()
