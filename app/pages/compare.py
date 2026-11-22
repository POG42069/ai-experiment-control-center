"""Compare page — side-by-side metric curves, comparability warnings, and hyperparameter diff."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from app.ui.charts import (
    build_comparison_chart,
    build_metric_chart,
    color_for_runs,
)
from app.ui.components import (
    MISSING,
    ensure_standalone_chrome,
    fmt_duration,
    fmt_float,
    fmt_text,
    render_demo_banner,
    render_empty_state,
)
from app.ui.state import (
    get_selected_project,
    get_service,
    mark_refreshed,
    parse_query_run_ids,
)
from ecc.service import STANDARD_CURVE_METRICS

DEFAULT_COMPARE_METRICS = [
    "eval/macro_f1",
    "eval/loss",
    "train/loss",
    "eval/accuracy",
]

HYPERPARAM_COLUMNS = [
    "platform",
    "account",
    "gpu",
    "model",
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "prompt_strategy",
    "seed",
    "dataset_version",
]


def _normalize_cell_for_diff(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _find_differing_columns(df: pd.DataFrame, columns: list[str]) -> set[str]:
    differing: set[str] = set()
    for col in columns:
        if col not in df.columns:
            continue
        unique_vals = {_normalize_cell_for_diff(v) for v in df[col].tolist()}
        if len(unique_vals) > 1:
            differing.add(col)
    return differing


def _display_value(val: Any) -> Any:
    if val is None:
        return MISSING
    if isinstance(val, float) and not math.isfinite(val):
        return MISSING
    return val


def render_page() -> None:
    service = get_service()
    ensure_standalone_chrome(service)

    selected_project = get_selected_project()
    all_runs = service.list_runs(project=None)
    mark_refreshed()

    st.markdown("## 📈 Compare Experiments")

    if len(all_runs) < 2:
        render_empty_state(
            "Need at least 2 runs to compare",
            "Run or configure at least two experiments to use side-by-side comparison.",
        )
        return

    run_by_id = {r.run_id: r for r in all_runs}
    all_ids = [r.run_id for r in all_runs]

    # Prefill from query params -> session state -> first 2-3 runs in active project
    query_ids = [rid for rid in parse_query_run_ids() if rid in run_by_id]
    if query_ids:
        default_ids = query_ids[:10]
    else:
        saved_ids = [
            rid
            for rid in st.session_state.get("ecc_compare_run_ids", [])
            if rid in run_by_id
        ]
        if len(saved_ids) >= 2:
            default_ids = saved_ids[:10]
        else:
            proj_runs = (
                [r.run_id for r in all_runs if r.project == selected_project]
                if selected_project
                else all_ids
            )
            default_ids = proj_runs[: min(3, len(proj_runs))]
            if len(default_ids) < 2:
                default_ids = all_ids[: min(3, len(all_ids))]

    selected_run_ids = st.multiselect(
        "Select runs to compare (2–10)",
        options=all_ids,
        default=default_ids,
        max_selections=10,
        format_func=lambda rid: f"[{run_by_id[rid].status.value}] {run_by_id[rid].name}",
        key="compare_run_selector",
    )
    st.session_state["ecc_compare_run_ids"] = selected_run_ids

    selected_runs = [run_by_id[rid] for rid in selected_run_ids if rid in run_by_id]
    render_demo_banner(selected_runs)

    if len(selected_run_ids) < 2:
        st.info("Select between 2 and 10 runs to compare.")
        return
    if len(selected_run_ids) > 10:
        st.warning("Please select at most 10 runs.")
        return

    # Discover available metrics across selected runs
    extra_metrics: list[str] = []
    for r in selected_runs:
        for k in (r.latest or {}).keys():
            if k not in STANDARD_CURVE_METRICS and k not in extra_metrics:
                extra_metrics.append(k)
    metric_options = list(dict.fromkeys([*DEFAULT_COMPARE_METRICS, *STANDARD_CURVE_METRICS, *extra_metrics]))

    c_met, c_axis, c_smooth = st.columns([2.6, 1.0, 1.4])
    with c_met:
        selected_metrics = st.multiselect(
            "Metrics",
            options=metric_options,
            default=[m for m in DEFAULT_COMPARE_METRICS if m in metric_options],
            key="compare_metrics_select",
        )
    with c_axis:
        axis = st.radio(
            "X-axis",
            options=["step", "epoch"],
            horizontal=True,
            key="compare_axis_radio",
        )
    with c_smooth:
        smoothing = st.slider(
            "Display smoothing (EMA)",
            min_value=0.0,
            max_value=0.95,
            value=0.0,
            step=0.05,
            key="compare_ema_slider",
            help="Exponential moving average applied strictly on display; underlying metrics are unchanged.",
        )

    primary_metric = selected_metrics[0] if selected_metrics else "eval/macro_f1"
    comparison = service.compare_runs(selected_run_ids, metric=primary_metric, axis=axis)

    # Comparability warnings at top
    warnings: list[str] = comparison.get("warnings", [])
    for w in warnings:
        st.warning(w)

    # Metric charts with consistent colors per run
    curves: dict[str, pd.DataFrame] = comparison.get("curves", {})
    color_map = color_for_runs(selected_run_ids)
    run_names = {r.run_id: r.name for r in selected_runs}

    if not selected_metrics:
        st.info("Select at least one metric to plot comparison curves.")
    else:
        st.markdown("### Metric Curves")
        extra_metric_dfs: dict[str, pd.DataFrame] | None = None
        for idx in range(0, len(selected_metrics), 2):
            cols = st.columns(2)
            for offset in range(2):
                if idx + offset >= len(selected_metrics):
                    continue
                metric = selected_metrics[idx + offset]
                with cols[offset]:
                    if metric in curves:
                        fig = build_comparison_chart(
                            curves[metric],
                            metric=metric,
                            axis=axis,
                            smoothing=smoothing,
                            run_order=selected_run_ids,
                            color_map=color_map,
                        )
                    else:
                        if extra_metric_dfs is None:
                            extra_metric_dfs = {
                                rid: service.get_run_metrics(rid)
                                for rid in selected_run_ids
                            }
                        fig = build_metric_chart(
                            extra_metric_dfs,
                            metric=metric,
                            axis=axis,
                            smoothing=smoothing,
                            run_names=run_names,
                            color_map=color_map,
                        )
                    st.plotly_chart(fig, width="stretch", theme=None)

    table_df: pd.DataFrame = comparison["table"]

    # Hyperparameter / diff table
    st.markdown("### Hyperparameter & Configuration Diff")
    hp_cols = [c for c in HYPERPARAM_COLUMNS if c in table_df.columns]
    # Also include extra common non-ecc config keys if present across runs
    extra_hp_keys: list[str] = []
    for r in selected_runs:
        for k in (r.config or {}).keys():
            if not k.startswith("ecc/") and k not in hp_cols and k not in extra_hp_keys:
                extra_hp_keys.append(k)

    hp_rows: list[dict[str, Any]] = []
    for i, r in enumerate(selected_runs):
        base_row = table_df.iloc[i].to_dict() if i < len(table_df) else {}
        row: dict[str, Any] = {"name": r.name}
        for c in hp_cols:
            row[c] = _display_value(base_row.get(c))
        for k in extra_hp_keys:
            row[k] = _display_value((r.config or {}).get(k))
        hp_rows.append(row)
    hp_df = pd.DataFrame(hp_rows)

    all_hp_cols = [c for c in hp_df.columns if c != "name"]
    differing_cols = _find_differing_columns(hp_df, all_hp_cols)

    diff_only = st.toggle(
        "Show only differing columns",
        value=False,
        key="compare_diff_only",
    )
    if differing_cols:
        st.caption(
            "Differing columns (highlighted): "
            + ", ".join(f"`{c}`" for c in sorted(differing_cols))
        )
    else:
        st.caption("All displayed hyperparameters match across the selected runs.")

    display_hp_cols = (
        ["name", *[c for c in all_hp_cols if c in differing_cols]]
        if diff_only and differing_cols
        else ["name", *all_hp_cols]
    )
    hp_view = hp_df[display_hp_cols]

    def _highlight_diff_col(col: pd.Series) -> list[str]:
        if col.name in differing_cols:
            return ["background-color: rgba(210, 153, 34, 0.18); color: #F2CC60;"] * len(col)
        return [""] * len(col)

    st.dataframe(
        hp_view.style.apply(_highlight_diff_col, axis=0),
        width="stretch",
        hide_index=True,
    )

    # Best-epoch / best-metric table
    st.markdown("### Best Epoch & Summary Metrics")
    best_rows: list[dict[str, Any]] = []
    for i, r in enumerate(selected_runs):
        t_row = table_df.iloc[i].to_dict() if i < len(table_df) else {}
        best_rows.append(
            {
                "Run": r.name,
                "Status": r.status.value,
                "Best metric": fmt_text(r.best_metric_name),
                "Best value": fmt_float(r.best_metric, 4),
                "Best step": str(r.best_step) if r.best_step is not None else MISSING,
                "Best epoch": fmt_float(r.best_epoch, 2),
                "Final train/loss": fmt_float(t_row.get("final train/loss"), 4),
                "Final eval/loss": fmt_float(t_row.get("final eval/loss"), 4),
                "eval/accuracy": fmt_float(t_row.get("eval/accuracy"), 4),
                "eval/macro_f1": fmt_float(t_row.get("eval/macro_f1"), 4),
                "Max GPU Mem (GB)": fmt_float(t_row.get("max gpu/mem_alloc_gb"), 2),
                "Duration": fmt_duration(r.duration_s),
            }
        )
    st.dataframe(pd.DataFrame(best_rows), width="stretch", hide_index=True)

    # CSV download
    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download comparison CSV",
        data=csv_bytes,
        file_name="ecc_comparison.csv",
        mime="text/csv",
        key="compare_download_csv",
    )


render_page()
