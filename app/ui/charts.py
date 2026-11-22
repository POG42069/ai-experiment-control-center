"""Pure Plotly figure builders and EMA display smoothing for ECC charts."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

BG_DARK = "#0E1117"
BG_SECONDARY = "#161B22"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED = "#8B949E"
GRID_COLOR = "#21262D"
AXIS_LINE_COLOR = "#30363D"

QUALITATIVE_PALETTE: list[str] = [
    "#4F8BF9",  # primary blue
    "#3FB950",  # green
    "#F78166",  # coral
    "#A371F7",  # purple
    "#D29922",  # amber
    "#39C5CF",  # cyan
    "#F778BA",  # pink
    "#79C0FF",  # sky
    "#FFA657",  # orange
    "#56D364",  # lime
]

_VALID_AXES = {"step", "epoch"}


def _validate_axis(axis: str) -> str:
    if axis not in _VALID_AXES:
        raise ValueError(f"axis must be 'step' or 'epoch', got {axis!r}")
    return axis


def color_for_runs(run_ids: Sequence[str]) -> dict[str, str]:
    """Deterministically assign fixed qualitative palette colors to run IDs."""
    mapping: dict[str, str] = {}
    for idx, rid in enumerate(dict.fromkeys(run_ids)):
        mapping[rid] = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
    return mapping


def apply_ema(
    values: Sequence[float] | pd.Series,
    smoothing: float,
) -> list[float]:
    """Apply exponential moving average smoothing purely for display.

    - ``smoothing <= 0`` returns raw float values unchanged.
    - ``smoothing`` is clamped to ``[0.0, 0.99]``.
    - Non-finite values (NaN / Inf) pass through as ``nan`` without corrupting EMA state.
    - Recurrence for finite points: ``ema_t = alpha * ema_{t-1} + (1 - alpha) * x_t``
      with ``ema_0 = x_0``.
    """
    raw = [float(v) if v is not None else float("nan") for v in values]
    if not raw:
        return []
    alpha = max(0.0, min(0.99, float(smoothing)))
    if alpha <= 0.0:
        return raw

    out: list[float] = []
    ema: float | None = None
    for v in raw:
        if not math.isfinite(v):
            out.append(float("nan"))
            continue
        if ema is None:
            ema = v
        else:
            ema = alpha * ema + (1.0 - alpha) * v
        out.append(ema)
    return out


def _apply_dark_layout(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    yaxis_tickformat: str | None = None,
    empty_message: str | None = None,
) -> go.Figure:
    yaxis_cfg: dict[str, object] = {
        "title": yaxis_title,
        "gridcolor": GRID_COLOR,
        "zerolinecolor": AXIS_LINE_COLOR,
        "linecolor": AXIS_LINE_COLOR,
    }
    if yaxis_tickformat:
        yaxis_cfg["tickformat"] = yaxis_tickformat

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY), x=0.01),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_SECONDARY,
        font=dict(
            color=TEXT_PRIMARY,
            family='ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            size=11,
        ),
        hovermode="x unified",
        margin=dict(l=52, r=20, t=44, b=42),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_PRIMARY),
        ),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=AXIS_LINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        yaxis=yaxis_cfg,
        height=320,
    )
    if empty_message:
        fig.add_annotation(
            text=empty_message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color=TEXT_MUTED),
        )
    return fig


def _axis_label(axis: str) -> str:
    return "Step" if axis == "step" else "Epoch"


# ---------------------------------------------------------------------------
# Single-run detail chart builders
# ---------------------------------------------------------------------------
def build_loss_chart(
    df: pd.DataFrame,
    axis: str = "step",
    smoothing: float = 0.0,
) -> go.Figure:
    """Build Train vs Eval Loss chart for a single run."""
    axis = _validate_axis(axis)
    fig = go.Figure()
    has_data = False

    if df is not None and not df.empty and axis in df.columns:
        if "train/loss" in df.columns:
            sub = df[[axis, "train/loss"]].dropna()
            if not sub.empty:
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=sub[axis].tolist(),
                        y=apply_ema(sub["train/loss"], smoothing),
                        mode="lines",
                        name="train/loss",
                        line=dict(color="#4F8BF9", width=2),
                    )
                )
        if "eval/loss" in df.columns:
            sub = df[[axis, "eval/loss"]].dropna()
            if not sub.empty:
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=sub[axis].tolist(),
                        y=apply_ema(sub["eval/loss"], smoothing),
                        mode="lines+markers",
                        name="eval/loss",
                        line=dict(color="#F78166", width=2),
                        marker=dict(size=5),
                    )
                )

    return _apply_dark_layout(
        fig,
        title="Train vs Eval Loss",
        xaxis_title=_axis_label(axis),
        yaxis_title="Loss",
        empty_message=None if has_data else "No loss data logged yet",
    )


def build_eval_metrics_chart(
    df: pd.DataFrame,
    axis: str = "step",
    smoothing: float = 0.0,
) -> go.Figure:
    """Build Eval Accuracy & Macro-F1 chart for a single run."""
    axis = _validate_axis(axis)
    fig = go.Figure()
    has_data = False

    series_spec = [
        ("eval/macro_f1", "#3FB950"),
        ("eval/accuracy", "#58A6FF"),
    ]
    if df is not None and not df.empty and axis in df.columns:
        for col, color in series_spec:
            if col in df.columns:
                sub = df[[axis, col]].dropna()
                if not sub.empty:
                    has_data = True
                    fig.add_trace(
                        go.Scatter(
                            x=sub[axis].tolist(),
                            y=apply_ema(sub[col], smoothing),
                            mode="lines+markers",
                            name=col,
                            line=dict(color=color, width=2),
                            marker=dict(size=5),
                        )
                    )

    return _apply_dark_layout(
        fig,
        title="Eval Accuracy & Macro-F1",
        xaxis_title=_axis_label(axis),
        yaxis_title="Score",
        empty_message=None if has_data else "No eval accuracy / F1 data logged yet",
    )


def build_lr_chart(
    df: pd.DataFrame,
    axis: str = "step",
) -> go.Figure:
    """Build Learning Rate schedule chart for a single run."""
    axis = _validate_axis(axis)
    fig = go.Figure()
    has_data = False

    if df is not None and not df.empty and axis in df.columns and "train/lr" in df.columns:
        sub = df[[axis, "train/lr"]].dropna()
        if not sub.empty:
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=sub[axis].tolist(),
                    y=sub["train/lr"].astype(float).tolist(),
                    mode="lines",
                    name="train/lr",
                    line=dict(color="#A371F7", width=2),
                )
            )

    return _apply_dark_layout(
        fig,
        title="Learning Rate",
        xaxis_title=_axis_label(axis),
        yaxis_title="LR",
        yaxis_tickformat=".2e",
        empty_message=None if has_data else "No learning rate data logged yet",
    )


def build_gpu_memory_chart(
    df: pd.DataFrame,
    axis: str = "step",
    total_mem_gb: float | None = None,
) -> go.Figure:
    """Build GPU memory allocation chart for a single run."""
    axis = _validate_axis(axis)
    fig = go.Figure()
    has_data = False

    if df is not None and not df.empty and axis in df.columns:
        for col, color in (
            ("gpu/mem_alloc_gb", "#39C5CF"),
            ("gpu/mem_reserved_gb", "#D29922"),
        ):
            if col in df.columns:
                sub = df[[axis, col]].dropna()
                if not sub.empty:
                    has_data = True
                    fig.add_trace(
                        go.Scatter(
                            x=sub[axis].tolist(),
                            y=sub[col].astype(float).tolist(),
                            mode="lines",
                            name=col,
                            line=dict(color=color, width=2),
                        )
                    )

    if has_data and total_mem_gb is not None and math.isfinite(float(total_mem_gb)):
        fig.add_hline(
            y=float(total_mem_gb),
            line_dash="dash",
            line_color="#F85149",
            annotation_text=f"GPU total ({float(total_mem_gb):.1f} GB)",
            annotation_font_color="#F85149",
        )

    return _apply_dark_layout(
        fig,
        title="GPU Memory (GB)",
        xaxis_title=_axis_label(axis),
        yaxis_title="GB",
        empty_message=None if has_data else "No GPU memory telemetry logged",
    )


# ---------------------------------------------------------------------------
# Multi-run comparison chart builders
# ---------------------------------------------------------------------------
def build_comparison_chart(
    curve_df: pd.DataFrame | None,
    metric: str,
    axis: str = "step",
    smoothing: float = 0.0,
    run_order: Sequence[str] | None = None,
    color_map: Mapping[str, str] | None = None,
) -> go.Figure:
    """Build a multi-run comparison line chart from long-format curve DataFrame.

    Expects ``curve_df`` with columns ``run_id``, ``name``, ``x``, ``value``
    (as returned by ``Service.compare_runs()['curves'][metric]``).
    Produces exactly one Plotly trace per run present in ``curve_df``.
    """
    axis = _validate_axis(axis)
    fig = go.Figure()

    if curve_df is None or curve_df.empty:
        return _apply_dark_layout(
            fig,
            title=metric,
            xaxis_title=_axis_label(axis),
            yaxis_title=metric,
            empty_message=f"No data available for {metric}",
        )

    present_ids = list(dict.fromkeys(curve_df["run_id"].astype(str).tolist()))
    ordered_ids = (
        [rid for rid in run_order if rid in present_ids]
        + [rid for rid in present_ids if run_order is not None and rid not in run_order]
        if run_order is not None
        else present_ids
    )
    palette_map = dict(color_map) if color_map is not None else color_for_runs(ordered_ids)

    is_sparse_eval = metric.startswith("eval/")
    for idx, rid in enumerate(ordered_ids):
        sub = curve_df[curve_df["run_id"].astype(str) == rid].dropna(subset=["x", "value"])
        if sub.empty:
            continue
        sub = sub.sort_values("x", kind="mergesort")
        run_name = str(sub["name"].iloc[0]) if "name" in sub.columns else rid
        color = palette_map.get(rid, QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)])
        y_vals = apply_ema(sub["value"], smoothing)
        fig.add_trace(
            go.Scatter(
                x=sub["x"].tolist(),
                y=y_vals,
                mode="lines+markers" if is_sparse_eval else "lines",
                name=run_name,
                line=dict(color=color, width=2),
                marker=dict(size=5) if is_sparse_eval else None,
            )
        )

    title = f"{metric} (EMA {smoothing:.2f})" if smoothing > 0 else metric
    return _apply_dark_layout(
        fig,
        title=title,
        xaxis_title=_axis_label(axis),
        yaxis_title=metric,
        yaxis_tickformat=".2e" if metric == "train/lr" else None,
        empty_message=None if fig.data else f"No data available for {metric}",
    )


def build_metric_chart(
    runs_metrics: Mapping[str, pd.DataFrame],
    metric: str,
    axis: str = "step",
    smoothing: float = 0.0,
    run_names: Mapping[str, str] | None = None,
    color_map: Mapping[str, str] | None = None,
) -> go.Figure:
    """Build a multi-run comparison chart directly from ``{run_id: metrics_df}``."""
    axis = _validate_axis(axis)
    frames: list[pd.DataFrame] = []
    run_ids = list(runs_metrics.keys())
    for rid in run_ids:
        df = runs_metrics[rid]
        if df is None or df.empty or axis not in df.columns or metric not in df.columns:
            continue
        sub = df[[axis, metric]].dropna()
        if sub.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "run_id": rid,
                    "name": (run_names or {}).get(rid, rid),
                    "x": sub[axis].to_numpy(),
                    "value": sub[metric].to_numpy(),
                }
            )
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["run_id", "name", "x", "value"])
    )
    resolved_colors = dict(color_map) if color_map is not None else color_for_runs(run_ids)
    return build_comparison_chart(
        combined,
        metric=metric,
        axis=axis,
        smoothing=smoothing,
        run_order=run_ids,
        color_map=resolved_colors,
    )
