"""Unit tests for pure Plotly chart builders and EMA display smoothing."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from app.ui.charts import (
    QUALITATIVE_PALETTE,
    apply_ema,
    build_comparison_chart,
    build_eval_metrics_chart,
    build_gpu_memory_chart,
    build_loss_chart,
    build_lr_chart,
    build_metric_chart,
    color_for_runs,
)
from ecc.service import Service


def test_apply_ema_identity_and_small_series() -> None:
    series = [1.0, 2.0, 3.0, 4.0]
    assert apply_ema(series, 0.0) == [1.0, 2.0, 3.0, 4.0]
    assert apply_ema(series, -0.5) == [1.0, 2.0, 3.0, 4.0]
    assert apply_ema([], 0.5) == []

    # Exact recurrence: ema_0 = 1.0, ema_1 = 0.5*1 + 0.5*2 = 1.5,
    # ema_2 = 0.5*1.5 + 0.5*3 = 2.25, ema_3 = 0.5*2.25 + 0.5*4 = 3.125
    smoothed = apply_ema(series, 0.5)
    assert smoothed == pytest.approx([1.0, 1.5, 2.25, 3.125], rel=1e-9)


def test_apply_ema_preserves_nans() -> None:
    series = [1.0, float("nan"), 3.0]
    out = apply_ema(series, 0.5)
    assert out[0] == pytest.approx(1.0)
    assert math.isnan(out[1])
    assert out[2] == pytest.approx(2.0)


def test_comparison_chart_one_trace_per_run_and_axis_switch(
    demo_service: Service,
) -> None:
    run_ids = [
        "demo:demo/demo-shared-task/run-a",
        "demo:demo/demo-shared-task/run-b",
        "demo:demo/demo-shared-task/run-c",
    ]
    colors = color_for_runs(run_ids)
    assert len(colors) == 3
    assert colors[run_ids[0]] == QUALITATIVE_PALETTE[0]
    assert colors[run_ids[1]] == QUALITATIVE_PALETTE[1]
    assert colors[run_ids[2]] == QUALITATIVE_PALETTE[2]

    cmp_step = demo_service.compare_runs(run_ids, metric="eval/macro_f1", axis="step")
    curve_step = cmp_step["curves"]["eval/macro_f1"]
    orig_values = curve_step["value"].tolist()

    fig_step = build_comparison_chart(
        curve_step,
        metric="eval/macro_f1",
        axis="step",
        smoothing=0.0,
        run_order=run_ids,
        color_map=colors,
    )
    assert isinstance(fig_step, go.Figure)
    assert len(fig_step.data) == 3
    assert fig_step.layout.xaxis.title.text == "Step"
    assert [trace.line.color for trace in fig_step.data] == [
        colors[rid] for rid in run_ids
    ]

    # Smoothing changes displayed y values without mutating input DataFrame
    fig_smooth = build_comparison_chart(
        curve_step,
        metric="eval/macro_f1",
        axis="step",
        smoothing=0.6,
        run_order=run_ids,
        color_map=colors,
    )
    assert isinstance(fig_smooth, go.Figure)
    assert len(fig_smooth.data) == 3
    assert curve_step["value"].tolist() == orig_values
    assert list(fig_smooth.data[0].y) != list(fig_step.data[0].y)

    # Axis switch to epoch updates x-axis title and x values
    cmp_epoch = demo_service.compare_runs(run_ids, metric="eval/macro_f1", axis="epoch")
    fig_epoch = build_comparison_chart(
        cmp_epoch["curves"]["eval/macro_f1"],
        metric="eval/macro_f1",
        axis="epoch",
        smoothing=0.0,
        run_order=run_ids,
        color_map=colors,
    )
    assert isinstance(fig_epoch, go.Figure)
    assert len(fig_epoch.data) == 3
    assert fig_epoch.layout.xaxis.title.text == "Epoch"
    assert max(fig_epoch.data[0].x) < max(fig_step.data[0].x)


def test_build_metric_chart_and_axis_validation() -> None:
    df1 = pd.DataFrame(
        {"step": [10, 20, 30], "epoch": [0.1, 0.2, 0.3], "train/loss": [2.0, 1.5, 1.0]}
    )
    df2 = pd.DataFrame(
        {"step": [10, 20, 30], "epoch": [0.1, 0.2, 0.3], "train/loss": [1.8, 1.2, 0.9]}
    )

    fig_step = build_metric_chart(
        {"r1": df1, "r2": df2},
        metric="train/loss",
        axis="step",
        run_names={"r1": "Run 1", "r2": "Run 2"},
    )
    assert isinstance(fig_step, go.Figure)
    assert len(fig_step.data) == 2
    assert list(fig_step.data[0].x) == [10, 20, 30]
    assert fig_step.layout.xaxis.title.text == "Step"

    fig_epoch = build_metric_chart(
        {"r1": df1, "r2": df2},
        metric="train/loss",
        axis="epoch",
        run_names={"r1": "Run 1", "r2": "Run 2"},
    )
    assert isinstance(fig_epoch, go.Figure)
    assert len(fig_epoch.data) == 2
    assert list(fig_epoch.data[0].x) == pytest.approx([0.1, 0.2, 0.3])
    assert fig_epoch.layout.xaxis.title.text == "Epoch"

    with pytest.raises(ValueError, match="axis"):
        build_metric_chart({"r1": df1}, metric="train/loss", axis="time")


def test_single_run_charts_and_empty_inputs(demo_service: Service) -> None:
    df = demo_service.get_run_metrics("demo:demo/demo-shared-task/run-a")

    loss_fig = build_loss_chart(df, axis="step")
    eval_fig = build_eval_metrics_chart(df, axis="epoch")
    lr_fig = build_lr_chart(df, axis="step")
    gpu_fig = build_gpu_memory_chart(df, axis="step", total_mem_gb=15.0)

    for fig in (loss_fig, eval_fig, lr_fig, gpu_fig):
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    assert len(loss_fig.data) == 2
    assert len(eval_fig.data) == 2
    assert eval_fig.layout.xaxis.title.text == "Epoch"

    # Empty frames should never raise
    empty = pd.DataFrame(columns=["step", "epoch"])
    for fig in (
        build_loss_chart(empty),
        build_eval_metrics_chart(empty),
        build_lr_chart(empty),
        build_gpu_memory_chart(empty),
        build_comparison_chart(pd.DataFrame(), metric="eval/macro_f1"),
    ):
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
