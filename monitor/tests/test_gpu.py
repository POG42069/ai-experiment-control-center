"""Unit tests for experiment_monitor.gpu."""

from __future__ import annotations

import sys
import types

import pytest

from experiment_monitor.gpu import gpu_snapshot, gpu_static_info


def test_gpu_snapshot_without_gpu() -> None:
    snap = gpu_snapshot()
    assert isinstance(snap, dict)
    static = gpu_static_info()
    assert isinstance(static, dict)


def test_gpu_helpers_with_mock_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    gb = 1024**3

    class _Props:
        total_memory = 16 * gb

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda idx=0: "Tesla T4",
        get_device_properties=lambda idx=0: _Props(),
        memory_allocated=lambda: int(2.5 * gb),
        memory_reserved=lambda: int(4.0 * gb),
        max_memory_allocated=lambda: int(3.0 * gb),
        utilization=lambda: 78,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    static = gpu_static_info()
    assert static == {
        "ecc/gpu_count": 2,
        "ecc/gpu_name": "Tesla T4",
        "ecc/gpu_mem_total_gb": 16.0,
    }

    snap = gpu_snapshot()
    assert snap["gpu/mem_alloc_gb"] == 2.5
    assert snap["gpu/mem_reserved_gb"] == 4.0
    assert snap["gpu/max_mem_alloc_gb"] == 3.0
    assert snap["gpu/util_pct"] == 78.0


def test_gpu_snapshot_utilization_failure_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    gb = 1024**3

    def _boom() -> int:
        raise RuntimeError("pynvml not available")

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: gb,
        memory_reserved=lambda: 2 * gb,
        max_memory_allocated=lambda: gb,
        utilization=_boom,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    snap = gpu_snapshot()
    assert snap["gpu/mem_alloc_gb"] == 1.0
    assert "gpu/util_pct" not in snap
