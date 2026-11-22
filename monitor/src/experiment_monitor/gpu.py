"""Optional GPU telemetry helpers (never raises when torch/CUDA is absent)."""

from __future__ import annotations

from typing import Any

_BYTES_PER_GB = 1024**3


def _get_cuda() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda
    except Exception:
        return None


def gpu_static_info() -> dict[str, Any]:
    """Return static GPU config keys (`ecc/gpu_*`) or ``{}`` when no GPU is available."""
    cuda = _get_cuda()
    if cuda is None:
        return {}
    info: dict[str, Any] = {}
    try:
        info["ecc/gpu_count"] = int(cuda.device_count())
    except Exception:
        pass
    try:
        info["ecc/gpu_name"] = str(cuda.get_device_name(0))
    except Exception:
        pass
    try:
        props = cuda.get_device_properties(0)
        info["ecc/gpu_mem_total_gb"] = round(float(props.total_memory) / _BYTES_PER_GB, 2)
    except Exception:
        pass
    return info


def gpu_snapshot() -> dict[str, float]:
    """Return dynamic GPU metrics (`gpu/*`) or ``{}`` when torch/CUDA is unavailable."""
    cuda = _get_cuda()
    if cuda is None:
        return {}
    metrics: dict[str, float] = {}
    try:
        metrics["gpu/mem_alloc_gb"] = round(float(cuda.memory_allocated()) / _BYTES_PER_GB, 4)
        metrics["gpu/mem_reserved_gb"] = round(float(cuda.memory_reserved()) / _BYTES_PER_GB, 4)
        metrics["gpu/max_mem_alloc_gb"] = round(float(cuda.max_memory_allocated()) / _BYTES_PER_GB, 4)
    except Exception:
        pass
    try:
        util = cuda.utilization()
        if util is not None:
            metrics["gpu/util_pct"] = float(util)
    except Exception:
        pass
    return metrics
