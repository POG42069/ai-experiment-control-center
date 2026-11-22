"""Backend data models for AI Experiment Control Center (ARCHITECTURE §4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    STALE = "STALE"
    OFFLINE = "OFFLINE"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CRASHED = "CRASHED"
    KILLED = "KILLED"
    UNKNOWN = "UNKNOWN"


def to_jsonable(value: Any) -> Any:
    """Recursively convert values into JSON-safe Python primitives.

    - Datetimes -> ISO-8601 UTC strings
    - Enums -> enum.value
    - float / numpy floating NaN or Inf -> None
    - numpy integers/booleans/arrays -> native Python types
    - objects with to_dict() -> dict
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    # Handle numpy/pandas scalar types without importing numpy/pandas eagerly.
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    # Handle pandas NA / NaT gracefully.
    cls_name = type(value).__name__
    if cls_name in ("NAType", "NaTType"):
        return None
    return str(value)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}


@dataclass
class RunSummary:
    run_id: str
    source: str
    project: str
    name: str
    status: RunStatus
    raw_state: str | None
    platform: str | None
    account: str | None
    gpu: str | None
    model_name: str | None
    config: dict[str, Any]
    tags: list[str]
    step: int | None
    total_steps: int | None
    epoch: float | None
    total_epochs: float | None
    progress: float | None
    latest: dict[str, float]
    best_metric_name: str | None
    best_metric: float | None
    best_step: int | None
    best_epoch: float | None
    started_at: datetime | None
    ended_at: datetime | None
    last_update: datetime | None
    duration_s: float | None
    eta_s: float | None
    notebook_url: str | None
    source_url: str | None
    git_branch: str | None
    git_sha: str | None
    dataset_version: str | None
    seed: int | None
    is_demo: bool

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass
class LogLine:
    ts: datetime | None
    stream: str  # "stdout" | "stderr" | "kaggle" | "system"
    text: str
    level: str  # "INFO" | "WARNING" | "ERROR"

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass
class Project:
    slug: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    baseline_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)
