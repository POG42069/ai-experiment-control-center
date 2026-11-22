"""Run status derivation, ETA, and progress helpers (ARCHITECTURE §5)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ecc.models import RunStatus

DEFAULT_STALE_AFTER_S = 180
DEFAULT_OFFLINE_AFTER_S = 900

_EXACT_STATE_MAP: dict[str, RunStatus] = {
    # W&B states
    "finished": RunStatus.FINISHED,
    "failed": RunStatus.FAILED,
    "crashed": RunStatus.CRASHED,
    "killed": RunStatus.KILLED,
    "preempted": RunStatus.KILLED,
    "pending": RunStatus.QUEUED,
    # Kaggle states (lowercased)
    "queued": RunStatus.QUEUED,
    "new_script": RunStatus.QUEUED,
    "complete": RunStatus.FINISHED,
    "error": RunStatus.FAILED,
}


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def derive_status(
    raw_state: str | None,
    last_update: datetime | None,
    now: datetime,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    offline_after_s: float = DEFAULT_OFFLINE_AFTER_S,
) -> RunStatus:
    """Derive canonical RunStatus from a source state and heartbeat timestamp.

    Never returns RUNNING without a fresh heartbeat.
    """
    if not raw_state or not isinstance(raw_state, str):
        return RunStatus.UNKNOWN

    norm = raw_state.strip().lower()
    if not norm:
        return RunStatus.UNKNOWN

    if norm in _EXACT_STATE_MAP:
        return _EXACT_STATE_MAP[norm]

    if norm.startswith("cancel_"):
        return RunStatus.KILLED

    if norm == "running":
        if last_update is None:
            return RunStatus.UNKNOWN
        age_s = (_to_utc(now) - _to_utc(last_update)).total_seconds()
        if age_s <= stale_after_s:
            return RunStatus.RUNNING
        if age_s <= offline_after_s:
            return RunStatus.STALE
        return RunStatus.OFFLINE

    return RunStatus.UNKNOWN


def compute_eta(
    step: int | None,
    total_steps: int | None,
    started_at: datetime | None,
    last_update: datetime | None,
) -> float | None:
    """Linear ETA estimate in seconds: elapsed / step * (total_steps - step).

    Returns None if insufficient/invalid data or step <= 0.
    Returns 0.0 if step >= total_steps.
    """
    if (
        step is None
        or total_steps is None
        or started_at is None
        or last_update is None
    ):
        return None
    if step <= 0 or total_steps <= 0:
        return None
    if step >= total_steps:
        return 0.0

    elapsed_s = (_to_utc(last_update) - _to_utc(started_at)).total_seconds()
    if elapsed_s <= 0:
        return None

    eta = (elapsed_s / float(step)) * float(total_steps - step)
    if not math.isfinite(eta):
        return None
    return eta


def compute_progress(
    step: int | None,
    total_steps: int | None,
    explicit: float | None = None,
) -> float | None:
    """Compute progress ratio clamped to [0.0, 1.0]."""
    if explicit is not None:
        try:
            val = float(explicit)
        except (TypeError, ValueError):
            val = float("nan")
        if math.isfinite(val):
            return max(0.0, min(1.0, val))

    if step is not None and total_steps is not None and total_steps > 0:
        val = float(step) / float(total_steps)
        if math.isfinite(val):
            return max(0.0, min(1.0, val))

    return None
