"""RunSource protocol and shared log-level classification."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ecc.models import LogLine, RunSummary

_ERROR_TOKENS = (
    "Traceback",
    "Error",
    "Exception",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ERROR",
)
_WARNING_TOKENS = ("Warning", "WARN")


def classify_log_level(text: str) -> str:
    """Derive log level ('ERROR' | 'WARNING' | 'INFO') from log line text.

    Note: stderr defaults to INFO unless error/warning patterns match.
    """
    for token in _ERROR_TOKENS:
        if token in text:
            return "ERROR"
    for token in _WARNING_TOKENS:
        if token in text:
            return "WARNING"
    return "INFO"


@runtime_checkable
class RunSource(Protocol):
    """Read-only interface for any run source (W&B, Demo, Kaggle, ...)."""

    name: str

    def list_runs(self, project: str | None = None) -> list[RunSummary]: ...

    def get_run(self, run_id: str) -> RunSummary | None: ...

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame: ...

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]: ...

    def list_projects(self) -> list[str]: ...

    def health(self) -> tuple[bool, str]: ...
