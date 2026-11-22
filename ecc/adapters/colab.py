"""Google Colab capability stub for ECC MVP (ARCHITECTURE §1).

MVP Decision
------------
Native Google Colab API integration is intentionally disabled in the MVP.
Colab runs are monitored through ``experiment_monitor`` / W&B instrumentation
and stored notebook URLs only.

Optional Future Adapter Design (Phase 3 — Read-Only)
----------------------------------------------------
If a native Colab runtime-inventory adapter is implemented in a future phase,
it must follow these strict multi-account and security boundaries:

1. **Isolated per-account Google OAuth**:
   Each Google / Colab Pro account uses its own OAuth token directory under
   ``%LOCALAPPDATA%\\ai-ecc\\google\\<nickname>\\`` so multiple Colab Pro
   accounts can be queried without switching browser or CLI logins.
2. **Strict read-only endpoint allow-list**:
   Only ``GET https://colaboratory.googleapis.com/v1beta/runtimes`` (runtime
   name + accelerator spec such as T4/L4/A100) and
   ``GET https://colaboratory.googleapis.com/v1beta/subscription`` may be
   called.
3. **No mutating or destructive operations**:
   Never call create-runtime, delete-runtime, assign/unassign, or
   ``execute_code`` endpoints.
4. **Honest capability reporting**:
   Even with the future adapter, Colab's API does not expose notebook
   execution status, console logs, or training progress; those remain
   provided exclusively by W&B instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ecc.models import LogLine, RunSummary, to_jsonable

COLAB_NOT_INTEGRATED_MSG = (
    "Native Colab API not integrated in MVP — monitored via W&B instrumentation"
)
COLAB_CAPABILITIES: tuple[str, ...] = ()
COLAB_LIMITATIONS: tuple[str, ...] = (
    "no official API for notebook execution status/logs",
    "runtime list API needs Google Cloud OAuth and is not enabled in MVP",
)


@dataclass(frozen=True)
class ColabCapability:
    """Explicit description of what ECC does and does not support for Colab."""

    platform: str = "colab"
    native_integrated: bool = False
    capabilities: list[str] = field(default_factory=lambda: list(COLAB_CAPABILITIES))
    limitations: list[str] = field(default_factory=lambda: list(COLAB_LIMITATIONS))
    status_message: str = COLAB_NOT_INTEGRATED_MSG

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "native_integrated": self.native_integrated,
            "capabilities": to_jsonable(self.capabilities),
            "limitations": to_jsonable(self.limitations),
            "status_message": self.status_message,
        }


COLAB_CAPABILITY = ColabCapability()


class ColabSource:
    """Read-only RunSource stub for Google Colab (always empty in MVP)."""

    name: str = "colab"
    capability: ColabCapability = COLAB_CAPABILITY

    def list_runs(self, project: str | None = None) -> list[RunSummary]:
        return []

    def get_run(self, run_id: str) -> RunSummary | None:
        return None

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["step", "epoch"])

    def get_logs(self, run_id: str, tail: int = 500) -> list[LogLine]:
        return []

    def list_projects(self) -> list[str]:
        return []

    def health(self) -> tuple[bool, str]:
        return False, COLAB_NOT_INTEGRATED_MSG
