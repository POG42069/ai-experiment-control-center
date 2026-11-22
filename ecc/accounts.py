"""Connected account management (Kaggle, Colab, W&B) with keyring-backed secrets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ecc.adapters.colab import (
    COLAB_CAPABILITIES,
    COLAB_LIMITATIONS,
    COLAB_NOT_INTEGRATED_MSG,
)
from ecc.adapters.kaggle import KaggleCLI
from ecc.config import Settings
from ecc.models import to_jsonable
from ecc.redact import redact
from ecc.secrets import SecretStore
from ecc.store import Store

VALID_PLATFORMS: frozenset[str] = frozenset({"kaggle", "colab", "wandb"})
NICKNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")

PLATFORM_CAPABILITIES: dict[str, list[str]] = {
    "kaggle": [
        "list my kernels",
        "kernel status (QUEUED/RUNNING/COMPLETE/ERROR)",
        "execution logs (persisted; live via kaggle kernels logs -f)",
    ],
    "colab": list(COLAB_CAPABILITIES),
    "wandb": [
        "runs, metrics, config, summary, heartbeat, console log file",
    ],
}

PLATFORM_LIMITATIONS: dict[str, list[str]] = {
    "kaggle": [
        "no GPU telemetry",
        "no structured metrics (use experiment_monitor/W&B)",
    ],
    "colab": list(COLAB_LIMITATIONS),
    "wandb": [],
}


def validate_nickname(nickname: str) -> str:
    """Validate account nickname against ``^[a-z0-9][a-z0-9_-]{1,31}$``."""
    if not isinstance(nickname, str) or not NICKNAME_RE.match(nickname):
        raise ValueError(
            f"Invalid account nickname {nickname!r}: must match {NICKNAME_RE.pattern} "
            "(2-32 lowercase alphanumeric, '-', or '_' characters, starting with alphanumeric)."
        )
    return nickname


def validate_platform(platform: str) -> str:
    """Validate account platform is one of kaggle, colab, wandb."""
    if not isinstance(platform, str) or platform not in VALID_PLATFORMS:
        raise ValueError(
            f"Invalid platform {platform!r}: must be one of {sorted(VALID_PLATFORMS)}"
        )
    return platform


@dataclass
class AccountStatus:
    """Public non-sensitive status of a configured account."""

    nickname: str
    platform: str
    username: str
    enabled: bool
    has_secret: bool
    last_check_at: str | None
    last_check_ok: bool | None
    last_check_msg: str | None
    capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "platform": self.platform,
            "username": self.username,
            "enabled": self.enabled,
            "has_secret": self.has_secret,
            "last_check_at": to_jsonable(self.last_check_at),
            "last_check_ok": self.last_check_ok,
            "last_check_msg": self.last_check_msg,
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
        }


def _default_wandb_api_factory(api_key: str) -> Any:
    import wandb

    return wandb.Api(api_key=api_key, timeout=15)


class AccountManager:
    """Manages connected accounts across SQLite metadata and keyring secrets."""

    def __init__(
        self,
        store: Store,
        secrets: SecretStore,
        settings: Settings | None = None,
        *,
        kaggle_cli_factory: Callable[[str, Callable[[], str | None]], KaggleCLI] | None = None,
        wandb_api_factory: Callable[[str], Any] = _default_wandb_api_factory,
    ) -> None:
        self.store = store
        self.secrets = secrets
        self.settings = settings
        self._kaggle_cli_factory = kaggle_cli_factory or (
            lambda nick, getter: KaggleCLI(nick, token_getter=getter)
        )
        self._wandb_api_factory = wandb_api_factory

        if self.settings is not None and self.settings.accounts:
            self.sync_from_config()

    def sync_from_config(self) -> None:
        """Sync non-sensitive ``[[accounts]]`` entries from config into SQLite."""
        if self.settings is None or not self.settings.accounts:
            return
        existing = {row["nickname"]: row for row in self.store.list_accounts()}
        for acc in self.settings.accounts:
            validate_nickname(acc.nickname)
            validate_platform(acc.platform)
            prev = existing.get(acc.nickname)
            enabled = prev["enabled"] if prev is not None else True
            self.store.upsert_account(
                nickname=acc.nickname,
                platform=acc.platform,
                username=acc.username,
                secret_ref=self.secrets.key(acc.platform, acc.nickname),
                enabled=enabled,
            )

    def _row_to_status(self, row: dict[str, Any]) -> AccountStatus:
        platform = str(row["platform"])
        nickname = str(row["nickname"])
        return AccountStatus(
            nickname=nickname,
            platform=platform,
            username=str(row.get("username", "")),
            enabled=bool(row.get("enabled", True)),
            has_secret=self.secrets.has(platform, nickname),
            last_check_at=row.get("last_check_at"),
            last_check_ok=row.get("last_check_ok"),
            last_check_msg=row.get("last_check_msg"),
            capabilities=list(PLATFORM_CAPABILITIES.get(platform, [])),
            limitations=list(PLATFORM_LIMITATIONS.get(platform, [])),
        )

    def list(self) -> list[AccountStatus]:
        """List all configured accounts with capabilities, limitations, and secret status."""
        return [self._row_to_status(row) for row in self.store.list_accounts()]

    def get(self, nickname: str) -> AccountStatus | None:
        """Look up a single account by nickname."""
        for row in self.store.list_accounts():
            if row["nickname"] == nickname:
                return self._row_to_status(row)
        return None

    def add_or_update(
        self,
        nickname: str,
        platform: str,
        username: str = "",
        secret: str | None = None,
        enabled: bool = True,
    ) -> AccountStatus:
        """Create or update an account.

        Stores ``secret`` (if provided) exclusively in ``SecretStore`` (keyring)
        and saves only the non-sensitive ``secret_ref`` in SQLite.
        """
        validate_nickname(nickname)
        validate_platform(platform)
        secret_ref = self.secrets.key(platform, nickname)

        if secret is not None and secret != "":
            self.secrets.set(platform, nickname, secret)

        self.store.upsert_account(
            nickname=nickname,
            platform=platform,
            username=username.strip() if isinstance(username, str) else "",
            secret_ref=secret_ref,
            enabled=enabled,
        )
        status = self.get(nickname)
        assert status is not None
        return status

    def remove(self, nickname: str) -> bool:
        """Delete both the keyring secret entry and the SQLite account row."""
        existing = next(
            (row for row in self.store.list_accounts() if row["nickname"] == nickname),
            None,
        )
        if existing is not None:
            self.secrets.delete(existing["platform"], nickname)

        if hasattr(self.store, "delete_account"):
            return bool(self.store.delete_account(nickname))  # type: ignore[attr-defined]

        conn = self.store._connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM accounts WHERE nickname = ?", (nickname,))
                return cur.rowcount > 0
        finally:
            self.store._close(conn)

    def _check_kaggle(self, nickname: str) -> tuple[bool, str]:
        token = self.secrets.get("kaggle", nickname)
        if not token:
            return False, "No Kaggle API token found in keyring or environment"
        cli = self._kaggle_cli_factory(
            nickname, lambda n=nickname: self.secrets.get("kaggle", n)
        )
        try:
            return cli.check()
        except Exception as exc:
            msg = str(exc).replace(token, "[REDACTED]")
            return False, redact(f"{type(exc).__name__}: {msg}")

    def _check_wandb(self, nickname: str) -> tuple[bool, str]:
        api_key = self.secrets.get("wandb", nickname)
        if not api_key:
            return False, "No W&B API key found in keyring or environment"
        try:
            api = self._wandb_api_factory(api_key)
            viewer = getattr(api, "viewer", None)
            default_entity = getattr(api, "default_entity", None)
            identity = None
            if viewer is not None:
                identity = getattr(viewer, "username", None) or getattr(viewer, "entity", None)
            if not identity and default_entity:
                identity = str(default_entity)
            if identity:
                return True, f"W&B API authenticated (entity={identity})"
            return True, "W&B API authenticated"
        except Exception as exc:
            msg = str(exc).replace(api_key, "[REDACTED]")
            return False, redact(f"{type(exc).__name__}: {msg}")

    def check(self, nickname: str) -> AccountStatus:
        """Run a read-only credential/capability check and record the result in SQLite."""
        row = next(
            (r for r in self.store.list_accounts() if r["nickname"] == nickname),
            None,
        )
        if row is None:
            raise KeyError(f"Unknown account: {nickname}")

        platform = row["platform"]
        if platform == "colab":
            ok, msg = False, COLAB_NOT_INTEGRATED_MSG
        elif platform == "kaggle":
            ok, msg = self._check_kaggle(nickname)
        elif platform == "wandb":
            ok, msg = self._check_wandb(nickname)
        else:
            ok, msg = False, f"Unsupported platform: {platform}"

        self.store.record_account_check(nickname, ok=ok, msg=msg)
        updated = self.get(nickname)
        assert updated is not None
        return updated

    def check_all(self) -> list[AccountStatus]:
        """Check every account in the store and return updated statuses."""
        results: list[AccountStatus] = []
        for row in self.store.list_accounts():
            results.append(self.check(row["nickname"]))
        return results
