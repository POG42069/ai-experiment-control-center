"""Configuration loader for ecc_config.toml (no secrets allowed)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from ecc.status import DEFAULT_OFFLINE_AFTER_S, DEFAULT_STALE_AFTER_S


class ConfigError(ValueError):
    """Raised when ecc_config.toml is invalid or contains forbidden secret-like keys."""


_FORBIDDEN_KEY_SUBSTRINGS = ("key", "token", "password", "secret")
_VALID_DEMO_MODES = {"auto", "on", "off"}
_VALID_PLATFORMS = {"kaggle", "colab", "wandb"}
DEFAULT_ALLOWED_ACCELERATORS: tuple[str, ...] = ("NvidiaTeslaT4", "TpuV5E8", "none")


def default_db_path() -> Path:
    """Return %LOCALAPPDATA%\\ai-ecc\\ecc.db if set, else ~/.ai-ecc/ecc.db."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "ai-ecc" / "ecc.db"
    return Path.home() / ".ai-ecc" / "ecc.db"


@dataclass
class AccountConfig:
    nickname: str
    platform: str
    username: str = ""


@dataclass
class GeneralSettings:
    refresh_seconds: int = 15
    stale_after_s: int = DEFAULT_STALE_AFTER_S
    offline_after_s: int = DEFAULT_OFFLINE_AFTER_S
    demo_mode: str = "auto"
    db_path: Path = field(default_factory=default_db_path)
    kaggle_lookback_hours: int = 72


@dataclass
class WandbSettings:
    entity: str = ""
    projects: list[str] = field(default_factory=list)


@dataclass
class LaunchSettings:
    enabled: bool = False
    runner_slug: str = "ecc-runner"
    default_accelerator: str = "NvidiaTeslaT4"
    allowed_accelerators: list[str] = field(
        default_factory=lambda: list(DEFAULT_ALLOWED_ACCELERATORS)
    )
    max_timeout_hours: int = 12
    wandb_credential_label: str = "WANDB_API_KEY"
    github_credential_label: str = "GITHUB_TOKEN"
    default_wandb_project: str = ""
    # Private Kaggle dataset holding ecc_credentials.json (Kaggle Secrets are not delivered to
    # API-pushed versions). Placeholders {owner}/{account}; empty string disables it.
    credentials_dataset: str = ""

    @property
    def wandb_secret_label(self) -> str:
        return self.wandb_credential_label


@dataclass
class Settings:
    general: GeneralSettings = field(default_factory=GeneralSettings)
    wandb: WandbSettings = field(default_factory=WandbSettings)
    launch: LaunchSettings = field(default_factory=LaunchSettings)
    accounts: list[AccountConfig] = field(default_factory=list)
    config_path: Path | None = None

    @property
    def refresh_seconds(self) -> int:
        return self.general.refresh_seconds

    @property
    def stale_after_s(self) -> int:
        return self.general.stale_after_s

    @property
    def offline_after_s(self) -> int:
        return self.general.offline_after_s

    @property
    def demo_mode(self) -> str:
        return self.general.demo_mode

    @property
    def db_path(self) -> Path:
        return self.general.db_path

    @property
    def kaggle_lookback_hours(self) -> int:
        return self.general.kaggle_lookback_hours


def _check_no_secret_keys(obj: Any, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_str = str(k)
            full_path = f"{prefix}.{key_str}" if prefix else key_str
            lower_key = key_str.lower()
            for forbidden in _FORBIDDEN_KEY_SUBSTRINGS:
                if forbidden in lower_key:
                    raise ConfigError(
                        f"Forbidden secret-like config key '{full_path}'. "
                        "Store secrets in keyring (service 'ai-ecc') or environment variables, "
                        "never in ecc_config.toml."
                    )
            _check_no_secret_keys(v, full_path)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _check_no_secret_keys(item, f"{prefix}[{idx}]")


def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path is not None:
        return Path(config_path)

    env_path = os.environ.get("ECC_CONFIG")
    if env_path:
        return Path(env_path)

    cwd_candidate = Path.cwd() / "ecc_config.toml"
    if cwd_candidate.is_file():
        return cwd_candidate

    repo_root_candidate = Path(__file__).resolve().parent.parent / "ecc_config.toml"
    if repo_root_candidate.is_file():
        return repo_root_candidate

    return None


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Config '{name}' must be an integer, got {type(value).__name__}")
    return value


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load ECC settings from ecc_config.toml (or defaults if missing)."""
    resolved = _resolve_config_path(config_path)
    if resolved is None or not resolved.is_file():
        return Settings(config_path=None)

    try:
        raw_bytes = resolved.read_bytes()
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ConfigError(f"Failed to parse config file {resolved}: {exc}") from exc

    _check_no_secret_keys(data)

    # [general]
    gen_raw = data.get("general", {})
    if not isinstance(gen_raw, dict):
        raise ConfigError("[general] must be a TOML table")

    refresh_seconds = _parse_int(gen_raw.get("refresh_seconds", 15), "general.refresh_seconds")
    if not (10 <= refresh_seconds <= 300):
        raise ConfigError(
            f"general.refresh_seconds must be between 10 and 300, got {refresh_seconds}"
        )

    stale_after_s = _parse_int(
        gen_raw.get("stale_after_s", DEFAULT_STALE_AFTER_S), "general.stale_after_s"
    )
    offline_after_s = _parse_int(
        gen_raw.get("offline_after_s", DEFAULT_OFFLINE_AFTER_S), "general.offline_after_s"
    )
    if stale_after_s <= 0:
        raise ConfigError(f"general.stale_after_s must be > 0, got {stale_after_s}")
    if offline_after_s < stale_after_s:
        raise ConfigError(
            f"general.offline_after_s ({offline_after_s}) must be >= stale_after_s ({stale_after_s})"
        )

    demo_mode = gen_raw.get("demo_mode", "auto")
    if not isinstance(demo_mode, str) or demo_mode not in _VALID_DEMO_MODES:
        raise ConfigError(
            f"general.demo_mode must be one of {sorted(_VALID_DEMO_MODES)}, got {demo_mode!r}"
        )

    db_path_raw = gen_raw.get("db_path", "")
    if not isinstance(db_path_raw, str):
        raise ConfigError("general.db_path must be a string")
    if db_path_raw.strip():
        db_path = Path(os.path.expandvars(db_path_raw.strip())).expanduser()
    else:
        db_path = default_db_path()

    kaggle_lookback_hours = _parse_int(
        gen_raw.get("kaggle_lookback_hours", 72), "general.kaggle_lookback_hours"
    )
    if kaggle_lookback_hours <= 0:
        raise ConfigError(
            f"general.kaggle_lookback_hours must be > 0, got {kaggle_lookback_hours}"
        )

    general = GeneralSettings(
        refresh_seconds=refresh_seconds,
        stale_after_s=stale_after_s,
        offline_after_s=offline_after_s,
        demo_mode=demo_mode,
        db_path=db_path,
        kaggle_lookback_hours=kaggle_lookback_hours,
    )

    # [wandb]
    wb_raw = data.get("wandb", {})
    if not isinstance(wb_raw, dict):
        raise ConfigError("[wandb] must be a TOML table")

    entity = wb_raw.get("entity", "")
    if not isinstance(entity, str):
        raise ConfigError("wandb.entity must be a string")

    projects_raw = wb_raw.get("projects", [])
    if not isinstance(projects_raw, list) or not all(isinstance(p, str) for p in projects_raw):
        raise ConfigError("wandb.projects must be a list of strings")
    projects = [p.strip() for p in projects_raw if p.strip()]

    wandb_settings = WandbSettings(entity=entity.strip(), projects=projects)

    # [launch]
    launch_raw = data.get("launch", {})
    if not isinstance(launch_raw, dict):
        raise ConfigError("[launch] must be a TOML table")

    launch_enabled = launch_raw.get("enabled", False)
    if not isinstance(launch_enabled, bool):
        raise ConfigError(
            f"launch.enabled must be a boolean, got {type(launch_enabled).__name__}"
        )

    runner_slug = launch_raw.get("runner_slug", "ecc-runner")
    if not isinstance(runner_slug, str) or len(runner_slug.strip()) < 5:
        raise ConfigError("launch.runner_slug must be a string of at least 5 characters")
    runner_slug = runner_slug.strip()

    allowed_acc_raw = launch_raw.get(
        "allowed_accelerators", list(DEFAULT_ALLOWED_ACCELERATORS)
    )
    if (
        not isinstance(allowed_acc_raw, list)
        or not allowed_acc_raw
        or not all(isinstance(a, str) and a.strip() for a in allowed_acc_raw)
    ):
        raise ConfigError("launch.allowed_accelerators must be a non-empty list of strings")
    allowed_accelerators = [a.strip() for a in allowed_acc_raw]

    default_accelerator = launch_raw.get("default_accelerator", "NvidiaTeslaT4")
    if not isinstance(default_accelerator, str) or not default_accelerator.strip():
        raise ConfigError("launch.default_accelerator must be a non-empty string")
    default_accelerator = default_accelerator.strip()
    if default_accelerator not in allowed_accelerators:
        raise ConfigError(
            f"launch.default_accelerator ({default_accelerator!r}) must be in "
            f"launch.allowed_accelerators ({allowed_accelerators!r})"
        )

    max_timeout_hours = _parse_int(
        launch_raw.get("max_timeout_hours", 12), "launch.max_timeout_hours"
    )
    if max_timeout_hours <= 0:
        raise ConfigError(
            f"launch.max_timeout_hours must be > 0, got {max_timeout_hours}"
        )

    wandb_cred_label = launch_raw.get("wandb_credential_label", "WANDB_API_KEY")
    if not isinstance(wandb_cred_label, str) or not wandb_cred_label.strip():
        raise ConfigError("launch.wandb_credential_label must be a non-empty string")

    github_cred_label = launch_raw.get("github_credential_label", "GITHUB_TOKEN")
    if not isinstance(github_cred_label, str):
        raise ConfigError("launch.github_credential_label must be a string")

    default_wandb_project = launch_raw.get("default_wandb_project", "")
    if not isinstance(default_wandb_project, str):
        raise ConfigError("launch.default_wandb_project must be a string")

    credentials_dataset = launch_raw.get("credentials_dataset", "")
    if not isinstance(credentials_dataset, str):
        raise ConfigError("launch.credentials_dataset must be a string")

    launch_settings = LaunchSettings(
        credentials_dataset=credentials_dataset.strip(),
        enabled=launch_enabled,
        runner_slug=runner_slug,
        default_accelerator=default_accelerator,
        allowed_accelerators=allowed_accelerators,
        max_timeout_hours=max_timeout_hours,
        wandb_credential_label=wandb_cred_label.strip(),
        github_credential_label=github_cred_label.strip(),
        default_wandb_project=default_wandb_project.strip(),
    )

    # [[accounts]]
    accounts_raw = data.get("accounts", [])
    if not isinstance(accounts_raw, list):
        raise ConfigError("[[accounts]] must be an array of tables")

    accounts: list[AccountConfig] = []
    for idx, acc in enumerate(accounts_raw):
        if not isinstance(acc, dict):
            raise ConfigError(f"accounts[{idx}] must be a table")
        nickname = acc.get("nickname", "")
        platform = acc.get("platform", "")
        username = acc.get("username", "")
        if not isinstance(nickname, str) or not nickname.strip():
            raise ConfigError(f"accounts[{idx}].nickname must be a non-empty string")
        if not isinstance(platform, str) or platform not in _VALID_PLATFORMS:
            raise ConfigError(
                f"accounts[{idx}].platform must be one of {sorted(_VALID_PLATFORMS)}, got {platform!r}"
            )
        if not isinstance(username, str):
            raise ConfigError(f"accounts[{idx}].username must be a string")
        accounts.append(
            AccountConfig(
                nickname=nickname.strip(),
                platform=platform,
                username=username.strip(),
            )
        )

    return Settings(
        general=general,
        wandb=wandb_settings,
        launch=launch_settings,
        accounts=accounts,
        config_path=resolved,
    )
