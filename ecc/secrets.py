"""Secure secret storage backed by OS keyring with environment variable fallback.

Secrets are stored under keyring service ``"ai-ecc"`` with username
``f"{platform}:{nickname}"``. Values are never logged or exposed in ``__repr__``.
"""

from __future__ import annotations

import os
import re
from typing import Any

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE_NAME = "ai-ecc"
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")


def secret_key(platform: str, nickname: str) -> str:
    """Return canonical keyring username ``f"{platform}:{nickname}"``."""
    plat = platform.strip().lower()
    nick = nickname.strip()
    if not plat or not nick:
        raise ValueError("platform and nickname must be non-empty")
    return f"{plat}:{nick}"


def env_var_name(platform: str, nickname: str) -> str:
    """Return fallback environment variable name ``ECC_SECRET_<PLATFORM>_<NICKNAME>``."""
    plat = _NON_ALNUM_RE.sub("_", platform.strip()).upper()
    nick = _NON_ALNUM_RE.sub("_", nickname.strip()).upper()
    if not plat or not nick:
        raise ValueError("platform and nickname must be non-empty")
    return f"ECC_SECRET_{plat}_{nick}"


class SecretStore:
    """Wrapper around ``keyring`` (service ``"ai-ecc"``) with env-var fallback."""

    def __init__(self, backend: Any = None, service: str = SERVICE_NAME) -> None:
        self._backend = backend
        self._service = service

    @property
    def service(self) -> str:
        return self._service

    def key(self, platform: str, nickname: str) -> str:
        return secret_key(platform, nickname)

    def env_var(self, platform: str, nickname: str) -> str:
        return env_var_name(platform, nickname)

    def set(self, platform: str, nickname: str, value: str) -> str:
        """Store a secret in keyring and return its non-sensitive ``secret_ref``."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Secret value must be a non-empty string")
        username = secret_key(platform, nickname)
        if self._backend is not None:
            self._backend.set_password(self._service, username, value)
        else:
            keyring.set_password(self._service, username, value)
        return username

    def get(self, platform: str, nickname: str) -> str | None:
        """Retrieve a secret from keyring, falling back to ``ECC_SECRET_*`` env var."""
        username = secret_key(platform, nickname)
        val: str | None = None
        try:
            if self._backend is not None:
                val = self._backend.get_password(self._service, username)
            else:
                val = keyring.get_password(self._service, username)
        except KeyringError:
            val = None

        if val:
            return val

        env_val = os.environ.get(env_var_name(platform, nickname))
        if env_val:
            return env_val
        return None

    def delete(self, platform: str, nickname: str) -> bool:
        """Delete a secret from keyring. Returns True if removed, False if absent."""
        username = secret_key(platform, nickname)
        try:
            if self._backend is not None:
                self._backend.delete_password(self._service, username)
            else:
                keyring.delete_password(self._service, username)
            return True
        except (PasswordDeleteError, KeyError):
            return False

    def has(self, platform: str, nickname: str) -> bool:
        """Return True if a non-empty secret is available in keyring or env."""
        return bool(self.get(platform, nickname))

    def __repr__(self) -> str:
        backend_name = (
            type(self._backend).__name__ if self._backend is not None else "keyring.default"
        )
        return f"SecretStore(service={self._service!r}, backend={backend_name!r})"
