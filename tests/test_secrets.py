"""Tests for ecc.secrets (using an in-memory fake keyring backend only)."""

from __future__ import annotations

import keyring
import keyring.backend
import pytest
from keyring.errors import PasswordDeleteError

from ecc.secrets import SERVICE_NAME, SecretStore, env_var_name, secret_key


class InMemoryKeyring(keyring.backend.KeyringBackend):
    """Fake in-memory keyring backend so tests never touch Windows Credential Manager."""

    priority = 10

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._data.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._data[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self._data:
            raise PasswordDeleteError(f"No password for {service}/{username}")
        del self._data[key]


@pytest.fixture
def fake_keyring() -> InMemoryKeyring:
    original = keyring.get_keyring()
    fake = InMemoryKeyring()
    keyring.set_keyring(fake)
    try:
        yield fake
    finally:
        keyring.set_keyring(original)


class TestSecretStore:
    def test_set_get_has_delete_via_global_fake_keyring(
        self, fake_keyring: InMemoryKeyring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ECC_SECRET_KAGGLE_KAGGLE_1", raising=False)
        store = SecretStore()

        assert store.has("kaggle", "kaggle-1") is False
        assert store.get("kaggle", "kaggle-1") is None

        ref = store.set("kaggle", "kaggle-1", "KGAT_testsecret123456")
        assert ref == "kaggle:kaggle-1"
        assert fake_keyring.get_password(SERVICE_NAME, "kaggle:kaggle-1") == "KGAT_testsecret123456"
        assert store.has("kaggle", "kaggle-1") is True
        assert store.get("kaggle", "kaggle-1") == "KGAT_testsecret123456"

        assert store.delete("kaggle", "kaggle-1") is True
        assert store.has("kaggle", "kaggle-1") is False
        assert store.get("kaggle", "kaggle-1") is None

        # Deleting a non-existent entry is idempotent and returns False
        assert store.delete("kaggle", "kaggle-1") is False

    def test_direct_backend_injection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ECC_SECRET_WANDB_MAIN_WB", raising=False)
        backend = InMemoryKeyring()
        store = SecretStore(backend=backend)
        store.set("wandb", "main-wb", "wandb_v1_secretvalue1234567890")
        assert store.get("wandb", "main-wb") == "wandb_v1_secretvalue1234567890"
        assert store.delete("wandb", "main-wb") is True
        assert store.get("wandb", "main-wb") is None

    def test_env_var_fallback_and_precedence(
        self, fake_keyring: InMemoryKeyring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert env_var_name("kaggle", "kaggle-1") == "ECC_SECRET_KAGGLE_KAGGLE_1"
        assert env_var_name("wandb", "team.acc-2") == "ECC_SECRET_WANDB_TEAM_ACC_2"
        assert secret_key("Kaggle", "kaggle-1") == "kaggle:kaggle-1"

        monkeypatch.setenv("ECC_SECRET_KAGGLE_KAGGLE_1", "KGAT_from_env_123456")
        store = SecretStore()

        # Falls back to env when keyring is empty
        assert store.has("kaggle", "kaggle-1") is True
        assert store.get("kaggle", "kaggle-1") == "KGAT_from_env_123456"

        # Keyring takes priority over env var when set
        store.set("kaggle", "kaggle-1", "KGAT_from_keyring_999")
        assert store.get("kaggle", "kaggle-1") == "KGAT_from_keyring_999"

        # After deleting from keyring, falls back to env again
        store.delete("kaggle", "kaggle-1")
        assert store.get("kaggle", "kaggle-1") == "KGAT_from_env_123456"

    def test_repr_never_leaks_secret_values(self, fake_keyring: InMemoryKeyring) -> None:
        secret_val = "KGAT_ultra_sensitive_token_987654321"
        store = SecretStore(backend=fake_keyring)
        store.set("kaggle", "kaggle-1", secret_val)

        r = repr(store)
        s = str(store)
        assert secret_val not in r
        assert secret_val not in s
        assert "SecretStore" in r

    def test_rejects_empty_secret(self, fake_keyring: InMemoryKeyring) -> None:
        store = SecretStore()
        with pytest.raises(ValueError):
            store.set("kaggle", "kaggle-1", "   ")
