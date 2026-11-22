"""Tests for ecc.accounts (fake keyring + fake runners, zero network calls)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import keyring
import keyring.backend
import pytest
from keyring.errors import PasswordDeleteError

from ecc.accounts import (
    PLATFORM_CAPABILITIES,
    PLATFORM_LIMITATIONS,
    AccountManager,
    validate_nickname,
)
from ecc.adapters.colab import COLAB_NOT_INTEGRATED_MSG
from ecc.adapters.kaggle import KaggleCLI
from ecc.config import AccountConfig, GeneralSettings, Settings, WandbSettings
from ecc.secrets import SecretStore
from ecc.store import Store


class _FakeKeyring(keyring.backend.KeyringBackend):
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
            raise PasswordDeleteError("missing")
        del self._data[key]


@pytest.fixture
def isolated_secrets() -> SecretStore:
    original = keyring.get_keyring()
    fake = _FakeKeyring()
    keyring.set_keyring(fake)
    try:
        yield SecretStore()
    finally:
        keyring.set_keyring(original)


class TestNicknameValidation:
    @pytest.mark.parametrize(
        "valid",
        ["k1", "kaggle-1", "colab_pro_2", "a0-b_c", "00", "a" + "b" * 31],
    )
    def test_valid_nicknames(self, valid: str) -> None:
        assert validate_nickname(valid) == valid

    @pytest.mark.parametrize(
        "invalid",
        [
            "",
            "a",  # too short (min 2 chars)
            "Kaggle-1",  # uppercase disallowed
            "-kaggle1",  # must start with alnum
            "_kaggle1",  # must start with alnum
            "kaggle 1",  # space disallowed
            "kaggle.1",  # dot disallowed
            "a" * 33,  # too long (max 32 chars)
        ],
    )
    def test_invalid_nicknames(self, invalid: str) -> None:
        with pytest.raises(ValueError):
            validate_nickname(invalid)


class TestAccountManager:
    def test_sqlite_never_contains_secret_bytes(
        self, tmp_path: Path, isolated_secrets: SecretStore
    ) -> None:
        db_path = tmp_path / "accounts_secret_check.db"
        store = Store(db_path)
        manager = AccountManager(store=store, secrets=isolated_secrets)

        secret_value = "KGAT_NEVER_WRITE_THIS_TO_SQLITE_999888777"
        status = manager.add_or_update(
            nickname="kaggle-1",
            platform="kaggle",
            username="alice",
            secret=secret_value,
        )
        assert status.has_secret is True
        assert status.capabilities == PLATFORM_CAPABILITIES["kaggle"]
        assert status.limitations == PLATFORM_LIMITATIONS["kaggle"]

        # Verify SQLite row stores only secret_ref, never the secret itself
        rows = store.list_accounts()
        assert len(rows) == 1
        assert rows[0]["secret_ref"] == "kaggle:kaggle-1"

        # Grep raw SQLite DB file bytes for the secret string
        raw_bytes = db_path.read_bytes()
        assert secret_value.encode("utf-8") not in raw_bytes

        # Remove deletes both keyring entry and SQLite row
        assert manager.remove("kaggle-1") is True
        assert isolated_secrets.has("kaggle", "kaggle-1") is False
        assert manager.list() == []
        assert manager.remove("kaggle-1") is False

    def test_config_sync_on_startup(
        self, temp_store: Store, isolated_secrets: SecretStore
    ) -> None:
        settings = Settings(
            general=GeneralSettings(demo_mode="off"),
            wandb=WandbSettings(),
            accounts=[
                AccountConfig(nickname="kaggle-1", platform="kaggle", username="alice"),
                AccountConfig(nickname="colab-1", platform="colab", username="bob"),
            ],
        )
        manager = AccountManager(store=temp_store, secrets=isolated_secrets, settings=settings)
        listed = manager.list()
        assert [a.nickname for a in listed] == ["colab-1", "kaggle-1"]
        assert listed[0].capabilities == []
        assert listed[0].limitations == PLATFORM_LIMITATIONS["colab"]
        assert listed[1].has_secret is False
        assert listed[1].capabilities == PLATFORM_CAPABILITIES["kaggle"]

    def test_check_kaggle_wandb_colab_and_check_all(
        self, temp_store: Store, isolated_secrets: SecretStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        calls: list[list[str]] = []

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(list(argv))
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="ref,title,author,lastRunTime,totalVotes\nalice/nb1,NB1,alice,2026-11-22 10:00:00,0\n",
                stderr="",
            )

        def fake_kaggle_factory(nick: str, getter: Any) -> KaggleCLI:
            return KaggleCLI(nick, token_getter=getter, kaggle_exe="fake-kaggle", runner=fake_runner)

        def fake_wandb_factory(api_key: str) -> Any:
            assert api_key == "wandb_v1_fakekey1234567890abcdef"
            return SimpleNamespace(
                viewer=SimpleNamespace(username="researcher"),
                default_entity="researcher",
            )

        manager = AccountManager(
            store=temp_store,
            secrets=isolated_secrets,
            kaggle_cli_factory=fake_kaggle_factory,
            wandb_api_factory=fake_wandb_factory,
        )
        manager.add_or_update("kaggle-1", "kaggle", "alice", secret="KGAT_validtoken123456")
        manager.add_or_update("wandb-1", "wandb", "researcher", secret="wandb_v1_fakekey1234567890abcdef")
        manager.add_or_update("colab-1", "colab", "colab-user", secret=None)

        results = {st.nickname: st for st in manager.check_all()}

        assert results["kaggle-1"].last_check_ok is True
        assert any(
            c[1:] == ["kernels", "list", "--mine", "--page-size", "1"] for c in calls
        )

        assert results["wandb-1"].last_check_ok is True
        assert "researcher" in (results["wandb-1"].last_check_msg or "")

        assert results["colab-1"].last_check_ok is False
        assert results["colab-1"].last_check_msg == COLAB_NOT_INTEGRATED_MSG
        assert results["colab-1"].capabilities == []

    def test_check_redacts_secret_on_failure(
        self, temp_store: Store, isolated_secrets: SecretStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        secret = "KGAT_leaked_in_stderr_99999"

        def failing_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=1,
                stdout="",
                stderr=f"401 Unauthorized for token {secret}",
            )

        manager = AccountManager(
            store=temp_store,
            secrets=isolated_secrets,
            kaggle_cli_factory=lambda n, g: KaggleCLI(
                n, token_getter=g, kaggle_exe="fake-kaggle", runner=failing_runner
            ),
        )
        manager.add_or_update("kaggle-bad", "kaggle", "bob", secret=secret)
        st = manager.check("kaggle-bad")
        assert st.last_check_ok is False
        assert st.last_check_msg is not None
        assert secret not in st.last_check_msg
        assert "[REDACTED]" in st.last_check_msg
