"""Tests for ecc.config and ecc.store."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecc.config import ConfigError, load_config
from ecc.store import Store


class TestConfig:
    def test_example_config_parses_cleanly(self) -> None:
        example_path = Path(__file__).resolve().parent.parent / "ecc_config.example.toml"
        settings = load_config(example_path)
        assert settings.refresh_seconds == 15
        assert settings.stale_after_s == 180
        assert settings.offline_after_s == 900
        assert settings.demo_mode == "auto"
        assert settings.wandb.entity == ""
        assert settings.wandb.projects == []

    def test_missing_config_uses_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ECC_CONFIG", raising=False)
        settings = load_config(tmp_path / "does_not_exist.toml")
        assert settings.refresh_seconds == 15
        assert settings.stale_after_s == 180
        assert settings.offline_after_s == 900
        assert settings.demo_mode == "auto"
        assert settings.wandb.entity == ""
        assert settings.wandb.projects == []
        assert settings.accounts == []

    def test_loads_valid_toml_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_file = tmp_path / "ecc_config.toml"
        db_file = tmp_path / "custom.db"
        cfg_file.write_text(
            f"""
[general]
refresh_seconds = 30
stale_after_s = 120
offline_after_s = 600
demo_mode = "on"
db_path = "{db_file.as_posix()}"

[wandb]
entity = "my-team"
projects = ["proj-a", "proj-b"]

[[accounts]]
nickname = "kaggle-1"
platform = "kaggle"
username = "alice"
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("ECC_CONFIG", str(cfg_file))
        settings = load_config()
        assert settings.refresh_seconds == 30
        assert settings.stale_after_s == 120
        assert settings.offline_after_s == 600
        assert settings.demo_mode == "on"
        assert settings.db_path == db_file
        assert settings.wandb.entity == "my-team"
        assert settings.wandb.projects == ["proj-a", "proj-b"]
        assert len(settings.accounts) == 1
        assert settings.accounts[0].nickname == "kaggle-1"
        assert settings.accounts[0].platform == "kaggle"

    @pytest.mark.parametrize(
        "bad_toml",
        [
            '[wandb]\napi_key = "nope"\n',
            '[general]\nmy_token = "nope"\n',
            '[general]\ndb_password = "nope"\n',
            '[[accounts]]\nnickname = "k1"\nplatform = "kaggle"\nclient_secret = "nope"\n',
            'secret_thing = "nope"\n',
        ],
    )
    def test_rejects_secret_like_keys(self, tmp_path: Path, bad_toml: str) -> None:
        cfg_file = tmp_path / "bad.toml"
        cfg_file.write_text(bad_toml, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(cfg_file)

    @pytest.mark.parametrize(
        "bad_toml",
        [
            "[general]\nrefresh_seconds = 5\n",
            "[general]\nrefresh_seconds = 301\n",
            '[general]\ndemo_mode = "maybe"\n',
            "[general]\nstale_after_s = 300\noffline_after_s = 100\n",
        ],
    )
    def test_rejects_invalid_values(self, tmp_path: Path, bad_toml: str) -> None:
        cfg_file = tmp_path / "invalid.toml"
        cfg_file.write_text(bad_toml, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(cfg_file)


class TestStore:
    def test_schema_idempotent_and_user_version(self, tmp_path: Path) -> None:
        db = tmp_path / "sub" / "ecc.db"
        s1 = Store(db)
        assert s1.user_version() == 3
        s2 = Store(db)
        assert s2.user_version() == 3

    def test_project_crud_and_baseline(self, temp_store: Store) -> None:
        p = temp_store.upsert_project(
            slug="shared-task",
            name="Shared Task 4",
            description="VLM experiments",
            wandb_entity="me",
            wandb_project="shared-task",
            tags=["vlm", "qwen"],
            notes="Initial notes",
        )
        assert p["slug"] == "shared-task"
        assert p["tags"] == ["vlm", "qwen"]
        assert p["baseline_run_id"] is None

        temp_store.set_baseline("shared-task", "demo:demo/shared-task/run-d")
        fetched = temp_store.get_project("shared-task")
        assert fetched is not None
        assert fetched["baseline_run_id"] == "demo:demo/shared-task/run-d"

        # Upsert without baseline preserves existing baseline
        temp_store.upsert_project("shared-task", "Shared Task 4 Renamed", tags=["vlm"])
        fetched2 = temp_store.get_project("shared-task")
        assert fetched2 is not None
        assert fetched2["name"] == "Shared Task 4 Renamed"
        assert fetched2["baseline_run_id"] == "demo:demo/shared-task/run-d"

        assert len(temp_store.list_projects()) == 1
        assert temp_store.delete_project("shared-task") is True
        assert temp_store.delete_project("shared-task") is False
        assert temp_store.get_project("shared-task") is None

    def test_run_notes_crud(self, temp_store: Store) -> None:
        assert temp_store.get_run_note("r1") is None
        saved = temp_store.set_run_note("r1", notes="Promising curve", tags=["best", "lora"], pinned=True)
        assert saved["pinned"] is True
        assert saved["tags"] == ["best", "lora"]
        fetched = temp_store.get_run_note("r1")
        assert fetched is not None
        assert fetched["notes"] == "Promising curve"

    def test_accounts_and_redacted_truncated_check(self, temp_store: Store) -> None:
        acc = temp_store.upsert_account("kaggle-1", "kaggle", username="alice")
        assert acc["secret_ref"] == "kaggle:kaggle-1"
        assert acc["enabled"] is True

        long_secret_msg = (
            "Auth error for KGAT_1234567890abcdef and key "
            "0123456789abcdef0123456789abcdef01234567 " + ("x" * 400)
        )
        temp_store.record_account_check("kaggle-1", ok=False, msg=long_secret_msg)
        accounts = temp_store.list_accounts()
        assert len(accounts) == 1
        rec = accounts[0]
        assert rec["last_check_ok"] is False
        assert rec["last_check_msg"] is not None
        assert len(rec["last_check_msg"]) <= 300
        assert "KGAT_1234567890abcdef" not in rec["last_check_msg"]
        assert "0123456789abcdef0123456789abcdef01234567" not in rec["last_check_msg"]

    def test_kernel_links(self, temp_store: Store) -> None:
        temp_store.link_kernel("alice/nb-1", "kaggle-1", "shared-task", "wandb:me/shared-task/abc")
        temp_store.link_kernel("bob/nb-2", "kaggle-2", "cv-proj", None)
        assert len(temp_store.list_kernel_links()) == 2
        assert len(temp_store.list_kernel_links(account="kaggle-1")) == 1
        assert len(temp_store.list_kernel_links(project="cv-proj")) == 1
