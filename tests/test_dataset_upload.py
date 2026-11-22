"""Tests for private Kaggle dataset upload, guards, allow-list, and CLI commands."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ecc import cli as ecc_cli
from ecc.config import GeneralSettings, LaunchSettings, Settings
from ecc.launch.datasets import (
    MAX_DATASET_SIZE_BYTES,
    DatasetUploadRefused,
    DatasetUploadService,
    KaggleDatasetUploader,
    build_monitor_wheel_folder,
    scan_folder_for_secrets,
)
from ecc.launch.github import GitHubReader
from ecc.launch.service import LaunchService
from ecc.secrets import SecretStore
from ecc.service import Service
from ecc.store import Store


class _InMemoryKeyring:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._data[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._data.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        del self._data[(service, username)]


def _make_sample_folder(base: Path, name: str = "pkg") -> Path:
    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.csv").write_text("id,label\n1,pos\n2,neg\n", encoding="utf-8")
    (folder / "readme.txt").write_text("Private dataset notes.\n", encoding="utf-8")
    return folder


class TestDatasetUploaderAllowList:
    @pytest.mark.parametrize(
        "bad_args",
        [
            ["datasets", "delete", "alice/my-ds"],
            ["datasets", "download", "alice/my-ds"],
            ["datasets", "create", "-p", "/tmp/x", "--public"],
            ["datasets", "create", "-p", "/tmp/x", "--public=true"],
            ["datasets", "create", "-p", "/tmp/x", "-u"],
            ["datasets", "create", "-p", "/tmp/x", "-qu"],
            ["datasets", "version", "-p", "/tmp/x", "-m", "v2", "-u"],
            ["datasets", "version", "-p", "/tmp/x", "-m", "v2", "--public"],
            ["datasets", "create", "-p", "/tmp/x", "-m", "not-allowed-on-create"],
            ["datasets", "create", "-p", "/tmp/x", "-r", "invalid_mode"],
            ["datasets", "create", "-p", "/tmp/x", "--dir-mode", "bad"],
            ["datasets", "create", "-p", "/tmp/x", "--delete-old-versions"],
            ["kernels", "push", "-p", "/tmp/x"],
            ["kernels", "list", "--mine"],
            ["competitions", "submit"],
            ["datasets"],
        ],
    )
    def test_allow_list_rejects_disallowed_commands_and_flags(
        self, bad_args: list[str]
    ) -> None:
        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_test_token_12345678",
            runner=lambda *a, **kw: pytest.fail("runner must not be called for disallowed args"),
        )
        with pytest.raises(PermissionError):
            uploader._exec(bad_args)

    def test_allow_list_accepts_valid_commands_and_injects_token_only_in_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAGGLE_USERNAME", "leaked_global")
        monkeypatch.setenv("KAGGLE_KEY", "leaked_global_key")
        monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_leaked_global_token")

        calls: list[dict[str, Any]] = []
        secret_tok = "KGAT_per_account_token_ABCDEF12"

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append({"argv": argv, "env": kwargs["env"]})
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout=f"ready {secret_tok}",
                stderr="",
            )

        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: secret_tok,
            kaggle_exe="kaggle-fake",
            runner=fake_runner,
        )
        rc, stdout, _ = uploader._exec(["datasets", "status", "alice/experiment-monitor"])
        assert rc == 0
        assert len(calls) == 1
        assert all(secret_tok not in part for part in calls[0]["argv"])
        assert calls[0]["env"]["KAGGLE_API_TOKEN"] == secret_tok
        assert "KAGGLE_USERNAME" not in calls[0]["env"]
        assert "KAGGLE_KEY" not in calls[0]["env"]
        assert "kaggle-1" in calls[0]["env"]["KAGGLE_CONFIG_DIR"]
        # Token scrubbed from output
        assert secret_tok not in stdout
        assert "[REDACTED]" in stdout

    def test_token_in_argv_rejected(self) -> None:
        tok = "KGAT_never_in_argv_12345"
        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: tok,
            runner=lambda *a, **kw: pytest.fail("runner must not be invoked"),
        )
        with pytest.raises(PermissionError, match="never be passed in CLI arguments"):
            uploader._exec(["datasets", "version", "-p", "/tmp/stage", "-m", tok])


class TestSecretAndSizeGuards:
    @pytest.mark.parametrize(
        "secret_relpath",
        [
            ".env",
            ".env.local",
            "sub/kaggle.json",
            "access_token_prod.txt",
            "keys/server.pem",
            "ssh/id_rsa",
            "ssh/id_rsa.pub",
        ],
    )
    def test_secret_filename_guard_rejects_and_reports_relative_path(
        self, tmp_path: Path, secret_relpath: str
    ) -> None:
        folder = _make_sample_folder(tmp_path)
        bad_file = folder / secret_relpath
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("dummy content", encoding="utf-8")

        found = scan_folder_for_secrets(folder)
        assert secret_relpath in found

        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_tok_12345678",
            runner=lambda *a, **kw: pytest.fail("runner must not be called"),
        )
        with pytest.raises(DatasetUploadRefused) as exc_info:
            uploader.upload(folder, "my-dataset", "My Dataset Title", "alice")
        assert secret_relpath in str(exc_info.value)

    def test_secret_content_guard_rejects_hf_token_without_leaking_content(
        self, tmp_path: Path
    ) -> None:
        folder = _make_sample_folder(tmp_path)
        fake_hf = "hf_AbCdEfGhIjKlMnOpQrSt1234"
        nested = folder / "configs" / "notes.yaml"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text(f"note: do not upload {fake_hf}\n", encoding="utf-8")

        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_tok_12345678",
            runner=lambda *a, **kw: pytest.fail("runner must not be called"),
        )
        with pytest.raises(DatasetUploadRefused) as exc_info:
            uploader.upload(folder, "my-dataset", "My Dataset Title", "alice")
        msg = str(exc_info.value)
        assert "configs/notes.yaml" in msg
        assert fake_hf not in msg

    def test_binary_wheel_passes_secret_scan(self, tmp_path: Path) -> None:
        folder = tmp_path / "wheel_dir"
        folder.mkdir()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("experiment_monitor/__init__.py", "VERSION = '0.1.0'\n")
        # Zip files naturally contain NUL bytes -> treated as binary
        (folder / "experiment_monitor-0.1.0-py3-none-any.whl").write_bytes(buf.getvalue())
        assert scan_folder_for_secrets(folder) == []

    def test_size_guard_enforces_limit_and_default_is_20_gb(self, tmp_path: Path) -> None:
        assert MAX_DATASET_SIZE_BYTES == 20 * 1024 * 1024 * 1024

        folder = tmp_path / "big_dir"
        folder.mkdir()
        (folder / "payload.bin").write_bytes(b"\x00" * 2048)

        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_tok_12345678",
            runner=lambda *a, **kw: pytest.fail("runner must not be called"),
            max_size_bytes=1024,
        )
        with pytest.raises(DatasetUploadRefused, match="exceeds maximum limit"):
            uploader.upload(folder, "my-dataset", "My Dataset Title", "alice")

    @pytest.mark.parametrize("bad_slug", ["ab", "UpperCase", "-leading", "has space", "a" * 51])
    def test_slug_validation(self, tmp_path: Path, bad_slug: str) -> None:
        folder = _make_sample_folder(tmp_path)
        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_tok_12345678",
            runner=lambda *a, **kw: pytest.fail("runner must not be called"),
        )
        with pytest.raises(DatasetUploadRefused, match="Invalid dataset slug"):
            uploader.upload(folder, bad_slug, "Valid Title Here", "alice")

    @pytest.mark.parametrize("bad_title", ["short", "x" * 51, "line1\nline2"])
    def test_title_validation(self, tmp_path: Path, bad_title: str) -> None:
        folder = _make_sample_folder(tmp_path)
        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_tok_12345678",
            runner=lambda *a, **kw: pytest.fail("runner must not be called"),
        )
        with pytest.raises(DatasetUploadRefused, match="Dataset title"):
            uploader.upload(folder, "valid-slug", bad_title, "alice")


class TestCreateVsVersionAndStaging:
    def test_creates_when_status_not_found_and_leaves_source_folder_unchanged(
        self, tmp_path: Path
    ) -> None:
        folder = _make_sample_folder(tmp_path)
        before_files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))
        observed: list[dict[str, Any]] = []

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            cmd = argv[1:3]
            entry: dict[str, Any] = {"argv": argv, "env": kwargs["env"]}
            if cmd == ["datasets", "status"]:
                observed.append(entry)
                return subprocess.CompletedProcess(
                    args=argv,
                    returncode=1,
                    stdout="",
                    stderr="404 - Not Found",
                )
            if cmd == ["datasets", "create"]:
                p_idx = argv.index("-p")
                stage_dir = Path(argv[p_idx + 1])
                entry["stage_dir"] = stage_dir
                entry["metadata"] = json.loads(
                    (stage_dir / "dataset-metadata.json").read_text(encoding="utf-8")
                )
                entry["staged_files"] = sorted(
                    p.relative_to(stage_dir).as_posix() for p in stage_dir.rglob("*")
                )
                observed.append(entry)
                return subprocess.CompletedProcess(
                    args=argv,
                    returncode=0,
                    stdout="Your private Dataset is being created. https://www.kaggle.com/datasets/alice/experiment-monitor",
                    stderr="",
                )
            raise AssertionError(f"Unexpected command: {argv}")

        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_secret_create_12345",
            kaggle_exe="kaggle-fake",
            runner=fake_runner,
        )
        res = uploader.upload(
            folder=folder,
            slug="experiment-monitor",
            title="experiment monitor",
            owner="alice",
        )
        assert res.ok is True
        assert res.action == "created"
        assert res.ref == "alice/experiment-monitor"
        assert res.url == "https://www.kaggle.com/datasets/alice/experiment-monitor"
        assert res.size_bytes > 0

        # Calls: status -> create
        assert [c["argv"][1:3] for c in observed] == [
            ["datasets", "status"],
            ["datasets", "create"],
        ]
        create_argv = observed[1]["argv"]
        assert "-r" in create_argv and "zip" in create_argv and "-q" in create_argv
        assert "--public" not in create_argv and "-u" not in create_argv

        # dataset-metadata.json verified in staging
        meta = observed[1]["metadata"]
        assert meta == {
            "title": "experiment monitor",
            "id": "alice/experiment-monitor",
            "licenses": [{"name": "other"}],
        }
        assert "isPrivate" not in meta or meta["isPrivate"] is True
        assert "data.csv" in observed[1]["staged_files"]

        # Staging dir cleaned up; source folder completely untouched
        assert not observed[1]["stage_dir"].exists()
        after_files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))
        assert after_files == before_files
        assert not (folder / "dataset-metadata.json").exists()

    def test_versions_when_status_indicates_existing_dataset(self, tmp_path: Path) -> None:
        folder = _make_sample_folder(tmp_path)
        observed_argvs: list[list[str]] = []

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            observed_argvs.append(argv)
            if argv[1:3] == ["datasets", "status"]:
                return subprocess.CompletedProcess(
                    args=argv, returncode=0, stdout="ready\n", stderr=""
                )
            if argv[1:3] == ["datasets", "version"]:
                return subprocess.CompletedProcess(
                    args=argv,
                    returncode=0,
                    stdout="Dataset version is being created.",
                    stderr="",
                )
            raise AssertionError(f"Unexpected command: {argv}")

        uploader = KaggleDatasetUploader(
            nickname="kaggle-1",
            token_getter=lambda: "KGAT_secret_version_12345",
            kaggle_exe="kaggle-fake",
            runner=fake_runner,
        )
        res = uploader.upload(
            folder=folder,
            slug="experiment-monitor",
            title="experiment monitor",
            owner="alice",
            new_version_message="v2 wheel",
        )
        assert res.ok is True
        assert res.action == "versioned"
        assert res.ref == "alice/experiment-monitor"
        assert [a[1:3] for a in observed_argvs] == [
            ["datasets", "status"],
            ["datasets", "version"],
        ]
        ver_argv = observed_argvs[1]
        assert "-m" in ver_argv
        assert ver_argv[ver_argv.index("-m") + 1] == "v2 wheel"
        assert "-r" in ver_argv and "zip" in ver_argv and "-q" in ver_argv


class TestCliDatasetCommandsAndAudit:
    def _setup_cli_ds_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        enabled: bool = True,
        exists: bool = False,
    ) -> dict[str, Any]:
        store = Store(tmp_path / "ecc.db")
        store.upsert_account("kaggle-1", "kaggle", username="alice", enabled=True)

        secrets = SecretStore(backend=_InMemoryKeyring())
        secrets.set("kaggle", "kaggle-1", "KGAT_cli_ds_secret_99999")

        settings = Settings(
            general=GeneralSettings(demo_mode="off", db_path=tmp_path / "ecc.db"),
            launch=LaunchSettings(enabled=enabled),
        )
        read_svc = Service(sources=[], store=store, settings=settings, secrets=secrets)
        launch_svc = LaunchService(
            store=store,
            secrets=secrets,
            settings=settings,
            read_service=read_svc,
            github=GitHubReader(runner=lambda *a, **kw: None),
        )

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[1:3] == ["datasets", "status"]:
                if exists:
                    return subprocess.CompletedProcess(
                        args=argv, returncode=0, stdout="ready\n", stderr=""
                    )
                return subprocess.CompletedProcess(
                    args=argv, returncode=1, stdout="", stderr="not found"
                )
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="Dataset uploaded. token=KGAT_cli_ds_secret_99999",
                stderr="",
            )

        ds_svc = DatasetUploadService(
            store=store,
            secrets=secrets,
            settings=settings,
            launch_service=launch_svc,
            uploader_factory=lambda nick, getter: KaggleDatasetUploader(
                nick, getter, kaggle_exe="kaggle-fake", runner=fake_runner
            ),
        )

        monkeypatch.setattr(ecc_cli, "build_service", lambda config_path=None: read_svc)
        monkeypatch.setattr(ecc_cli, "build_launch_service", lambda rs: launch_svc)
        monkeypatch.setattr(ecc_cli, "build_dataset_upload_service", lambda rs, ls: ds_svc)
        return {"store": store, "ds_svc": ds_svc}

    def test_cli_dataset_upload_audits_and_redacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = self._setup_cli_ds_env(tmp_path, monkeypatch, enabled=True, exists=False)
        folder = _make_sample_folder(tmp_path)

        rc = ecc_cli.main(
            [
                "dataset",
                "upload",
                "--account",
                "kaggle-1",
                "--folder",
                str(folder),
                "--slug",
                "experiment-monitor",
                "--title",
                "experiment monitor",
            ]
        )
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["ok"] is True
        assert out["ref"] == "alice/experiment-monitor"
        assert out["created"] is True
        assert out["versioned"] is False
        assert out["size_mb"] >= 0
        assert "KGAT_cli_ds_secret_99999" not in out["message"]

        rows = env["store"].list_dataset_uploads()
        assert len(rows) == 1
        assert rows[0]["account"] == "kaggle-1"
        assert rows[0]["ref"] == "alice/experiment-monitor"
        assert rows[0]["ok"] is True
        assert rows[0]["source"] == "claude"
        assert rows[0]["size_bytes"] > 0
        assert "KGAT_cli_ds_secret_99999" not in rows[0]["message"]

    def test_cli_dataset_upload_disabled_refused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._setup_cli_ds_env(tmp_path, monkeypatch, enabled=False)
        folder = _make_sample_folder(tmp_path)

        rc = ecc_cli.main(
            [
                "dataset",
                "upload",
                "--account",
                "kaggle-1",
                "--folder",
                str(folder),
                "--slug",
                "experiment-monitor",
                "--title",
                "experiment monitor",
            ]
        )
        out = json.loads(capsys.readouterr().out)
        assert rc == 3
        assert out["error"] == "upload_refused"
        assert "disabled" in out["reason"].lower()

    def test_build_monitor_wheel_helper_and_cli(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "experiment_monitor-0.1.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nTag: py3-none-any\n",
            )
            zf.writestr("experiment_monitor/__init__.py", "__version__ = '0.1.0'\n")
        fake_whl = buf.getvalue()

        out_dir = tmp_path / "wheel_out"
        whl_path = build_monitor_wheel_folder(out_dir, wheel_builder=lambda: fake_whl)
        assert whl_path.name == "experiment_monitor-0.1.0-py3-none-any.whl"
        assert whl_path.read_bytes() == fake_whl

        # Non-empty directory must be refused
        with pytest.raises(DatasetUploadRefused, match="must be empty"):
            build_monitor_wheel_folder(out_dir, wheel_builder=lambda: fake_whl)

        # CLI wrapper
        cli_out_dir = tmp_path / "cli_wheel_out"
        monkeypatch.setattr(
            ecc_cli,
            "build_monitor_wheel_folder",
            lambda out: build_monitor_wheel_folder(out, wheel_builder=lambda: fake_whl),
        )
        monkeypatch.setattr(
            ecc_cli,
            "build_service",
            lambda config_path=None: Service(
                sources=[], store=Store(":memory:"), settings=Settings()
            ),
        )
        rc = ecc_cli.main(["dataset", "build-monitor-wheel", "--out", str(cli_out_dir)])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["ok"] is True
        assert out["wheel_file"].endswith("experiment_monitor-0.1.0-py3-none-any.whl")
