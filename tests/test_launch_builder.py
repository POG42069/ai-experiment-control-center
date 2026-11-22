"""Tests for ecc.launch.builder and ecc.launch.kaggle_launcher (all via fakes)."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ecc.launch.builder import (
    WheelBuildError,
    build_kernel_folder,
    build_wheel,
    clear_wheel_cache,
)
from ecc.launch.kaggle_launcher import (
    KaggleLauncher,
    KaggleLaunchError,
    parse_push_output,
)
from ecc.launch.runner_template import RunnerParams


class TestBuilder:
    def test_build_wheel_caches_by_content_hash_and_invalidates_on_change(
        self, tmp_path: Path
    ) -> None:
        clear_wheel_cache()
        mon_dir = tmp_path / "monitor"
        src_dir = mon_dir / "src" / "experiment_monitor"
        src_dir.mkdir(parents=True)
        (mon_dir / "pyproject.toml").write_text('[project]\nname = "experiment-monitor"\n', encoding="utf-8")
        core_py = src_dir / "core.py"
        core_py.write_text("VALUE = 1\n", encoding="utf-8")

        calls: list[list[str]] = []

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(list(argv))
            assert argv[1:4] == ["-m", "pip", "wheel"]
            assert "--no-deps" in argv
            w_idx = argv.index("-w")
            out_dir = Path(argv[w_idx + 1])
            (out_dir / "experiment_monitor-0.1.0-py3-none-any.whl").write_bytes(
                f"wheel-build-{len(calls)}".encode("utf-8")
            )
            return subprocess.CompletedProcess(argv, 0, "Built wheel", "")

        b1 = build_wheel(mon_dir, runner=fake_runner, python_exe="fake-python")
        b2 = build_wheel(mon_dir, runner=fake_runner, python_exe="fake-python")
        assert b1 == b2 == b"wheel-build-1"
        assert len(calls) == 1

        # Modify source file -> hash changes -> rebuilds
        core_py.write_text("VALUE = 2\n", encoding="utf-8")
        b3 = build_wheel(mon_dir, runner=fake_runner, python_exe="fake-python")
        assert b3 == b"wheel-build-2"
        assert len(calls) == 2

    def test_build_wheel_error_is_redacted(self, tmp_path: Path) -> None:
        clear_wheel_cache()
        mon_dir = tmp_path / "monitor"
        mon_dir.mkdir()
        fake_tok = "ghp_secretTokenLeak1234567890abc"

        def failing_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", f"pip failed with {fake_tok}")

        with pytest.raises(WheelBuildError) as exc_info:
            build_wheel(mon_dir, runner=failing_runner)
        assert fake_tok not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value)

    @pytest.mark.parametrize(
        "accelerator,preflight,exp_gpu,exp_tpu",
        [
            ("NvidiaTeslaT4", False, True, False),
            ("TpuV5E8", False, False, True),
            ("none", False, False, False),
            ("NvidiaTeslaT4", True, False, False),
        ],
    )
    def test_build_kernel_folder_metadata_and_valid_runner(
        self,
        tmp_path: Path,
        accelerator: str,
        preflight: bool,
        exp_gpu: bool,
        exp_tpu: bool,
    ) -> None:
        params = RunnerParams(
            launch_id="abc123",
            account="kaggle-1",
            owner="alice",
            runner_slug="ecc-runner",
            repo="alice/vlm-exp",
            branch="main",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            command="python train.py",
            monitor_mode="runner",
            project="shared-task-4",
            run_name="exp-1",
            preflight=preflight,
            accelerator=accelerator,
        )
        dest = build_kernel_folder(params, b"PK\x03\x04fake-wheel", tmp_path / "out")

        runner_file = dest / "runner.py"
        meta_file = dest / "kernel-metadata.json"
        assert runner_file.is_file()
        assert meta_file.is_file()

        ast.parse(runner_file.read_text(encoding="utf-8"))

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta == {
            "id": "alice/ecc-runner",
            "title": "ecc-runner",
            "code_file": "ecc-runner.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": exp_gpu,
            "enable_tpu": exp_tpu,
            "enable_internet": True,
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }


class TestKaggleLauncher:
    @pytest.mark.parametrize(
        "disallowed_args",
        [
            ["kernels", "delete", "alice/ecc-runner"],
            ["kernels", "status", "alice/ecc-runner"],
            ["kernels", "list", "--mine"],
            ["kernels", "logs", "alice/ecc-runner"],
            ["datasets", "create", "-p", "."],
            ["competitions", "submit"],
            ["kernels", "push", "-p", ".", "--no-run"],
            ["kernels", "push", "-p", ".", "--unknown-flag", "1"],
            ["kernels", "push", "extra-positional"],
            ["kernels"],
        ],
    )
    def test_allow_list_rejects_disallowed_commands(self, disallowed_args: list[str]) -> None:
        called = False

        def should_not_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal called
            called = True
            raise AssertionError("runner must not be called for disallowed commands")

        launcher = KaggleLauncher(
            "kaggle-1",
            token_getter=lambda: "KGAT_dummy_12345678",
            runner=should_not_run,
        )
        with pytest.raises(PermissionError):
            launcher.run(disallowed_args)
        assert called is False

    def test_push_env_isolation_token_not_in_argv_and_success_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        monkeypatch.setenv("KAGGLE_USERNAME", "foreign")
        monkeypatch.setenv("KAGGLE_KEY", "foreign-key")
        token = "KGAT_launcher_secret_987654321"
        captured: dict[str, Any] = {}

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["argv"] = list(argv)
            captured["env"] = dict(kwargs["env"])
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout=(
                    "Kernel version 7 successfully pushed.  "
                    "Please check progress at https://www.kaggle.com/code/alice/ecc-runner\n"
                ),
                stderr="",
            )

        launcher = KaggleLauncher(
            "kaggle-1",
            token_getter=lambda: token,
            kaggle_exe="fake-kaggle.exe",
            runner=fake_runner,
        )
        res = launcher.push(tmp_path / "pkg", accelerator="NvidiaTeslaT4", timeout_s=3600)

        assert res.ok is True
        assert res.version == 7
        assert res.url == "https://www.kaggle.com/code/alice/ecc-runner"

        assert captured["argv"] == [
            "fake-kaggle.exe",
            "kernels",
            "push",
            "-p",
            str(tmp_path / "pkg"),
            "--accelerator",
            "NvidiaTeslaT4",
            "-t",
            "3600",
        ]
        assert all(token not in arg for arg in captured["argv"])
        assert captured["env"]["KAGGLE_API_TOKEN"] == token
        assert "KAGGLE_USERNAME" not in captured["env"]
        assert "KAGGLE_KEY" not in captured["env"]
        assert Path(captured["env"]["KAGGLE_CONFIG_DIR"]) == (
            tmp_path / "localappdata" / "ai-ecc" / "kaggle" / "kaggle-1"
        )

    def test_push_omits_accelerator_when_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        captured_argv: list[str] = []

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured_argv.extend(argv)
            return subprocess.CompletedProcess(
                argv,
                0,
                "Kernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/alice/ecc-runner",
                "",
            )

        launcher = KaggleLauncher("kaggle-1", token_getter=lambda: "KGAT_tok12345678", runner=fake_runner)
        res = launcher.push(tmp_path, accelerator="none", timeout_s=900)
        assert res.ok is True
        assert "--accelerator" not in captured_argv
        assert "-t" in captured_argv

    def test_parse_push_output_error_and_redaction(self) -> None:
        tok = "custom_raw_kaggle_secret_xyz"
        err_res = parse_push_output(
            f"Kernel push error: 403 Forbidden for token {tok}",
            returncode=0,
            token=tok,
        )
        assert err_res.ok is False
        assert err_res.version is None
        assert err_res.url is None
        assert tok not in err_res.message
        assert "[REDACTED]" in err_res.message

    def test_missing_token_raises_launch_error(self, tmp_path: Path) -> None:
        launcher = KaggleLauncher("kaggle-1", token_getter=lambda: None)
        with pytest.raises(KaggleLaunchError):
            launcher.push(tmp_path)
