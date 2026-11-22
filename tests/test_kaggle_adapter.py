"""Tests for ecc.adapters.kaggle (all via fake subprocess runner — no real CLI/network)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ecc.adapters.kaggle import (
    KaggleCLI,
    KaggleCLIError,
    KaggleSource,
    parse_kaggle_logs,
    parse_status_output,
)
from ecc.config import GeneralSettings, Settings
from ecc.models import RunStatus
from ecc.store import Store


class TestKaggleCLI:
    def test_env_isolation_and_token_never_in_argv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        monkeypatch.setenv("KAGGLE_USERNAME", "foreign-user")
        monkeypatch.setenv("KAGGLE_KEY", "foreign-key-12345")
        monkeypatch.setenv("KAGGLE_PROXY", "http://should-be-removed")

        captured: dict[str, Any] = {}
        token = "KGAT_account1_secret_token_999"

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["argv"] = list(argv)
            captured["env"] = dict(kwargs["env"])
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="ref,title,author,lastRunTime,totalVotes\nalice/demo,Demo,alice,2026-11-22 10:00:00,3\n",
                stderr="",
            )

        cli = KaggleCLI(
            "kaggle-1",
            token_getter=lambda: token,
            kaggle_exe="fake-kaggle.exe",
            runner=fake_runner,
        )
        kernels = cli.list_kernels()
        assert len(kernels) == 1
        assert kernels[0].ref == "alice/demo"
        assert kernels[0].url == "https://www.kaggle.com/code/alice/demo"

        # Token must be in env, absent from argv
        assert captured["env"]["KAGGLE_API_TOKEN"] == token
        assert all(token not in arg for arg in captured["argv"])

        # Foreign KAGGLE_* vars stripped
        assert "KAGGLE_USERNAME" not in captured["env"]
        assert "KAGGLE_KEY" not in captured["env"]
        assert "KAGGLE_PROXY" not in captured["env"]

        # Per-account config dir created and PYTHONIOENCODING set
        cfg_dir = Path(captured["env"]["KAGGLE_CONFIG_DIR"])
        assert cfg_dir == tmp_path / "localappdata" / "ai-ecc" / "kaggle" / "kaggle-1"
        assert cfg_dir.is_dir()
        assert captured["env"]["PYTHONIOENCODING"] == "utf-8"

    @pytest.mark.parametrize(
        "disallowed_args",
        [
            ["kernels", "push", "-p", "."],
            ["kernels", "delete", "alice/demo"],
            ["kernels", "pull", "alice/demo"],
            ["datasets", "list"],
            ["kernels", "logs", "-f", "alice/demo"],
            ["kernels", "logs", "--follow", "alice/demo"],
            ["kernels", "logs", "-qf", "alice/demo"],
            ["competitions", "submit"],
            ["kernels"],
        ],
    )
    def test_allow_list_rejects_disallowed_commands(
        self, disallowed_args: list[str]
    ) -> None:
        called = False

        def should_not_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal called
            called = True
            raise AssertionError("runner must not be called for disallowed commands")

        cli = KaggleCLI(
            "kaggle-1",
            token_getter=lambda: "KGAT_dummy_12345678",
            runner=should_not_run,
        )
        with pytest.raises(PermissionError):
            cli.run(disallowed_args)
        assert called is False

    def test_stderr_fake_token_is_redacted_in_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        fake_token = "KGAT_my_secret_token_abcdef123456"
        custom_token = "custom_raw_secret_xyz_789"

        for tok in (fake_token, custom_token):
            def failing_runner(argv: list[str], _tok: str = tok, **kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(
                    args=argv,
                    returncode=1,
                    stdout="",
                    stderr=f"Request failed with credentials {_tok}",
                )

            cli = KaggleCLI(
                "kaggle-1",
                token_getter=lambda _tok=tok: _tok,
                kaggle_exe="fake-kaggle",
                runner=failing_runner,
            )
            with pytest.raises(KaggleCLIError) as exc_info:
                cli.status("alice/demo")
            msg = str(exc_info.value)
            assert tok not in msg
            assert "[REDACTED]" in msg

    def test_csv_parsing_tolerates_unknown_and_missing_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        csv_output = (
            "ref,title,lastRunTime,unknownExtraCol\n"
            "alice/qwen-ft,Qwen Fine-Tune,2026-11-22 11:30:00,extra1\n"
            "bob/minimal,,2026-11-21T09:15:00Z,extra2\n"
        )

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=csv_output, stderr="")

        cli = KaggleCLI("kaggle-1", token_getter=lambda: "KGAT_tok12345678", runner=fake_runner)
        kernels = cli.list_kernels()
        assert len(kernels) == 2
        assert kernels[0].ref == "alice/qwen-ft"
        assert kernels[0].title == "Qwen Fine-Tune"
        assert kernels[0].author == "alice"
        assert kernels[0].last_run_time == datetime(2026, 11, 22, 11, 30, 0, tzinfo=timezone.utc)
        assert kernels[0].url == "https://www.kaggle.com/code/alice/qwen-ft"
        assert kernels[1].title == "minimal"
        assert kernels[1].author == "bob"


class TestParsers:
    @pytest.mark.parametrize(
        "text,expected_status,expected_msg",
        [
            ('alice/nb has status "KernelWorkerStatus.RUNNING"', "RUNNING", None),
            ('alice/nb has status "RUNNING"', "RUNNING", None),
            (
                'alice/nb has status "KernelWorkerStatus.ERROR"\nFailure message: "CUDA out of memory"',
                "ERROR",
                "CUDA out of memory",
            ),
            ('alice/nb has status "QUEUED"', "QUEUED", None),
            ('alice/nb has status "KernelWorkerStatus.COMPLETE"', "COMPLETE", None),
            ('alice/nb has status "KernelWorkerStatus.CANCEL_REQUESTED"', "CANCEL_REQUESTED", None),
            ('alice/nb has status "NEW_SCRIPT"', "NEW_SCRIPT", None),
            ("", "UNKNOWN", None),
        ],
    )
    def test_parse_status_output(
        self, text: str, expected_status: str, expected_msg: str | None
    ) -> None:
        st, msg = parse_status_output(text)
        assert st == expected_status
        assert msg == expected_msg

    def test_parse_logs_plain_text(self) -> None:
        raw = (
            "2026-11-22T11:00:00Z Starting epoch 1\n"
            "WARNING: gradient clipping active\n"
            "RuntimeError: CUDA out of memory\n"
        )
        lines = parse_kaggle_logs(raw, tail_lines=10)
        assert len(lines) == 3
        assert all(line.stream == "kaggle" for line in lines)
        assert lines[0].ts == datetime(2026, 11, 22, 11, 0, 0, tzinfo=timezone.utc)
        assert lines[0].level == "INFO"
        assert lines[1].level == "WARNING"
        assert lines[2].level == "ERROR"

    def test_parse_logs_json_array(self) -> None:
        payload = [
            {"stream_name": "stdout", "time": 1.25, "data": "step 10 loss=0.9\nstep 20 loss=0.7\n"},
            {"stream_name": "stderr", "time": "2026-11-22T11:05:00Z", "data": "ValueError: NaN loss\n"},
        ]
        lines = parse_kaggle_logs(json.dumps(payload), tail_lines=10)
        assert len(lines) == 3
        assert all(line.stream == "kaggle" for line in lines)
        assert lines[0].ts is None  # relative elapsed seconds not turned into 1970 epoch
        assert lines[0].text == "step 10 loss=0.9"
        assert lines[1].text == "step 20 loss=0.7"
        assert lines[2].ts == datetime(2026, 11, 22, 11, 5, 0, tzinfo=timezone.utc)
        assert lines[2].level == "ERROR"


class TestKaggleSource:
    def test_status_mapping_lookback_cache_and_failure_isolation(
        self, temp_store: Store, fixed_now: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        temp_store.upsert_account("kaggle-1", "kaggle", username="alice")
        temp_store.upsert_account("kaggle-2", "kaggle", username="bob")
        temp_store.link_kernel("alice/k-run", "kaggle-1", "shared-task-4", None)

        recent_ts = (fixed_now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        old_ts = (fixed_now - timedelta(hours=200)).strftime("%Y-%m-%d %H:%M:%S")

        csv_1 = (
            "ref,title,author,lastRunTime,totalVotes\n"
            f"alice/k-run,Run Kernel,alice,{recent_ts},0\n"
            f"alice/k-q,Queued Kernel,alice,{recent_ts},0\n"
            f"alice/k-err,Err Kernel,alice,{recent_ts},0\n"
            f"alice/k-done,Done Kernel,alice,{recent_ts},0\n"
            f"alice/k-old,Ancient Kernel,alice,{old_ts},0\n"
        )
        status_map = {
            "alice/k-run": 'alice/k-run has status "KernelWorkerStatus.RUNNING"',
            "alice/k-q": 'alice/k-q has status "QUEUED"',
            "alice/k-err": 'alice/k-err has status "KernelWorkerStatus.ERROR"\nFailure message: "OOM"',
            "alice/k-done": 'alice/k-done has status "KernelWorkerStatus.COMPLETE"',
        }
        call_counts = {"list": 0, "status": 0}

        def runner_ok(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[1:3] == ["kernels", "list"]:
                call_counts["list"] += 1
                return subprocess.CompletedProcess(args=argv, returncode=0, stdout=csv_1, stderr="")
            if argv[1:3] == ["kernels", "status"]:
                call_counts["status"] += 1
                ref = argv[3]
                assert ref != "alice/k-old", "status must not be fetched for kernels outside lookback"
                return subprocess.CompletedProcess(args=argv, returncode=0, stdout=status_map[ref], stderr="")
            raise AssertionError(f"Unexpected argv: {argv}")

        def runner_fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=1,
                stdout="",
                stderr="403 Forbidden for token KGAT_bob_secret_12345678",
            )

        cli_1 = KaggleCLI("kaggle-1", token_getter=lambda: "KGAT_alice_12345678", runner=runner_ok)
        cli_2 = KaggleCLI("kaggle-2", token_getter=lambda: "KGAT_bob_secret_12345678", runner=runner_fail)

        settings = Settings(general=GeneralSettings(kaggle_lookback_hours=72))
        src = KaggleSource(
            accounts=[("kaggle-1", cli_1), ("kaggle-2", cli_2)],
            store=temp_store,
            settings=settings,
            now_fn=lambda: fixed_now,
        )

        runs = src.list_runs()
        by_id = {r.run_id: r for r in runs}

        # Account 1's 4 recent kernels succeeded despite Account 2 failing
        assert len(by_id) == 4
        assert "kaggle:kaggle-1/alice/k-old" not in by_id

        r_run = by_id["kaggle:kaggle-1/alice/k-run"]
        assert r_run.status == RunStatus.RUNNING
        assert r_run.raw_state == "RUNNING (Kaggle API)"
        assert r_run.last_update == fixed_now
        assert r_run.project == "shared-task-4"

        assert by_id["kaggle:kaggle-1/alice/k-q"].status == RunStatus.QUEUED
        assert by_id["kaggle:kaggle-1/alice/k-q"].project == "unassigned"
        assert by_id["kaggle:kaggle-1/alice/k-err"].status == RunStatus.FAILED
        assert by_id["kaggle:kaggle-1/alice/k-done"].status == RunStatus.FINISHED

        # Failing account error recorded via store.record_account_check (with redaction)
        acc_rows = {a["nickname"]: a for a in temp_store.list_accounts()}
        assert acc_rows["kaggle-2"]["last_check_ok"] is False
        assert "KGAT_bob_secret_12345678" not in (acc_rows["kaggle-2"]["last_check_msg"] or "")

        # Second call hits 30s TTL cache for kaggle-1
        _ = src.list_runs()
        assert call_counts["list"] == 1
        assert call_counts["status"] == 4
