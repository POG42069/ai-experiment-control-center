"""Integration tests for experiment_monitor.core using W&B offline mode (no network)."""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest
import wandb

import experiment_monitor as em
from experiment_monitor.redact import REDACTED


@pytest.fixture(autouse=True)
def _wandb_offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb_config"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb_cache"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb_data"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    yield
    if wandb.run is not None:
        wandb.finish()


def test_init_config_preserves_user_keys_and_adds_ecc_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    user_cfg = {"learning_rate": 3e-4, "lora_r": 16, "batch_size": 8}
    user_cfg_snapshot = dict(user_cfg)

    with em.init(
        project="shared-task-4",
        name="run-config-test",
        config=user_cfg,
        account="kaggle-1",
        notebook_url="https://www.kaggle.com/code/user/nb",
        kaggle_kernel="user/nb",
        dataset_version="v2",
        dataset_split="90/10 seed42",
        seed=42,
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        prompt_strategy="vi-default",
        total_steps=200,
        total_epochs=3,
        eval_signature="val-v2-f1",
        tags=["baseline"],
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
        mode="offline",
    ) as mon:
        # Caller dict must never be mutated.
        assert user_cfg == user_cfg_snapshot

        cfg = dict(mon.run.config)
        # User keys untouched
        assert cfg["learning_rate"] == 3e-4
        assert cfg["lora_r"] == 16
        assert cfg["batch_size"] == 8
        # ECC contract keys
        assert cfg["ecc/schema_version"] == 1
        assert cfg["ecc/project"] == "shared-task-4"
        assert cfg["ecc/platform"] == "kaggle"
        assert cfg["ecc/account"] == "kaggle-1"
        assert cfg["ecc/notebook_url"] == "https://www.kaggle.com/code/user/nb"
        assert cfg["ecc/kaggle_kernel"] == "user/nb"
        assert cfg["ecc/dataset_version"] == "v2"
        assert cfg["ecc/dataset_split"] == "90/10 seed42"
        assert cfg["ecc/seed"] == 42
        assert cfg["ecc/model_name"] == "Qwen/Qwen3-VL-4B-Instruct"
        assert cfg["ecc/prompt_strategy"] == "vi-default"
        assert cfg["ecc/total_steps"] == 200
        assert cfg["ecc/total_epochs"] == 3
        assert cfg["ecc/eval_signature"] == "val-v2-f1"
        assert "ecc/git_branch" in cfg
        assert "ecc/git_sha" in cfg

    assert mon.final_summary["ecc/end_reason"] == "finished"


def test_init_reuses_active_wandb_run(tmp_path: Path) -> None:
    outer_run = wandb.init(project="reuse-proj", mode="offline", dir=str(tmp_path))
    try:
        mon = em.init(
            project="reuse-proj",
            config={"lr": 1e-4},
            account="colab-1",
            capture_console=False,
            gpu_telemetry=False,
            heartbeat_interval_s=0,
            mode="offline",
        )
        assert mon.run is outer_run
        assert mon.run.config["ecc/account"] == "colab-1"
        assert mon.run.config["lr"] == 1e-4
        mon.finish()
        # Outer run was not created by Monitor, so it remains active until outer finishes it.
        assert wandb.run is outer_run
    finally:
        outer_run.finish()


def test_best_metric_tracking_max_and_min() -> None:
    with em.init(
        project="best-track-max",
        total_steps=100,
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
        mode="offline",
    ) as mon:
        mon.track_best("eval/macro_f1", mode="max")
        mon.log({"eval/macro_f1": 0.70, "train/epoch": 1.0}, step=25)
        mon.log({"eval/macro_f1": 0.85, "train/epoch": 2.0}, step=50)
        mon.log({"eval/macro_f1": 0.80, "train/epoch": 3.0}, step=75)
        mon.set_progress(step=75, epoch=3.0)

        assert mon.run.summary["ecc/best_metric_name"] == "eval/macro_f1"
        assert mon.run.summary["ecc/best_metric"] == pytest.approx(0.85)
        assert mon.run.summary["ecc/best_step"] == 50
        assert mon.run.summary["ecc/best_epoch"] == pytest.approx(2.0)
        assert mon.run.summary["ecc/progress"] == pytest.approx(0.75)

    with em.init(
        project="best-track-min",
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
        mode="offline",
    ) as mon2:
        mon2.track_best("eval/loss", mode="min")
        mon2.log({"eval/loss": 1.2, "train/epoch": 1.0}, step=10)
        mon2.log({"eval/loss": 0.45, "train/epoch": 2.0}, step=20)
        mon2.log({"eval/loss": 0.60, "train/epoch": 3.0}, step=30)

        assert mon2.run.summary["ecc/best_metric_name"] == "eval/loss"
        assert mon2.run.summary["ecc/best_metric"] == pytest.approx(0.45)
        assert mon2.run.summary["ecc/best_step"] == 20
        assert mon2.run.summary["ecc/best_epoch"] == pytest.approx(2.0)


def test_log_checkpoint_records_size_without_copying(tmp_path: Path) -> None:
    ckpt_file = tmp_path / "weights.pt"
    ckpt_file.write_bytes(b"a" * (512 * 1024))  # 0.5 MB

    ckpt_dir = tmp_path / "checkpoint-100"
    ckpt_dir.mkdir()
    (ckpt_dir / "part1.bin").write_bytes(b"b" * (256 * 1024))
    (ckpt_dir / "part2.bin").write_bytes(b"c" * (256 * 1024))

    with em.init(
        project="ckpt-test",
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
        mode="offline",
    ) as mon:
        run_dir = Path(mon.run.dir)
        e1 = mon.log_checkpoint(ckpt_file, step=50, metric=0.77)
        e2 = mon.log_checkpoint(ckpt_dir, step=100, metric=0.83)

        assert e1["size_mb"] == pytest.approx(0.5, abs=0.01)
        assert e2["size_mb"] == pytest.approx(0.5, abs=0.01)
        assert len(mon.run.summary["ecc/checkpoints"]) == 2
        assert mon.run.summary["ecc/checkpoints"][0]["step"] == 50
        assert mon.run.summary["ecc/checkpoints"][1]["metric"] == pytest.approx(0.83)

        # Must NOT copy or upload checkpoint files into run.dir.
        assert not (run_dir / "weights.pt").exists()
        assert not (run_dir / "checkpoint-100").exists()


def test_context_manager_exception_redacts_logs_traceback_and_restores_streams() -> None:
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    fake_secret = "hf_" + "B" * 34
    long_msg = f"boom with secret {fake_secret} " + ("z" * 600)

    mon_ref = None
    summary_at_finish: dict[str, object] = {}
    exit_codes: list[int] = []
    with pytest.raises(RuntimeError, match="boom with secret"):
        with em.init(
            project="exc-test",
            capture_console=True,
            gpu_telemetry=False,
            heartbeat_interval_s=0,
            mode="offline",
        ) as mon:
            mon_ref = mon
            orig_finish = mon.run.finish

            def _spy_finish(exit_code: int | None = None, **kwargs: object) -> None:
                summary_at_finish.update(dict(mon.run.summary))
                if exit_code is not None:
                    exit_codes.append(exit_code)
                orig_finish(exit_code=exit_code, **kwargs)

            mon.run.finish = _spy_finish  # type: ignore[method-assign]
            print(f"about to fail with {fake_secret}")
            raise RuntimeError(long_msg)

    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr
    assert mon_ref is not None

    assert exit_codes == [1]
    assert summary_at_finish["ecc/end_reason"] == "exception"
    assert summary_at_finish["ecc/exception_type"] == "RuntimeError"
    exc_msg = str(summary_at_finish["ecc/exception_message"])
    assert len(exc_msg) <= 500
    assert fake_secret not in exc_msg
    assert REDACTED in exc_msg
    assert mon_ref.final_summary["ecc/end_reason"] == "exception"

    assert mon_ref.console_log_path is not None
    log_text = Path(mon_ref.console_log_path).read_text(encoding="utf-8")
    assert "RuntimeError" in log_text
    assert "Traceback (most recent call last):" in log_text
    assert fake_secret not in log_text
    assert REDACTED in log_text


def test_context_manager_keyboard_interrupt() -> None:
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    mon_ref = None
    summary_at_finish: dict[str, object] = {}
    with pytest.raises(KeyboardInterrupt):
        with em.init(
            project="interrupt-test",
            capture_console=True,
            gpu_telemetry=False,
            heartbeat_interval_s=0,
            mode="offline",
        ) as mon:
            mon_ref = mon
            orig_finish = mon.run.finish

            def _spy_finish(exit_code: int | None = None, **kwargs: object) -> None:
                summary_at_finish.update(dict(mon.run.summary))
                orig_finish(exit_code=exit_code, **kwargs)

            mon.run.finish = _spy_finish  # type: ignore[method-assign]
            raise KeyboardInterrupt("user cancelled")

    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr
    assert mon_ref is not None
    assert summary_at_finish["ecc/end_reason"] == "interrupted"
    assert mon_ref.final_summary["ecc/end_reason"] == "interrupted"


def test_heartbeat_thread_updates_summary_and_stops_on_finish() -> None:
    mon = em.init(
        project="hb-test",
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0.2,
        mode="offline",
    )
    try:
        hb1 = mon.run.summary["ecc/heartbeat"]
        assert isinstance(hb1, float)
        time.sleep(0.55)
        hb2 = mon.run.summary["ecc/heartbeat"]
        assert hb2 > hb1
        assert mon._hb_thread is not None
        assert mon._hb_thread.is_alive()
    finally:
        mon.finish()

    assert mon._hb_thread is not None
    assert not mon._hb_thread.is_alive()


def test_login_from_notebook_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_keys: list[str] = []

    def _fake_login(*, key: str | None = None, relogin: bool | None = None) -> bool:
        assert relogin is False
        assert key is not None
        captured_keys.append(key)
        return True

    monkeypatch.setattr(wandb, "login", _fake_login)

    # 1. Missing everywhere -> False
    monkeypatch.delenv("TEST_WANDB_SECRET", raising=False)
    assert em.login_from_notebook_secrets("TEST_WANDB_SECRET") is False

    # 2. Environment fallback -> True
    fake_env_key = "0123456789" * 4
    monkeypatch.setenv("TEST_WANDB_SECRET", fake_env_key)
    assert em.login_from_notebook_secrets("TEST_WANDB_SECRET") is True
    assert captured_keys[-1] == fake_env_key

    # 3. Kaggle secrets takes precedence
    fake_kaggle_key = "abcdef0123" * 4
    fake_kaggle_mod = types.ModuleType("kaggle_secrets")

    class _UserSecretsClient:
        def get_secret(self, name: str) -> str:
            assert name == "TEST_WANDB_SECRET"
            return fake_kaggle_key

    fake_kaggle_mod.UserSecretsClient = _UserSecretsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle_mod)
    assert em.login_from_notebook_secrets("TEST_WANDB_SECRET") is True
    assert captured_keys[-1] == fake_kaggle_key

    # Secret must never be printed.
    out = capsys.readouterr()
    assert fake_env_key not in out.out and fake_env_key not in out.err
    assert fake_kaggle_key not in out.out and fake_kaggle_key not in out.err
