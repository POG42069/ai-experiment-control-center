"""Unit tests for experiment_monitor.hf (pure mapping + optional transformers callback)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_monitor.hf import make_hf_callback, map_hf_logs


def test_map_hf_logs_pure() -> None:
    raw = {
        "loss": 0.42,
        "learning_rate": 2e-5,
        "grad_norm": 1.15,
        "epoch": 1.5,
        "eval_loss": 0.35,
        "eval_accuracy": 0.91,
        "eval_macro_f1": 0.88,
        "eval_runtime": 12.3,
        "eval_custom_score": 0.77,
        "train_runtime": 120.0,
    }
    mapped = map_hf_logs(raw)
    assert mapped["train/loss"] == 0.42
    assert mapped["train/lr"] == 2e-5
    assert mapped["train/grad_norm"] == 1.15
    assert mapped["train/epoch"] == 1.5
    assert mapped["eval/loss"] == 0.35
    assert mapped["eval/accuracy"] == 0.91
    assert mapped["eval/macro_f1"] == 0.88
    assert mapped["eval/runtime"] == 12.3
    assert mapped["eval/custom_score"] == 0.77
    assert mapped["train_runtime"] == 120.0
    assert "loss" not in mapped
    assert "eval_loss" not in mapped


def test_map_hf_logs_f1_macro_alias() -> None:
    mapped = map_hf_logs({"eval_f1_macro": 0.84})
    assert mapped == {"eval/macro_f1": 0.84}


def test_hf_callback_with_transformers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")

    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb_config"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb_cache"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb_data"))
    monkeypatch.setenv("WANDB_SILENT", "true")

    import experiment_monitor as em

    ckpt_dir = tmp_path / "out" / "checkpoint-10"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.bin").write_bytes(b"0" * 1024)

    with em.init(
        "test-hf",
        mode="offline",
        capture_console=False,
        gpu_telemetry=False,
        heartbeat_interval_s=0,
    ) as mon:
        cb = make_hf_callback(mon)
        assert isinstance(cb, transformers.TrainerCallback)

        args = SimpleNamespace(num_train_epochs=3.0, output_dir=str(tmp_path / "out"))
        state = SimpleNamespace(max_steps=100, global_step=10, epoch=0.3)
        control = SimpleNamespace()

        cb.on_train_begin(args, state, control)
        assert mon.total_steps == 100
        assert mon.total_epochs == 3.0

        cb.on_log(args, state, control, logs={"loss": 0.5, "eval_macro_f1": 0.81, "epoch": 0.3})
        assert mon.run.summary["ecc/best_metric"] == 0.81

        cb.on_save(args, state, control)
        ckpts = mon.run.summary["ecc/checkpoints"]
        assert len(ckpts) == 1
        assert ckpts[0]["step"] == 10
        assert ckpts[0]["path"] == os.path.join(str(tmp_path / "out"), "checkpoint-10")
