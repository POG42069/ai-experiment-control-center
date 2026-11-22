"""Tests for ecc.models, ecc.status, and ecc.redact."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from ecc.models import LogLine, Project, RunStatus, RunSummary
from ecc.redact import REDACTED, redact
from ecc.status import compute_eta, compute_progress, derive_status


class TestDeriveStatus:
    @pytest.mark.parametrize(
        "raw_state,expected",
        [
            # W&B states
            ("finished", RunStatus.FINISHED),
            ("FINISHED", RunStatus.FINISHED),
            ("failed", RunStatus.FAILED),
            ("crashed", RunStatus.CRASHED),
            ("killed", RunStatus.KILLED),
            ("preempted", RunStatus.KILLED),
            ("pending", RunStatus.QUEUED),
            # Kaggle states
            ("QUEUED", RunStatus.QUEUED),
            ("NEW_SCRIPT", RunStatus.QUEUED),
            ("COMPLETE", RunStatus.FINISHED),
            ("ERROR", RunStatus.FAILED),
            ("CANCEL_REQUESTED", RunStatus.KILLED),
            ("CANCEL_ACKNOWLEDGED", RunStatus.KILLED),
            # Unknown / invalid
            (None, RunStatus.UNKNOWN),
            ("", RunStatus.UNKNOWN),
            ("   ", RunStatus.UNKNOWN),
            ("something-weird", RunStatus.UNKNOWN),
        ],
    )
    def test_non_running_states(
        self, raw_state: str | None, expected: RunStatus, fixed_now: datetime
    ) -> None:
        assert derive_status(raw_state, None, fixed_now) == expected

    def test_running_without_heartbeat_is_unknown(self, fixed_now: datetime) -> None:
        assert derive_status("running", None, fixed_now) == RunStatus.UNKNOWN
        assert derive_status("RUNNING", None, fixed_now) == RunStatus.UNKNOWN

    def test_running_heartbeat_thresholds(self, fixed_now: datetime) -> None:
        # Slight future clock skew -> RUNNING
        assert derive_status("running", fixed_now + timedelta(seconds=5), fixed_now) == RunStatus.RUNNING
        # Fresh heartbeat -> RUNNING
        assert derive_status("running", fixed_now - timedelta(seconds=30), fixed_now) == RunStatus.RUNNING
        # Exact stale boundary (<= 180s) -> RUNNING
        assert derive_status("RUNNING", fixed_now - timedelta(seconds=180), fixed_now) == RunStatus.RUNNING
        # Just over stale boundary -> STALE
        assert derive_status("running", fixed_now - timedelta(seconds=181), fixed_now) == RunStatus.STALE
        # Exact offline boundary (<= 900s) -> STALE
        assert derive_status("running", fixed_now - timedelta(seconds=900), fixed_now) == RunStatus.STALE
        # Over offline boundary -> OFFLINE
        assert derive_status("running", fixed_now - timedelta(seconds=901), fixed_now) == RunStatus.OFFLINE

    def test_custom_thresholds(self, fixed_now: datetime) -> None:
        hb = fixed_now - timedelta(seconds=60)
        assert derive_status("running", hb, fixed_now, stale_after_s=30, offline_after_s=120) == RunStatus.STALE
        assert derive_status("running", hb, fixed_now, stale_after_s=30, offline_after_s=50) == RunStatus.OFFLINE

    def test_naive_datetimes_treated_as_utc(self) -> None:
        now = datetime(2026, 11, 22, 12, 0, 0)
        hb = datetime(2026, 11, 22, 11, 59, 0)
        assert derive_status("running", hb, now) == RunStatus.RUNNING


class TestEtaAndProgress:
    def test_compute_eta_linear(self, fixed_now: datetime) -> None:
        started = fixed_now - timedelta(seconds=200)
        eta = compute_eta(step=250, total_steps=1000, started_at=started, last_update=fixed_now)
        assert eta == pytest.approx(600.0)

    def test_compute_eta_edge_cases(self, fixed_now: datetime) -> None:
        started = fixed_now - timedelta(seconds=100)
        assert compute_eta(None, 1000, started, fixed_now) is None
        assert compute_eta(100, None, started, fixed_now) is None
        assert compute_eta(100, 1000, None, fixed_now) is None
        assert compute_eta(100, 1000, started, None) is None
        assert compute_eta(0, 1000, started, fixed_now) is None
        assert compute_eta(-5, 1000, started, fixed_now) is None
        assert compute_eta(100, 0, started, fixed_now) is None
        assert compute_eta(1000, 1000, started, fixed_now) == 0.0
        assert compute_eta(1200, 1000, started, fixed_now) == 0.0
        # Zero or negative elapsed
        assert compute_eta(100, 1000, fixed_now, fixed_now) is None
        assert compute_eta(100, 1000, fixed_now + timedelta(seconds=10), fixed_now) is None

    def test_compute_progress_clamping_and_fallback(self) -> None:
        assert compute_progress(250, 1000) == pytest.approx(0.25)
        assert compute_progress(1500, 1000) == 1.0
        assert compute_progress(-10, 1000) == 0.0
        assert compute_progress(100, 0) is None
        assert compute_progress(None, 1000) is None
        assert compute_progress(100, 1000, explicit=0.75) == pytest.approx(0.75)
        assert compute_progress(100, 1000, explicit=1.5) == 1.0
        assert compute_progress(100, 1000, explicit=-0.2) == 0.0
        # Non-finite explicit falls back to step/total_steps
        assert compute_progress(500, 1000, explicit=float("nan")) == pytest.approx(0.5)
        assert compute_progress(500, 1000, explicit=float("inf")) == pytest.approx(0.5)


class TestModelsSerialization:
    def test_to_dict_json_safe(self, fixed_now: datetime) -> None:
        summary = RunSummary(
            run_id="demo:demo/p/r1",
            source="demo",
            project="p",
            name="r1",
            status=RunStatus.RUNNING,
            raw_state="running",
            platform="kaggle",
            account="kaggle-1",
            gpu="T4",
            model_name="Qwen",
            config={"lr": 1e-4, "bad_nan": float("nan"), "bad_inf": float("inf")},
            tags=["t1"],
            step=10,
            total_steps=100,
            epoch=0.5,
            total_epochs=5.0,
            progress=0.1,
            latest={"train/loss": 0.5, "eval/loss": float("nan")},
            best_metric_name="eval/macro_f1",
            best_metric=float("inf"),
            best_step=10,
            best_epoch=0.5,
            started_at=fixed_now,
            ended_at=None,
            last_update=fixed_now,
            duration_s=60.0,
            eta_s=540.0,
            notebook_url=None,
            source_url=None,
            git_branch="main",
            git_sha="abc",
            dataset_version="v1",
            seed=42,
            is_demo=True,
        )
        d = summary.to_dict()
        assert d["status"] == "RUNNING"
        assert d["started_at"] == "2026-11-22T12:00:00+00:00"
        assert d["config"]["bad_nan"] is None
        assert d["config"]["bad_inf"] is None
        assert d["latest"]["eval/loss"] is None
        assert d["best_metric"] is None
        # Must serialize cleanly under strict JSON (allow_nan=False)
        encoded = json.dumps(d, allow_nan=False)
        assert "RUNNING" in encoded

        line = LogLine(ts=fixed_now, stream="stdout", text="hello", level="INFO").to_dict()
        assert line["ts"] == "2026-11-22T12:00:00+00:00"
        proj = Project(slug="p1", name="P1", tags=["a"]).to_dict()
        assert proj["slug"] == "p1"
        assert math.isnan(float("nan"))


class TestRedact:
    @pytest.mark.parametrize(
        "raw,secret_fragment",
        [
            ("wandb key a1b2c3d4e5f60718293a4b5c6d7e8f9012345678 failed", "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"),
            ("kaggle KGAT_1234567890abcdef rejected", "KGAT_1234567890abcdef"),
            ("hf token hf_AbCdEfGhIjKlMnOpQrSt", "hf_AbCdEfGhIjKlMnOpQrSt"),
            ("openai sk-proj-abcdef1234567890xyz", "sk-proj-abcdef1234567890xyz"),
            ("gh ghp_1234567890abcdef1234567890", "ghp_1234567890abcdef1234567890"),
            ("gho gho_1234567890abcdef1234567890", "gho_1234567890abcdef1234567890"),
            ("pat github_pat_11AA22BB33CC44DD55EE66FF", "github_pat_11AA22BB33CC44DD55EE66FF"),
            ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
            ("url?password=superSecret99&x=1", "superSecret99"),
            ("api_key=my_private_val", "my_private_val"),
            ("token: tok_999888777", "tok_999888777"),
            ("secret=\"top_secret_val\"", "top_secret_val"),
        ],
    )
    def test_redacts_secret_patterns(self, raw: str, secret_fragment: str) -> None:
        cleaned = redact(raw)
        assert secret_fragment not in cleaned
        assert REDACTED in cleaned

    def test_leaves_normal_text_untouched(self) -> None:
        msg = "Epoch 2/3 | Step 400/1000 | train/loss=0.4521 | eval/macro_f1=0.812"
        assert redact(msg) == msg
