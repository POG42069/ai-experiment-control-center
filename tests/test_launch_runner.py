"""Tests for ecc.launch.runner_template (AST validity, no secret leaks, helpers)."""

from __future__ import annotations

import ast
import base64

import pytest

from ecc.launch.runner_template import (
    REDACTED,
    RunnerParams,
    build_clone_url,
    redact_runner_text,
    render_runner,
)


def _sample_params(*, monitor_mode: str = "runner", preflight: bool = False) -> RunnerParams:
    return RunnerParams(
        launch_id="a1b2c3d4e5f60718293a4b5c6d7e8f90",
        account="kaggle-1",
        owner="alice",
        runner_slug="ecc-runner",
        repo="alice/vlm-exp",
        branch="feat/lora-v2",
        commit_sha="0123456789abcdef0123456789abcdef01234567",
        command="python train.py --epochs 3 --lr 2e-4",
        monitor_mode=monitor_mode,
        project="shared-task-4",
        run_name="qwen3-lora-r16",
        requirements_file="requirements.txt",
        preflight=preflight,
        wandb_credential_label="WANDB_API_KEY",
        github_credential_label="GITHUB_TOKEN",
        accelerator="none" if preflight else "NvidiaTeslaT4",
    )


class TestRunnerRender:
    @pytest.mark.parametrize(
        "monitor_mode,preflight",
        [
            ("runner", True),
            ("runner", False),
            ("script", False),
        ],
    )
    def test_rendered_script_parses_cleanly_for_all_modes(
        self, monitor_mode: str, preflight: bool
    ) -> None:
        wheel_b64 = base64.b64encode(b"PK\x03\x04fake-wheel-payload").decode("ascii")
        params = _sample_params(monitor_mode=monitor_mode, preflight=preflight)
        source = render_runner(params, wheel_b64)

        tree = ast.parse(source)
        assert tree is not None

        # Must reference secret LABELS, never secret values
        assert "WANDB_API_KEY" in source
        assert "GITHUB_TOKEN" in source
        assert "https://www.kaggle.com/code/alice/ecc-runner" in source
        assert "Add-ons -> Secrets" in source
        assert "sys.exit" in source or "raise SystemExit" in source
        assert "ECC_PROJECT" in source
        assert "ECC_LAUNCH_ID" in source
        assert "ECC_GIT_REPO" in source

    def test_rendered_script_contains_no_secret_values_only_labels(self) -> None:
        fake_wandb_val = "wandb_v1_SECRETVALUE_NEVER_EMBED_12345678"
        fake_gh_val = "ghp_SECRETGITHUBVALUE_NEVER_EMBED_123456"
        params = RunnerParams(
            launch_id="deadbeef",
            account="kaggle-2",
            owner="bob",
            runner_slug="ecc-runner",
            repo="bob/private-repo",
            branch="main",
            commit_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            command="python run.py",
            monitor_mode="script",
            project="proj-x",
            run_name="exp-1",
            preflight=True,
            wandb_credential_label="CUSTOM_WANDB_LABEL",
            github_credential_label="CUSTOM_GH_LABEL",
        )
        wheel_b64 = base64.b64encode(b"fake-whl").decode("ascii")
        source = render_runner(params, wheel_b64)

        ast.parse(source)
        assert "CUSTOM_WANDB_LABEL" in source
        assert "CUSTOM_GH_LABEL" in source
        assert fake_wandb_val not in source
        assert fake_gh_val not in source

    def test_invalid_monitor_mode_rejected(self) -> None:
        params = _sample_params(monitor_mode="invalid")
        with pytest.raises(ValueError):
            render_runner(params, "")


class TestRunnerHelpers:
    def test_build_clone_url_public_and_authenticated(self) -> None:
        assert build_clone_url("alice/repo", None) == "https://github.com/alice/repo.git"
        assert build_clone_url("alice/repo", "") == "https://github.com/alice/repo.git"
        assert build_clone_url("alice/repo", "   ") == "https://github.com/alice/repo.git"

        tok = "ghp_fakeToken1234567890abcdef"
        assert (
            build_clone_url("alice/repo", tok)
            == f"https://x-access-token:{tok}@github.com/alice/repo.git"
        )

    def test_redact_runner_text_masks_tokens_urls_and_preserves_commit_sha(self) -> None:
        gh_tok = "ghp_superSecretGitHubToken123456"
        wb_key = "0123456789abcdef0123456789abcdef01234567"
        commit_sha = "fedcba9876543210fedcba9876543210fedcba98"
        url = build_clone_url("alice/private", gh_tok)

        raw = (
            f"fatal: could not read from {url}\n"
            f"WANDB_API_KEY={wb_key}\n"
            f"also raw {wb_key} and Bearer abcdefgh12345678\n"
            f"commit {commit_sha} checked out\n"
        )
        cleaned = redact_runner_text(raw, secrets=[wb_key, gh_tok])

        assert gh_tok not in cleaned
        assert wb_key not in cleaned
        assert "abcdefgh12345678" not in cleaned
        assert f"https://x-access-token:{REDACTED}@github.com/alice/private.git" in cleaned
        # Commit SHA (not in secrets) must remain intact
        assert commit_sha in cleaned
