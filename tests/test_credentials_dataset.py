"""Credentials-dataset wiring: config, dataset_sources in kernel metadata, runner fallback code."""

import ast
import json
from types import SimpleNamespace

import pytest

from ecc.launch.builder import build_kernel_folder
from ecc.launch.runner_template import RunnerParams, render_runner
from ecc.launch.service import LaunchRefused, LaunchService


def _svc(template):
    svc = LaunchService.__new__(LaunchService)
    svc.settings = SimpleNamespace(launch=SimpleNamespace(credentials_dataset=template))
    return svc


def test_dataset_sources_template():
    assert _svc("{owner}/ecc-credentials").credentials_dataset_sources("carol", "kaggle-3") == (
        "carol/ecc-credentials",
    )
    assert _svc("").credentials_dataset_sources("carol", "kaggle-3") == ()


@pytest.mark.parametrize("bad", ["{user}/x", "no-slash", "{owner}/a b"])
def test_bad_dataset_templates(bad):
    with pytest.raises(LaunchRefused):
        _svc(bad).credentials_dataset_sources("carol", "kaggle-3")


def _params(**kw):
    base = dict(launch_id="l", account="kaggle-3", owner="carol", runner_slug="mmculture-qa-new-carol",
                repo="o/r", branch="main", commit_sha="b" * 40, command="python t.py", preflight=True)
    base.update(kw)
    return RunnerParams(**base)


def test_metadata_attaches_credentials_dataset(tmp_path):
    folder = build_kernel_folder(_params(dataset_sources=("carol/ecc-credentials",)), b"w", tmp_path)
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["carol/ecc-credentials"]


def test_runner_reads_credentials_file_before_secrets():
    src = render_runner(_params(), "QUJD")
    ast.parse(src)
    assert "ecc_credentials.json" in src
    assert src.index("_load_credentials_file()") < src.index("_get_kaggle_secret(WANDB_CREDENTIAL_LABEL)")


def test_config_accepts_credentials_dataset(tmp_path):
    from ecc.config import load_config

    cfg = tmp_path / "c.toml"
    cfg.write_text('[launch]\nenabled = true\ncredentials_dataset = "{owner}/ecc-credentials"\n', encoding="utf-8")
    assert load_config(cfg).launch.credentials_dataset == "{owner}/ecc-credentials"
