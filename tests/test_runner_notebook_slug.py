"""Per-account runner slug templates and the one-cell notebook packaging of the runner."""

import ast
import json
from types import SimpleNamespace

import pytest

from ecc.launch.builder import build_kernel_folder, runner_notebook
from ecc.launch.runner_template import RunnerParams, render_runner
from ecc.launch.service import LaunchRefused, LaunchService


def _svc(template):
    svc = LaunchService.__new__(LaunchService)
    svc.settings = SimpleNamespace(launch=SimpleNamespace(runner_slug=template))
    return svc


def test_slug_template_owner_and_account():
    assert _svc("mmculture-qa-new-{owner}").runner_slug_for("alice", "kaggle-1") == "mmculture-qa-new-alice"
    assert _svc("runner-{account}").runner_slug_for("x", "kaggle-2") == "runner-kaggle-2"
    assert _svc("ecc-runner").runner_slug_for("x", "kaggle-1") == "ecc-runner"


@pytest.mark.parametrize("bad", ["run-{user}", "bad slug {owner}", "{owner}/../x"])
def test_bad_slug_templates_refused(bad):
    with pytest.raises(LaunchRefused):
        _svc(bad).runner_slug_for("alice", "kaggle-1")


def _params(**kw):
    base = dict(
        launch_id="abc", account="kaggle-1", owner="alice", runner_slug="mmculture-qa-new-alice",
        repo="o/r", branch="main", commit_sha="a" * 40, command="python train.py", preflight=True,
    )
    base.update(kw)
    return RunnerParams(**base)


def test_notebook_wraps_runner_and_raises_on_failure():
    nb = runner_notebook(render_runner(_params(), "QUJD"))
    src = nb["cells"][0]["source"]
    ast.parse(src)
    assert "raise SystemExit(main())" not in src
    assert "raise RuntimeError" in src
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 1


def test_kernel_folder_is_notebook_type_with_per_account_slug(tmp_path):
    folder = build_kernel_folder(_params(), b"wheel-bytes", tmp_path)
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "alice/mmculture-qa-new-alice"
    assert meta["kernel_type"] == "notebook"
    assert meta["code_file"] == "mmculture-qa-new-alice.ipynb"
    assert (folder / meta["code_file"]).is_file()
    assert meta["enable_gpu"] is False  # preflight


def test_header_prints_short_sha_only():
    src = render_runner(_params(), "QUJD")
    assert "COMMIT_SHA[:12]" in src
