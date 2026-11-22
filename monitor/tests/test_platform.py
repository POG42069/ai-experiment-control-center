"""Unit tests for experiment_monitor.platform."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from experiment_monitor.platform import detect_git_info, detect_platform

_PLATFORM_ENV_KEYS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE",
    "COLAB_RELEASE_TAG",
    "COLAB_GPU",
)


def _clear_platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PLATFORM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)


def test_detect_platform_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_platform_env(monkeypatch)
    assert detect_platform() == "local"


def test_detect_platform_kaggle_run_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert detect_platform() == "kaggle"


def test_detect_platform_kaggle_url_base(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_URL_BASE", "https://www.kaggle.com")
    assert detect_platform() == "kaggle"


def test_detect_platform_colab_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-20261122")
    assert detect_platform() == "colab"

    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("COLAB_GPU", "1")
    assert detect_platform() == "colab"


def test_detect_platform_colab_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_platform_env(monkeypatch)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "google.colab" else None,
    )
    assert detect_platform() == "colab"


def test_detect_git_info_non_git_dir(tmp_path: Path) -> None:
    branch, sha = detect_git_info(cwd=str(tmp_path))
    assert branch == ""
    assert sha == ""
