"""Streamlit AppTest end-to-end UI tests (offline demo mode, temp SQLite DB)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui.components import DEMO_BANNER_TEXT
from app.ui.state import clear_service_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"


@pytest.fixture
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary demo-mode ecc_config.toml with a temp SQLite db_path."""
    db_path = (tmp_path / "ui_test_ecc.db").as_posix()
    cfg_path = tmp_path / "ecc_config.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[general]",
                "refresh_seconds = 15",
                "stale_after_s = 180",
                "offline_after_s = 900",
                'demo_mode = "on"',
                f'db_path = "{db_path}"',
                "",
                "[wandb]",
                'entity = ""',
                "projects = []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ECC_CONFIG", str(cfg_path))
    clear_service_cache()
    yield cfg_path
    clear_service_cache()


def test_overview_page_shows_demo_banner_and_runs_cleanly(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages" / "overview.py"))
    at.run(timeout=30)

    assert not at.exception
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert DEMO_BANNER_TEXT in markdown_text


def test_main_entrypoint_loads_default_overview(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "main.py"))
    at.run(timeout=30)

    assert not at.exception
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert DEMO_BANNER_TEXT in markdown_text


def test_run_detail_page_with_demo_run_query_param(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages" / "run_detail.py"))
    at.query_params["run"] = "demo:demo/demo-shared-task/run-a"
    at.run(timeout=30)

    assert not at.exception
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert DEMO_BANNER_TEXT in markdown_text
    assert "Qwen3-VL-4B LoRA r16/a32" in markdown_text


def test_run_detail_page_detects_oom_exception_callout(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages" / "run_detail.py"))
    at.query_params["run"] = "demo:demo/demo-shared-task/run-e"
    at.run(timeout=30)

    assert not at.exception
    assert len(at.error) >= 1
    error_text = "\n".join(e.value for e in at.error)
    assert "CUDA out of memory" in error_text or "Traceback" in error_text


def test_compare_page_with_three_demo_runs_shows_warning(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages" / "compare.py"))
    at.query_params["runs"] = ",".join(
        [
            "demo:demo/demo-shared-task/run-a",
            "demo:demo/demo-shared-task/run-b",
            "demo:demo/demo-shared-task/run-c",
        ]
    )
    at.run(timeout=30)

    assert not at.exception
    assert len(at.warning) >= 1
    warning_text = "\n".join(w.value for w in at.warning)
    assert "eval_signature" in warning_text


def test_projects_page_runs_cleanly(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages" / "projects.py"))
    at.run(timeout=30)

    assert not at.exception
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert DEMO_BANNER_TEXT in markdown_text


def test_accounts_page_runs_cleanly_and_recommends_cli(demo_env: Path) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages" / "accounts.py"))
    at.run(timeout=30)

    assert not at.exception
    info_text = "\n".join(i.value for i in at.info)
    assert "add_account.py" in info_text


def test_ui_code_has_no_direct_network_or_sdk_imports() -> None:
    forbidden = {"wandb", "kaggle", "requests", "urllib", "httpx", "aiohttp", "socket"}
    for py_file in APP_DIR.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in forbidden, f"{py_file} imports forbidden module {top}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top not in forbidden, f"{py_file} imports from forbidden module {top}"
