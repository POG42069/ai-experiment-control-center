"""Validation and offline smoke tests for example Kaggle & Colab notebooks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from ecc.redact import redact

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

NOTEBOOK_CASES = [
    ("kaggle_example.ipynb", "kaggle-1"),
    ("colab_example.ipynb", "colab-1"),
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _cell_source_text(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _load_notebook(filename: str) -> tuple[str, dict]:
    path = NOTEBOOKS_DIR / filename
    assert path.is_file(), f"Missing notebook: {path}"
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return raw, data


@pytest.mark.parametrize("filename,expected_account", NOTEBOOK_CASES)
def test_notebook_nbformat4_structure(filename: str, expected_account: str) -> None:
    _raw, nb = _load_notebook(filename)

    assert nb.get("nbformat") == 4
    assert isinstance(nb.get("nbformat_minor"), int) and nb["nbformat_minor"] >= 0
    assert isinstance(nb.get("metadata"), dict)

    cells = nb.get("cells")
    assert isinstance(cells, list) and len(cells) >= 5

    valid_types = {"markdown", "code", "raw"}
    code_cells = 0
    for idx, cell in enumerate(cells):
        assert isinstance(cell, dict), f"Cell {idx} is not a dict"
        ctype = cell.get("cell_type")
        assert ctype in valid_types, f"Cell {idx} has invalid cell_type: {ctype!r}"
        assert isinstance(cell.get("metadata"), dict), f"Cell {idx} missing metadata dict"
        assert "source" in cell, f"Cell {idx} missing source"
        src = cell["source"]
        assert isinstance(src, (str, list)), f"Cell {idx} source must be str or list"
        if isinstance(src, list):
            assert all(isinstance(line, str) for line in src)
        if ctype == "code":
            code_cells += 1
            assert cell.get("outputs") == [], f"Code cell {idx} must have empty outputs"
            assert cell.get("execution_count") is None, f"Code cell {idx} must have null execution_count"

    assert code_cells >= 4
    full_source = "\n".join(_cell_source_text(c) for c in cells)
    assert f'account="{expected_account}"' in full_source


@pytest.mark.parametrize("filename,_expected_account", NOTEBOOK_CASES)
def test_notebook_contains_no_secrets_or_emails(filename: str, _expected_account: str) -> None:
    raw, nb = _load_notebook(filename)

    # Neither the raw JSON nor any individual cell source may trigger ecc.redact
    assert redact(raw) == raw, f"{filename} contains a secret-like pattern matched by ecc.redact"
    for idx, cell in enumerate(nb["cells"]):
        text = _cell_source_text(cell)
        assert redact(text) == text, f"{filename} cell {idx} triggered ecc.redact"
        assert not _EMAIL_RE.search(text), f"{filename} cell {idx} contains an e-mail address"


@pytest.mark.parametrize("filename,_expected_account", NOTEBOOK_CASES)
def test_notebook_contains_required_instrumentation_and_valid_python(
    filename: str, _expected_account: str
) -> None:
    _raw, nb = _load_notebook(filename)
    code_sources = [
        _cell_source_text(c) for c in nb["cells"] if c.get("cell_type") == "code"
    ]
    joined_code = "\n".join(code_sources)

    assert "em.init(" in joined_code
    assert "make_hf_callback" in joined_code
    assert "login_from_notebook_secrets" in joined_code
    assert "resume_id=" in joined_code
    assert "<YOUR_GH_USER>" in joined_code
    assert any("# ECC-SMOKE" in src for src in code_sources)

    # Every non-magic Python code cell must parse cleanly with ast.parse
    for idx, src in enumerate(code_sources):
        non_magic_lines = [
            line
            for line in src.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        ]
        cleaned = "\n".join(non_magic_lines).strip()
        if cleaned:
            ast.parse(cleaned, filename=f"{filename}:code_cell_{idx}")


@pytest.mark.parametrize("filename,_expected_account", NOTEBOOK_CASES)
def test_notebook_smoke_cells_execute_offline(
    filename: str,
    _expected_account: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    _raw, nb = _load_notebook(filename)
    smoke_cells = [
        _cell_source_text(c)
        for c in nb["cells"]
        if c.get("cell_type") == "code" and "# ECC-SMOKE" in _cell_source_text(c)
    ]
    assert smoke_cells, f"No # ECC-SMOKE code cell found in {filename}"

    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))
    monkeypatch.setenv("WANDB_SILENT", "true")

    namespace: dict = {}
    for idx, src in enumerate(smoke_cells):
        code = compile(src, f"{filename}::ECC-SMOKE-{idx}", "exec")
        exec(code, namespace)
