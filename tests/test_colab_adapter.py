"""Tests for ecc.adapters.colab capability stub."""

from __future__ import annotations

import ecc.adapters.colab as colab_mod
from ecc.adapters.colab import COLAB_CAPABILITY, ColabCapability, ColabSource
from ecc.sources.base import RunSource


class TestColabAdapter:
    def test_colab_source_implements_run_source_and_reports_unhealthy(self) -> None:
        src = ColabSource()
        assert isinstance(src, RunSource)
        assert src.name == "colab"
        assert src.list_runs() == []
        assert src.list_runs(project="any") == []
        assert src.get_run("colab:colab-1/foo") is None
        assert src.get_metrics("colab:colab-1/foo").empty
        assert src.get_logs("colab:colab-1/foo") == []
        assert src.list_projects() == []

        ok, msg = src.health()
        assert ok is False
        assert "not integrated" in msg.lower()
        assert "w&b" in msg.lower()

    def test_colab_capability_strings_match_contract(self) -> None:
        cap = ColabCapability()
        assert cap == COLAB_CAPABILITY
        assert cap.native_integrated is False
        assert cap.capabilities == []
        assert cap.limitations == [
            "no official API for notebook execution status/logs",
            "runtime list API needs Google Cloud OAuth and is not enabled in MVP",
        ]
        as_dict = cap.to_dict()
        assert as_dict["platform"] == "colab"
        assert as_dict["capabilities"] == []
        assert len(as_dict["limitations"]) == 2

    def test_module_docstring_documents_phase3_readonly_oauth_design(self) -> None:
        doc = colab_mod.__doc__ or ""
        assert "/v1beta/runtimes" in doc
        assert "/v1beta/subscription" in doc
        assert "LOCALAPPDATA" in doc
        assert "google" in doc.lower()
