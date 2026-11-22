"""Streamlit multipage entrypoint for AI Experiment Control Center.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.ui.components import inject_global_css, render_sidebar
from app.ui.state import get_service, main_entrypoint_context


def main() -> None:
    st.set_page_config(
        page_title="AI Experiment Control Center",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()

    service = get_service()
    render_sidebar(service)

    pages = [
        st.Page(
            "pages/overview.py",
            title="Overview",
            icon="📊",
            default=True,
            url_path="overview",
        ),
        st.Page(
            "pages/launch.py",
            title="Launch",
            icon="🚀",
            url_path="launch",
        ),
        st.Page(
            "pages/run_detail.py",
            title="Run Detail",
            icon="🔍",
            url_path="run_detail",
        ),
        st.Page(
            "pages/compare.py",
            title="Compare",
            icon="📈",
            url_path="compare",
        ),
        st.Page(
            "pages/projects.py",
            title="Projects",
            icon="📁",
            url_path="projects",
        ),
        st.Page(
            "pages/accounts.py",
            title="Connected Accounts",
            icon="🔑",
            url_path="accounts",
        ),
    ]

    nav = st.navigation(pages)
    with main_entrypoint_context():
        nav.run()


if __name__ == "__main__":
    main()
