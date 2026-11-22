"""Streamlit session state, cached service access, and navigation helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Iterator

import streamlit as st

from ecc.accounts import AccountManager
from ecc.launch.service import LaunchService, build_launch_service
from ecc.secrets import SecretStore
from ecc.service import Service, build_service

_IN_MAIN_ENTRYPOINT: ContextVar[bool] = ContextVar("_IN_MAIN_ENTRYPOINT", default=False)

MIN_REFRESH_SECONDS = 10
MAX_REFRESH_SECONDS = 60


@contextmanager
def main_entrypoint_context() -> Iterator[None]:
    """Mark execution as running inside app/main.py so pages skip duplicate chrome."""
    token = _IN_MAIN_ENTRYPOINT.set(True)
    try:
        yield
    finally:
        _IN_MAIN_ENTRYPOINT.reset(token)


def in_main_entrypoint() -> bool:
    """Return True when the current page is executing inside app/main.py."""
    return _IN_MAIN_ENTRYPOINT.get()


@st.cache_resource(show_spinner=False)
def _cached_service(config_key: str) -> Service:
    return build_service(config_key or None)


def get_service() -> Service:
    """Return the cached ECC Service keyed by the active ECC_CONFIG path."""
    config_key = os.environ.get("ECC_CONFIG", "")
    return _cached_service(config_key)


@st.cache_resource(show_spinner=False)
def _cached_launch_service(config_key: str) -> LaunchService:
    return build_launch_service(_cached_service(config_key))


def get_launch_service() -> LaunchService:
    """Return the cached LaunchService keyed by the active ECC_CONFIG path."""
    config_key = os.environ.get("ECC_CONFIG", "")
    return _cached_launch_service(config_key)


def clear_service_cache() -> None:
    """Clear the cached Service and LaunchService so config/account changes take effect."""
    _cached_service.clear()
    _cached_launch_service.clear()


def get_account_manager(service: Service) -> AccountManager:
    """Build a public AccountManager bound to the active service's store and settings."""
    return AccountManager(
        store=service.store,
        secrets=SecretStore(),
        settings=service.settings,
    )


def default_refresh_interval(service: Service) -> int:
    """Clamp configured refresh_seconds into the UI slider range [10, 60]."""
    raw = int(getattr(service.settings, "refresh_seconds", 15))
    return max(MIN_REFRESH_SECONDS, min(MAX_REFRESH_SECONDS, raw))


def init_ui_state(service: Service) -> None:
    """Initialize shared sidebar/refresh keys in st.session_state."""
    if "ecc_selected_project" not in st.session_state:
        st.session_state["ecc_selected_project"] = "All"
    if "ecc_auto_refresh" not in st.session_state:
        st.session_state["ecc_auto_refresh"] = True
    if "ecc_refresh_interval" not in st.session_state:
        st.session_state["ecc_refresh_interval"] = default_refresh_interval(service)
    if "ecc_last_refreshed" not in st.session_state:
        st.session_state["ecc_last_refreshed"] = datetime.now().strftime("%H:%M:%S")


def mark_refreshed() -> str:
    """Record current wall-clock HH:MM:SS as last refresh timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["ecc_last_refreshed"] = ts
    return ts


def get_selected_project() -> str | None:
    """Return the selected project slug, or None when 'All' is selected."""
    val = st.session_state.get("ecc_selected_project", "All")
    if not val or val == "All":
        return None
    return str(val)


def get_refresh_settings(service: Service) -> tuple[bool, int]:
    """Return (auto_refresh_enabled, interval_seconds)."""
    init_ui_state(service)
    enabled = bool(st.session_state.get("ecc_auto_refresh", True))
    interval = int(
        st.session_state.get("ecc_refresh_interval", default_refresh_interval(service))
    )
    interval = max(MIN_REFRESH_SECONDS, min(MAX_REFRESH_SECONDS, interval))
    return enabled, interval


def parse_query_run_ids() -> list[str]:
    """Parse comparison run IDs from query params (?runs=a,b,c or repeated ?run=)."""
    ids: list[str] = []
    raw_list: list[str] = []
    if hasattr(st.query_params, "get_all"):
        raw_list.extend(st.query_params.get_all("runs"))
        raw_list.extend(st.query_params.get_all("run_ids"))
        if not raw_list:
            raw_list.extend(st.query_params.get_all("run"))
    else:
        for key in ("runs", "run_ids", "run"):
            val = st.query_params.get(key)
            if val:
                raw_list.append(str(val))
                break

    for item in raw_list:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in ids:
                ids.append(cleaned)
    return ids


def navigate_to_run_detail(run_id: str) -> None:
    """Switch to the Run Detail page with the target run_id in query params."""
    st.session_state["ecc_active_run_id"] = run_id
    try:
        st.switch_page("pages/run_detail.py", query_params={"run": run_id})
    except Exception:
        st.query_params["run"] = run_id
        st.rerun()


def navigate_to_compare(run_ids: list[str]) -> None:
    """Switch to the Compare page with the selected run_ids in query params."""
    cleaned = [rid for rid in run_ids if rid]
    st.session_state["ecc_compare_run_ids"] = cleaned
    joined = ",".join(cleaned)
    try:
        st.switch_page("pages/compare.py", query_params={"runs": joined})
    except Exception:
        st.query_params["runs"] = joined
        st.rerun()
