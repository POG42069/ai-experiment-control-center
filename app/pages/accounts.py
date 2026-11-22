"""Connected Accounts page — multi-account status, read-only health checks, and keyring-backed onboarding."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from app.ui.components import (
    MISSING,
    ensure_standalone_chrome,
    fmt_text,
    render_empty_state,
)
from app.ui.state import (
    clear_service_cache,
    get_account_manager,
    get_service,
    mark_refreshed,
)
from ecc.accounts import AccountStatus


def _format_check_result(acc: AccountStatus) -> str:
    if acc.last_check_ok is None:
        return "Not checked"
    prefix = "OK" if acc.last_check_ok else "Failed"
    msg = (acc.last_check_msg or "").strip()
    return f"{prefix}: {msg}" if msg else prefix


def render_page() -> None:
    service = get_service()
    ensure_standalone_chrome(service)
    manager = get_account_manager(service)
    mark_refreshed()

    st.markdown("## 🔑 Connected Accounts")
    st.info(
        "Security note: Secrets are stored exclusively in your OS Credential Manager "
        "(`keyring`, service `ai-ecc`) and are never written to SQLite, config files, or displayed here. "
        "For the safest onboarding path (secret never enters browser memory), run "
        "`scripts\\add_account.py` in your terminal (`getpass`)."
    )

    accounts = manager.list()

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown("### Configured Accounts")
    with hdr_r:
        if st.button(
            "Check all",
            key="accounts_check_all_btn",
            disabled=not bool(accounts),
            width="stretch",
        ):
            results = manager.check_all()
            ok_cnt = sum(1 for r in results if r.last_check_ok)
            st.success(f"Checked {len(results)} account(s): {ok_cnt} OK.")
            accounts = manager.list()

    if not accounts:
        render_empty_state(
            "No connected accounts yet",
            "Add a Kaggle, W&B, or Colab account below (or via scripts\\add_account.py).",
        )
    else:
        table_rows: list[dict[str, Any]] = []
        for acc in accounts:
            table_rows.append(
                {
                    "Nickname": acc.nickname,
                    "Platform": acc.platform,
                    "Username": fmt_text(acc.username),
                    "Secret stored": "Yes" if acc.has_secret else "No",
                    "Last check time": fmt_text(acc.last_check_at),
                    "Last check result": _format_check_result(acc),
                    "Capabilities": "; ".join(acc.capabilities) if acc.capabilities else MISSING,
                    "Limitations": "; ".join(acc.limitations) if acc.limitations else MISSING,
                }
            )
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

        st.markdown("#### Per-Account Checks")
        for acc in accounts:
            c1, c2, c3, c4 = st.columns([1.4, 1.0, 2.8, 1.0])
            with c1:
                st.markdown(f"**`{acc.nickname}`** ({acc.platform})")
            with c2:
                secret_badge = "Secret: Yes" if acc.has_secret else "Secret: No"
                st.caption(secret_badge)
            with c3:
                st.caption(_format_check_result(acc))
            with c4:
                if st.button(
                    "Check",
                    key=f"account_check_{acc.nickname}",
                    width="stretch",
                ):
                    checked = manager.check(acc.nickname)
                    if checked.last_check_ok:
                        st.success(f"`{acc.nickname}`: {checked.last_check_msg}")
                    else:
                        st.warning(f"`{acc.nickname}`: {checked.last_check_msg}")
                    st.rerun()

    st.divider()
    col_add, col_remove = st.columns([1.4, 1.0])

    with col_add:
        st.markdown("### Add or Update Account")
        with st.form("add_account_form", clear_on_submit=True):
            platform = st.selectbox(
                "Platform",
                options=["kaggle", "wandb", "colab"],
                key="add_account_platform",
            )
            nickname = st.text_input(
                "Account nickname",
                placeholder="e.g. kaggle-1, wandb-main, colab-1",
                help="2–32 lowercase alphanumeric, '-', or '_' characters (never use your e-mail).",
                key="add_account_nickname",
            )
            username = st.text_input(
                "Username / alias (optional)",
                placeholder="Non-sensitive label or platform handle",
                key="add_account_username",
            )
            secret_input = st.text_input(
                "API Token / Key (stored in keyring, never displayed)",
                type="password",
                placeholder="Leave blank to keep existing secret or for Colab",
                key="add_account_secret",
            )
            submitted = st.form_submit_button("Save account", type="primary")
            if submitted:
                cleaned_nick = nickname.strip().lower()
                cleaned_user = username.strip()
                secret_to_store = secret_input if secret_input else None
                # Never echo or persist secret_input in session/UI messages.
                try:
                    manager.add_or_update(
                        nickname=cleaned_nick,
                        platform=platform,
                        username=cleaned_user,
                        secret=secret_to_store,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    clear_service_cache()
                    st.success(
                        f"Saved account `{cleaned_nick}` ({platform}). "
                        + (
                            "Credential stored in keyring."
                            if secret_to_store
                            else "No secret updated."
                        )
                    )
                    st.rerun()

    with col_remove:
        st.markdown("### Remove Account")
        if not accounts:
            st.caption("No accounts to remove.")
        else:
            rm_nick = st.selectbox(
                "Account to remove",
                options=[a.nickname for a in accounts],
                key="remove_account_select",
            )
            confirm_rm = st.checkbox(
                f"I confirm removing `{rm_nick}` and deleting its keyring secret.",
                value=False,
                key="remove_account_confirm",
            )
            if st.button(
                "Remove account",
                type="secondary",
                disabled=not confirm_rm,
                key="remove_account_btn",
                width="stretch",
            ):
                removed = manager.remove(rm_nick)
                clear_service_cache()
                if removed:
                    st.success(f"Removed account `{rm_nick}`.")
                else:
                    st.warning(f"Account `{rm_nick}` was not found.")
                st.rerun()

    st.divider()
    st.markdown("### Platform Setup & Limitations")
    h1, h2, h3 = st.columns(3)
    with h1:
        with st.container(border=True):
            st.markdown("#### 🔷 Kaggle")
            st.markdown(
                "- Create an API token at **[kaggle.com/settings/api](https://www.kaggle.com/settings/api)**.\n"
                "- **Read-only use:** ECC only runs `kaggle kernels list`, `status`, and `logs`.\n"
                "- Limitations: No GPU telemetry or structured training metrics natively — pair with `experiment_monitor` + W&B."
            )
    with h2:
        with st.container(border=True):
            st.markdown("#### 🟡 Weights & Biases (W&B)")
            st.markdown(
                "- Obtain your API key from **[wandb.ai/authorize](https://wandb.ai/authorize)**.\n"
                "- Primary source of truth for structured metrics, config, heartbeats, and live `ecc_console.log`.\n"
                "- Configure `entity` and `projects` in `ecc_config.toml` (never put the API key in TOML)."
            )
    with h3:
        with st.container(border=True):
            st.markdown("#### 🟠 Google Colab")
            st.markdown(
                "- **Not natively integrated**; add `experiment_monitor` to the notebook to stream metrics and console logs via W&B.\n"
                "- Colab has no official API for notebook execution status or logs.\n"
                "- You can still register a Colab nickname here as a label and store notebook URLs via `ecc/notebook_url`."
            )


render_page()
