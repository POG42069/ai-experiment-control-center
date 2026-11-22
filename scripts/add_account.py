"""Interactive CLI for adding, removing, checking, and listing ECC accounts.

Secrets are read with ``getpass.getpass()`` (never echoed, never passed via
argv) and stored in the OS keyring via ``SecretStore`` (service ``"ai-ecc"``).
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

# Allow running directly as `python scripts/add_account.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ecc.accounts import AccountManager, AccountStatus, validate_nickname, validate_platform
from ecc.config import ConfigError, load_config
from ecc.redact import redact
from ecc.secrets import SecretStore
from ecc.service import _open_store_with_fallback


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _format_table(accounts: list[AccountStatus]) -> str:
    if not accounts:
        return "No accounts configured."

    headers = ["NICKNAME", "PLATFORM", "USERNAME", "ENABLED", "SECRET", "LAST_CHECK", "MESSAGE"]
    rows: list[list[str]] = []
    for acc in accounts:
        if acc.last_check_ok is True:
            check_str = "OK"
        elif acc.last_check_ok is False:
            check_str = "FAIL"
        else:
            check_str = "—"
        msg = redact(acc.last_check_msg or "—")
        if len(msg) > 60:
            msg = msg[:57] + "..."
        rows.append(
            [
                acc.nickname,
                acc.platform,
                acc.username or "—",
                "yes" if acc.enabled else "no",
                "set" if acc.has_secret else "missing",
                check_str,
                msg,
            ]
        )

    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [
        fmt.format(*headers),
        "  ".join("-" * w for w in widths),
        *(fmt.format(*r) for r in rows),
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/add_account.py",
        description="Manage connected ECC accounts (secrets stored in OS keyring, never argv).",
    )
    parser.add_argument(
        "--platform",
        choices=["kaggle", "colab", "wandb"],
        default=None,
        help="Account platform (kaggle | colab | wandb)",
    )
    parser.add_argument(
        "--nickname",
        default=None,
        help="Account nickname (2-32 chars, ^[a-z0-9][a-z0-9_-]{1,31}$)",
    )
    parser.add_argument(
        "--username",
        default="",
        help="Non-sensitive platform username/alias",
    )
    parser.add_argument(
        "--remove",
        nargs="?",
        const=True,
        default=None,
        metavar="NICKNAME",
        help="Remove an account and delete its keyring secret",
    )
    parser.add_argument(
        "--list",
        dest="list_accounts",
        action="store_true",
        help="Print configured accounts table (without secrets)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip credential verification check after adding/updating",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to ecc_config.toml",
    )
    parser.add_argument(
        "--from-clipboard",
        action="store_true",
        help="Read the secret from the Windows clipboard (never echoed), then clear the clipboard",
    )
    return parser


def _read_secret_from_clipboard() -> str | None:
    """Read the secret from the Windows clipboard and clear it afterwards. Never prints the value."""
    import subprocess

    ps = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    try:
        proc = subprocess.run(
            ps + ["[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"],
            capture_output=True,
            timeout=15,
        )
        value = proc.stdout.decode("utf-8", errors="replace").strip()
    finally:
        subprocess.run(ps + ["Set-Clipboard -Value ' '"], capture_output=True, timeout=15)
    if not value or any(ch.isspace() for ch in value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_config(args.config)
    except ConfigError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 1

    store = _open_store_with_fallback(settings.db_path)
    secrets = SecretStore()
    manager = AccountManager(store=store, secrets=secrets, settings=settings)

    if args.list_accounts:
        print(_format_table(manager.list()))
        return 0

    if args.remove is not None:
        target = args.remove if isinstance(args.remove, str) else args.nickname
        if not target:
            sys.stderr.write("Error: --remove requires a nickname (via --remove NICK or --nickname NICK)\n")
            return 1
        removed = manager.remove(target)
        if removed:
            print(f"Removed account '{target}' and deleted its keyring entry (if present).")
            return 0
        print(f"Account '{target}' was not found.")
        return 1

    if not args.platform or not args.nickname:
        parser.print_help(sys.stderr)
        sys.stderr.write("\nError: --platform and --nickname are required (or use --list / --remove).\n")
        return 1

    try:
        validate_platform(args.platform)
        validate_nickname(args.nickname)
    except ValueError as exc:
        sys.stderr.write(f"Validation error: {exc}\n")
        return 1

    secret: str | None = None
    if args.platform in {"kaggle", "wandb"}:
        if args.from_clipboard:
            input(
                f"Copy the {args.platform} token for '{args.nickname}' to the clipboard now, "
                "then press Enter here... "
            )
            secret = _read_secret_from_clipboard()
            if secret is None:
                sys.stderr.write(
                    "Clipboard is empty or does not look like a single token. "
                    "Copy the token first, then run this command again.\n"
                )
                return 1
            print(f"Read {args.platform} secret from clipboard ({len(secret)} chars); clipboard cleared.")
        else:
            prompt = f"Enter {args.platform} API secret for '{args.nickname}' (input hidden): "
            try:
                entered = getpass.getpass(prompt, echo_char="*").strip()  # Python 3.14+: show '*'
            except TypeError:
                entered = getpass.getpass(prompt).strip()
            secret = entered if entered else None

    manager.add_or_update(
        nickname=args.nickname,
        platform=args.platform,
        username=args.username,
        secret=secret,
    )

    if args.no_check:
        print(f"Saved account '{args.nickname}' ({args.platform}).")
        return 0

    status = manager.check(args.nickname)
    state_label = "OK" if status.last_check_ok else "FAIL"
    print(f"[{state_label}] {status.nickname} ({status.platform}): {redact(status.last_check_msg)}")
    return 0 if status.last_check_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
