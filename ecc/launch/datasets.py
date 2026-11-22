"""Private Kaggle dataset uploader with strict allow-list, secret/size guards, and audit log.

Security rules:
- Dedicated allow-list permitting ONLY ``("datasets", "create")``,
  ``("datasets", "version")``, and ``("datasets", "status")``.
- Allowed flags ONLY: ``-p``, ``-m``, ``-r`` (zip|tar|skip), ``--dir-mode``, ``-q``.
- ``--public`` / ``-u`` (and any public flag variant) MUST be rejected — datasets are
  always private.
- Token is injected strictly via subprocess ``env=`` (never argv) with an isolated
  per-account ``KAGGLE_CONFIG_DIR`` and all inherited ``KAGGLE_*`` env vars stripped.
- Source folder is copied into a temporary staging directory before writing
  ``dataset-metadata.json`` so the user's folder is never modified.
- Folders containing obvious secret filenames or text files (<= 1 MB) matching
  ``ecc.redact`` patterns are refused without printing file contents.
- Folders exceeding 20 GB are refused.
- All subprocess outputs and error messages are scrubbed through ``ecc.redact.redact``
  and active-token replacement.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ecc.config import Settings
from ecc.launch.builder import build_wheel, default_monitor_dir
from ecc.launch.service import LaunchService
from ecc.redact import REDACTED, redact
from ecc.secrets import SecretStore
from ecc.store import Store

ALLOWED_DATASET_SUBCOMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("datasets", "create"),
        ("datasets", "version"),
        ("datasets", "status"),
    }
)

ALLOWED_DIR_MODES: frozenset[str] = frozenset({"zip", "tar", "skip"})

_VALUE_FLAGS: frozenset[str] = frozenset({"-p", "-m", "-r", "--dir-mode"})
_BOOL_FLAGS: frozenset[str] = frozenset({"-q"})

MAX_DATASET_SIZE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB
SECRET_SCAN_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,49}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_DATASET_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}/[a-z0-9][a-z0-9-]{2,49}$")
_DATASET_URL_RE = re.compile(r"(https://www\.kaggle\.com/datasets/[A-Za-z0-9_./-]+)")

TITLE_MIN_LEN = 6
TITLE_MAX_LEN = 50


class DatasetUploadRefused(RuntimeError):
    """Raised when a dataset upload violates safety, policy, or validation rules."""


class KaggleDatasetError(RuntimeError):
    """Raised when invoking Kaggle CLI for a dataset operation fails unexpectedly."""


@dataclass(frozen=True)
class UploadResult:
    ok: bool
    ref: str
    url: str | None
    message: str
    size_bytes: int = 0
    action: str = "created"  # "created" | "versioned" | "failed"

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ref": self.ref,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_mb,
            "action": self.action,
            "created": self.action == "created" and self.ok,
            "versioned": self.action == "versioned" and self.ok,
            "message": self.message,
        }


def _default_kaggle_exe() -> str:
    candidate = Path(sys.executable).parent / "kaggle.exe"
    if candidate.is_file():
        return str(candidate)
    return "kaggle"


def _sanitize_text(text: str, token: str | None = None) -> str:
    cleaned = text or ""
    if token:
        cleaned = cleaned.replace(token, REDACTED)
    return redact(cleaned)


def validate_dataset_slug(slug: str) -> str:
    """Validate Kaggle dataset slug against ``^[a-z0-9][a-z0-9-]{2,49}$``."""
    if not isinstance(slug, str):
        raise DatasetUploadRefused("Dataset slug must be a string")
    s = slug.strip()
    if not _SLUG_RE.match(s):
        raise DatasetUploadRefused(
            f"Invalid dataset slug {slug!r}: must match ^[a-z0-9][a-z0-9-]{{2,49}}$ "
            "(3-50 chars, lowercase letters, digits, hyphens, starting with alphanumeric)"
        )
    return s


def validate_dataset_title(title: str) -> str:
    """Validate Kaggle dataset title length (6-50 characters, single line)."""
    if not isinstance(title, str):
        raise DatasetUploadRefused("Dataset title must be a string")
    if "\n" in title or "\r" in title:
        raise DatasetUploadRefused("Dataset title must be a single line")
    t = title.strip()
    if not (TITLE_MIN_LEN <= len(t) <= TITLE_MAX_LEN):
        raise DatasetUploadRefused(
            f"Dataset title length must be between {TITLE_MIN_LEN} and {TITLE_MAX_LEN} "
            f"characters, got {len(t)}"
        )
    return t


def validate_dataset_owner(owner: str) -> str:
    """Validate Kaggle dataset owner username."""
    if not isinstance(owner, str):
        raise DatasetUploadRefused("Dataset owner must be a string")
    o = owner.strip()
    if not _OWNER_RE.match(o):
        raise DatasetUploadRefused(f"Invalid dataset owner {owner!r}")
    return o


def _is_secret_filename(filename: str) -> bool:
    name = filename.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name == "kaggle.json":
        return True
    if name.startswith("access_token"):
        return True
    if name.endswith(".pem"):
        return True
    if name.startswith("id_rsa"):
        return True
    return False


def scan_folder_for_secrets(folder: str | Path) -> list[str]:
    """Return sorted relative POSIX paths of files that look like secrets.

    Checks both forbidden filenames (``.env``, ``kaggle.json``, ``access_token*``,
    ``*.pem``, ``id_rsa*``) and content of UTF-8 text files <= 1 MiB against
    ``ecc.redact`` patterns. Never returns file contents.
    """
    root = Path(folder)
    offending: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_secret_filename(path.name):
            offending.append(rel)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > SECRET_SCAN_MAX_BYTES:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        # Skip binary files (e.g. .whl zip archives, images, weights).
        if b"\x00" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if redact(text) != text:
            offending.append(rel)
    return offending


def compute_folder_size(folder: str | Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for all regular files under ``folder``."""
    root = Path(folder)
    total = 0
    count = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
            count += 1
    return total, count


class KaggleDatasetUploader:
    """Dedicated private Kaggle dataset uploader with strict allow-list and guards."""

    def __init__(
        self,
        nickname: str,
        token_getter: Callable[[], str | None],
        kaggle_exe: str | Path | None = None,
        runner: Callable[..., Any] = subprocess.run,
        timeout_s: float = 900,
        max_size_bytes: int = MAX_DATASET_SIZE_BYTES,
    ) -> None:
        self.nickname = nickname
        self._token_getter = token_getter
        self.kaggle_exe = str(kaggle_exe) if kaggle_exe is not None else _default_kaggle_exe()
        self._runner = runner
        self.timeout_s = timeout_s
        self.max_size_bytes = max_size_bytes

    def _config_dir(self) -> Path:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            target = Path(local_app_data) / "ai-ecc" / "kaggle" / self.nickname
        else:
            target = Path.home() / ".ai-ecc" / "kaggle" / self.nickname
        try:
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "ai-ecc" / "kaggle" / self.nickname
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def build_env(self, token: str) -> dict[str, str]:
        """Build isolated subprocess environment identical to ``KaggleCLI.build_env``."""
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("KAGGLE_")}
        env["KAGGLE_API_TOKEN"] = token
        env["KAGGLE_CONFIG_DIR"] = str(self._config_dir())
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def _validate_command(args: Sequence[str]) -> None:
        if len(args) < 2:
            raise PermissionError(f"Disallowed Kaggle dataset command: {list(args)!r}")

        cmd_pair = (args[0].strip().lower(), args[1].strip().lower())
        if cmd_pair not in ALLOWED_DATASET_SUBCOMMANDS:
            raise PermissionError(
                f"Disallowed Kaggle dataset subcommand {cmd_pair!r}; "
                f"only {sorted(ALLOWED_DATASET_SUBCOMMANDS)} are permitted"
            )

        if cmd_pair == ("datasets", "status"):
            positionals: list[str] = []
            for raw_tok in args[2:]:
                tok = str(raw_tok).strip()
                if tok.startswith("-"):
                    if tok in _BOOL_FLAGS:
                        continue
                    raise PermissionError(
                        f"Disallowed flag for kaggle datasets status: {tok!r}"
                    )
                positionals.append(tok)
            if len(positionals) != 1 or not _DATASET_REF_RE.match(positionals[0]):
                raise PermissionError(
                    f"datasets status requires a single owner/slug argument, got {positionals!r}"
                )
            return

        # ("datasets", "create") or ("datasets", "version")
        idx = 2
        while idx < len(args):
            tok = str(args[idx]).strip()
            if not tok.startswith("-"):
                raise PermissionError(
                    f"Unexpected positional argument in kaggle {' '.join(cmd_pair)}: {tok!r}"
                )

            flag_name, has_eq, eq_val = tok.partition("=")
            flag_lower = flag_name.lower()

            # Explicitly reject public flags and short-flag clusters containing 'u'.
            if (
                flag_lower in {"--public", "-u"}
                or (
                    flag_name.startswith("-")
                    and not flag_name.startswith("--")
                    and "u" in flag_lower[1:]
                )
            ):
                raise PermissionError(
                    f"Public dataset flag {tok!r} is forbidden; datasets must always be private"
                )

            if flag_name in _BOOL_FLAGS:
                if has_eq:
                    raise PermissionError(f"Boolean flag {flag_name!r} does not take a value")
                idx += 1
                continue

            if flag_name not in _VALUE_FLAGS:
                raise PermissionError(f"Disallowed flag for KaggleDatasetUploader: {tok!r}")

            if flag_name == "-m" and cmd_pair != ("datasets", "version"):
                raise PermissionError("Flag '-m' is only permitted for 'datasets version'")

            if has_eq:
                val = eq_val
                idx += 1
            else:
                if idx + 1 >= len(args):
                    raise PermissionError(f"Flag {tok!r} requires a value")
                val = str(args[idx + 1])
                idx += 2

            if flag_name in {"-r", "--dir-mode"}:
                if val.strip().lower() not in ALLOWED_DIR_MODES:
                    raise PermissionError(
                        f"Disallowed dir-mode {val!r}; must be one of {sorted(ALLOWED_DIR_MODES)}"
                    )

    def _exec(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Validate and execute an allow-listed Kaggle dataset CLI command."""
        arg_list = [str(a) for a in args]
        self._validate_command(arg_list)

        token = self._token_getter()
        if not token:
            raise KaggleDatasetError(
                f"No Kaggle API token available for account '{self.nickname}'"
            )

        if any(token in a for a in arg_list):
            raise PermissionError("Kaggle API token must never be passed in CLI arguments")

        env = self.build_env(token)
        argv = [self.kaggle_exe, *arg_list]

        try:
            proc = self._runner(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleDatasetError(
                f"Kaggle dataset CLI timed out after {self.timeout_s}s for account '{self.nickname}'"
            ) from exc
        except OSError as exc:
            safe_exc = _sanitize_text(str(exc), token)
            raise KaggleDatasetError(
                f"Failed to invoke Kaggle CLI for account '{self.nickname}': {safe_exc}"
            ) from exc

        returncode = int(getattr(proc, "returncode", 0))
        stdout = _sanitize_text(getattr(proc, "stdout", "") or "", token)
        stderr = _sanitize_text(getattr(proc, "stderr", "") or "", token)
        return returncode, stdout, stderr

    def dataset_exists(self, ref: str) -> bool:
        """Check whether ``owner/slug`` already exists via ``kaggle datasets status``."""
        rc, stdout, stderr = self._exec(["datasets", "status", ref])
        combined = f"{stdout}\n{stderr}".lower()
        if rc != 0:
            return False
        if (
            "not found" in combined
            or "404" in combined
            or "does not exist" in combined
            or "no dataset" in combined
        ):
            return False
        return True

    def upload(
        self,
        folder: str | Path,
        slug: str,
        title: str,
        owner: str,
        new_version_message: str | None = None,
    ) -> UploadResult:
        """Validate, stage, and create or version a private Kaggle dataset."""
        src_folder = Path(folder)
        if not src_folder.exists() or not src_folder.is_dir():
            raise DatasetUploadRefused(f"Upload folder does not exist or is not a directory: {src_folder}")

        clean_slug = validate_dataset_slug(slug)
        clean_title = validate_dataset_title(title)
        clean_owner = validate_dataset_owner(owner)
        ref = f"{clean_owner}/{clean_slug}"
        default_url = f"https://www.kaggle.com/datasets/{ref}"

        # Guard 1: secret files & secret-like content
        offending = scan_folder_for_secrets(src_folder)
        if offending:
            raise DatasetUploadRefused(
                "Dataset upload refused: potential secret file(s) detected: "
                + ", ".join(offending)
            )

        # Guard 2: size & non-empty folder
        total_bytes, file_count = compute_folder_size(src_folder)
        if file_count == 0:
            raise DatasetUploadRefused(f"Upload folder is empty: {src_folder}")
        if total_bytes > self.max_size_bytes:
            limit_gb = self.max_size_bytes / (1024**3)
            actual_gb = total_bytes / (1024**3)
            raise DatasetUploadRefused(
                f"Upload folder size ({actual_gb:.2f} GB) exceeds maximum limit ({limit_gb:.2f} GB)"
            )

        # Existence check via allow-listed `datasets status`
        exists = self.dataset_exists(ref)

        # Stage into an isolated temp directory (never modify src_folder)
        tmp_root = Path(tempfile.mkdtemp(prefix="ecc-dataset-"))
        staging = tmp_root / "staging"
        try:
            shutil.copytree(src_folder, staging)
            metadata = {
                "title": clean_title,
                "id": ref,
                "licenses": [{"name": "other"}],
            }
            (staging / "dataset-metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            if exists:
                msg = (new_version_message or "Update dataset").strip() or "Update dataset"
                cmd = ["datasets", "version", "-p", str(staging), "-m", msg, "-r", "zip", "-q"]
                action = "versioned"
            else:
                cmd = ["datasets", "create", "-p", str(staging), "-r", "zip", "-q"]
                action = "created"

            rc, stdout, stderr = self._exec(cmd)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part).strip()
        um = _DATASET_URL_RE.search(combined)
        url = um.group(1).rstrip(".,;)") if um else default_url

        if rc != 0:
            return UploadResult(
                ok=False,
                ref=ref,
                url=url,
                message=combined or f"Kaggle datasets {action} failed (exit code {rc})",
                size_bytes=total_bytes,
                action="failed",
            )

        return UploadResult(
            ok=True,
            ref=ref,
            url=url,
            message=combined or f"Dataset {ref} {action} successfully.",
            size_bytes=total_bytes,
            action=action,
        )

    def __repr__(self) -> str:
        return f"KaggleDatasetUploader(nickname={self.nickname!r}, kaggle_exe={self.kaggle_exe!r})"


class DatasetUploadService:
    """High-level gated & audited private dataset upload service."""

    def __init__(
        self,
        store: Store,
        secrets: SecretStore,
        settings: Settings,
        launch_service: LaunchService,
        uploader_factory: Callable[[str, Callable[[], str | None]], KaggleDatasetUploader] | None = None,
    ) -> None:
        self.store = store
        self.secrets = secrets
        self.settings = settings
        self.launch_service = launch_service
        self._uploader_factory = uploader_factory or (
            lambda nick, getter: KaggleDatasetUploader(nick, token_getter=getter)
        )

    def is_enabled(self) -> bool:
        return bool(self.settings.launch.enabled)

    def _find_account(self, nickname: str) -> dict[str, Any] | None:
        for acc in self.store.list_accounts():
            if acc.get("nickname") == nickname:
                return acc
        return None

    def upload(
        self,
        *,
        account: str,
        folder: str | Path,
        slug: str,
        title: str,
        message: str | None = None,
        source: str = "claude",
    ) -> UploadResult:
        if not self.is_enabled():
            raise DatasetUploadRefused(
                "Write operations are disabled. Set [launch] enabled = true in ecc_config.toml."
            )

        acc = self._find_account(account)
        if acc is None:
            raise DatasetUploadRefused(f"Unknown account '{account}'")
        if acc.get("platform") != "kaggle":
            raise DatasetUploadRefused(
                f"Account '{account}' is platform '{acc.get('platform')}'; "
                "only Kaggle accounts can upload datasets"
            )
        if not acc.get("enabled", True):
            raise DatasetUploadRefused(f"Account '{account}' is disabled")

        if not self.secrets.has("kaggle", account):
            raise DatasetUploadRefused(f"Missing Kaggle API token for account '{account}'")

        owner = self.launch_service.account_owner(account)
        if not owner:
            raise DatasetUploadRefused(
                f"Kaggle username is unknown for account '{account}'; "
                "set username via scripts/add_account.py --username"
            )

        uploader = self._uploader_factory(
            account,
            lambda n=account: self.secrets.get("kaggle", n),
        )
        result = uploader.upload(
            folder=folder,
            slug=slug,
            title=title,
            owner=owner,
            new_version_message=message,
        )

        self.store.record_dataset_upload(
            account=account,
            ref=result.ref,
            folder=str(Path(folder).resolve()),
            size_bytes=result.size_bytes,
            ok=result.ok,
            message=redact(result.message),
            source=source,
        )
        return result


def _infer_wheel_filename(
    wheel_bytes: bytes,
    fallback: str = "experiment_monitor-0.1.0-py3-none-any.whl",
) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes), "r") as zf:
            dist_info = next(
                (n.split("/", 1)[0] for n in zf.namelist() if ".dist-info/" in n),
                None,
            )
            if not dist_info:
                return fallback
            dist_ver = dist_info[: -len(".dist-info")]
            tag = "py3-none-any"
            wheel_meta_name = f"{dist_info}/WHEEL"
            if wheel_meta_name in zf.namelist():
                for line in (
                    zf.read(wheel_meta_name)
                    .decode("utf-8", errors="replace")
                    .splitlines()
                ):
                    if line.lower().startswith("tag:"):
                        t = line.split(":", 1)[1].strip()
                        if t:
                            tag = t
                            break
            return f"{dist_ver}-{tag}.whl"
    except Exception:
        return fallback


def build_monitor_wheel_folder(
    out_dir: str | Path,
    *,
    monitor_dir: str | Path | None = None,
    wheel_builder: Callable[[], bytes] | None = None,
) -> Path:
    """Build the ``experiment_monitor`` wheel into an empty folder and return the wheel path."""
    out_path = Path(out_dir)
    if out_path.exists():
        if not out_path.is_dir():
            raise DatasetUploadRefused(f"Output path exists and is not a directory: {out_path}")
        if any(out_path.iterdir()):
            raise DatasetUploadRefused(f"Output folder must be empty: {out_path}")
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    if wheel_builder is not None:
        wheel_bytes = wheel_builder()
    else:
        src = Path(monitor_dir) if monitor_dir is not None else default_monitor_dir()
        wheel_bytes = build_wheel(src)

    wheel_name = _infer_wheel_filename(wheel_bytes)
    wheel_file = out_path / wheel_name
    wheel_file.write_bytes(wheel_bytes)
    return wheel_file
