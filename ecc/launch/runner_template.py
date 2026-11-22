"""Self-contained Kaggle runner script template and helper functions.

The rendered script targets Python 3.10+ on Kaggle using only the Python standard
library prior to installing the embedded ``experiment_monitor`` wheel. It never
contains actual secret values — only Kaggle Secret label names.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Sequence

REDACTED = "[REDACTED]"

_INLINE_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(https?://x-access-token:)[^@\s/]+(@)", re.IGNORECASE),
        rf"\1{REDACTED}\2",
    ),
    (
        re.compile(
            r"(?i)(?<!x-access-)\b(password|passwd|token|api[_-]?key|secret)\s*([:=])\s*(?!\[REDACTED\])(['\"]?)[^\s'\",;)}\]]+\3"
        ),
        rf"\1\2{REDACTED}",
    ),
    (
        re.compile(r"(?i)\b(Bearer)\s+(?!\[REDACTED\])[A-Za-z0-9._~+/=-]{8,}"),
        rf"\1 {REDACTED}",
    ),
    (re.compile(r"\bKGAT_[A-Za-z0-9_-]{8,}\b"), REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), REDACTED),
    (re.compile(r"\b(?:ghp|gho)_[A-Za-z0-9]{16,}\b"), REDACTED),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"), REDACTED),
    (re.compile(r"\bwandb_v\d+_[A-Za-z0-9_-]{16,}\b"), REDACTED),
)


@dataclass(frozen=True)
class RunnerParams:
    """Parameters baked into the rendered Kaggle runner script (NO secret values)."""

    launch_id: str
    account: str
    owner: str
    runner_slug: str
    repo: str
    branch: str
    commit_sha: str
    command: str
    monitor_mode: str = "runner"
    project: str = ""
    run_name: str = ""
    requirements_file: str = ""
    preflight: bool = False
    wandb_credential_label: str = "WANDB_API_KEY"
    github_credential_label: str = "GITHUB_TOKEN"
    wheel_filename: str = "experiment_monitor-0.1.0-py3-none-any.whl"
    accelerator: str = "none"
    dataset_sources: tuple[str, ...] = ()

    @property
    def kernel_ref(self) -> str:
        return f"{self.owner}/{self.runner_slug}"

    @property
    def notebook_url(self) -> str:
        return f"https://www.kaggle.com/code/{self.owner}/{self.runner_slug}"


def build_clone_url(repo: str, github_token: str | None = None) -> str:
    """Return HTTPS GitHub clone URL, embedding ``x-access-token`` when provided."""
    tok = github_token.strip() if isinstance(github_token, str) else ""
    if tok:
        return f"https://x-access-token:{tok}@github.com/{repo}.git"
    return f"https://github.com/{repo}.git"


def redact_runner_text(text: str | None, secrets: Sequence[str] = ()) -> str:
    """Redact explicit secret values and well-known token patterns from runner output."""
    if not text:
        return "" if text is None else text
    result = text
    for sec in secrets:
        if sec:
            result = result.replace(sec, REDACTED)
    for pattern, repl in _INLINE_REDACT_PATTERNS:
        result = pattern.sub(repl, result)
    return result


def _wrap_b64(wheel_b64: str, width: int = 100) -> str:
    cleaned = "".join(wheel_b64.split())
    if not cleaned:
        return '""'
    chunks = [cleaned[i : i + width] for i in range(0, len(cleaned), width)]
    lines = ["("]
    for ch in chunks:
        lines.append(f"    {ch!r}")
    lines.append(")")
    return "\n".join(lines)


def render_runner(params: RunnerParams, wheel_b64: str) -> str:
    """Render a self-contained Kaggle Python 3.10+ runner script.

    Security: ``params`` only carries Kaggle Secret *labels*; no secret values are
    embedded anywhere in the generated script.
    """
    if params.monitor_mode not in {"runner", "script"}:
        raise ValueError(
            f"monitor_mode must be 'runner' or 'script', got {params.monitor_mode!r}"
        )

    wheel_literal = _wrap_b64(wheel_b64)

    script = textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        """Auto-generated ECC Kaggle Runner script (no secrets embedded)."""

        from __future__ import annotations

        import base64
        import os
        import re
        import shlex
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        # ------------------------------------------------------------------
        # Launch parameters (rendered by ecc.launch.runner_template)
        # ------------------------------------------------------------------
        LAUNCH_ID = {params.launch_id!r}
        ACCOUNT = {params.account!r}
        OWNER = {params.owner!r}
        RUNNER_SLUG = {params.runner_slug!r}
        KERNEL_REF = {params.kernel_ref!r}
        NOTEBOOK_URL = {params.notebook_url!r}
        REPO = {params.repo!r}
        BRANCH = {params.branch!r}
        COMMIT_SHA = {params.commit_sha!r}
        COMMAND = {params.command!r}
        MONITOR_MODE = {params.monitor_mode!r}
        PROJECT = {params.project!r}
        RUN_NAME = {params.run_name!r}
        REQUIREMENTS_FILE = {params.requirements_file!r}
        PREFLIGHT = {bool(params.preflight)!r}
        ACCELERATOR = {params.accelerator!r}
        WANDB_CREDENTIAL_LABEL = {params.wandb_credential_label!r}
        GITHUB_CREDENTIAL_LABEL = {params.github_credential_label!r}
        WHEEL_FILENAME = {params.wheel_filename!r}

        WHEEL_B64 = __WHEEL_B64_PLACEHOLDER__

        REDACTED = "[REDACTED]"
        _REDACT_PATTERNS = (
            (re.compile(r"(https?://x-access-token:)[^@\\s/]+(@)", re.IGNORECASE), r"\\1[REDACTED]\\2"),
            (
                re.compile(
                    r"(?i)(?<!x-access-)\\b(password|passwd|token|api[_-]?key|secret)\\s*([:=])\\s*(?!\\[REDACTED\\])(['\\"]?)[^\\s'\\",;)\\}}\\]]+\\3"
                ),
                r"\\1\\2[REDACTED]",
            ),
            (re.compile(r"(?i)\\b(Bearer)\\s+(?!\\[REDACTED\\])[A-Za-z0-9._~+/=-]{{8,}}"), r"\\1 [REDACTED]"),
            (re.compile(r"\\bKGAT_[A-Za-z0-9_-]{{8,}}\\b"), REDACTED),
            (re.compile(r"\\bhf_[A-Za-z0-9]{{8,}}\\b"), REDACTED),
            (re.compile(r"\\bsk-[A-Za-z0-9_-]{{12,}}\\b"), REDACTED),
            (re.compile(r"\\b(?:ghp|gho)_[A-Za-z0-9]{{16,}}\\b"), REDACTED),
            (re.compile(r"\\bgithub_pat_[A-Za-z0-9_]{{16,}}\\b"), REDACTED),
            (re.compile(r"\\bwandb_v\\d+_[A-Za-z0-9_-]{{16,}}\\b"), REDACTED),
        )


        def _redact(text: str, secrets: list[str]) -> str:
            if not text:
                return ""
            out = text
            for sec in secrets:
                if sec:
                    out = out.replace(sec, REDACTED)
            for pattern, repl in _REDACT_PATTERNS:
                out = pattern.sub(repl, out)
            return out


        def _build_clone_url(repo: str, github_token: str | None) -> str:
            tok = github_token.strip() if isinstance(github_token, str) else ""
            if tok:
                return f"https://x-access-token:{{tok}}@github.com/{{repo}}.git"
            return f"https://github.com/{{repo}}.git"


        def _get_kaggle_secret(label: str) -> str | None:
            if not label:
                return None
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

                val = UserSecretsClient().get_secret(label)
                if isinstance(val, str) and val.strip():
                    print(f"[ECC] Kaggle Secret '{{label}}': found", flush=True)
                    return val.strip()
                print(f"[ECC] Kaggle Secret '{{label}}': empty value", flush=True)
            except Exception as exc:
                # Diagnostic only: the service error never contains the secret value itself.
                cause = exc.__cause__ or exc.__context__
                detail = ""
                if cause is not None:
                    code = getattr(cause, "code", None)
                    body = ""
                    try:
                        body = cause.read().decode("utf-8", "replace") if hasattr(cause, "read") else ""
                    except Exception:
                        body = ""
                    detail = f" | cause={{type(cause).__name__}} code={{code}} body={{_redact(body, [])[:300]}}"
                print(
                    f"[ECC] Kaggle Secret '{{label}}': unavailable ({{type(exc).__name__}}: "
                    f"{{_redact(str(exc), [])[:200]}}){{detail}}",
                    flush=True,
                )
                return None
            return None


        def _load_credentials_file() -> dict[str, str]:
            """Read ecc_credentials.json from an attached private dataset; never prints values."""
            import glob
            import json

            paths = sorted(glob.glob("/kaggle/input/**/ecc_credentials.json", recursive=True))
            if not paths:
                print("[ECC] credentials dataset: ecc_credentials.json not attached", flush=True)
                return {{}}
            try:
                data = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[ECC] credentials dataset: unreadable ({{type(exc).__name__}})", flush=True)
                return {{}}
            out = {{
                str(k): str(v).strip()
                for k, v in (data.items() if isinstance(data, dict) else [])
                if isinstance(v, str) and v.strip()
            }}
            print(f"[ECC] credentials dataset: found keys {{sorted(out)}}", flush=True)
            return out


        def _run_step(
            argv: list[str],
            *,
            secrets: list[str],
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> int:
            proc = subprocess.run(
                argv,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if proc.stdout:
                print(_redact(proc.stdout.rstrip(), secrets), flush=True)
            return int(proc.returncode)


        def _run_command_streaming(
            command: str,
            *,
            cwd: Path,
            env: dict[str, str],
            secrets: list[str],
        ) -> int:
            shell_ops = ("|", "&&", "||", ";", ">", "<")
            use_shell = any(op in command for op in shell_ops)
            cmd_arg: str | list[str] = command if use_shell else shlex.split(command)
            proc = subprocess.Popen(
                cmd_arg,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                shell=use_shell,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(_redact(line.rstrip("\\r\\n"), secrets), flush=True)
            return int(proc.wait())


        def main() -> int:
            print("=== ECC Kaggle Runner ===", flush=True)
            print(f"launch_id    : {{LAUNCH_ID}}", flush=True)
            print(f"account      : {{ACCOUNT}} ({{KERNEL_REF}})", flush=True)
            print(f"repo         : {{REPO}}@{{BRANCH}}", flush=True)
            # Short SHA on purpose: a full 40-hex SHA looks like a legacy W&B key to log redactors.
            print(f"commit       : {{COMMIT_SHA[:12]}}", flush=True)
            print(f"command      : {{COMMAND}}", flush=True)
            print(f"monitor_mode : {{MONITOR_MODE}} (preflight={{PREFLIGHT}})", flush=True)
            print("=========================", flush=True)

            # 0. Diagnostics: outbound network + presence (not values) of Kaggle service env vars
            import urllib.request

            for _url in ("https://github.com", "https://api.wandb.ai"):
                try:
                    with urllib.request.urlopen(_url, timeout=10) as _resp:
                        print(f"[ECC] network {{_url}}: HTTP {{_resp.status}}", flush=True)
                except Exception as _exc:
                    print(f"[ECC] network {{_url}}: FAILED ({{type(_exc).__name__}})", flush=True)
            for _name in ("KAGGLE_USER_SECRETS_TOKEN", "KAGGLE_IAP_TOKEN", "KAGGLE_DATA_PROXY_URL",
                          "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE"):
                print(f"[ECC] env {{_name}} present: {{bool(os.environ.get(_name))}}", flush=True)

            # 1. Read credentials at runtime (never embedded in code):
            #    a) private credentials dataset (ecc_credentials.json) attached via dataset_sources,
            #    b) Kaggle Secrets (only delivered to versions saved from the Kaggle UI).
            creds = _load_credentials_file()
            wandb_key = creds.get(WANDB_CREDENTIAL_LABEL) or _get_kaggle_secret(WANDB_CREDENTIAL_LABEL)
            github_token = creds.get(GITHUB_CREDENTIAL_LABEL) or _get_kaggle_secret(GITHUB_CREDENTIAL_LABEL)

            if not wandb_key:
                print(
                    f"[ECC] ERROR: credential '{{WANDB_CREDENTIAL_LABEL}}' not found.\\n"
                    f"Create the private dataset ecc-credentials on this account with "
                    f"`scripts/setup_credentials_dataset.py`, then launch Preflight again.",
                    flush=True,
                )
                return 3

            active_secrets = [s for s in (wandb_key, github_token) if s]

            # 2. Checkout exact commit SHA into /kaggle/working/src
            work_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
            src_dir = work_dir / "src"
            if src_dir.exists():
                shutil.rmtree(src_dir)

            git_env = dict(os.environ)
            git_env["GIT_TERMINAL_PROMPT"] = "0"
            clone_url = _build_clone_url(REPO, github_token)

            print(f"[ECC] Fetching {{REPO}} @ {{COMMIT_SHA[:7]}} into {{src_dir}} ...", flush=True)
            rc_init = _run_step(["git", "init", str(src_dir)], secrets=active_secrets, env=git_env)
            rc_fetch = (
                _run_step(
                    ["git", "-C", str(src_dir), "fetch", "--depth", "1", clone_url, COMMIT_SHA],
                    secrets=active_secrets,
                    env=git_env,
                )
                if rc_init == 0
                else rc_init
            )
            rc_checkout = (
                _run_step(
                    ["git", "-C", str(src_dir), "checkout", "FETCH_HEAD"],
                    secrets=active_secrets,
                    env=git_env,
                )
                if rc_fetch == 0
                else rc_fetch
            )
            if rc_checkout != 0:
                print(
                    f"[ECC] ERROR: Failed to fetch/checkout {{REPO}}@{{COMMIT_SHA}}.\\n"
                    f"Hint: If '{{REPO}}' is a private repository, attach '{{GITHUB_CREDENTIAL_LABEL}}' "
                    f"via {{NOTEBOOK_URL}} -> Add-ons -> Secrets, then retry.",
                    flush=True,
                )
                return 4

            # 3. Install embedded experiment_monitor wheel (+ optional requirements)
            wheel_path = work_dir / WHEEL_FILENAME
            wheel_path.write_bytes(base64.b64decode(WHEEL_B64))
            print(f"[ECC] Installing embedded wheel {{wheel_path.name}} ...", flush=True)
            rc_whl = _run_step(
                [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel_path)],
                secrets=active_secrets,
            )
            if rc_whl != 0:
                print("[ECC] ERROR: Failed to install embedded experiment_monitor wheel.", flush=True)
                return rc_whl

            if REQUIREMENTS_FILE:
                req_path = src_dir / REQUIREMENTS_FILE
                if req_path.is_file():
                    print(f"[ECC] Installing requirements from {{REQUIREMENTS_FILE}} ...", flush=True)
                    rc_req = _run_step(
                        [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                        secrets=active_secrets,
                    )
                    if rc_req != 0:
                        print(f"[ECC] ERROR: Failed to install requirements ({{REQUIREMENTS_FILE}}).", flush=True)
                        return rc_req
                else:
                    print(
                        f"[ECC] WARNING: requirements_file '{{REQUIREMENTS_FILE}}' not found in repo; skipping.",
                        flush=True,
                    )

            # 4. Prepare child environment with ECC_* defaults + WANDB_API_KEY
            child_env = dict(os.environ)
            ecc_env = {{
                "ECC_PROJECT": PROJECT,
                "ECC_ACCOUNT": ACCOUNT,
                "ECC_KAGGLE_KERNEL": KERNEL_REF,
                "ECC_NOTEBOOK_URL": NOTEBOOK_URL,
                "ECC_GIT_BRANCH": BRANCH,
                "ECC_GIT_SHA": COMMIT_SHA,
                "ECC_GIT_REPO": REPO,
                "ECC_LAUNCH_ID": LAUNCH_ID,
                "ECC_RUN_NAME": RUN_NAME,
                "WANDB_API_KEY": wandb_key,
            }}
            child_env.update(ecc_env)
            os.environ.update(ecc_env)

            # 5. Preflight mode: environment + repo + monitor + W&B checks only (no training)
            if PREFLIGHT:
                print("[ECC] === Preflight checks ===", flush=True)
                print(f"[ECC] Python: {{sys.version.splitlines()[0]}}", flush=True)
                try:
                    import torch  # type: ignore[import-not-found]

                    cuda_ok = bool(torch.cuda.is_available())
                    cuda_ver = getattr(torch.version, "cuda", None)
                    dev_count = torch.cuda.device_count() if cuda_ok else 0
                    print(
                        f"[ECC] PyTorch: {{torch.__version__}}, CUDA={{cuda_ver}}, "
                        f"cuda_available={{cuda_ok}}, devices={{dev_count}}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[ECC] PyTorch import skipped/failed: {{type(exc).__name__}}", flush=True)

                if shutil.which("nvidia-smi"):
                    _run_step(["nvidia-smi", "-L"], secrets=active_secrets)
                else:
                    print("[ECC] nvidia-smi not found (CPU session).", flush=True)

                file_count = sum(
                    1
                    for p in src_dir.rglob("*")
                    if p.is_file() and ".git" not in p.relative_to(src_dir).parts
                )
                print(f"[ECC] Repo checkout OK: {{file_count}} file(s) in {{src_dir}}", flush=True)

                import experiment_monitor as em  # type: ignore[import-not-found]

                print(f"[ECC] experiment_monitor import OK: {{em.__file__}}", flush=True)

                import wandb  # type: ignore[import-not-found]

                login_ok = bool(wandb.login(key=wandb_key, relogin=True))
                print(f"[ECC] W&B login check: {{login_ok}}", flush=True)
                print("[ECC] Preflight completed (no training started).", flush=True)
                return 0

            # 6. Training execution: "runner" vs "script" monitoring mode
            if MONITOR_MODE == "runner":
                import wandb  # type: ignore[import-not-found]
                import experiment_monitor as em  # type: ignore[import-not-found]

                wandb.login(key=wandb_key, relogin=True)
                mon = em.init(
                    project=PROJECT,
                    name=RUN_NAME or None,
                    account=ACCOUNT,
                    kaggle_kernel=KERNEL_REF,
                    notebook_url=NOTEBOOK_URL,
                    config={{
                        "ecc/launch_id": LAUNCH_ID,
                        "command": COMMAND,
                        "ecc/git_repo": REPO,
                    }},
                    tags=["ecc-launch"],
                )
                rc = 1
                try:
                    rc = _run_command_streaming(
                        COMMAND,
                        cwd=src_dir,
                        env=child_env,
                        secrets=active_secrets,
                    )
                finally:
                    mon.finish(exit_code=rc)
                return rc

            # monitor_mode == "script": user script calls em.init() using ECC_* env defaults
            return _run_command_streaming(
                COMMAND,
                cwd=src_dir,
                env=child_env,
                secrets=active_secrets,
            )


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )
    return script.replace("__WHEEL_B64_PLACEHOLDER__", wheel_literal, 1)
