"""Core Monitor class, init(), and notebook secret authentication helper."""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
from typing import Any

import wandb

from .console import ConsoleCapture
from .gpu import gpu_snapshot, gpu_static_info
from .platform import detect_git_info, detect_platform
from .redact import redact

SCHEMA_VERSION = 1


def login_from_notebook_secrets(secret_name: str = "WANDB_API_KEY") -> bool:
    """Authenticate W&B using Kaggle Secrets, Colab Userdata, or environment variables.

    Never prints, logs, or returns the secret value. Returns ``True`` if login succeeded.
    """
    key: str | None = None

    # 1. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        candidate = UserSecretsClient().get_secret(secret_name)
        if isinstance(candidate, str) and candidate.strip():
            key = candidate.strip()
    except Exception:
        pass

    # 2. Google Colab Secrets
    if not key:
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            candidate = userdata.get(secret_name)
            if isinstance(candidate, str) and candidate.strip():
                key = candidate.strip()
        except Exception:
            pass

    # 3. Environment variable
    if not key:
        candidate = os.environ.get(secret_name)
        if isinstance(candidate, str) and candidate.strip():
            key = candidate.strip()

    if not key:
        return False

    try:
        return bool(wandb.login(key=key, relogin=False))
    except Exception:
        return False


def _compute_path_size_mb(path: str | os.PathLike[str]) -> float:
    path_str = os.fspath(path)
    total_bytes = 0
    if os.path.isfile(path_str):
        total_bytes = os.path.getsize(path_str)
    elif os.path.isdir(path_str):
        for root, _dirs, files in os.walk(path_str):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if os.path.isfile(fpath):
                        total_bytes += os.path.getsize(fpath)
                except OSError:
                    pass
    return round(total_bytes / (1024 * 1024), 4)


class Monitor:
    """Thin observation wrapper around a ``wandb.Run`` following the ECC §3 contract."""

    def __init__(
        self,
        run: Any,
        *,
        owns_run: bool = True,
        total_steps: int | None = None,
        total_epochs: float | None = None,
        capture_console: bool = True,
        gpu_telemetry: bool = True,
        heartbeat_interval_s: float = 30.0,
        console_max_bytes: int = 2_000_000,
    ) -> None:
        self.run = run
        self._owns_run = owns_run
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self._gpu_telemetry = gpu_telemetry
        self._heartbeat_interval_s = float(heartbeat_interval_s) if heartbeat_interval_s else 0.0

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._finished = False
        self._end_reason_set = False

        self._last_step: int | None = None
        self._last_committed_step: int | None = None
        self._last_epoch: float | None = None
        self.final_summary: dict[str, Any] = {}

        self._best_metric_name: str | None = "eval/macro_f1"
        self._best_mode: str = "max"
        self._best_value: float | None = None
        self._best_step: int | None = None
        self._best_epoch: float | None = None

        self._checkpoints: list[dict[str, Any]] = []

        self._console: ConsoleCapture | None = None
        self.console_log_path: str | None = None
        if capture_console and getattr(run, "dir", None):
            self._console = ConsoleCapture(
                run.dir,
                max_bytes=console_max_bytes,
                wandb_run=run,
            )
            self.console_log_path = self._console.start()

        # Record initial heartbeat before starting daemon thread.
        self._emit_heartbeat()

        self._hb_thread: threading.Thread | None = None
        if self._heartbeat_interval_s > 0:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="ecc-heartbeat",
                daemon=True,
            )
            self._hb_thread.start()

    def _emit_heartbeat(self) -> None:
        with self._lock:
            if self._finished:
                return
            try:
                self.run.summary["ecc/heartbeat"] = time.time()
            except Exception:
                pass
            if self._gpu_telemetry:
                snap = gpu_snapshot()
                if snap:
                    try:
                        for k, v in snap.items():
                            self.run.summary[k] = v
                        self.run.log(snap, commit=False)
                    except Exception:
                        pass

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval_s):
            self._emit_heartbeat()

    def track_best(self, metric_name: str = "eval/macro_f1", mode: str = "max") -> Monitor:
        """Configure which metric and direction (``max`` or ``min``) to track as best."""
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        with self._lock:
            self._best_metric_name = metric_name
            self._best_mode = mode
            self._best_value = None
            self._best_step = None
            self._best_epoch = None
        return self

    def set_totals(
        self,
        total_steps: int | None = None,
        total_epochs: float | None = None,
    ) -> None:
        """Update expected total steps/epochs and sync them to ``run.config``."""
        updates: dict[str, Any] = {}
        with self._lock:
            if total_steps is not None:
                self.total_steps = int(total_steps)
                updates["ecc/total_steps"] = self.total_steps
            if total_epochs is not None:
                self.total_epochs = float(total_epochs)
                updates["ecc/total_epochs"] = self.total_epochs
            if updates:
                try:
                    self.run.config.update(updates, allow_val_change=True)
                except Exception:
                    pass

    def _update_best_locked(
        self,
        metrics: dict[str, Any],
        step: int | None,
        epoch: float | None,
    ) -> None:
        if not self._best_metric_name or self._best_metric_name not in metrics:
            return
        raw_val = metrics[self._best_metric_name]
        if raw_val is None or isinstance(raw_val, bool):
            return
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            return
        if math.isnan(val) or math.isinf(val):
            return

        improved = (
            self._best_value is None
            or (self._best_mode == "max" and val > self._best_value)
            or (self._best_mode == "min" and val < self._best_value)
        )
        if improved:
            self._best_value = val
            self._best_step = step if step is not None else self._last_step
            self._best_epoch = epoch if epoch is not None else self._last_epoch
            self.run.summary["ecc/best_metric_name"] = self._best_metric_name
            self.run.summary["ecc/best_metric"] = self._best_value
            self.run.summary["ecc/best_step"] = self._best_step
            self.run.summary["ecc/best_epoch"] = self._best_epoch

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        *,
        commit: bool | None = None,
    ) -> None:
        """Log metrics to W&B, recording ``train/global_step`` and updating best tracking."""
        payload = dict(metrics)
        with self._lock:
            if step is not None:
                self._last_step = int(step)
                payload.setdefault("train/global_step", int(step))
            elif "train/global_step" in payload and payload["train/global_step"] is not None:
                try:
                    self._last_step = int(payload["train/global_step"])
                except (TypeError, ValueError):
                    pass

            epoch = payload.get("train/epoch")
            if epoch is not None:
                try:
                    self._last_epoch = float(epoch)
                except (TypeError, ValueError):
                    pass

            if commit is None:
                self.run.log(payload, step=step)
                if step is not None:
                    self._last_committed_step = int(step)
            else:
                self.run.log(payload, step=step, commit=commit)
                if step is not None and commit:
                    self._last_committed_step = int(step)

            self._update_best_locked(payload, step=self._last_step, epoch=self._last_epoch)

    def set_progress(
        self,
        step: int,
        total_steps: int | None = None,
        epoch: float | None = None,
    ) -> None:
        """Log normalized progress ``ecc/progress``, ``train/global_step``, and ``train/epoch``."""
        step_int = int(step)
        with self._lock:
            self._last_step = step_int
            if total_steps is not None:
                self.total_steps = int(total_steps)
                try:
                    self.run.config.update({"ecc/total_steps": self.total_steps}, allow_val_change=True)
                except Exception:
                    pass
            if epoch is not None:
                self._last_epoch = float(epoch)

            progress: float | None = None
            if self.total_steps and self.total_steps > 0:
                progress = min(max(float(step_int) / float(self.total_steps), 0.0), 1.0)
            elif epoch is not None and self.total_epochs and self.total_epochs > 0:
                progress = min(max(float(epoch) / float(self.total_epochs), 0.0), 1.0)

            payload: dict[str, Any] = {"train/global_step": step_int}
            if epoch is not None:
                payload["train/epoch"] = float(epoch)
            if progress is not None:
                payload["ecc/progress"] = progress

            if self._last_committed_step is not None and step_int <= self._last_committed_step:
                self.run.log(payload, commit=False)
            else:
                self.run.log(payload, step=step_int)
                self._last_committed_step = step_int

            # Mirror latest progress fields in summary so dashboards see them immediately.
            try:
                self.run.summary["train/global_step"] = step_int
                if epoch is not None:
                    self.run.summary["train/epoch"] = float(epoch)
                if progress is not None:
                    self.run.summary["ecc/progress"] = progress
            except Exception:
                pass

    def log_checkpoint(
        self,
        path: str | os.PathLike[str],
        step: int,
        metric: float | None = None,
        metric_name: str | None = None,
    ) -> dict[str, Any]:
        """Record checkpoint metadata in ``run.summary['ecc/checkpoints']`` without uploading."""
        size_mb = _compute_path_size_mb(path)
        entry: dict[str, Any] = {
            "step": int(step),
            "path": os.fspath(path),
            "size_mb": size_mb,
            "metric": float(metric) if metric is not None else None,
        }
        if metric_name is not None:
            entry["metric_name"] = metric_name

        with self._lock:
            self._checkpoints.append(entry)
            self.run.summary["ecc/checkpoints"] = list(self._checkpoints)
        return entry

    def finish(self, exit_code: int = 0) -> None:
        """Stop heartbeat & console capture, finalize summary, and finish owned W&B run."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._stop_event.set()

        if self._hb_thread is not None and self._hb_thread.is_alive():
            if threading.current_thread() is not self._hb_thread:
                self._hb_thread.join(timeout=5.0)

        try:
            if not self._end_reason_set:
                self.run.summary["ecc/end_reason"] = "finished" if exit_code == 0 else "exception"
                self._end_reason_set = True
            self.run.summary["ecc/heartbeat"] = time.time()
            self.final_summary = dict(self.run.summary)
        except Exception:
            pass

        if self._console is not None:
            self._console.stop()

        if self._owns_run:
            self.run.finish(exit_code=exit_code)

    def __enter__(self) -> Monitor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            try:
                self.run.summary["ecc/end_reason"] = "finished"
                self._end_reason_set = True
            except Exception:
                pass
            self.finish(exit_code=0)
            return False

        is_interrupt = issubclass(exc_type, KeyboardInterrupt)
        end_reason = "interrupted" if is_interrupt else "exception"
        exc_message = redact(str(exc_val))[:500]

        try:
            self.run.summary["ecc/end_reason"] = end_reason
            self.run.summary["ecc/exception_type"] = exc_type.__name__
            self.run.summary["ecc/exception_message"] = exc_message
            self._end_reason_set = True
        except Exception:
            pass

        if self._console is not None:
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._console.write_direct("stderr", tb_text)

        self.finish(exit_code=1)
        return False


def _env_nonempty(var_name: str) -> str | None:
    val = os.environ.get(var_name)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def init(
    project: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    *,
    account: str | None = None,
    notebook_url: str | None = None,
    kaggle_kernel: str | None = None,
    dataset_version: str | None = None,
    dataset_split: str | None = None,
    seed: int | None = None,
    model_name: str | None = None,
    prompt_strategy: str | None = None,
    total_steps: int | None = None,
    total_epochs: float | None = None,
    eval_signature: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    entity: str | None = None,
    resume_id: str | None = None,
    capture_console: bool = True,
    gpu_telemetry: bool = True,
    heartbeat_interval_s: float = 30,
    console_max_bytes: int = 2_000_000,
    mode: str | None = None,
) -> Monitor:
    """Initialize (or reuse active) W&B run with ECC §3 config keys and return a ``Monitor``.

    When optional metadata arguments are ``None``, falls back to ``ECC_*`` environment
    variables (and ``WANDB_PROJECT`` for ``project``). Explicit arguments always win.
    """
    resolved_project: str | None = project
    if resolved_project is None:
        resolved_project = _env_nonempty("ECC_PROJECT") or _env_nonempty("WANDB_PROJECT")
    if not isinstance(resolved_project, str) or not resolved_project.strip():
        raise ValueError(
            "project must be provided via argument or ECC_PROJECT / WANDB_PROJECT environment variable"
        )
    resolved_project = resolved_project.strip()

    resolved_name = name if name is not None else _env_nonempty("ECC_RUN_NAME")
    resolved_account = account if account is not None else _env_nonempty("ECC_ACCOUNT")
    resolved_kaggle_kernel = (
        kaggle_kernel if kaggle_kernel is not None else _env_nonempty("ECC_KAGGLE_KERNEL")
    )
    resolved_notebook_url = (
        notebook_url if notebook_url is not None else _env_nonempty("ECC_NOTEBOOK_URL")
    )

    merged: dict[str, Any] = dict(config) if config is not None else {}

    detected_branch, detected_sha = detect_git_info()
    git_branch = _env_nonempty("ECC_GIT_BRANCH") or detected_branch
    git_sha = _env_nonempty("ECC_GIT_SHA") or detected_sha

    ecc_config: dict[str, Any] = {
        "ecc/schema_version": SCHEMA_VERSION,
        "ecc/project": resolved_project,
        "ecc/platform": detect_platform(),
        "ecc/git_branch": git_branch,
        "ecc/git_sha": git_sha,
    }

    optional_fields = {
        "ecc/account": resolved_account,
        "ecc/notebook_url": resolved_notebook_url,
        "ecc/kaggle_kernel": resolved_kaggle_kernel,
        "ecc/dataset_version": dataset_version,
        "ecc/dataset_split": dataset_split,
        "ecc/seed": seed,
        "ecc/model_name": model_name,
        "ecc/prompt_strategy": prompt_strategy,
        "ecc/total_steps": total_steps,
        "ecc/total_epochs": total_epochs,
        "ecc/eval_signature": eval_signature,
    }
    for key, val in optional_fields.items():
        if val is not None:
            ecc_config[key] = val

    env_launch_id = _env_nonempty("ECC_LAUNCH_ID")
    if env_launch_id is not None and "ecc/launch_id" not in merged:
        ecc_config["ecc/launch_id"] = env_launch_id

    env_git_repo = _env_nonempty("ECC_GIT_REPO")
    if env_git_repo is not None and "ecc/git_repo" not in merged:
        ecc_config["ecc/git_repo"] = env_git_repo

    if gpu_telemetry:
        ecc_config.update(gpu_static_info())

    merged.update(ecc_config)

    existing_run = wandb.run
    if existing_run is not None:
        run = existing_run
        run.config.update(merged, allow_val_change=True)
        owns_run = False
    else:
        run = wandb.init(
            project=resolved_project,
            entity=entity,
            name=resolved_name,
            config=merged,
            tags=list(tags) if tags is not None else None,
            id=resume_id,
            resume="allow" if resume_id else None,
            mode=mode,
        )
        owns_run = True

    return Monitor(
        run,
        owns_run=owns_run,
        total_steps=total_steps,
        total_epochs=total_epochs,
        capture_console=capture_console,
        gpu_telemetry=gpu_telemetry,
        heartbeat_interval_s=heartbeat_interval_s,
        console_max_bytes=console_max_bytes,
    )
