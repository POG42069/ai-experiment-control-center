"""Thread-safe stdout/stderr tee into ``ecc_console.log`` with redaction and ring-capping."""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, TextIO

from .redact import redact

CONSOLE_LOG_FILENAME = "ecc_console.log"


def _utc_timestamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class _StreamTee:
    """File-like proxy that forwards writes to the original stream and the console sink."""

    def __init__(self, original: TextIO, stream_name: str, sink: ConsoleCapture) -> None:
        self._original = original
        self._stream_name = stream_name
        self._sink = sink

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:
        return getattr(self._original, "errors", None)

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        written = self._original.write(data)
        if data:
            self._sink._ingest(self._stream_name, data)
        return written if isinstance(written, int) else len(data)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._original.flush()
        finally:
            self._sink.flush()

    def isatty(self) -> bool:
        try:
            return bool(self._original.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ConsoleCapture:
    """Tees ``sys.stdout`` and ``sys.stderr`` into ``<run_dir>/ecc_console.log``."""

    def __init__(
        self,
        run_dir: str,
        *,
        max_bytes: int = 2_000_000,
        wandb_run: Any | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, CONSOLE_LOG_FILENAME)
        self.max_bytes = max(int(max_bytes), 128)
        self._wandb_run = wandb_run
        self._lock = threading.RLock()
        self._buffers: dict[str, str] = {"stdout": "", "stderr": ""}
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None
        self._active = False
        self._saved_to_wandb = False
        self._size = 0

    def start(self) -> str:
        with self._lock:
            if self._active:
                return self.log_path
            os.makedirs(self.run_dir, exist_ok=True)
            if not os.path.exists(self.log_path):
                with open(self.log_path, "wb"):
                    pass
            self._size = os.path.getsize(self.log_path)

            if not self._saved_to_wandb and self._wandb_run is not None:
                self._register_wandb_live_save()

            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = _StreamTee(self._orig_stdout, "stdout", self)  # type: ignore[assignment]
            sys.stderr = _StreamTee(self._orig_stderr, "stderr", self)  # type: ignore[assignment]
            self._active = True
            return self.log_path

    def _register_wandb_live_save(self) -> None:
        try:
            import wandb

            wandb.save(self.log_path, base_path=self.run_dir, policy="live")
            self._saved_to_wandb = True
        except Exception:
            # Never crash training if wandb.save fails (e.g. disabled mode).
            pass

    def _ingest(self, stream_name: str, data: str) -> None:
        with self._lock:
            if not self._active:
                return
            normalized = data.replace("\r\n", "\n")
            buf = self._buffers.get(stream_name, "") + normalized
            lines: list[str] = []
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                lines.append(line)
            self._buffers[stream_name] = buf
            if lines:
                self._write_lines_locked(stream_name, lines)

    def write_direct(self, stream_name: str, text: str) -> None:
        """Write text directly to the console log file (redacted & prefixed) without echoing."""
        with self._lock:
            if not os.path.exists(self.log_path):
                os.makedirs(self.run_dir, exist_ok=True)
            normalized = text.replace("\r\n", "\n")
            lines = normalized.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]
            if lines:
                self._write_lines_locked(stream_name, lines)

    def _write_lines_locked(self, stream_name: str, lines: list[str]) -> None:
        ts = _utc_timestamp()
        encoded_chunks: list[bytes] = []
        for line in lines:
            safe_line = redact(line)
            formatted = f"{ts} [{stream_name}] {safe_line}\n"
            encoded_chunks.append(formatted.encode("utf-8", errors="replace"))
        payload = b"".join(encoded_chunks)
        if not payload:
            return
        try:
            with open(self.log_path, "ab") as fh:
                fh.write(payload)
                fh.flush()
            self._size += len(payload)
            if self._size > self.max_bytes:
                self._cap_file_locked()
        except Exception:
            pass

    def _cap_file_locked(self) -> None:
        keep_bytes = max(int(self.max_bytes * 0.75), 1)
        try:
            with open(self.log_path, "rb") as fh:
                data = fh.read()
            if len(data) <= self.max_bytes:
                self._size = len(data)
                return
            tail = data[-keep_bytes:]
            nl_idx = tail.find(b"\n")
            if nl_idx != -1 and nl_idx + 1 < len(tail):
                tail = tail[nl_idx + 1 :]
            # Rewrite in place to avoid rename permission errors on Windows.
            with open(self.log_path, "r+b") as fh:
                fh.seek(0)
                fh.write(tail)
                fh.truncate()
                fh.flush()
            self._size = len(tail)
        except Exception:
            pass

    def flush(self) -> None:
        # Partial lines remain buffered until newline or stop() so split writes stay on one line.
        pass

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            for stream_name in ("stdout", "stderr"):
                rem = self._buffers.get(stream_name, "")
                if rem:
                    self._write_lines_locked(stream_name, [rem])
                    self._buffers[stream_name] = ""
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
            self._orig_stdout = None
            self._orig_stderr = None
            self._active = False
