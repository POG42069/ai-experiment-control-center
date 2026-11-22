"""Unit tests for experiment_monitor.console."""

from __future__ import annotations

import io
import os
import re
import sys
import threading
from pathlib import Path

from experiment_monitor.console import ConsoleCapture
from experiment_monitor.redact import REDACTED

_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \[(stdout|stderr)\] ")


class _FakeTTYStream(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 42


def test_console_tee_prefixes_redacts_and_restores(tmp_path: Path) -> None:
    orig_out, orig_err = sys.stdout, sys.stderr
    fake_out = _FakeTTYStream()
    fake_err = _FakeTTYStream()
    sys.stdout = fake_out
    sys.stderr = fake_err

    fake_hf_key = "hf_" + "Z" * 34
    cap = ConsoleCapture(str(tmp_path), max_bytes=100_000, wandb_run=None)
    try:
        log_path = cap.start()
        assert sys.stdout.isatty() is True
        assert sys.stdout.fileno() == 42

        # Partial writes should coalesce into one line.
        sys.stdout.write("hello ")
        sys.stdout.write(f"token={fake_hf_key}\n")
        sys.stderr.write("warning line\n")
        sys.stdout.write("trailing partial")
        sys.stdout.flush()
    finally:
        cap.stop()
        assert sys.stdout is fake_out
        assert sys.stderr is fake_err
        sys.stdout = orig_out
        sys.stderr = orig_err

    # Original streams still received output.
    assert "hello " in fake_out.getvalue()
    assert "warning line\n" in fake_err.getvalue()

    content = Path(log_path).read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) == 3
    for line in lines:
        assert _PREFIX_RE.match(line), f"Bad prefix: {line!r}"

    assert "[stdout] hello " in lines[0]
    assert fake_hf_key not in content
    assert REDACTED in lines[0]
    assert "[stderr] warning line" in lines[1]
    assert "[stdout] trailing partial" in lines[2]


def test_console_caps_file_size_and_keeps_tail(tmp_path: Path) -> None:
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    max_bytes = 600
    cap = ConsoleCapture(str(tmp_path), max_bytes=max_bytes, wandb_run=None)
    try:
        log_path = cap.start()
        for idx in range(40):
            print(f"line-{idx:03d}-" + ("x" * 30))
    finally:
        cap.stop()
        sys.stdout = orig_out
        sys.stderr = orig_err

    size = os.path.getsize(log_path)
    assert size <= max_bytes
    text = Path(log_path).read_text(encoding="utf-8")
    assert "line-039-" in text
    assert "line-000-" not in text


def test_console_thread_safety(tmp_path: Path) -> None:
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    cap = ConsoleCapture(str(tmp_path), max_bytes=500_000, wandb_run=None)
    try:
        log_path = cap.start()

        def _worker(tid: int) -> None:
            for i in range(25):
                print(f"thread-{tid}-msg-{i}")

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        cap.stop()
        sys.stdout = orig_out
        sys.stderr = orig_err

    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    for line in lines:
        assert _PREFIX_RE.match(line)
