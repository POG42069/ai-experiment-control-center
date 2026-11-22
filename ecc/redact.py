"""Secret redaction utilities for logs, error messages, and store entries."""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Patterns ordered from most specific to most general.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Key-value secret assignments: password=..., token: ..., api_key="...", secret=...
    (
        re.compile(
            r"(?i)\b(password|passwd|token|api[_-]?key|secret)\s*([:=])\s*(['\"]?)[^\s'\",;)}\]]+\3"
        ),
        rf"\1\2{REDACTED}",
    ),
    # HTTP Authorization Bearer tokens
    (
        re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
        rf"\1 {REDACTED}",
    ),
    # Kaggle API tokens (KGAT_...)
    (
        re.compile(r"\bKGAT_[A-Za-z0-9_-]{8,}\b"),
        REDACTED,
    ),
    # Hugging Face tokens (hf_...)
    (
        re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
        REDACTED,
    ),
    # OpenAI / Anthropic / similar sk-... keys
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        REDACTED,
    ),
    # GitHub personal/OAuth/fine-grained tokens
    (
        re.compile(r"\b(?:ghp|gho)_[A-Za-z0-9]{16,}\b"),
        REDACTED,
    ),
    (
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"),
        REDACTED,
    ),
    # W&B v1 prefixed API keys
    (
        re.compile(r"\bwandb_v\d+_[A-Za-z0-9_-]{16,}\b"),
        REDACTED,
    ),
    # Classic 40-hex W&B API keys
    (
        re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])"),
        REDACTED,
    ),
]


def redact(text: str | None) -> str:
    """Replace secret-like patterns in text with [REDACTED]."""
    if not text:
        return "" if text is None else text
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result
