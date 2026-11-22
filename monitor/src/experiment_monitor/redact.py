"""Secret redaction utilities for console logs and exception messages."""

from __future__ import annotations

import re

REDACTED = "***REDACTED***"

# Specific high-signal secret token formats.
_TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Hugging Face tokens: hf_ followed by 30+ alphanumeric chars
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), REDACTED),
    # Kaggle access tokens: KGAT_...
    (re.compile(r"\bKGAT_[A-Za-z0-9_\-]{8,}\b"), REDACTED),
    # OpenAI / Anthropic / similar sk-... keys (20+ chars after sk-)
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), REDACTED),
    # GitHub personal / OAuth / fine-grained tokens
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), REDACTED),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), REDACTED),
    # Newer W&B v1 API keys
    (re.compile(r"\bwandb_v\d+_[A-Za-z0-9_=-]{20,}\b"), REDACTED),
    # Bearer <token>
    (re.compile(r"(?i)\b(Bearer\s+)([A-Za-z0-9._~+/=-]{8,})"), rf"\1{REDACTED}"),
    # Classic W&B API keys (40 hex chars, bounded by non-hex)
    (re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{40}(?![A-Fa-f0-9])"), REDACTED),
]

# Key-value assignments like password=..., token=..., api_key=... (including JSON/YAML styles)
_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    (
        \b(?:
            [A-Za-z0-9_.-]*(?:password|passwd|api[_-]?key|secret)
            | [A-Za-z0-9_.-]*(?:access|auth|refresh|bearer|api|kaggle|hf|wandb|github|gh)[_-]token
            | token
        )\b
        ["']?\s*[:=]\s*
    )
    (["']?)
    ([^\s,"'&;}\])]+)
    \2
    """
)


def _replace_assignment(match: re.Match[str]) -> str:
    prefix, quote, _value = match.group(1), match.group(2), match.group(3)
    return f"{prefix}{quote}{REDACTED}{quote}"


def redact(text: str | None) -> str:
    """Redact secret-looking tokens and credential assignments from ``text``."""
    if text is None:
        return ""
    result = str(text)
    if not result:
        return result
    for pattern, repl in _TOKEN_PATTERNS:
        result = pattern.sub(repl, result)
    result = _ASSIGNMENT_PATTERN.sub(_replace_assignment, result)
    return result
