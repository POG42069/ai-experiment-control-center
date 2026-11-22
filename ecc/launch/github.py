"""Read-only GitHub CLI wrapper (``gh``) for listing repos, branches, and commits.

Security rules:
- Only two ``gh`` command shapes are permitted via an explicit allow-list:
  1. ``gh repo list ...``
  2. ``gh api repos/...`` (GET only; rejects non-GET ``-X``/``--method`` and rejects
     mutation/body flags ``-f``, ``-F``, ``--field``, ``--raw-field``, ``--input``).
- All repository, owner, branch, and ref inputs are validated against strict regexes.
- Error messages are always scrubbed through ``ecc.redact.redact``.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ecc.redact import redact

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,254}$")

DEFAULT_CACHE_TTL_S = 60.0

_API_FORBIDDEN_FLAGS = frozenset({"-f", "-F", "--field", "--raw-field", "--input"})
_API_FORBIDDEN_PREFIXES = ("-f", "-F", "--field=", "--raw-field=", "--input=")
_API_VALUE_FLAGS = frozenset({"-X", "--method", "-H", "--header", "-q", "--jq", "-t", "--template"})
_REPO_LIST_ALLOWED_FLAGS = frozenset({"--limit", "--json"})


class GitHubError(RuntimeError):
    """Raised when a read-only GitHub CLI invocation fails."""


@dataclass(frozen=True)
class RepoInfo:
    name_with_owner: str
    is_private: bool
    default_branch: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_with_owner": self.name_with_owner,
            "is_private": self.is_private,
            "default_branch": self.default_branch,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class BranchInfo:
    name: str
    sha: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha": self.sha,
        }


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    short_sha: str
    message_first_line: str
    author: str
    date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha": self.sha,
            "short_sha": self.short_sha,
            "message_first_line": self.message_first_line,
            "author": self.author,
            "date": self.date,
        }


def validate_repo(repo: str) -> str:
    """Validate ``owner/repo`` against ``^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$``."""
    if not isinstance(repo, str) or not REPO_RE.fullmatch(repo):
        raise ValueError(
            f"Invalid GitHub repository {repo!r}: expected 'owner/repo' with safe characters"
        )
    owner, name = repo.split("/", 1)
    if (
        ".." in repo
        or owner in {".", ".."}
        or name in {".", ".."}
        or owner.startswith("-")
        or name.startswith("-")
    ):
        raise ValueError(f"Invalid GitHub repository {repo!r}: unsafe path segment")
    return repo


def validate_owner(owner: str) -> str:
    """Validate a GitHub user or organization login."""
    if not isinstance(owner, str) or not OWNER_RE.fullmatch(owner):
        raise ValueError(f"Invalid GitHub owner {owner!r}")
    return owner


def validate_branch(branch: str) -> str:
    """Validate a git branch name (no whitespace, no '..', no leading '-')."""
    if not isinstance(branch, str) or not BRANCH_RE.fullmatch(branch):
        raise ValueError(
            f"Invalid git branch name {branch!r}: must not contain spaces, '..', or leading '-'"
        )
    if (
        ".." in branch
        or "//" in branch
        or "@{" in branch
        or branch.startswith("/")
        or branch.endswith("/")
        or branch.endswith(".")
        or branch.endswith(".lock")
    ):
        raise ValueError(f"Invalid git branch name {branch!r}: unsafe ref pattern")
    return branch


def validate_ref(ref: str) -> str:
    """Validate a git branch, tag, or commit SHA."""
    return validate_branch(ref)


def _decode_json_stream(text: str) -> Any:
    """Decode a single JSON document or concatenated JSON arrays from ``gh api --paginate``."""
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = 0
        length = len(stripped)
        chunks: list[Any] = []
        while idx < length:
            while idx < length and stripped[idx].isspace():
                idx += 1
            if idx >= length:
                break
            obj, end = decoder.raw_decode(stripped, idx)
            chunks.append(obj)
            idx = end
        if len(chunks) == 1:
            return chunks[0]
        if all(isinstance(c, list) for c in chunks):
            merged: list[Any] = []
            for c in chunks:
                merged.extend(c)
            return merged
        return chunks


def _parse_commit_entry(item: dict[str, Any]) -> CommitInfo:
    sha = str(item.get("sha") or "").strip()
    short_sha = sha[:7] if sha else ""
    commit_obj = item.get("commit") if isinstance(item.get("commit"), dict) else {}
    raw_msg = str(commit_obj.get("message") or "")
    first_line = raw_msg.splitlines()[0].strip() if raw_msg else ""

    author_obj = commit_obj.get("author") if isinstance(commit_obj.get("author"), dict) else {}
    committer_obj = (
        commit_obj.get("committer") if isinstance(commit_obj.get("committer"), dict) else {}
    )
    top_author = item.get("author") if isinstance(item.get("author"), dict) else {}

    author = str(
        author_obj.get("name")
        or top_author.get("login")
        or committer_obj.get("name")
        or ""
    ).strip()
    date = str(author_obj.get("date") or committer_obj.get("date") or "").strip()

    return CommitInfo(
        sha=sha,
        short_sha=short_sha,
        message_first_line=first_line,
        author=author,
        date=date,
    )


class GitHubReader:
    """Read-only ``gh`` CLI reader with strict allow-list and 60s TTL cache."""

    def __init__(
        self,
        runner: Callable[..., Any] = subprocess.run,
        gh_exe: str = "gh",
        timeout_s: float = 20,
        *,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runner = runner
        self.gh_exe = str(gh_exe)
        self.timeout_s = timeout_s
        self._cache_ttl_s = cache_ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def _cache_get(self, key: tuple[Any, ...]) -> tuple[bool, Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False, None
            expires_at, val = entry
            if self._clock() >= expires_at:
                self._cache.pop(key, None)
                return False, None
            return True, val

    def _cache_set(self, key: tuple[Any, ...], val: Any) -> None:
        if self._cache_ttl_s <= 0:
            return
        with self._lock:
            self._cache[key] = (self._clock() + self._cache_ttl_s, val)

    @staticmethod
    def _validate_command(args: Sequence[str]) -> None:
        if len(args) < 2:
            raise PermissionError(f"Disallowed gh CLI command: {list(args)!r}")

        cmd0 = args[0].strip().lower()
        cmd1 = args[1].strip().lower()

        if cmd0 == "repo" and cmd1 == "list":
            idx = 2
            while idx < len(args):
                tok = args[idx]
                if tok.startswith("-"):
                    flag_name = tok.split("=", 1)[0]
                    if flag_name not in _REPO_LIST_ALLOWED_FLAGS:
                        raise PermissionError(f"Disallowed flag for 'gh repo list': {tok!r}")
                    if "=" not in tok:
                        idx += 2
                        continue
                else:
                    validate_owner(tok)
                idx += 1
            return

        if cmd0 == "api":
            endpoint: str | None = None
            idx = 1
            while idx < len(args):
                tok = args[idx]
                if tok in _API_FORBIDDEN_FLAGS or any(
                    tok.startswith(p) for p in _API_FORBIDDEN_PREFIXES
                ):
                    raise PermissionError(
                        f"Disallowed write/body flag for 'gh api': {tok!r}"
                    )

                if tok in {"-X", "--method"}:
                    if idx + 1 >= len(args):
                        raise PermissionError("Missing HTTP method after -X/--method")
                    method = args[idx + 1].strip().upper()
                    if method != "GET":
                        raise PermissionError(
                            f"Only GET requests are allowed for 'gh api', got {method!r}"
                        )
                    idx += 2
                    continue

                if tok.startswith("--method="):
                    method = tok.split("=", 1)[1].strip().upper()
                    if method != "GET":
                        raise PermissionError(
                            f"Only GET requests are allowed for 'gh api', got {method!r}"
                        )
                    idx += 1
                    continue

                if tok.startswith("-X") and len(tok) > 2:
                    method = tok[2:].lstrip("=").strip().upper()
                    if method != "GET":
                        raise PermissionError(
                            f"Only GET requests are allowed for 'gh api', got {method!r}"
                        )
                    idx += 1
                    continue

                if tok.startswith("-") and not tok.startswith("--"):
                    # Reject combined short flags smuggling -f, -F, or -X.
                    short_chars = tok[1:]
                    if any(ch in short_chars for ch in ("f", "F", "X")):
                        raise PermissionError(f"Disallowed short flag for 'gh api': {tok!r}")

                if tok in _API_VALUE_FLAGS:
                    idx += 2
                    continue

                if tok.startswith("-"):
                    idx += 1
                    continue

                if endpoint is None:
                    endpoint = tok
                else:
                    raise PermissionError(f"Unexpected extra positional argument to 'gh api': {tok!r}")
                idx += 1

            if endpoint is None:
                raise PermissionError("'gh api' requires an endpoint starting with 'repos/'")
            if not endpoint.startswith("repos/") or ".." in endpoint or any(
                ch.isspace() for ch in endpoint
            ):
                raise PermissionError(
                    f"Disallowed 'gh api' endpoint {endpoint!r}; only 'repos/...' paths are permitted"
                )
            return

        raise PermissionError(
            f"Disallowed gh CLI subcommand {(cmd0, cmd1)!r}; "
            "only read-only ('repo', 'list') and GET ('api', 'repos/...') are permitted"
        )

    def run(self, args: Sequence[str]) -> str:
        """Execute an allow-listed read-only ``gh`` command and return stdout."""
        arg_list = [str(a) for a in args]
        self._validate_command(arg_list)

        argv = [self.gh_exe, *arg_list]
        try:
            proc = self._runner(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubError(f"gh CLI timed out after {self.timeout_s}s") from exc
        except OSError as exc:
            raise GitHubError(f"Failed to invoke gh CLI: {redact(str(exc))}") from exc

        returncode = getattr(proc, "returncode", 0)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise GitHubError(f"gh CLI failed (exit {returncode}): {redact(detail)}")
        return stdout

    def list_repos(self, owner: str | None = None) -> list[RepoInfo]:
        owner_norm = validate_owner(owner.strip()) if owner and owner.strip() else None
        cache_key = ("repos", owner_norm)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached)

        args = ["repo", "list"]
        if owner_norm:
            args.append(owner_norm)
        args.extend(
            [
                "--limit",
                "100",
                "--json",
                "nameWithOwner,isPrivate,defaultBranchRef,updatedAt",
            ]
        )
        raw = self.run(args)
        try:
            data = _decode_json_stream(raw)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"Invalid JSON from gh repo list: {redact(str(exc))}") from exc

        if not isinstance(data, list):
            raise GitHubError("Unexpected response shape from gh repo list (expected JSON list)")

        repos: list[RepoInfo] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name_with_owner = str(entry.get("nameWithOwner") or "").strip()
            if not name_with_owner:
                continue
            branch_ref = entry.get("defaultBranchRef")
            default_branch = (
                str(branch_ref.get("name") or "").strip()
                if isinstance(branch_ref, dict)
                else ""
            )
            repos.append(
                RepoInfo(
                    name_with_owner=name_with_owner,
                    is_private=bool(entry.get("isPrivate", False)),
                    default_branch=default_branch,
                    updated_at=str(entry.get("updatedAt") or "").strip(),
                )
            )
        self._cache_set(cache_key, repos)
        return list(repos)

    def list_branches(self, repo: str) -> list[BranchInfo]:
        repo_norm = validate_repo(repo)
        cache_key = ("branches", repo_norm)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached)

        raw = self.run(["api", f"repos/{repo_norm}/branches", "--paginate"])
        try:
            data = _decode_json_stream(raw)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"Invalid JSON from gh api branches: {redact(str(exc))}") from exc

        if not isinstance(data, list):
            raise GitHubError("Unexpected response shape from gh api branches (expected JSON list)")

        branches: list[BranchInfo] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            commit_obj = entry.get("commit") if isinstance(entry.get("commit"), dict) else {}
            sha = str(commit_obj.get("sha") or "").strip()
            if name:
                branches.append(BranchInfo(name=name, sha=sha))

        self._cache_set(cache_key, branches)
        return list(branches)

    def list_commits(self, repo: str, branch: str, n: int = 15) -> list[CommitInfo]:
        repo_norm = validate_repo(repo)
        branch_norm = validate_branch(branch)
        if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= 100):
            raise ValueError(f"n must be an integer between 1 and 100, got {n!r}")

        cache_key = ("commits", repo_norm, branch_norm, n)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return list(cached)

        encoded_branch = urllib.parse.quote(branch_norm, safe="")
        raw = self.run(
            ["api", f"repos/{repo_norm}/commits?sha={encoded_branch}&per_page={n}"]
        )
        try:
            data = _decode_json_stream(raw)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"Invalid JSON from gh api commits: {redact(str(exc))}") from exc

        if not isinstance(data, list):
            raise GitHubError("Unexpected response shape from gh api commits (expected JSON list)")

        commits = [_parse_commit_entry(item) for item in data if isinstance(item, dict)]
        self._cache_set(cache_key, commits)
        return list(commits)

    def resolve_commit(self, repo: str, ref: str) -> CommitInfo:
        repo_norm = validate_repo(repo)
        ref_norm = validate_ref(ref)

        cache_key = ("commit", repo_norm, ref_norm)
        hit, cached = self._cache_get(cache_key)
        if hit:
            return cached

        raw = self.run(["api", f"repos/{repo_norm}/commits/{ref_norm}"])
        try:
            data = _decode_json_stream(raw)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"Invalid JSON from gh api commit: {redact(str(exc))}") from exc

        if not isinstance(data, dict) or not data.get("sha"):
            raise GitHubError(
                f"Could not resolve commit ref {ref_norm!r} in {repo_norm!r}"
            )

        commit = _parse_commit_entry(data)
        self._cache_set(cache_key, commit)
        return commit
