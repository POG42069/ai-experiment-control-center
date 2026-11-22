"""Tests for ecc.launch.github (all via fake subprocess runner — no real gh/network)."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from ecc.launch.github import (
    GitHubError,
    GitHubReader,
    validate_branch,
    validate_repo,
)


class TestGitHubValidation:
    @pytest.mark.parametrize(
        "bad_repo",
        [
            "a/b; rm -rf",
            "a/b && echo hi",
            "--upload-pack=x",
            "owner/..",
            "../repo",
            "owner/-repo",
            "-owner/repo",
            "just-one-part",
            "a/b/c",
            "owner/repo with space",
            "",
        ],
    )
    def test_validate_repo_rejects_unsafe_inputs(self, bad_repo: str) -> None:
        with pytest.raises(ValueError):
            validate_repo(bad_repo)

    @pytest.mark.parametrize(
        "good_repo",
        [
            "octocat/hello-world",
            "my_org/model.v2-train",
            "User-123/repo_456",
        ],
    )
    def test_validate_repo_accepts_valid(self, good_repo: str) -> None:
        assert validate_repo(good_repo) == good_repo

    @pytest.mark.parametrize(
        "bad_branch",
        [
            "a/b; rm -rf",
            "--upload-pack=x",
            "-leading-dash",
            "..",
            "feat/..",
            "feat/../main",
            "has space",
            "branch\nnewline",
            "bad//double",
            "/leading-slash",
            "trailing-slash/",
            "bad@{0}",
            "name.lock",
            "",
        ],
    )
    def test_validate_branch_rejects_unsafe_inputs(self, bad_branch: str) -> None:
        with pytest.raises(ValueError):
            validate_branch(bad_branch)

    @pytest.mark.parametrize(
        "good_branch",
        [
            "main",
            "feat/v2-lora_1.0",
            "release/0.1.2",
            "0123456789abcdef0123456789abcdef01234567",
        ],
    )
    def test_validate_branch_accepts_valid(self, good_branch: str) -> None:
        assert validate_branch(good_branch) == good_branch


class TestGitHubAllowList:
    @pytest.mark.parametrize(
        "disallowed_args",
        [
            ["repo", "delete", "octocat/hello"],
            ["repo", "create", "octocat/new"],
            ["repo", "edit", "octocat/hello"],
            ["pr", "merge", "1"],
            ["auth", "token"],
            ["api", "-X", "POST", "repos/octocat/hello/issues"],
            ["api", "-XPOST", "repos/octocat/hello/issues"],
            ["api", "--method", "DELETE", "repos/octocat/hello"],
            ["api", "--method=PATCH", "repos/octocat/hello"],
            ["api", "-f", "title=hi", "repos/octocat/hello/issues"],
            ["api", "-F", "count=1", "repos/octocat/hello/issues"],
            ["api", "--field", "a=b", "repos/octocat/hello/issues"],
            ["api", "--field=a=b", "repos/octocat/hello/issues"],
            ["api", "--raw-field", "a=b", "repos/octocat/hello/issues"],
            ["api", "--input", "body.json", "repos/octocat/hello/issues"],
            ["api", "user"],
            ["api", "orgs/octocat/repos"],
            ["api", "/repos/octocat/hello/branches"],
            ["api", "repos/octocat/hello/../../user"],
            ["api", "-Xf", "repos/octocat/hello/branches"],
            ["api"],
        ],
    )
    def test_allow_list_rejects_disallowed_commands(self, disallowed_args: list[str]) -> None:
        called = False

        def should_not_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal called
            called = True
            raise AssertionError("runner must not be called for disallowed gh commands")

        reader = GitHubReader(runner=should_not_run)
        with pytest.raises(PermissionError):
            reader.run(disallowed_args)
        assert called is False

    def test_allow_list_permits_explicit_get_on_repos_path(self) -> None:
        captured: list[list[str]] = []

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured.append(list(argv))
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="[]", stderr="")

        reader = GitHubReader(runner=fake_runner, gh_exe="fake-gh")
        out = reader.run(["api", "-X", "GET", "repos/octocat/hello/branches"])
        assert out == "[]"
        assert captured == [["fake-gh", "api", "-X", "GET", "repos/octocat/hello/branches"]]


class TestGitHubReaderParsingAndCache:
    def test_list_repos_branches_commits_resolve_and_ttl_cache(self) -> None:
        now = [1000.0]
        calls: list[list[str]] = []

        repos_json = json.dumps(
            [
                {
                    "nameWithOwner": "alice/vlm-exp",
                    "isPrivate": True,
                    "defaultBranchRef": {"name": "main"},
                    "updatedAt": "2026-11-22T10:00:00Z",
                },
                {
                    "nameWithOwner": "alice/empty-repo",
                    "isPrivate": False,
                    "defaultBranchRef": None,
                    "updatedAt": "2026-11-20T08:00:00Z",
                },
            ]
        )
        # Simulated paginated output (two concatenated JSON arrays)
        branches_json = (
            json.dumps([{"name": "main", "commit": {"sha": "1111111111111111111111111111111111111111"}}])
            + "\n"
            + json.dumps(
                [{"name": "feat/lora", "commit": {"sha": "2222222222222222222222222222222222222222"}}]
            )
        )
        commits_json = json.dumps(
            [
                {
                    "sha": "abcdef1234567890abcdef1234567890abcdef12",
                    "commit": {
                        "message": "Add cosine LR scheduler\n\nDetailed body line",
                        "author": {"name": "Alice", "date": "2026-11-22T09:30:00Z"},
                    },
                }
            ]
        )
        single_commit_json = json.dumps(
            {
                "sha": "abcdef1234567890abcdef1234567890abcdef12",
                "commit": {
                    "message": "Add cosine LR scheduler\n\nDetails",
                    "author": {"name": "Alice", "date": "2026-11-22T09:30:00Z"},
                },
            }
        )

        def fake_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(list(argv))
            if argv[1:3] == ["repo", "list"]:
                return subprocess.CompletedProcess(argv, 0, repos_json, "")
            if argv[1] == "api" and argv[2].endswith("/branches"):
                return subprocess.CompletedProcess(argv, 0, branches_json, "")
            if argv[1] == "api" and "/commits?" in argv[2]:
                return subprocess.CompletedProcess(argv, 0, commits_json, "")
            if argv[1] == "api" and "/commits/" in argv[2]:
                return subprocess.CompletedProcess(argv, 0, single_commit_json, "")
            raise AssertionError(f"Unexpected argv: {argv}")

        reader = GitHubReader(
            runner=fake_runner,
            gh_exe="gh",
            cache_ttl_s=60.0,
            clock=lambda: now[0],
        )

        repos = reader.list_repos(owner="alice")
        assert len(repos) == 2
        assert repos[0].name_with_owner == "alice/vlm-exp"
        assert repos[0].is_private is True
        assert repos[0].default_branch == "main"
        assert repos[1].default_branch == ""

        # Cached within 60s
        now[0] += 30.0
        _ = reader.list_repos(owner="alice")
        assert len(calls) == 1

        # Expired after 60s
        now[0] += 35.0
        _ = reader.list_repos(owner="alice")
        assert len(calls) == 2

        branches = reader.list_branches("alice/vlm-exp")
        assert [b.name for b in branches] == ["main", "feat/lora"]
        assert branches[1].sha == "2222222222222222222222222222222222222222"

        commits = reader.list_commits("alice/vlm-exp", "feat/lora", n=15)
        assert len(commits) == 1
        assert commits[0].short_sha == "abcdef1"
        assert commits[0].message_first_line == "Add cosine LR scheduler"
        assert commits[0].author == "Alice"
        assert commits[0].date == "2026-11-22T09:30:00Z"

        resolved = reader.resolve_commit("alice/vlm-exp", "feat/lora")
        assert resolved.sha == "abcdef1234567890abcdef1234567890abcdef12"
        assert resolved.short_sha == "abcdef1"

    def test_error_message_is_redacted(self) -> None:
        fake_gh_token = "ghp_1234567890abcdefghij1234567890"

        def failing_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=1,
                stdout="",
                stderr=f"HTTP 401 Unauthorized using token {fake_gh_token}",
            )

        reader = GitHubReader(runner=failing_runner)
        with pytest.raises(GitHubError) as exc_info:
            reader.list_repos()
        msg = str(exc_info.value)
        assert fake_gh_token not in msg
        assert "[REDACTED]" in msg
