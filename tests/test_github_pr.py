"""Tests for the GitHub PR comment poster."""

from __future__ import annotations

import os

import httpx
import respx

from agent_eval.ci.gate import _wrap_summary, post_to_github_pr


def test_wrap_summary_includes_marker():
    out = _wrap_summary("## body")
    assert "agent-eval ci-gate" in out
    assert "## body" in out


def test_post_pr_no_token_returns_none(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert post_to_github_pr("body") is None


def test_post_pr_no_pr_number_returns_none(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.delenv("PR_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_PR_NUMBER", raising=False)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")  # not a PR ref
    assert post_to_github_pr("body") is None


@respx.mock
def test_post_pr_success(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.setenv("PR_NUMBER", "42")
    respx.post("https://api.github.com/repos/org/repo/issues/42/comments").mock(
        return_value=httpx.Response(201, json={"id": 1})
    )
    pr = post_to_github_pr("body")
    assert pr == 42


@respx.mock
def test_post_pr_failure_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.setenv("PR_NUMBER", "42")
    respx.post("https://api.github.com/repos/org/repo/issues/42/comments").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    try:
        post_to_github_pr("body")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "401" in str(e)


def test_pr_number_parsed_from_github_ref(monkeypatch):
    """When PR_NUMBER not set, should parse from GITHUB_REF=refs/pull/N/merge."""
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.delenv("PR_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_PR_NUMBER", raising=False)
    monkeypatch.setenv("GITHUB_REF", "refs/pull/123/merge")
    with respx.mock:
        respx.post("https://api.github.com/repos/org/repo/issues/123/comments").mock(
            return_value=httpx.Response(201, json={"id": 1})
        )
        pr = post_to_github_pr("body")
        assert pr == 123
