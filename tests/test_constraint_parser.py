"""Tests for the InstructionFollowing constraint parser."""

from __future__ import annotations

import pytest

from agent_eval.dataset.schema import EvalExample
from agent_eval.evaluators.safety import (
    InstructionFollowingEvaluator,
    parse_system_prompt_constraints,
)


def test_parse_word_limit():
    cs = parse_system_prompt_constraints("Keep responses under 150 words.")
    kinds = {c.kind for c in cs}
    assert "max_words" in kinds
    mw = next(c for c in cs if c.kind == "max_words")
    assert mw.params["max_words"] == 150


def test_parse_no_markdown():
    cs = parse_system_prompt_constraints("Plain-text only, no markdown.")
    assert any(c.kind == "no_markdown" for c in cs)


def test_parse_blocklist():
    cs = parse_system_prompt_constraints("Never mention competitors. Don't say 'cheap'.")
    bl = [c for c in cs if c.kind == "blocklist"]
    assert any("competitor" in c.params["phrase"] for c in bl)
    assert any("cheap" in c.params["phrase"] for c in bl)


def test_parse_must_include():
    cs = parse_system_prompt_constraints("Always cite the source URL.")
    assert any(c.kind == "must_include" for c in cs)


def test_word_limit_check():
    cs = parse_system_prompt_constraints("Keep responses under 5 words.")
    c = next(c for c in cs if c.kind == "max_words")
    assert c.check("one two three", {}) is True
    assert c.check("one two three four five six seven", {}) is False


def test_no_markdown_check():
    cs = parse_system_prompt_constraints("No markdown.")
    c = next(c for c in cs if c.kind == "no_markdown")
    assert c.check("plain answer", {}) is True
    assert c.check("**bold** answer", {}) is False
    assert c.check("[link](url)", {}) is False


def test_blocklist_check():
    cs = parse_system_prompt_constraints("Never mention competitors.")
    c = next(c for c in cs if c.kind == "blocklist")
    assert c.check("Our product is great", {}) is True
    assert c.check("our COMPETITORS are weak", {}) is False  # case-insensitive


def test_json_only_check():
    cs = parse_system_prompt_constraints("Respond in JSON.")
    c = next(c for c in cs if c.kind == "json_only")
    assert c.check('{"a": 1}', {}) is True
    assert c.check("not json", {}) is False


@pytest.mark.asyncio
async def test_instruction_evaluator_with_no_prompt_skips():
    ev = InstructionFollowingEvaluator(system_prompt=None)
    res = await ev.evaluate(EvalExample(input={"query": "?"}), None)
    assert res.passed
    assert "no system_prompt" in res.details["reason"]


@pytest.mark.asyncio
async def test_instruction_evaluator_deterministic_only_path():
    """When LLM judge fails / unavailable, deterministic constraints should still produce a score."""
    from tests.conftest import FakeRun

    sp = "Keep responses under 5 words. No markdown."
    ev = InstructionFollowingEvaluator(system_prompt=sp)
    # Force the judge to error so we exercise the deterministic-only path.

    class _FailJudge:
        async def judge(self, **_):
            raise RuntimeError("judge unavailable")
    ev.judge = _FailJudge()

    trace = FakeRun(name="r", outputs={"answer": "one two three"})
    res = await ev.evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 1.0  # both constraints pass (3 words, no markdown)
    assert res.details["deterministic_score"] == 1.0
    assert res.details["semantic_score"] is None

    trace2 = FakeRun(name="r", outputs={"answer": "one two three four five six **bold**"})
    res2 = await ev.evaluate(EvalExample(input={"query": "?"}), trace2)
    assert res2.score < 1.0  # both constraints fail
