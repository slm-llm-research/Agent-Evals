"""Tests for the TRACE-style CoT trajectory judge."""

from __future__ import annotations

import pytest

from agent_eval.judges.chain_of_thought_judge import (
    TrajectoryJudge,
    TrajectoryStep,
    _add_to_evidence,
    _claim_supported_by_bank,
    _detect_redundancy,
)


def test_evidence_bank_lru():
    from collections import OrderedDict

    bank: OrderedDict[str, str] = OrderedDict()
    for i in range(5):
        _add_to_evidence(bank, f"claim {i}", max_size=3)
    # Only 3 most recent kept.
    assert len(bank) == 3
    assert "claim 4" in bank.values()
    assert "claim 0" not in bank.values()


def test_claim_supported_by_bank_overlap():
    from collections import OrderedDict

    bank: OrderedDict[str, str] = OrderedDict()
    bank["h1"] = "Paris is the capital of France"
    assert _claim_supported_by_bank("Paris is the capital of France indeed", bank)
    assert not _claim_supported_by_bank("Madrid Spain Barcelona Sevilla", bank)


def test_redundancy_detection():
    steps = [
        TrajectoryStep(node="search", input={"q": "x"}, is_tool=True),
        TrajectoryStep(node="search", input={"q": "x"}, is_tool=True),  # duplicate
        TrajectoryStep(node="search", input={"q": "y"}, is_tool=True),
    ]
    redundancy = _detect_redundancy(steps)
    assert len(redundancy) == 1
    assert redundancy[0]["step_index"] == 1


@pytest.mark.asyncio
async def test_empty_trajectory():
    judge = TrajectoryJudge()
    res = await judge.judge_trajectory(query="?", trajectory=[])
    assert "empty" in res.improvement_suggestions[0]


@pytest.mark.asyncio
async def test_tool_step_scoring_no_llm_needed():
    """Pure-tool trajectory should score deterministically without an LLM call."""
    steps = [
        TrajectoryStep(node="search", input={"q": "x"}, output={"r": "ok"}, is_tool=True),
        TrajectoryStep(node="search", input={"q": "y"}, output={"r": "ok"}, is_tool=True),
    ]
    judge = TrajectoryJudge()
    res = await judge.judge_trajectory(query="?", trajectory=steps)
    # Each step gets 1.0 (no errors, no redundancy).
    assert all(s.relevance == 1.0 for s in res.per_step_scores)
    assert all(s.correctness == 1.0 for s in res.per_step_scores)


@pytest.mark.asyncio
async def test_tool_step_redundancy_penalty():
    steps = [
        TrajectoryStep(node="search", input={"q": "x"}, is_tool=True),
        TrajectoryStep(node="search", input={"q": "x"}, is_tool=True),  # duplicate
    ]
    judge = TrajectoryJudge()
    res = await judge.judge_trajectory(query="?", trajectory=steps)
    assert res.redundant_calls
    # Second step is penalized on efficiency.
    assert res.per_step_scores[1].efficiency < 1.0


@pytest.mark.asyncio
async def test_adaptivity_after_recovery():
    """Error followed by changed retry → adaptivity > 0."""
    steps = [
        TrajectoryStep(node="search", input={"q": "x"}, error="boom", is_tool=True),
        TrajectoryStep(node="search", input={"q": "different"}, output={"r": "ok"}, is_tool=True),
    ]
    judge = TrajectoryJudge()
    res = await judge.judge_trajectory(query="?", trajectory=steps)
    assert res.overall_adaptivity == 1.0  # 1 error, 1 recovery
