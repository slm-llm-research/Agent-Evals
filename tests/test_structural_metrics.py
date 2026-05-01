"""Tests for v2 structural metrics: Tool F1, Node F1, SSI."""

from __future__ import annotations

import pytest

from agent_eval.dataset.schema import EvalExample
from agent_eval.evaluators.trajectory import (
    NodeF1Evaluator,
    StructuralSimilarityEvaluator,
    ToolF1Evaluator,
    _f1,
    _order_bonus,
)


def test_f1_perfect_match():
    p, r, f = _f1(["a", "b"], ["a", "b"])
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_f1_disjoint():
    p, r, f = _f1(["a"], ["b"])
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_f1_partial():
    p, r, f = _f1(["a", "b", "c"], ["a", "b"])
    # actual = {a,b,c}, expected = {a,b}, tp=2
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(2 * (2/3) * 1.0 / (2/3 + 1.0))


def test_order_bonus():
    assert _order_bonus(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _order_bonus(["c", "b", "a"], ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert _order_bonus([], ["a"]) == 0.0


@pytest.mark.asyncio
async def test_tool_f1_evaluator(fake_trace_research):
    ex = EvalExample(
        input={"query": "Compare emissions"},
        expected_tool_sequence=["web_search", "web_search", "web_search", "summarize"],
    )
    res = await ToolF1Evaluator().evaluate(ex, fake_trace_research)
    assert 0.95 <= res.score <= 1.0
    assert res.passed
    assert "f1" in res.details


@pytest.mark.asyncio
async def test_tool_f1_skipped_without_expected():
    ex = EvalExample(input={"query": "Hello"})
    res = await ToolF1Evaluator().evaluate(ex, None)
    assert res.score == 1.0
    assert "skipped" in res.details["reason"]


@pytest.mark.asyncio
async def test_node_f1_evaluator(fake_trace_research):
    ex = EvalExample(
        input={"query": "Compare emissions"},
        expected_task_graph={"plan": ["web_search"], "web_search": ["summarize"], "summarize": []},
    )
    res = await NodeF1Evaluator().evaluate(ex, fake_trace_research)
    # actual graph nodes from trace = {research_agent, plan}; expected = {plan, web_search, summarize}.
    # intersection = {plan}; F1 = 2*(1/2)*(1/3)/(1/2+1/3) = 0.4. The evaluator measures correctly.
    assert res.score == 0.4
    assert "precision" in res.details


@pytest.mark.asyncio
async def test_ssi_evaluator(fake_trace_research):
    ex = EvalExample(
        input={"query": "Compare emissions"},
        expected_task_graph={"research_agent": ["plan"], "plan": ["web_search"], "web_search": ["summarize"], "summarize": []},
    )
    res = await StructuralSimilarityEvaluator().evaluate(ex, fake_trace_research)
    assert 0.0 <= res.score <= 1.0
    assert "graph_edit_distance" in res.details or res.details.get("note")


@pytest.mark.asyncio
async def test_ssi_skipped_without_expected_graph():
    ex = EvalExample(input={"query": "Hi"})
    res = await StructuralSimilarityEvaluator().evaluate(ex, None)
    assert res.score == 1.0
