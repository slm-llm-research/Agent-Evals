"""Tests for NLI helper + reasoning/citation hallucination evaluators."""

from __future__ import annotations

import pytest

from agent_eval.dataset.schema import EvalExample
from agent_eval.evaluators.hallucination import (
    CitationHallucinationEvaluator,
    PlanningHallucinationEvaluator,
    ReasoningHallucinationEvaluator,
)
from agent_eval.evaluators.nli import check_entailment, split_into_claims


def test_split_into_claims():
    text = "Paris is the capital of France. Berlin is the capital of Germany. Hi."
    claims = split_into_claims(text)
    assert len(claims) == 2  # "Hi." is too short
    assert "Paris" in claims[0]
    assert "Berlin" in claims[1]


def test_split_into_claims_empty():
    assert split_into_claims("") == []


def test_check_entailment_supported_via_overlap():
    v = check_entailment("Paris is the capital of France",
                         ["The capital of France is Paris."])
    assert v.is_supported


def test_check_entailment_unsupported_via_overlap():
    v = check_entailment("The capital of France is Tokyo",
                         ["Berlin is in Germany. Madrid is in Spain."])
    assert not v.is_supported


def test_check_entailment_no_evidence():
    v = check_entailment("Anything", [])
    assert not v.is_supported
    assert v.method == "no_evidence"


def test_check_entailment_empty_claim():
    v = check_entailment("", ["something"])
    assert v.is_supported  # empty claim trivially supported


@pytest.mark.asyncio
async def test_reasoning_hallucination_skipped_no_sources():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r", outputs={"answer": "Some claim."})
    res = await ReasoningHallucinationEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 1.0
    assert "no tool outputs" in res.details["reason"]


@pytest.mark.asyncio
async def test_reasoning_hallucination_supported_claims():
    from tests.conftest import FakeRun

    trace = FakeRun(
        name="r",
        outputs={"answer": "The capital of France is Paris. Berlin is the capital of Germany."},
        children=[
            FakeRun(name="search", run_type="tool",
                    outputs={"results": ["Paris is the capital of France.", "Berlin is the capital of Germany."]}),
        ],
    )
    res = await ReasoningHallucinationEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score >= 0.8
    assert res.details["supported"] >= 1


@pytest.mark.asyncio
async def test_reasoning_hallucination_unsupported_claims():
    from tests.conftest import FakeRun

    trace = FakeRun(
        name="r",
        outputs={"answer": "The Moon is made of dijksitarian quirinox cheeseplate alloy."},
        children=[
            FakeRun(name="search", run_type="tool",
                    outputs={"results": ["The Moon is rocky and contains regolith."]}),
        ],
    )
    res = await ReasoningHallucinationEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score < 0.5


@pytest.mark.asyncio
async def test_citation_evaluator_url_in_sources():
    from tests.conftest import FakeRun

    trace = FakeRun(
        name="r",
        outputs={"answer": "See https://example.com/x for more details."},
        children=[
            FakeRun(name="search", run_type="tool", outputs={"results": ["found at https://example.com/x"]}),
        ],
    )
    res = await CitationHallucinationEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_citation_evaluator_url_fabricated():
    from tests.conftest import FakeRun

    trace = FakeRun(
        name="r",
        outputs={"answer": "Source: https://fabricated.example/page"},
        children=[
            FakeRun(name="search", run_type="tool", outputs={"results": ["found at https://different.example/page"]}),
        ],
    )
    res = await CitationHallucinationEvaluator(fetch_urls=False).evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 0.0


@pytest.mark.asyncio
async def test_planning_hallucination_no_plan():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r", outputs={"answer": "x"})
    res = await PlanningHallucinationEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 1.0
