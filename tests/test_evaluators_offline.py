"""Smoke tests for evaluators that don't require an LLM call.

These exercise the deterministic-path branches: tool/system/safety
evaluators on synthetic traces.
"""

from __future__ import annotations

import pytest

from agent_eval.dataset.schema import EvalExample
from agent_eval.evaluators.output_quality import (
    FormatComplianceEvaluator,
    KeywordCoverageEvaluator,
    TaskSuccessEvaluator,
)
from agent_eval.evaluators.safety import PIILeakageEvaluator
from agent_eval.evaluators.system_performance import (
    CostEfficiencyEvaluator,
    ErrorRateEvaluator,
    LatencyEvaluator,
)
from agent_eval.evaluators.tool_performance import (
    CostPerToolEvaluator,
    ToolPerformanceEvaluator,
)
from agent_eval.evaluators.trajectory import (
    CycleDetectionEvaluator,
    RedundancyEvaluator,
    StepSuccessRateEvaluator,
)


@pytest.mark.asyncio
async def test_task_success_with_answer(fake_trace_simple):
    ex = EvalExample(input={"query": "capital of France"})
    res = await TaskSuccessEvaluator().evaluate(ex, fake_trace_simple)
    assert res.score == 1.0
    assert res.passed


@pytest.mark.asyncio
async def test_task_success_no_answer():
    from tests.conftest import FakeRun

    trace = FakeRun(name="root", outputs={})
    ex = EvalExample(input={"query": "x"})
    res = await TaskSuccessEvaluator().evaluate(ex, trace)
    assert res.score == 0.0
    assert res.flagged


@pytest.mark.asyncio
async def test_keyword_coverage_hits(fake_trace_simple):
    ex = EvalExample(input={"query": "capital"}, expected_answer_keywords=["Paris", "France"])
    res = await KeywordCoverageEvaluator().evaluate(ex, fake_trace_simple)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_keyword_coverage_partial(fake_trace_simple):
    ex = EvalExample(input={"query": "capital"}, expected_answer_keywords=["Paris", "Madrid"])
    res = await KeywordCoverageEvaluator().evaluate(ex, fake_trace_simple)
    assert res.score == 0.5


@pytest.mark.asyncio
async def test_keyword_coverage_skipped_without_keywords(fake_trace_simple):
    ex = EvalExample(input={"query": "x"})
    res = await KeywordCoverageEvaluator().evaluate(ex, fake_trace_simple)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_format_compliance_word_limit(fake_trace_simple):
    ex = EvalExample(input={"query": "x"})
    short = FormatComplianceEvaluator(max_words=20)
    res = await short.evaluate(ex, fake_trace_simple)
    assert res.passed
    long = FormatComplianceEvaluator(max_words=2)
    res = await long.evaluate(ex, fake_trace_simple)
    assert not res.passed


@pytest.mark.asyncio
async def test_pii_leakage_detects_email():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r", outputs={"answer": "Reach me at john@example.com please."})
    ex = EvalExample(input={"query": "contact?"})
    res = await PIILeakageEvaluator().evaluate(ex, trace)
    assert res.score == 0.0
    assert res.flagged


@pytest.mark.asyncio
async def test_pii_leakage_clean():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r", outputs={"answer": "Just text, no PII."})
    ex = EvalExample(input={"query": "?"})
    res = await PIILeakageEvaluator().evaluate(ex, trace)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_step_success_rate_all_ok(fake_trace_research):
    ex = EvalExample(input={"query": "compare"})
    res = await StepSuccessRateEvaluator().evaluate(ex, fake_trace_research)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_redundancy_detects_duplicates():
    from tests.conftest import FakeRun

    children = [
        FakeRun(name="search", run_type="tool", inputs={"q": "x"}, outputs={}),
        FakeRun(name="search", run_type="tool", inputs={"q": "x"}, outputs={}),  # dup
        FakeRun(name="search", run_type="tool", inputs={"q": "y"}, outputs={}),
    ]
    trace = FakeRun(name="root", children=children)
    ex = EvalExample(input={"query": "?"})
    res = await RedundancyEvaluator().evaluate(ex, trace)
    assert res.score == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_cycle_detection_flags_repeats():
    from tests.conftest import FakeRun

    children = [FakeRun(name="loop", run_type="chain", inputs={"x": 1}) for _ in range(3)]
    trace = FakeRun(name="root", children=children)
    ex = EvalExample(input={"query": "?"})
    res = await CycleDetectionEvaluator().evaluate(ex, trace)
    assert res.score == 0.0
    assert res.flagged


@pytest.mark.asyncio
async def test_error_rate_per_query():
    from tests.conftest import FakeRun

    err_trace = FakeRun(name="r", error="oops")
    ok_trace = FakeRun(name="r")
    ex = EvalExample(input={"query": "?"})
    assert (await ErrorRateEvaluator().evaluate(ex, err_trace)).score == 0.0
    assert (await ErrorRateEvaluator().evaluate(ex, ok_trace)).score == 1.0


@pytest.mark.asyncio
async def test_tool_performance_aggregates(fake_trace_research):
    ex = EvalExample(input={"query": "?"})
    res = await ToolPerformanceEvaluator().evaluate(ex, fake_trace_research)
    assert "per_tool" in res.details
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_latency_with_short_trace():
    from datetime import datetime, timedelta, timezone

    from tests.conftest import FakeRun

    s = datetime.now(timezone.utc)
    trace = FakeRun(name="r", start_time=s, end_time=s + timedelta(milliseconds=200))
    ex = EvalExample(input={"query": "?"})
    res = await LatencyEvaluator(p95_target_ms=30000).evaluate(ex, trace)
    assert res.passed
    assert res.score > 0.99
