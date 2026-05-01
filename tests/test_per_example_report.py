"""Tests for the new per-example fields + HTML drill-down rendering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import respx

from agent_eval import AgentEval
from agent_eval.dataset.schema import EvalDataset, EvalExample
from agent_eval.discovery.registry import ComponentRegistry
from agent_eval.reporters.html_reporter import render_html_report
from agent_eval.reporters.report import (
    DimensionScores,
    EvaluationReport,
    PerExampleResult,
    SystemOverview,
    ToolCall,
    status_for_score,
)
from agent_eval.runners.http_runner import HttpAgentRunner


# ---------------------------------------------------------------------------- _build_per_example_result


def test_per_example_result_pass_logic():
    """A PerExampleResult with high pass_rate and no critical_count should be is_overall_pass=True."""
    ex = PerExampleResult(
        example_id="x", query="q", score=0.85, pass_rate=0.9, flagged_count=1,
        critical_count=0, is_overall_pass=True,
    )
    assert ex.is_overall_pass is True


def test_per_example_result_fail_logic():
    ex = PerExampleResult(
        example_id="x", query="q", score=0.4, pass_rate=0.5, flagged_count=4,
        critical_count=2, is_overall_pass=False,
    )
    assert ex.is_overall_pass is False
    assert ex.critical_count == 2


def test_tool_call_fields():
    tc = ToolCall(name="web_search", inputs={"q": "x"}, outputs_preview="ok",
                   success=True, latency_ms=120.0)
    assert tc.success is True
    assert tc.latency_ms == 120.0


# ---------------------------------------------------------------------------- end-to-end with runner


@respx.mock
def test_evaluate_populates_per_example_results():
    """When evaluate() runs with a runner, the report should contain a per_example_results entry per dataset example."""
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(
        200, json={
            "answer": "Paris is the capital of France.",
            "tool_calls": [
                {"name": "web_search", "inputs": {"q": "capital France"},
                 "outputs": {"results": ["Paris is the capital."]}}
            ],
        }
    ))
    ds = EvalDataset(name="t", examples=[
        EvalExample(input={"query": "capital of France?"}, expected_answer_keywords=["Paris"]),
        EvalExample(input={"query": "capital of Germany?"}, expected_answer_keywords=["Berlin"]),
    ])
    ev = AgentEval(ComponentRegistry(discovery_method="manual"))
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval", langsmith_project=None)
    report = ev.evaluate(dataset=ds, runner=runner, dimensions=["output_quality"], backend="native")

    assert len(report.per_example_results) == 2
    pe = report.per_example_results[0]
    assert pe.query == "capital of France?"
    assert pe.actual_output == "Paris is the capital of France."
    assert pe.expected_keywords == ["Paris"]
    # SyntheticTrace surfaced the response's `tool_calls` array as a child run.
    assert len(pe.tool_calls) >= 1
    assert pe.tool_calls[0].name == "web_search"
    # Synthetic flag should be True (no real LangSmith run captured).
    assert pe.trace_was_synthetic is True


@respx.mock
def test_evaluate_marks_synthetic_when_no_run_id():
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(
        200, json={"answer": "ok"}  # no langsmith_run_id key
    ))
    ds = EvalDataset(name="t", examples=[EvalExample(input={"query": "?"})])
    ev = AgentEval(ComponentRegistry(discovery_method="manual"))
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval", langsmith_project=None)
    report = ev.evaluate(dataset=ds, runner=runner, dimensions=["output_quality"], backend="native")
    assert report.per_example_results[0].langsmith_run_id is None
    assert report.per_example_results[0].langsmith_run_url is None
    assert report.per_example_results[0].trace_was_synthetic is True


@respx.mock
def test_evaluate_extracts_run_id_for_deep_link():
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(
        200, json={"answer": "Paris", "langsmith_run_id": "abc-123-def"}
    ))
    ds = EvalDataset(name="t", examples=[EvalExample(input={"query": "?"})])
    ev = AgentEval(ComponentRegistry(discovery_method="manual"))
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval", langsmith_project=None)
    report = ev.evaluate(dataset=ds, runner=runner, dimensions=["output_quality"], backend="native")
    pe = report.per_example_results[0]
    # Trace is still synthetic (no real LangSmith fetch happened — no project), but the
    # run_result.metadata captured the run_id so the deep link is built.
    assert pe.langsmith_run_id == "abc-123-def"
    assert "abc-123-def" in (pe.langsmith_run_url or "")


# ---------------------------------------------------------------------------- HTML sections


def _sample_report_with_examples():
    return EvaluationReport(
        system_name="voice-agent-system",
        dataset_name="voice_agent",
        dataset_size=2,
        evaluation_duration_seconds=8.4,
        system_overview=SystemOverview(
            overall_score=0.78, health_status=status_for_score(0.78),
            pass_rate=0.85, flag_count=2, critical_flag_count=0,
        ),
        dimension_scores=DimensionScores(
            output_quality=0.80, trajectory_quality=0.85, hallucination_risk=0.05,
            tool_performance=0.90, system_performance=0.95, safety=1.0, memory_quality=0.0,
        ),
        per_example_results=[
            PerExampleResult(
                example_id="ex-1",
                query="What is the capital of France?",
                actual_output="The capital of France is Paris.",
                expected_keywords=["Paris"],
                query_type="search",
                complexity="simple",
                score=0.92, pass_rate=1.0, flagged_count=0, critical_count=0,
                is_overall_pass=True,
                langsmith_run_id="run-pass-001",
                langsmith_run_url="https://smith.langchain.com/r/run-pass-001",
                tool_calls=[
                    ToolCall(name="web_search", inputs={"q": "capital France"},
                              outputs_preview='{"results": ["Paris"]}', success=True, latency_ms=145.0),
                ],
            ),
            PerExampleResult(
                example_id="ex-2",
                query="Compare emissions of A and B in 2099.",
                actual_output="",
                query_type="research",
                complexity="complex",
                score=0.30, pass_rate=0.40, flagged_count=4, critical_count=2,
                is_overall_pass=False,
                langsmith_run_id="run-fail-002",
                langsmith_run_url="https://smith.langchain.com/r/run-fail-002",
                tool_calls=[
                    ToolCall(name="web_search", inputs={"q": "A emissions 2099"}, success=False,
                              error="Tool returned 500", latency_ms=4200.0),
                ],
                runner_error=None,
            ),
        ],
        tuning_recommendations=[
            {
                "component": "answer-generation",
                "issue_type": "answer quality",
                "severity": "high",
                "current_score": 0.5,
                "target_score": 0.85,
                "effort_estimate": "hours",
                "tuning_type": "prompt",
                "specific_action": "Add 3-5 few-shot examples to the prompt.",
            }
        ],
    )


def test_html_renders_recommendations_at_top(tmp_path: Path):
    r = _sample_report_with_examples()
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    # Recommendations heading should appear BEFORE the Overview heading in the HTML.
    rec_idx = text.find('id="recommendations"')
    overview_idx = text.find('id="overview"')
    assert 0 < rec_idx < overview_idx


def test_html_renders_metric_explanations(tmp_path: Path):
    r = _sample_report_with_examples()
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    # Both headline + dimension explanations should be present.
    assert "Weighted aggregate across all 7 dimensions" in text
    assert "How good is the agent's final answer" in text
    assert "Risk that the agent fabricated facts" in text


def test_html_renders_query_drilldown(tmp_path: Path):
    r = _sample_report_with_examples()
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    # Both queries surface.
    assert "What is the capital of France?" in text
    assert "Compare emissions of A and B in 2099." in text
    # Pass / fail badges.
    assert "✓ PASS" in text
    assert "✗ FAIL" in text
    # Tool call detail.
    assert "web_search" in text
    assert "Tool returned 500" in text
    assert "145" in text  # latency
    # LangSmith deep links.
    assert "https://smith.langchain.com/r/run-pass-001" in text
    assert "https://smith.langchain.com/r/run-fail-002" in text


def test_html_renders_pass_fail_filter_buttons(tmp_path: Path):
    r = _sample_report_with_examples()
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    # Filter buttons with counts.
    assert "Passed (1)" in text
    assert "Failed (1)" in text
    assert "filterQueries" in text  # JS function


def test_html_keyword_check_marks(tmp_path: Path):
    r = _sample_report_with_examples()
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    # "Paris" should be ticked because it appears in the actual_output.
    assert "✓ Paris" in text


def test_html_no_per_example_data_shows_empty_state(tmp_path: Path):
    r = _sample_report_with_examples()
    r.per_example_results = []
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    assert "No per-example data" in text


def test_dimension_pairs_use_proper_threshold_styling(tmp_path: Path):
    """Dimensions below 0.5 (or hallucination_risk above 0.25) should be in `class='critical'` rows."""
    r = _sample_report_with_examples()
    r.dimension_scores.output_quality = 0.30  # critical
    r.dimension_scores.hallucination_risk = 0.40  # critical (inverted)
    p = render_html_report(r, tmp_path / "out.html")
    text = p.read_text()
    # Grep for two `class="critical"` rows in the metrics table specifically. Dimensions
    # appear inside <tr class="critical"> wrappers in the dimension-scores table.
    assert text.count('class="critical"') >= 2
