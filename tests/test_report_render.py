"""Tests for report rendering — JSON + HTML."""

from __future__ import annotations

from pathlib import Path

from agent_eval.reporters.html_reporter import render_html_report
from agent_eval.reporters.json_reporter import render_json_report
from agent_eval.reporters.report import (
    ComponentScore,
    DimensionScores,
    EvaluationReport,
    Issue,
    SystemOverview,
    status_for_score,
)


def _sample_report():
    return EvaluationReport(
        system_name="test-system",
        dataset_name="dataset-x",
        dataset_size=10,
        evaluation_duration_seconds=12.5,
        system_overview=SystemOverview(
            overall_score=0.78,
            health_status=status_for_score(0.78),
            pass_rate=0.85,
            flag_count=3,
            critical_flag_count=1,
        ),
        dimension_scores=DimensionScores(
            output_quality=0.82,
            trajectory_quality=0.75,
            hallucination_risk=0.05,
            tool_performance=0.90,
            system_performance=0.65,
            safety=1.0,
            memory_quality=0.85,
        ),
        component_scores=[
            ComponentScore(component_name="orchestrator", overall_score=0.80, rank=1),
            ComponentScore(component_name="search_agent", overall_score=0.70, rank=2),
        ],
        flagged_issues=[
            Issue(severity="critical", component="orchestrator", metric="cycle_detected", score=0.0, description="cycle"),
        ],
        backends_used={"task_success_rate": "native", "answer_relevance": "native"},
    )


def test_json_render(tmp_path: Path):
    r = _sample_report()
    p = render_json_report(r, tmp_path / "report.json")
    assert p.exists()
    blob = p.read_text()
    assert "test-system" in blob
    assert "report_id" in blob


def test_html_render(tmp_path: Path):
    r = _sample_report()
    p = render_html_report(r, tmp_path / "report.html")
    assert p.exists()
    text = p.read_text()
    assert "<!doctype html>" in text
    assert "test-system" in text
    assert "Chart" in text
    assert "0.78" in text
    assert len(text) > 3000  # small sample report renders ~4-5KB; full reports easily exceed 10KB


def test_report_compare_detects_drop():
    r_now = _sample_report()
    prev = _sample_report()
    prev.dimension_scores.output_quality = 0.95
    diff = r_now.compare(prev)
    assert "output_quality" in diff.metrics_changed
    assert diff.metrics_changed["output_quality"]["pct_change"] < 0
