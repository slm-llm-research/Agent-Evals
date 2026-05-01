"""Tests for the TuningAdvisor's 6 signal types."""

from __future__ import annotations

from agent_eval.evaluators.base import EvaluatorResult
from agent_eval.reporters.report import (
    ComponentScore,
    DimensionScores,
    EvaluationReport,
    SystemOverview,
    status_for_score,
)
from agent_eval.reporters.tuning_advisor import TuningAdvisor


def _report(out=0.85, traj=0.85, hallu_risk=0.05, tool=0.85, sys_p=0.85, safety=1.0, mem=0.85,
            cycle_results=None, components=None):
    overall = (out + traj + (1 - hallu_risk) + tool + sys_p + safety + mem) / 7
    return EvaluationReport(
        system_name="test",
        dataset_name="d",
        dataset_size=10,
        system_overview=SystemOverview(
            overall_score=overall, health_status=status_for_score(overall),
            pass_rate=1.0, flag_count=0, critical_flag_count=0,
        ),
        dimension_scores=DimensionScores(
            output_quality=out, trajectory_quality=traj, hallucination_risk=hallu_risk,
            tool_performance=tool, system_performance=sys_p, safety=safety, memory_quality=mem,
        ),
        component_scores=components or [],
    )


def test_no_signals_when_healthy():
    advisor = TuningAdvisor()
    signals = advisor.analyze(_report())
    assert signals == []


def test_prompt_signal_on_low_output_quality():
    advisor = TuningAdvisor()
    signals = advisor.analyze(_report(out=0.5))
    assert any(s.tuning_type == "prompt" and "answer quality" in s.issue_type for s in signals)


def test_hallucination_signal_critical():
    advisor = TuningAdvisor()
    signals = advisor.analyze(_report(hallu_risk=0.4))
    hallu_signals = [s for s in signals if "hallucination" in s.issue_type]
    assert any(s.severity == "critical" for s in hallu_signals)


def test_architecture_signal_on_cycle():
    cycle_component = ComponentScore(
        component_name="orchestrator",
        overall_score=0.5,
        evaluator_results=[EvaluatorResult(
            evaluator_name="cycle_detected", component_name="orchestrator",
            score=0.0, passed=False, threshold=0.7, flagged=True, flag_reason="cycle detected",
        )],
        rank=1,
    )
    advisor = TuningAdvisor()
    signals = advisor.analyze(_report(components=[cycle_component]))
    assert any(s.tuning_type == "architecture" and "cycles" in s.issue_type for s in signals)


def test_memory_signal_on_low_recall():
    component = ComponentScore(
        component_name="memory",
        overall_score=0.5,
        evaluator_results=[
            EvaluatorResult(evaluator_name="memory_retrieval_recall", component_name="memory",
                            score=0.5, passed=False, threshold=0.7),
            EvaluatorResult(evaluator_name="memory_retrieval_precision", component_name="memory",
                            score=0.9, passed=True, threshold=0.7),
        ],
        rank=1,
    )
    advisor = TuningAdvisor()
    signals = advisor.analyze(_report(mem=0.6, components=[component]))
    memory_signals = [s for s in signals if s.tuning_type == "memory"]
    assert memory_signals
    assert any("recall" in s.issue_type for s in memory_signals)


def test_signals_sorted_by_severity():
    advisor = TuningAdvisor()
    signals = advisor.analyze(_report(out=0.4, hallu_risk=0.5, sys_p=0.5))
    severities = [s.severity for s in signals]
    valid_order = ["critical", "high", "medium", "low"]
    indices = [valid_order.index(s) for s in severities]
    assert indices == sorted(indices)
