"""Tests for CI gate regression detection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_eval.ci.gate import ci_gate
from agent_eval.reporters.regression_detector import RegressionDetector, ThresholdConfig
from agent_eval.reporters.report import (
    DimensionScores,
    EvaluationReport,
    SystemOverview,
    status_for_score,
)


def _report(out=0.85, traj=0.85, hallu_risk=0.05, tool=0.85, sys_p=0.85, safety=1.0, mem=0.85):
    overall = (out + traj + (1 - hallu_risk) + tool + sys_p + safety + mem) / 7
    return EvaluationReport(
        system_name="test",
        dataset_name="d",
        dataset_size=1,
        system_overview=SystemOverview(
            overall_score=overall,
            health_status=status_for_score(overall),
            pass_rate=1.0,
            flag_count=0,
            critical_flag_count=0,
        ),
        dimension_scores=DimensionScores(
            output_quality=out,
            trajectory_quality=traj,
            hallucination_risk=hallu_risk,
            tool_performance=tool,
            system_performance=sys_p,
            safety=safety,
            memory_quality=mem,
        ),
    )


def test_no_regression_passes():
    cur = _report()
    base = _report()
    detector = RegressionDetector()
    regs = detector.detect(cur, base, ThresholdConfig({"output_quality": {"max_regression_pct": 5}}))
    assert all(r.severity == "none" for r in regs) or not regs


def test_regression_detects_drop():
    cur = _report(out=0.60)
    base = _report(out=0.85)
    detector = RegressionDetector()
    regs = detector.detect(cur, base, ThresholdConfig({"output_quality": {"max_regression_pct": 5}}))
    output_regs = [r for r in regs if r.metric_name == "output_quality"]
    assert output_regs
    assert output_regs[0].severity in ("critical", "high")


def test_absolute_threshold_violation():
    cur = _report(safety=0.7)
    detector = RegressionDetector()
    violations = detector.check_absolute_thresholds(cur, ThresholdConfig({"safety": {"min": 0.95}}))
    assert any(v.metric_name == "safety" for v in violations)


def test_ci_gate_pass(tmp_path):
    cur_path = tmp_path / "cur.json"
    base_path = tmp_path / "base.json"
    thr_path = tmp_path / "thr.yaml"
    cur_path.write_text(_report().to_json())
    base_path.write_text(_report().to_json())
    thr_path.write_text(yaml.safe_dump({"output_quality": {"min": 0.7, "max_regression_pct": 5}}))
    code = ci_gate(EvaluationReport.load(cur_path), EvaluationReport.load(base_path), thr_path)
    assert code == 0


def test_ci_gate_fails_on_violation(tmp_path):
    cur_path = tmp_path / "cur.json"
    base_path = tmp_path / "base.json"
    thr_path = tmp_path / "thr.yaml"
    cur_path.write_text(_report(safety=0.6).to_json())
    base_path.write_text(_report(safety=1.0).to_json())
    thr_path.write_text(yaml.safe_dump({"safety": {"min": 0.95, "max_regression_pct": 5}}))
    code = ci_gate(EvaluationReport.load(cur_path), EvaluationReport.load(base_path), thr_path)
    assert code == 1
