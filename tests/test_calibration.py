"""Tests for judge calibration framework."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_eval.judges.base_judge import JudgeResult
from agent_eval.judges.calibration import (
    CalibrationSet,
    JudgeMonitor,
    _cohen_kappa,
    calibrate_judge,
    render_calibration_html,
)


def test_calibration_set_loads_all_dimensions():
    for dim in ("answer_quality", "tool_selection", "hallucination", "intent_resolution", "safety"):
        cs = CalibrationSet.load(dim)
        assert cs.dimension == dim
        assert len(cs.examples) >= 15  # Spec requires 20; we ship 20 each.
        for ex in cs.examples:
            assert 0.0 <= ex.human_score <= 1.0


def test_calibration_set_invalid_dim_raises():
    with pytest.raises(ValueError):
        CalibrationSet.load("nonexistent")


class _ScriptedJudge:
    """Returns scores from a fixed list, ignoring inputs."""

    name = "scripted"

    def __init__(self, scores):
        self.scores = list(scores)
        self.idx = 0

    async def judge(self, **_):
        s = self.scores[self.idx % len(self.scores)]
        self.idx += 1
        return JudgeResult(score=s)


@pytest.mark.asyncio
async def test_calibration_perfect_judge_is_reliable():
    cs = CalibrationSet.load("answer_quality")
    expected = [ex.human_score for ex in cs.examples]
    judge = _ScriptedJudge(expected)
    report = await calibrate_judge(judge, "answer_quality")
    assert report.is_reliable
    assert report.pearson_r > 0.95
    assert report.mae < 0.01


@pytest.mark.asyncio
async def test_calibration_inverted_judge_unreliable():
    cs = CalibrationSet.load("safety")
    inverted = [1.0 - ex.human_score for ex in cs.examples]
    judge = _ScriptedJudge(inverted)
    report = await calibrate_judge(judge, "safety")
    assert not report.is_reliable
    assert report.pearson_r < 0


def test_cohen_kappa_perfect_agreement():
    k = _cohen_kappa([0.0, 0.5, 1.0, 0.0, 0.5], [0.0, 0.5, 1.0, 0.0, 0.5])
    assert k == 1.0


def test_cohen_kappa_random():
    k = _cohen_kappa([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert k <= 0.0


def test_judge_monitor_no_drift():
    m = JudgeMonitor(window_days=7, alert_std_shift=0.15)
    base = datetime.now(timezone.utc) - timedelta(days=3)
    for i in range(20):
        m.observe("dim1", 0.8, ts=base + timedelta(hours=i))
    assert m.detect_drift() == []


def test_judge_monitor_detects_drift():
    m = JudgeMonitor(window_days=7, alert_std_shift=0.10)
    base = datetime.now(timezone.utc) - timedelta(days=3)
    # Stable then very noisy.
    for i in range(15):
        m.observe("dim1", 0.8, ts=base + timedelta(hours=i))
    for i in range(15):
        m.observe("dim1", 0.5 if i % 2 == 0 else 1.0, ts=base + timedelta(hours=15 + i))
    drift = m.detect_drift()
    assert any(d["dimension"] == "dim1" for d in drift)


def test_calibration_html_render(tmp_path: Path):
    from agent_eval.judges.calibration import CalibrationRunReport, CalibrationSuiteReport

    suite = CalibrationSuiteReport(
        judges={
            "answer_quality": CalibrationRunReport(
                dimension="answer_quality", n=20, pearson_r=0.85, mae=0.10,
                cohen_kappa=0.78, is_reliable=True,
            ),
            "safety": CalibrationRunReport(
                dimension="safety", n=20, pearson_r=0.45, mae=0.30,
                cohen_kappa=0.22, is_reliable=False, notes="below threshold",
            ),
        },
        overall_reliable=False,
    )
    p = render_calibration_html(suite, tmp_path / "cal.html")
    assert p.exists()
    text = p.read_text()
    assert "answer_quality" in text
    assert "0.450" in text  # safety pearson
