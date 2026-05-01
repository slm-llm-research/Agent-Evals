"""Judge calibration framework + drift monitor.

Spec section 4.4. Loads pre-labeled calibration sets shipped with the package,
runs each judge against them, computes Pearson r / Cohen's kappa / MAE, and
flags judges whose correlation drops below 0.7. Also provides:
  - position-bias mitigation for pairwise judges (run A>B + B>A, average)
  - JudgeMonitor: rolling drift tracking with std-shift alerts
  - HTML calibration report
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_eval.judges.base_judge import BaseJudge, CalibrationReport, _pearson


_DIMENSIONS = ("answer_quality", "tool_selection", "hallucination", "intent_resolution", "safety")


class CalibrationExample(BaseModel):
    id: str
    human_score: float
    notes: str = ""
    # Free-form fields per dimension — we pass these through as kwargs to the judge.
    payload: dict[str, Any] = {}


@dataclass
class CalibrationSet:
    dimension: str
    description: str
    examples: list[CalibrationExample]

    @classmethod
    def load(cls, dimension: str) -> CalibrationSet:
        if dimension not in _DIMENSIONS:
            raise ValueError(f"Unknown calibration dimension '{dimension}'. Valid: {_DIMENSIONS}")
        text = resources.files("agent_eval.data.calibration_sets").joinpath(f"{dimension}.json").read_text()
        data = json.loads(text)
        out = []
        for raw in data["examples"]:
            payload = {k: v for k, v in raw.items() if k not in ("id", "human_score", "notes")}
            out.append(CalibrationExample(
                id=raw["id"],
                human_score=float(raw["human_score"]),
                notes=raw.get("notes", ""),
                payload=payload,
            ))
        return cls(dimension=dimension, description=data.get("description", ""), examples=out)


@dataclass
class CalibrationRunReport:
    """Per-judge calibration result."""

    dimension: str
    n: int
    pearson_r: float
    mae: float
    cohen_kappa: float | None
    is_reliable: bool  # True iff pearson_r >= 0.7
    judge_scores: list[float] = field(default_factory=list)
    human_scores: list[float] = field(default_factory=list)
    per_example_disagreement: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n": self.n,
            "pearson_r": round(self.pearson_r, 3),
            "mae": round(self.mae, 3),
            "cohen_kappa": round(self.cohen_kappa, 3) if self.cohen_kappa is not None else None,
            "is_reliable": self.is_reliable,
            "judge_scores": self.judge_scores,
            "human_scores": self.human_scores,
            "per_example_disagreement": self.per_example_disagreement,
            "notes": self.notes,
        }


@dataclass
class CalibrationSuiteReport:
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    judges: dict[str, CalibrationRunReport] = field(default_factory=dict)
    overall_reliable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_reliable": self.overall_reliable,
            "judges": {k: v.to_dict() for k, v in self.judges.items()},
        }


# ---------------------------------------------------------------------------- runner


async def calibrate_judge(judge: BaseJudge, dimension: str | None = None) -> CalibrationRunReport:
    """Run a single judge against its calibration set."""
    dim = dimension or getattr(judge, "name", None)
    cs = CalibrationSet.load(dim)
    judge_scores: list[float] = []
    human_scores: list[float] = []
    disagreements: list[dict[str, Any]] = []

    for ex in cs.examples:
        try:
            result = await judge.judge(**ex.payload)
            score = float(result.score)
        except Exception as e:
            score = 0.5
            disagreements.append({"id": ex.id, "error": str(e)})
            continue
        judge_scores.append(score)
        human_scores.append(ex.human_score)
        if abs(score - ex.human_score) >= 0.3:
            disagreements.append({
                "id": ex.id,
                "judge": round(score, 3),
                "human": ex.human_score,
                "delta": round(score - ex.human_score, 3),
                "notes": ex.notes,
            })

    if not judge_scores:
        return CalibrationRunReport(dimension=dim, n=0, pearson_r=0.0, mae=0.0,
                                     cohen_kappa=None, is_reliable=False, notes="no successful runs")

    r = _pearson(judge_scores, human_scores)
    mae = sum(abs(a - b) for a, b in zip(judge_scores, human_scores)) / len(judge_scores)
    kappa = _cohen_kappa(judge_scores, human_scores)
    notes = "" if r >= 0.7 else f"Pearson r={r:.2f} < 0.7 — judge may be unreliable"
    return CalibrationRunReport(
        dimension=dim,
        n=len(judge_scores),
        pearson_r=r,
        mae=mae,
        cohen_kappa=kappa,
        is_reliable=r >= 0.7,
        judge_scores=judge_scores,
        human_scores=human_scores,
        per_example_disagreement=disagreements,
        notes=notes,
    )


async def calibrate_all_judges(judges: dict[str, BaseJudge] | None = None) -> CalibrationSuiteReport:
    """Run calibration across all standard judges. If `judges` is None, instantiates the defaults."""
    if judges is None:
        from agent_eval.judges.rubric_judge import (
            AnswerQualityJudge,
            HallucinationJudge,
            IntentResolutionJudge,
            SafetyJudge,
            ToolSelectionJudge,
        )

        judges = {
            "answer_quality": AnswerQualityJudge(),
            "tool_selection": ToolSelectionJudge(),
            "hallucination": HallucinationJudge(),
            "intent_resolution": IntentResolutionJudge(),
            "safety": SafetyJudge(),
        }

    suite = CalibrationSuiteReport()
    for dim, judge in judges.items():
        report = await calibrate_judge(judge, dim)
        suite.judges[dim] = report
        if not report.is_reliable:
            suite.overall_reliable = False
    return suite


# ---------------------------------------------------------------------------- position bias


async def pairwise_with_position_swap(
    judge: BaseJudge,
    *,
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    """Mitigate position bias for A-vs-B pairwise judges by running both orderings."""
    score_ab = (await judge.judge(prompt=prompt_template.format(item_a=item_a, item_b=item_b))).score
    score_ba = (await judge.judge(prompt=prompt_template.format(item_a=item_b, item_b=item_a))).score
    return {
        "score_a_first": score_ab,
        "score_b_first": 1.0 - score_ba,  # invert because B is now in A's slot
        "averaged": (score_ab + (1.0 - score_ba)) / 2.0,
        "agreement": abs(score_ab - (1.0 - score_ba)) < 0.15,
    }


# ---------------------------------------------------------------------------- monitor


@dataclass
class _DriftPoint:
    timestamp: datetime
    score: float


class JudgeMonitor:
    """Tracks per-judge score distributions over a rolling window. Alerts on >0.15 std shift."""

    def __init__(self, window_days: int = 7, alert_std_shift: float = 0.15):
        self.window = timedelta(days=window_days)
        self.alert_std_shift = alert_std_shift
        self._history: dict[str, list[_DriftPoint]] = {}

    def observe(self, dimension: str, score: float, ts: datetime | None = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        self._history.setdefault(dimension, []).append(_DriftPoint(ts, score))
        # Trim to window.
        cutoff = ts - self.window
        self._history[dimension] = [p for p in self._history[dimension] if p.timestamp >= cutoff]

    def detect_drift(self) -> list[dict[str, Any]]:
        out = []
        for dim, points in self._history.items():
            if len(points) < 10:
                continue
            mid = len(points) // 2
            old = [p.score for p in points[:mid]]
            new = [p.score for p in points[mid:]]
            old_std = statistics.pstdev(old) if old else 0.0
            new_std = statistics.pstdev(new) if new else 0.0
            shift = abs(new_std - old_std)
            if shift >= self.alert_std_shift:
                out.append({
                    "dimension": dim,
                    "old_std": round(old_std, 3),
                    "new_std": round(new_std, 3),
                    "std_shift": round(shift, 3),
                    "n_old": len(old),
                    "n_new": len(new),
                })
        return out

    def snapshot(self) -> dict[str, dict[str, float]]:
        out = {}
        for dim, points in self._history.items():
            scores = [p.score for p in points]
            if not scores:
                continue
            out[dim] = {
                "n": len(scores),
                "mean": round(statistics.mean(scores), 3),
                "std": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
            }
        return out


# ---------------------------------------------------------------------------- HTML report


def render_calibration_html(suite: CalibrationSuiteReport, output_path: str | Path) -> Path:
    """Render a single-file calibration report."""
    rows = []
    for dim, r in suite.judges.items():
        color = "#2e7d32" if r.is_reliable else "#c62828"
        rows.append(f"""
        <tr>
          <td><strong>{dim}</strong></td>
          <td>{r.n}</td>
          <td style="color: {color}; font-weight: 600;">{r.pearson_r:.3f}</td>
          <td>{r.mae:.3f}</td>
          <td>{('%.3f' % r.cohen_kappa) if r.cohen_kappa is not None else '—'}</td>
          <td>{'✅' if r.is_reliable else '❌'}</td>
          <td>{r.notes}</td>
        </tr>""")
    disagreement_rows = []
    for dim, r in suite.judges.items():
        for d in r.per_example_disagreement[:5]:
            disagreement_rows.append(f"""
            <tr>
              <td>{dim}</td>
              <td>{d.get('id', '?')}</td>
              <td>{d.get('judge', '—')}</td>
              <td>{d.get('human', '—')}</td>
              <td>{d.get('delta', '—')}</td>
              <td>{d.get('notes', '')}</td>
            </tr>""")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>agent-eval — judge calibration</title>
<style>
body {{ font: 14px -apple-system, sans-serif; max-width: 1000px; margin: 32px auto; color: #222; }}
h1 {{ margin-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 13px; }}
th {{ background: #fafafa; font-weight: 600; }}
.bad {{ background: #ffebee; }}
.muted {{ color: #777; font-size: 12px; }}
</style></head>
<body>
  <h1>Judge Calibration Report</h1>
  <div class="muted">Generated {suite.generated_at.strftime('%Y-%m-%d %H:%M UTC')} · Overall: {'✅ reliable' if suite.overall_reliable else '❌ unreliable judges detected'}</div>

  <h2>Per-judge calibration</h2>
  <table>
    <thead><tr><th>Dimension</th><th>n</th><th>Pearson r</th><th>MAE</th><th>Cohen κ</th><th>Reliable</th><th>Notes</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>Top disagreements per judge</h2>
  <p class="muted">Examples where judge and human scores differed by ≥0.3 — useful for prompt tuning.</p>
  <table>
    <thead><tr><th>Dim</th><th>Example</th><th>Judge</th><th>Human</th><th>Δ</th><th>Notes</th></tr></thead>
    <tbody>{''.join(disagreement_rows) or '<tr><td colspan="6" class="muted">No major disagreements.</td></tr>'}</tbody>
  </table>
</body></html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return Path(output_path)


# ---------------------------------------------------------------------------- helpers


def _cohen_kappa(judge: list[float], human: list[float], bins: int = 4) -> float | None:
    """Cohen's kappa on quantized scores (binning into `bins` buckets)."""
    if len(judge) != len(human) or len(judge) < 2:
        return None
    j_bins = [_quantize(s, bins) for s in judge]
    h_bins = [_quantize(s, bins) for s in human]
    n = len(j_bins)
    po = sum(1 for a, b in zip(j_bins, h_bins) if a == b) / n
    j_dist = {b: j_bins.count(b) / n for b in set(j_bins) | set(h_bins)}
    h_dist = {b: h_bins.count(b) / n for b in set(j_bins) | set(h_bins)}
    pe = sum(j_dist.get(b, 0) * h_dist.get(b, 0) for b in set(j_bins) | set(h_bins))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _quantize(score: float, bins: int) -> int:
    return min(bins - 1, int(score * bins))
