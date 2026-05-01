"""Regression detection — current report vs baseline report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent_eval.reporters.report import EvaluationReport

Severity = Literal["critical", "high", "medium", "low", "none"]


@dataclass
class Regression:
    metric_name: str
    component: str
    current_value: float
    baseline_value: float
    absolute_change: float
    pct_change: float
    severity: Severity
    affected_examples: list[str] = field(default_factory=list)


@dataclass
class ThresholdConfig:
    """Per-metric regression and absolute thresholds.

    YAML schema example:
      output_quality:
        min: 0.75
        max_regression_pct: 5
      hallucination:
        max: 0.10
        max_regression_pct: 10
    """

    raw: dict[str, Any] = field(default_factory=dict)

    def for_metric(self, metric: str) -> dict[str, Any]:
        return self.raw.get(metric, {})

    def regression_severity(self, metric: str, pct_change: float) -> Severity:
        cfg = self.for_metric(metric)
        # pct_change is signed: positive = improvement; we care about negative drops.
        drop = -pct_change if pct_change < 0 else 0.0
        if drop == 0:
            return "none"
        crit = cfg.get("critical_regression_pct", cfg.get("max_regression_pct", 10) * 2)
        high = cfg.get("max_regression_pct", 10)
        med = max(1, high * 0.5)
        if drop >= crit:
            return "critical"
        if drop >= high:
            return "high"
        if drop >= med:
            return "medium"
        return "low"

    @classmethod
    def from_yaml(cls, path: str) -> ThresholdConfig:
        import yaml

        with open(path) as f:
            return cls(raw=yaml.safe_load(f) or {})


@dataclass
class ThresholdViolation:
    metric_name: str
    component: str
    value: float
    threshold: float
    direction: Literal["below_min", "above_max"]
    severity: Severity = "high"


class RegressionDetector:
    def detect(self, current: EvaluationReport, baseline: EvaluationReport, thresholds: ThresholdConfig) -> list[Regression]:
        out: list[Regression] = []
        cur_dims = current.dimension_scores.model_dump()
        base_dims = baseline.dimension_scores.model_dump()
        for metric, cur_v in cur_dims.items():
            base_v = base_dims.get(metric, cur_v)
            abs_change = cur_v - base_v
            pct = ((cur_v - base_v) / base_v * 100) if base_v else 0.0
            sev = thresholds.regression_severity(metric, pct)
            if sev == "none":
                continue
            out.append(
                Regression(
                    metric_name=metric,
                    component="system",
                    current_value=cur_v,
                    baseline_value=base_v,
                    absolute_change=abs_change,
                    pct_change=pct,
                    severity=sev,
                )
            )
        # Component-level: any component whose score dropped > threshold.
        cur_components = {c.component_name: c for c in current.component_scores}
        for prev_c in baseline.component_scores:
            cur_c = cur_components.get(prev_c.component_name)
            if cur_c is None:
                continue
            abs_change = cur_c.overall_score - prev_c.overall_score
            pct = ((cur_c.overall_score - prev_c.overall_score) / prev_c.overall_score * 100) if prev_c.overall_score else 0.0
            sev = thresholds.regression_severity("component", pct)
            if sev != "none":
                out.append(
                    Regression(
                        metric_name="component_overall",
                        component=prev_c.component_name,
                        current_value=cur_c.overall_score,
                        baseline_value=prev_c.overall_score,
                        absolute_change=abs_change,
                        pct_change=pct,
                        severity=sev,
                    )
                )
        return out

    def check_absolute_thresholds(self, current: EvaluationReport, thresholds: ThresholdConfig) -> list[ThresholdViolation]:
        out: list[ThresholdViolation] = []
        dims = current.dimension_scores.model_dump()
        for metric, cfg in thresholds.raw.items():
            if not isinstance(cfg, dict):
                continue
            value = dims.get(metric)
            if value is None:
                continue
            min_v = cfg.get("min")
            max_v = cfg.get("max")
            if min_v is not None and value < min_v:
                out.append(ThresholdViolation(metric_name=metric, component="system", value=value, threshold=min_v, direction="below_min"))
            if max_v is not None and value > max_v:
                out.append(ThresholdViolation(metric_name=metric, component="system", value=value, threshold=max_v, direction="above_max"))
        return out

    def ci_format(self, regressions: list[Regression], violations: list[ThresholdViolation]) -> dict[str, Any]:
        critical = [r for r in regressions if r.severity == "critical"] + [v for v in violations if v.severity == "critical"]
        exit_code = 1 if (critical or violations) else 0
        lines: list[str] = []
        lines.append(f"## CI Gate: {'❌ FAILED' if exit_code else '✅ PASSED'}")
        if regressions:
            lines.append("\n### Regressions vs baseline")
            lines.append("| Metric | Component | Baseline | Current | Δ% | Severity |")
            lines.append("|---|---|---|---|---|---|")
            for r in regressions:
                lines.append(f"| {r.metric_name} | {r.component} | {r.baseline_value:.3f} | {r.current_value:.3f} | {r.pct_change:+.1f}% | {r.severity} |")
        if violations:
            lines.append("\n### Absolute threshold violations")
            for v in violations:
                lines.append(f"- **{v.metric_name}** on {v.component}: {v.value:.3f} {v.direction} {v.threshold}")
        if not regressions and not violations:
            lines.append("\nNo regressions detected.")
        return {"exit_code": exit_code, "summary": "\n".join(lines), "regressions": [r.__dict__ for r in regressions], "violations": [v.__dict__ for v in violations]}
