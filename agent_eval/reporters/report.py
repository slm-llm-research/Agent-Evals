"""EvaluationReport schema."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_eval.evaluators.base import EvaluatorResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


HealthStatus = Literal["excellent", "good", "needs_improvement", "critical"]
Severity = Literal["critical", "high", "medium", "low"]


def status_for_score(score: float) -> HealthStatus:
    if score >= 0.85:
        return "excellent"
    if score >= 0.70:
        return "good"
    if score >= 0.50:
        return "needs_improvement"
    return "critical"


class SystemOverview(BaseModel):
    overall_score: float
    health_status: HealthStatus
    pass_rate: float
    flag_count: int
    critical_flag_count: int


class ComponentScore(BaseModel):
    component_name: str
    component_type: str = "agent"  # agent / tool / mcp_server / memory / system
    overall_score: float
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
    rank: int = 0
    trend: float | None = None  # delta vs prior report (positive = improvement)


class DimensionScores(BaseModel):
    output_quality: float = 0.0
    trajectory_quality: float = 0.0
    hallucination_risk: float = 0.0  # higher = MORE risk (1 - faithfulness)
    tool_performance: float = 0.0
    system_performance: float = 0.0
    safety: float = 0.0
    memory_quality: float = 0.0


class Issue(BaseModel):
    severity: Severity
    component: str
    metric: str
    score: float
    description: str
    example_trace_id: str | None = None
    example_trace_url: str | None = None


class DatasetStats(BaseModel):
    n_examples: int = 0
    query_type_distribution: dict[str, int] = Field(default_factory=dict)
    complexity_distribution: dict[str, int] = Field(default_factory=dict)
    avg_complexity: str | None = None
    coverage: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """One tool invocation observed in the trace."""

    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs_preview: str = ""  # truncated string preview of outputs
    success: bool = True
    error: str | None = None
    latency_ms: float | None = None


class PerExampleResult(BaseModel):
    """Per-dataset-example record — the data drives the HTML drill-down view."""

    example_id: str
    query: str
    actual_output: str = ""
    expected_output: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    query_type: str = "general"
    complexity: str = "medium"
    score: float = 0.0
    pass_rate: float = 0.0
    flagged_count: int = 0
    critical_count: int = 0
    is_overall_pass: bool = False
    langsmith_run_id: str | None = None
    langsmith_run_url: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
    runner_latency_ms: float | None = None
    runner_error: str | None = None
    trace_was_synthetic: bool = False


class ReportDiff(BaseModel):
    metrics_changed: dict[str, dict[str, float]] = Field(default_factory=dict)  # metric -> {current, baseline, pct}
    overall_delta: float = 0.0
    new_critical_issues: list[Issue] = Field(default_factory=list)
    resolved_critical_issues: list[Issue] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=_utcnow)
    system_name: str = "unknown"
    dataset_name: str = ""
    dataset_size: int = 0
    evaluation_duration_seconds: float = 0.0
    package_version: str = "1.0.0"
    langsmith_project: str | None = None

    system_overview: SystemOverview
    component_scores: list[ComponentScore] = Field(default_factory=list)
    dimension_scores: DimensionScores = Field(default_factory=DimensionScores)
    flagged_issues: list[Issue] = Field(default_factory=list)
    tuning_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    dataset_stats: DatasetStats = Field(default_factory=DatasetStats)
    backends_used: dict[str, str] = Field(default_factory=dict)  # evaluator -> backend
    per_example_results: list[PerExampleResult] = Field(default_factory=list)

    def compare(self, previous: EvaluationReport) -> ReportDiff:
        diff = ReportDiff()
        cur = self.dimension_scores.model_dump()
        prv = previous.dimension_scores.model_dump()
        for metric in cur:
            c, p = float(cur[metric]), float(prv[metric])
            if abs(c - p) >= 1e-6:
                pct = ((c - p) / p * 100) if p else 0.0
                diff.metrics_changed[metric] = {"current": c, "baseline": p, "pct_change": round(pct, 2)}
        diff.overall_delta = self.system_overview.overall_score - previous.system_overview.overall_score
        prev_critical = {(i.metric, i.component) for i in previous.flagged_issues if i.severity == "critical"}
        cur_critical = {(i.metric, i.component) for i in self.flagged_issues if i.severity == "critical"}
        diff.new_critical_issues = [i for i in self.flagged_issues if i.severity == "critical" and (i.metric, i.component) not in prev_critical]
        diff.resolved_critical_issues = [i for i in previous.flagged_issues if i.severity == "critical" and (i.metric, i.component) not in cur_critical]
        return diff

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(indent=2, **kwargs)

    @classmethod
    def load(cls, path: str) -> EvaluationReport:
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text())
