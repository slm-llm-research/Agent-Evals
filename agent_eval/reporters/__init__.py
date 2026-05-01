"""Report generation, regression detection, tuning advisor."""

from agent_eval.reporters.html_reporter import render_html_report
from agent_eval.reporters.json_reporter import render_json_report
from agent_eval.reporters.regression_detector import Regression, RegressionDetector
from agent_eval.reporters.report import (
    ComponentScore,
    DimensionScores,
    EvaluationReport,
    Issue,
    PerExampleResult,
    ReportDiff,
    SystemOverview,
    ToolCall,
)
from agent_eval.reporters.tuning_advisor import TuningAdvisor, TuningSignal

__all__ = [
    "EvaluationReport",
    "SystemOverview",
    "ComponentScore",
    "DimensionScores",
    "Issue",
    "PerExampleResult",
    "ToolCall",
    "ReportDiff",
    "RegressionDetector",
    "Regression",
    "TuningAdvisor",
    "TuningSignal",
    "render_html_report",
    "render_json_report",
]
