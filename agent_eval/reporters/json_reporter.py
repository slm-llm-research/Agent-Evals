"""JSON reporter — canonical machine-readable output."""

from __future__ import annotations

from pathlib import Path

from agent_eval.reporters.report import EvaluationReport


def render_json_report(report: EvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json())
    return path
