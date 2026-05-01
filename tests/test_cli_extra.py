"""Extra CLI smoke tests for new commands (alerts / wizard / cost-estimate)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agent_eval.cli.main import cli


def test_alerts_help():
    r = CliRunner().invoke(cli, ["alerts", "--help"])
    assert r.exit_code == 0
    assert "ack" in r.output


def test_alerts_list_empty(tmp_path: Path):
    db = tmp_path / "empty.db"
    r = CliRunner().invoke(cli, ["alerts", "list", "--db", str(db)])
    assert r.exit_code == 0


def test_calibrate_help():
    r = CliRunner().invoke(cli, ["calibrate", "--help"])
    assert r.exit_code == 0
    assert "dimension" in r.output.lower()


def test_wizard_help():
    r = CliRunner().invoke(cli, ["wizard", "--help"])
    assert r.exit_code == 0


def test_cost_estimate_runs(tmp_path: Path):
    from agent_eval.discovery.registry import ComponentRegistry, ToolInfo

    reg = ComponentRegistry(tools=[ToolInfo(name="x")], discovery_method="manual")
    reg_path = tmp_path / "reg.json"
    reg_path.write_text(reg.to_json(indent=2))

    from agent_eval.dataset.schema import EvalDataset, EvalExample

    ds = EvalDataset(name="t", examples=[EvalExample(input={"query": "?"})])
    ds_path = tmp_path / "ds.json"
    ds_path.write_text(ds.model_dump_json(indent=2))

    r = CliRunner().invoke(cli, ["cost-estimate", "--registry", str(reg_path), "--dataset", str(ds_path)])
    assert r.exit_code == 0
    assert "estimated_usd" in r.output


def test_compare_command(tmp_path: Path):
    from agent_eval.reporters.report import (
        DimensionScores,
        EvaluationReport,
        SystemOverview,
        status_for_score,
    )

    def _build(score):
        return EvaluationReport(
            system_name="t", dataset_name="d", dataset_size=1,
            system_overview=SystemOverview(overall_score=score, health_status=status_for_score(score),
                                             pass_rate=score, flag_count=0, critical_flag_count=0),
            dimension_scores=DimensionScores(output_quality=score, trajectory_quality=score),
        )

    r1 = tmp_path / "a.json"; r2 = tmp_path / "b.json"
    r1.write_text(_build(0.85).to_json())
    r2.write_text(_build(0.65).to_json())

    res = CliRunner().invoke(cli, ["compare", "--report-1", str(r1), "--report-2", str(r2)])
    assert res.exit_code == 0
    assert "delta" in res.output.lower()
