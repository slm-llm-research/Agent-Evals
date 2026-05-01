"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agent_eval.cli.main import cli


def test_cli_help():
    res = CliRunner().invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "agent-eval" in res.output


def test_dataset_from_template_command(tmp_path: Path):
    output = tmp_path / "ds.json"
    res = CliRunner().invoke(cli, ["dataset", "from-template", "--template", "general_agent", "--output", str(output)])
    assert res.exit_code == 0, res.output
    assert output.exists()


def test_dataset_command_help():
    res = CliRunner().invoke(cli, ["dataset", "--help"])
    assert res.exit_code == 0


def test_evaluate_command_help():
    res = CliRunner().invoke(cli, ["evaluate", "--help"])
    assert res.exit_code == 0


def test_ci_gate_help():
    res = CliRunner().invoke(cli, ["ci-gate", "--help"])
    assert res.exit_code == 0
