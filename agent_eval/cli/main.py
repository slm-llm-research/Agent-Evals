"""agent-eval CLI."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agent_eval import AgentEval
from agent_eval.dataset.schema import EvalDataset
from agent_eval.discovery.mcp_inspector import MCPInspector
from agent_eval.discovery.registry import ComponentRegistry
from agent_eval.reporters.report import EvaluationReport

console = Console()


@click.group()
@click.version_option()
def cli():
    """agent-eval — generic agent system evaluation."""


# --------------------------------------------------------------------- discover


@cli.command()
@click.option("--mcp-url", help="MCP endpoint URL (primary discovery method).")
@click.option("--graph-module", help="Python import path 'package.module:graph_var' for LangGraph fallback.")
@click.option("--langsmith-project", help="LangSmith project for trace-mining fallback.")
@click.option("--output", default="registry.json", show_default=True, help="Where to write the registry JSON.")
def discover(mcp_url, graph_module, langsmith_project, output):
    """Discover the components of an agent system."""
    if mcp_url:
        registry = asyncio.run(MCPInspector(mcp_url).inspect())
    elif graph_module:
        mod_path, var = graph_module.split(":")
        mod = importlib.import_module(mod_path)
        graph = getattr(mod, var)
        from agent_eval.discovery.graph_inspector import GraphInspector

        registry = GraphInspector(graph).inspect()
    elif langsmith_project:
        from agent_eval.config import get_config
        from agent_eval.discovery.langsmith_inspector import LangSmithInspector

        registry = LangSmithInspector(get_config().get_langsmith_client(), langsmith_project).mine_components(n_traces=100)
    else:
        raise click.UsageError("Provide one of --mcp-url, --graph-module, or --langsmith-project.")
    Path(output).write_text(registry.to_json(indent=2))
    console.print(registry.summary())
    console.print(f"[green]Wrote {output}[/green]")


# ---------------------------------------------------------------------- dataset


@cli.group()
def dataset():
    """Dataset management commands."""


@dataset.command("generate")
@click.option("--registry", "registry_path", required=True, help="Path to registry JSON.")
@click.option("--mode", type=click.Choice(["auto", "semi", "manual", "template"]), default="auto", show_default=True)
@click.option("--template", help="Template name when --mode=template.")
@click.option("--n-examples", default=50, show_default=True, type=int)
@click.option("--langsmith-project", help="Project name (used in auto mode for trace harvesting).")
@click.option("--output", default="dataset.json", show_default=True)
def dataset_generate(registry_path, mode, template, n_examples, langsmith_project, output):
    registry = ComponentRegistry.model_validate_json(Path(registry_path).read_text())
    ev = AgentEval(registry, langsmith_project=langsmith_project)
    if mode == "semi":
        click.echo("[note] PATH B (semi-automatic) is deferred — falling back to manual wizard. See NEXT_STEPS.md.", err=True)
        ds = ev.generate_dataset(mode="manual")
    else:
        ds = ev.generate_dataset(mode=mode, n_examples=n_examples, template=template)
    Path(output).write_text(ds.model_dump_json(indent=2))
    console.print(f"[green]Wrote dataset with {len(ds.examples)} examples to {output}[/green]")


@dataset.command("from-template")
@click.option("--template", type=click.Choice(["search_agent", "research_agent", "voice_agent", "general_agent"]), required=True)
@click.option("--output", default="dataset.json", show_default=True)
@click.option("--customize", is_flag=True, help="Prompt to add additional examples to the template.")
def dataset_from_template(template, output, customize):
    ds = EvalDataset.from_template(template)
    if customize:
        from agent_eval.dataset.schema import EvalExample

        click.echo(f"\nLoaded {len(ds.examples)} starter examples from '{template}' template.")
        click.echo("Add custom examples (blank input to stop):")
        while True:
            q = click.prompt("Query", default="", show_default=False)
            if not q:
                break
            ds.examples.append(EvalExample(input={"query": q}, created_by="manual", tags=["custom"]))
    Path(output).write_text(ds.model_dump_json(indent=2))
    console.print(f"[green]Wrote {output} ({len(ds.examples)} examples)[/green]")


# --------------------------------------------------------------------- evaluate


@cli.command()
@click.option("--registry", "registry_path", help="Path to registry JSON. If absent, will auto-discover via MCP.")
@click.option("--mcp-url", help="If --registry is missing, discover via MCP.")
@click.option("--dataset", "dataset_path", required=True)
@click.option("--output-dir", default="./eval_results/", show_default=True)
@click.option("--dimensions", default="all", show_default=True, help="Comma-separated dimension names or 'all'.")
@click.option("--backend", type=click.Choice(["native", "deepeval", "ragas"]), default="native", show_default=True)
@click.option("--baseline", "baseline_path", help="Optional baseline report JSON to compute trends.")
@click.option("--langsmith-project", help="LangSmith project name.")
@click.option("--n-examples", type=int, help="Cap dataset to first N examples (smoke test).")
@click.option("--runner-url", help="HTTP runner: POST each example to this URL to get the agent's response/trace.")
@click.option("--runner-method", default="POST", show_default=True, help="HTTP method when --runner-url is set.")
@click.option("--runner-header", multiple=True, help="Extra header(s) for HTTP runner, e.g. 'Authorization: Bearer XYZ'.")
@click.option("--graph-module", help="LangGraph runner: 'pkg.module:graph_var' to invoke in-process.")
@click.option("--replay", is_flag=True, help="Replay runner: match dataset examples to existing LangSmith traces in --langsmith-project.")
@click.option("--max-concurrency", default=5, show_default=True, type=int)
def evaluate(
    registry_path, mcp_url, dataset_path, output_dir, dimensions, backend, baseline_path, langsmith_project, n_examples,
    runner_url, runner_method, runner_header, graph_module, replay, max_concurrency,
):
    """Run the full evaluation suite.

    Provide a runner to actually exercise the agent:
      --runner-url URL         HTTP runner (most common)
      --graph-module pkg:var   In-process LangGraph runner
      --replay                 Match against existing LangSmith traces

    Without any runner the report measures dataset testability only (a warning prints).
    """
    if registry_path:
        registry = ComponentRegistry.model_validate_json(Path(registry_path).read_text())
        ev = AgentEval(registry, langsmith_project=langsmith_project, mcp_url=mcp_url)
    elif mcp_url:
        ev = AgentEval.from_mcp(mcp_url, langsmith_project=langsmith_project or "")
        ev.discover()
    else:
        raise click.UsageError("Provide --registry or --mcp-url.")

    ds = EvalDataset.model_validate_json(Path(dataset_path).read_text())
    baseline = EvaluationReport.load(baseline_path) if baseline_path else None
    dims = [d.strip() for d in dimensions.split(",")] if dimensions != "all" else ["all"]

    # Build runner from flags.
    runner = None
    runner_flags = sum(1 for x in (runner_url, graph_module, replay) if x)
    if runner_flags > 1:
        raise click.UsageError("Provide at most one of --runner-url, --graph-module, --replay.")
    if runner_url:
        from agent_eval.runners.http_runner import HttpAgentRunner

        headers = {}
        for h in runner_header or []:
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip()] = v.strip()
        runner = HttpAgentRunner(
            endpoint_url=runner_url,
            method=runner_method,
            headers=headers,
            langsmith_project=langsmith_project,
            max_concurrency=max_concurrency,
        )
    elif graph_module:
        from agent_eval.runners.langgraph_runner import LangGraphRunner

        mod_path, var = graph_module.split(":")
        mod = importlib.import_module(mod_path)
        graph = getattr(mod, var)
        runner = LangGraphRunner(graph=graph, max_concurrency=max_concurrency)
    elif replay:
        if not langsmith_project:
            raise click.UsageError("--replay requires --langsmith-project.")
        from agent_eval.runners.langsmith_replay import LangSmithReplayRunner

        runner = LangSmithReplayRunner(project_name=langsmith_project, max_concurrency=max_concurrency)

    report = ev.evaluate(dataset=ds, dimensions=dims, backend=backend, baseline_report=baseline,
                          n_examples=n_examples, runner=runner)
    paths = ev.save_report(report, output_dir=output_dir)
    _print_overview(report)
    console.print(f"\n[green]Reports written:[/green]")
    for k, v in paths.items():
        console.print(f"  {k}: {v}")


def _print_overview(report: EvaluationReport) -> None:
    t = Table(title=f"Evaluation Overview — {report.system_name}")
    t.add_column("Metric"); t.add_column("Score", justify="right")
    t.add_row("Overall", f"{report.system_overview.overall_score:.3f}")
    t.add_row("Status", report.system_overview.health_status)
    t.add_row("Pass rate", f"{report.system_overview.pass_rate*100:.0f}%")
    t.add_row("Flags", str(report.system_overview.flag_count))
    t.add_row("Critical flags", str(report.system_overview.critical_flag_count))
    console.print(t)
    d = report.dimension_scores
    dt = Table(title="Dimensions")
    dt.add_column("Dimension"); dt.add_column("Score", justify="right")
    for k, v in d.model_dump().items():
        dt.add_row(k, f"{v:.3f}")
    console.print(dt)


# ---------------------------------------------------------------------- ci-gate


@cli.command("ci-gate")
@click.option("--report", "report_path", required=True)
@click.option("--baseline-report", "baseline_path", help="Baseline report (optional). If absent, only absolute thresholds are checked.")
@click.option("--thresholds", "thresholds_path", required=True)
@click.option("--output-summary", help="Write the markdown summary to this path (for posting to PRs).")
def ci_gate(report_path, baseline_path, thresholds_path, output_summary):
    """Compare current report to baseline; non-zero exit on regression."""
    from agent_eval.ci.gate import ci_gate as _gate

    cur = EvaluationReport.load(report_path)
    base = EvaluationReport.load(baseline_path) if baseline_path else None
    code = _gate(cur, base, thresholds_path, output_summary_path=output_summary)
    sys.exit(code)


# ----------------------------------------------------------------------- monitor


@cli.command()
@click.option("--registry", "registry_path", required=True)
@click.option("--project", required=True)
@click.option("--sampling-rate", default=0.1, show_default=True, type=float)
@click.option("--poll-interval", default=60, show_default=True, type=int)
def monitor(registry_path, project, sampling_rate, poll_interval):
    """Attach to a running LangSmith project and continuously evaluate traces."""
    registry = ComponentRegistry.model_validate_json(Path(registry_path).read_text())
    ev = AgentEval(registry, langsmith_project=project)
    online = ev.monitor(sampling_rate=sampling_rate)
    console.print(f"[cyan]Monitoring '{project}' at {sampling_rate*100:.0f}% sampling. Ctrl-C to stop.[/cyan]")
    try:
        asyncio.run(online.start_monitoring(project, poll_interval_seconds=poll_interval))
    except KeyboardInterrupt:
        online.stop()


# ------------------------------------------------------------------------ compare


@cli.command()
@click.option("--report-1", "r1_path", required=True)
@click.option("--report-2", "r2_path", required=True)
def compare(r1_path, r2_path):
    """Side-by-side compare two reports."""
    r1 = EvaluationReport.load(r1_path)
    r2 = EvaluationReport.load(r2_path)
    diff = r2.compare(r1)
    console.print(f"Overall delta: [bold]{diff.overall_delta:+.3f}[/bold]")
    t = Table(title="Dimension changes")
    t.add_column("Metric"); t.add_column("Baseline", justify="right"); t.add_column("Current", justify="right"); t.add_column("Δ%", justify="right")
    for m, vals in diff.metrics_changed.items():
        t.add_row(m, f"{vals['baseline']:.3f}", f"{vals['current']:.3f}", f"{vals['pct_change']:+.1f}")
    console.print(t)


# ------------------------------------------------------------------- cost-estimate


@cli.command("cost-estimate")
@click.option("--registry", "registry_path", required=True)
@click.option("--dataset", "dataset_path", required=True)
def cost_estimate(registry_path, dataset_path):
    """Estimate $ cost of running the eval suite."""
    from agent_eval.dataset.auto_generator import AutoDatasetGenerator

    registry = ComponentRegistry.model_validate_json(Path(registry_path).read_text())
    ds = EvalDataset.model_validate_json(Path(dataset_path).read_text())
    est = AutoDatasetGenerator(registry).estimate_evaluation_cost(ds)
    console.print(json.dumps(est.__dict__, indent=2))


# ----------------------------------------------------------------------- alerts


@cli.group()
def alerts():
    """Inspect / acknowledge alerts from the SQLite store."""


@alerts.command("list")
@click.option("--db", "db_path", default="agent_eval_metrics.db", show_default=True)
@click.option("--unacked", is_flag=True, help="Only show un-acknowledged alerts.")
@click.option("--limit", default=50, show_default=True, type=int)
def alerts_list(db_path, unacked, limit):
    from agent_eval.monitor.persistence import Store
    from rich.table import Table

    rows = Store(db_path).list_alerts(only_unacked=unacked, limit=limit)
    if not rows:
        console.print("[green]No alerts.[/green]")
        return
    t = Table(title=f"Alerts ({'unacked' if unacked else 'all'})")
    for col in ("id", "ts", "severity", "metric", "component", "title", "value", "ack"):
        t.add_column(col)
    for r in rows:
        sev_color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "white"}.get(r["severity"], "white")
        t.add_row(
            str(r["id"]), r["ts"][:19], f"[{sev_color}]{r['severity']}[/{sev_color}]",
            r["metric"] or "", r["component"] or "", r["title"][:60],
            f"{r['value']}" if r["value"] is not None else "",
            "✓" if r["acked"] else "—",
        )
    console.print(t)


@alerts.command("ack")
@click.argument("alert_id", type=int)
@click.option("--db", "db_path", default="agent_eval_metrics.db", show_default=True)
def alerts_ack(alert_id, db_path):
    from agent_eval.monitor.persistence import Store

    if Store(db_path).ack_alert(alert_id):
        console.print(f"[green]Acknowledged alert #{alert_id}[/green]")
    else:
        console.print(f"[red]No alert with id {alert_id}[/red]")


# ----------------------------------------------------------------- calibrate


@cli.command()
@click.option("--dimension", type=click.Choice(["answer_quality", "tool_selection", "hallucination", "intent_resolution", "safety", "all"]),
              default="all", show_default=True)
@click.option("--output-html", help="Write a calibration HTML report to this path.")
def calibrate(dimension, output_html):
    """Run shipped calibration sets against the standard judges. Reports Pearson r / MAE / Cohen's κ per judge."""
    from agent_eval.judges.calibration import calibrate_all_judges, calibrate_judge, render_calibration_html
    from agent_eval.judges.rubric_judge import (
        AnswerQualityJudge, HallucinationJudge, IntentResolutionJudge, SafetyJudge, ToolSelectionJudge,
    )
    from rich.table import Table

    judge_map = {
        "answer_quality": AnswerQualityJudge(),
        "tool_selection": ToolSelectionJudge(),
        "hallucination": HallucinationJudge(),
        "intent_resolution": IntentResolutionJudge(),
        "safety": SafetyJudge(),
    }
    if dimension == "all":
        suite = asyncio.run(calibrate_all_judges(judge_map))
    else:
        report = asyncio.run(calibrate_judge(judge_map[dimension], dimension))
        from agent_eval.judges.calibration import CalibrationSuiteReport

        suite = CalibrationSuiteReport(judges={dimension: report}, overall_reliable=report.is_reliable)

    t = Table(title="Calibration Report")
    for col in ("Dimension", "n", "Pearson r", "MAE", "Cohen κ", "Reliable"):
        t.add_column(col)
    for dim, r in suite.judges.items():
        color = "green" if r.is_reliable else "red"
        kappa = f"{r.cohen_kappa:.3f}" if r.cohen_kappa is not None else "—"
        t.add_row(dim, str(r.n), f"[{color}]{r.pearson_r:.3f}[/{color}]", f"{r.mae:.3f}", kappa,
                  "✅" if r.is_reliable else "❌")
    console.print(t)
    console.print(f"[{'green' if suite.overall_reliable else 'red'}]Overall: {'reliable' if suite.overall_reliable else 'unreliable judges detected'}[/]")
    if output_html:
        path = render_calibration_html(suite, output_html)
        console.print(f"[green]Wrote calibration HTML report to {path}[/green]")


# ------------------------------------------------------------------- wizard


@cli.command()
@click.option("--registry", "registry_path", required=True)
@click.option("--mode", type=click.Choice(["manual", "semi-auto"]), default="semi-auto", show_default=True,
              help="manual = Path C, semi-auto = Path B.")
@click.option("--n-candidates", default=30, show_default=True, type=int, help="(semi-auto only) candidates to review.")
@click.option("--langsmith-project", help="(semi-auto only) project to harvest real traces from.")
@click.option("--output", default="dataset.json", show_default=True)
def wizard(registry_path, mode, n_candidates, langsmith_project, output):
    """Interactive dataset wizard.

    --mode manual    : Path C — author every example by hand (5 mandatory categories).
    --mode semi-auto : Path B — review auto-generated candidates one by one.
    """
    from agent_eval.dataset.user_guide import SemiAutoWizard, UserWizard

    registry = ComponentRegistry.model_validate_json(Path(registry_path).read_text())
    if mode == "manual":
        try:
            from agent_eval.config import get_config

            llm = get_config().get_chat_model()
        except Exception:
            llm = None
        ds = UserWizard(registry, llm=llm).run()
    else:
        try:
            from agent_eval.config import get_config

            client = get_config().get_langsmith_client()
            llm = get_config().get_chat_model()
        except Exception:
            client = None
            llm = None
        ds = SemiAutoWizard(registry, langsmith_client=client, llm=llm,
                             project_name=langsmith_project).run(n_candidates=n_candidates)
    Path(output).write_text(ds.model_dump_json(indent=2))
    console.print(f"[green]Wrote {output} ({len(ds.examples)} examples)[/green]")


if __name__ == "__main__":
    cli()
