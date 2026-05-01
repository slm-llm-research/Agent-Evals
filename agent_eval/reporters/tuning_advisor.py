"""Tuning advisor — turns scores into prioritized actions.

Six signal types per spec section 7.4:
  (a) PromptTuningSignal       — judge low → add few-shot, tighten prompt
  (b) ModelDowngradeSignal     — high latency, decent accuracy → cheaper model
  (c) ToolConfigSignal         — tool quality low → tune params
  (d) ArchitectureSignal       — looping/cycles/over-iteration → add stop conditions
  (e) DataSignal               — low-coverage query types → add training/eval examples
  (f) MemorySignal             — memory metrics low → tuning options
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from agent_eval.evaluators.base import EvaluatorResult
from agent_eval.reporters.report import EvaluationReport

Severity = Literal["critical", "high", "medium", "low"]
TuningType = Literal["prompt", "model", "tool_config", "architecture", "data", "memory"]
Effort = Literal["minutes", "hours", "days", "weeks"]


@dataclass
class TuningSignal:
    component: str
    issue_type: str
    severity: Severity
    current_score: float
    target_score: float
    effort_estimate: Effort
    tuning_type: TuningType
    specific_action: str
    example_fix: str | None = None
    metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def _severity_for(score: float) -> Severity:
    if score < 0.5:
        return "critical"
    if score < 0.65:
        return "high"
    if score < 0.80:
        return "medium"
    return "low"


class TuningAdvisor:
    """Generates 6 categories of tuning signals from an EvaluationReport."""

    def analyze(self, report: EvaluationReport) -> list[TuningSignal]:
        signals: list[TuningSignal] = []
        signals.extend(self._prompt_signals(report))
        signals.extend(self._model_signals(report))
        signals.extend(self._tool_config_signals(report))
        signals.extend(self._architecture_signals(report))
        signals.extend(self._data_signals(report))
        signals.extend(self._memory_signals(report))
        signals.sort(key=lambda s: ("critical", "high", "medium", "low").index(s.severity))
        return signals

    # ------------------------------------------------------------- (a) prompt

    def _prompt_signals(self, report: EvaluationReport) -> list[TuningSignal]:
        out: list[TuningSignal] = []
        d = report.dimension_scores
        if d.output_quality < 0.85:
            out.append(TuningSignal(
                component="answer-generation node",
                issue_type="answer quality",
                severity=_severity_for(d.output_quality),
                current_score=d.output_quality,
                target_score=0.85,
                effort_estimate="hours",
                tuning_type="prompt",
                specific_action=(
                    "Add 3-5 few-shot examples of correct, well-formatted answers to the answer-generation prompt. "
                    "Tighten the output format clause (word limit, sections, JSON schema if structured output is needed). "
                    "Audit the system prompt for vague verbs ('help', 'try') — replace with specific instructions."
                ),
                example_fix="<system>\nYou are an X expert. Always:\n1. State the conclusion in the first sentence.\n2. Cite at least 1 source.\n3. Stay under 150 words.\n\nHere are 3 example responses:\n[good example 1]\n[good example 2]\n[bad example with annotation]\n</system>",
                metric="output_quality",
            ))
        if d.hallucination_risk > 0.10:
            out.append(TuningSignal(
                component="answer-generation node",
                issue_type="hallucination — prompt-level mitigation",
                severity="critical" if d.hallucination_risk > 0.25 else "high",
                current_score=1 - d.hallucination_risk,
                target_score=0.95,
                effort_estimate="days",
                tuning_type="prompt",
                specific_action=(
                    "Require the agent to ground every factual claim in a citation that appears in retrieved sources. "
                    "Add a post-hoc verification pass: 'Before responding, list each claim and the source it comes from.' "
                    "Lower temperature on the synthesis LLM."
                ),
                metric="hallucination_risk",
            ))
        return out

    # ------------------------------------------------------------- (b) model

    def _model_signals(self, report: EvaluationReport) -> list[TuningSignal]:
        out: list[TuningSignal] = []
        per_node = self._per_node_metrics(report)
        for node, metrics in per_node.items():
            avg_latency = metrics.get("avg_latency_ms", 0.0)
            quality = metrics.get("avg_quality", 1.0)
            if avg_latency > 5000 and quality > 0.80:
                out.append(TuningSignal(
                    component=node,
                    issue_type="model latency",
                    severity="medium",
                    current_score=1.0 - min(1.0, avg_latency / 30000),
                    target_score=0.85,
                    effort_estimate="hours",
                    tuning_type="model",
                    specific_action=(
                        f"Node '{node}' avg latency = {avg_latency:.0f}ms with quality {quality:.2f}. "
                        "Try downgrading to a smaller model (gpt-4o-mini or claude-haiku-4-5). "
                        "Estimated 40% cost + 60% latency reduction with <5% quality loss for routine reasoning."
                    ),
                    metric="end_to_end_latency",
                ))
        if report.dimension_scores.system_performance < 0.65:
            out.append(TuningSignal(
                component="system",
                issue_type="overall latency / cost",
                severity=_severity_for(report.dimension_scores.system_performance),
                current_score=report.dimension_scores.system_performance,
                target_score=0.85,
                effort_estimate="days",
                tuning_type="model",
                specific_action=(
                    "Profile per-node latency in the report. Move non-critical nodes (intent classification, "
                    "format polishing) to gpt-4o-mini or claude-haiku-4-5. Keep the synthesis node on the strong model."
                ),
                metric="system_performance",
            ))
        return out

    # --------------------------------------------------- (c) tool_config

    def _tool_config_signals(self, report: EvaluationReport) -> list[TuningSignal]:
        out: list[TuningSignal] = []
        # Look at per-tool quality from raw evaluator results.
        per_tool: dict[str, list[float]] = defaultdict(list)
        for c in report.component_scores:
            for ev in c.evaluator_results:
                if ev.evaluator_name in ("tool_success_rate", "tool_result_quality", "argument_correctness"):
                    if "per_tool" in ev.details:
                        for tname, t_metrics in ev.details["per_tool"].items():
                            per_tool[tname].append(t_metrics.get("success_rate", 1.0))
                    else:
                        per_tool[c.component_name].append(ev.score)
        for tool_name, scores in per_tool.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            if avg < 0.85:
                out.append(TuningSignal(
                    component=tool_name,
                    issue_type="tool reliability / quality",
                    severity=_severity_for(avg),
                    current_score=avg,
                    target_score=0.95,
                    effort_estimate="hours",
                    tuning_type="tool_config",
                    specific_action=(
                        f"Tool '{tool_name}' average score = {avg:.2f}. Try: (a) increase timeouts, "
                        "(b) add retries with backoff, (c) switch to a more constrained input schema, "
                        "(d) for search tools, switch to advanced/research mode if the API supports it."
                    ),
                    metric="tool_performance",
                ))
        if report.dimension_scores.tool_performance < 0.85 and not out:
            out.append(TuningSignal(
                component="tools",
                issue_type="tool reliability",
                severity=_severity_for(report.dimension_scores.tool_performance),
                current_score=report.dimension_scores.tool_performance,
                target_score=0.90,
                effort_estimate="hours",
                tuning_type="tool_config",
                specific_action=(
                    "Inspect per-tool success rates and error breakdown in the report. Add retries, "
                    "increase timeouts on slow tools, or replace tool argument templates with constrained schemas."
                ),
                metric="tool_performance",
            ))
        return out

    # ----------------------------------------------- (d) architecture

    def _architecture_signals(self, report: EvaluationReport) -> list[TuningSignal]:
        out: list[TuningSignal] = []
        cycle_count = sum(
            1
            for c in report.component_scores
            for ev in c.evaluator_results
            if ev.evaluator_name == "cycle_detected" and ev.flagged
        )
        if cycle_count > 0:
            out.append(TuningSignal(
                component="orchestrator / planner",
                issue_type="cycles / loops detected",
                severity="critical",
                current_score=0.0,
                target_score=1.0,
                effort_estimate="hours",
                tuning_type="architecture",
                specific_action=(
                    f"Cycle detected in {cycle_count} examples. Add an explicit max-iteration counter "
                    "and short-circuit when (node, input_hash) repeats ≥3 times. Add a stronger stop condition "
                    "in the conditional edge of the planning loop."
                ),
                example_fix="if state['_iteration_count'] >= 5: return 'finalize'  # in conditional edge",
                metric="cycle_detected",
            ))
        if report.dimension_scores.trajectory_quality < 0.75:
            out.append(TuningSignal(
                component="orchestrator / planner",
                issue_type="trajectory quality",
                severity=_severity_for(report.dimension_scores.trajectory_quality),
                current_score=report.dimension_scores.trajectory_quality,
                target_score=0.85,
                effort_estimate="days",
                tuning_type="architecture",
                specific_action=(
                    "Tighten the planner prompt to require explicit tool-selection rationale per step. "
                    "Add a stop condition for research depth (max 2 iterations). Cache repeated tool inputs."
                ),
                metric="trajectory_quality",
            ))
        return out

    # ------------------------------------------------------ (e) data

    def _data_signals(self, report: EvaluationReport) -> list[TuningSignal]:
        out: list[TuningSignal] = []
        # If a particular query_type underperforms, suggest more dataset coverage there.
        per_qt: dict[str, list[float]] = defaultdict(list)
        for c in report.component_scores:
            qt_tag = next((t for t in (getattr(c, "tags", None) or []) if t.startswith("qt:")), None)
            if not qt_tag:
                continue
            qt = qt_tag.split(":", 1)[1]
            per_qt[qt].append(c.overall_score)
        for qt, scores in per_qt.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            if avg < 0.70:
                out.append(TuningSignal(
                    component="dataset",
                    issue_type=f"low coverage for '{qt}' queries",
                    severity=_severity_for(avg),
                    current_score=avg,
                    target_score=0.85,
                    effort_estimate="days",
                    tuning_type="data",
                    specific_action=(
                        f"Query type '{qt}' scores {avg:.2f}. Add 10-20 examples of this query type to your "
                        "training/eval dataset. Consider fine-tuning or in-context examples specifically for this category."
                    ),
                    metric="query_type_coverage",
                ))
        # Generic data signal if nothing above fired.
        if not out and report.system_overview.flag_count > report.dataset_size * 0.3:
            out.append(TuningSignal(
                component="dataset",
                issue_type="high failure rate may indicate dataset gaps",
                severity="medium",
                current_score=report.system_overview.pass_rate,
                target_score=0.85,
                effort_estimate="hours",
                tuning_type="data",
                specific_action=(
                    "Examine the flagged examples for common patterns (intent type, complexity). "
                    "Add 10-20 examples of underperforming categories to the eval dataset to make the signal stronger."
                ),
            ))
        return out

    # ----------------------------------------------------- (f) memory

    def _memory_signals(self, report: EvaluationReport) -> list[TuningSignal]:
        out: list[TuningSignal] = []
        d = report.dimension_scores
        if not d.memory_quality:
            return out  # no memory backend evaluated
        per_metric = self._per_metric_scores(report)
        recall = per_metric.get("memory_retrieval_recall")
        precision = per_metric.get("memory_retrieval_precision")
        write = per_metric.get("memory_write_quality")
        staleness = per_metric.get("memory_staleness")

        if recall is not None and recall < 0.80:
            out.append(TuningSignal(
                component="memory backend (retrieval)",
                issue_type="low recall",
                severity=_severity_for(recall),
                current_score=recall,
                target_score=0.85,
                effort_estimate="days",
                tuning_type="memory",
                specific_action=(
                    f"Memory retrieval recall = {recall:.2f}. Tuning options: increase top_k (10→25), "
                    "switch to hybrid retrieval (dense + BM25 reranker), re-embed corpus with text-embedding-3-large, "
                    "or use query expansion (HyDE) before retrieval."
                ),
                metric="memory_retrieval_recall",
            ))
        if precision is not None and precision < 0.80:
            out.append(TuningSignal(
                component="memory backend (retrieval)",
                issue_type="low precision (noisy results)",
                severity=_severity_for(precision),
                current_score=precision,
                target_score=0.85,
                effort_estimate="hours",
                tuning_type="memory",
                specific_action=(
                    f"Memory retrieval precision = {precision:.2f}. Tuning options: lower top_k, "
                    "add a reranker (cross-encoder/ms-marco-MiniLM), tighten the similarity threshold, "
                    "or add metadata filtering before vector search."
                ),
                metric="memory_retrieval_precision",
            ))
        if write is not None and write < 0.70:
            out.append(TuningSignal(
                component="memory backend (writes)",
                issue_type="poor write granularity",
                severity=_severity_for(write),
                current_score=write,
                target_score=0.85,
                effort_estimate="hours",
                tuning_type="memory",
                specific_action=(
                    "Add a write-decision LLM filter: only persist messages that contain durable facts "
                    "(named entities, preferences, decisions). Don't store every conversational turn verbatim."
                ),
                metric="memory_write_quality",
            ))
        if staleness is not None and staleness < 0.80:
            out.append(TuningSignal(
                component="memory backend (freshness)",
                issue_type="stale memories surfacing",
                severity=_severity_for(staleness),
                current_score=staleness,
                target_score=0.90,
                effort_estimate="days",
                tuning_type="memory",
                specific_action=(
                    "Add a TTL or last-confirmed timestamp on stored facts. On retrieval, prefer recent facts "
                    "and de-rank older ones unless explicitly historical. Periodically prune contradicted facts."
                ),
                metric="memory_staleness",
            ))
        return out

    # ----------------------------------------------------- helpers

    def _per_node_metrics(self, report: EvaluationReport) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(dict)
        for c in report.component_scores:
            latencies = []
            qualities = []
            for ev in c.evaluator_results:
                if "latency_ms" in ev.details:
                    latencies.append(float(ev.details.get("latency_ms", 0)))
                if ev.evaluator_name in ("answer_relevance", "answer_faithfulness", "tool_result_quality"):
                    qualities.append(ev.score)
            if latencies:
                out[c.component_name]["avg_latency_ms"] = sum(latencies) / len(latencies)
            if qualities:
                out[c.component_name]["avg_quality"] = sum(qualities) / len(qualities)
        return out

    def _per_metric_scores(self, report: EvaluationReport) -> dict[str, float]:
        out: dict[str, list[float]] = defaultdict(list)
        for c in report.component_scores:
            for ev in c.evaluator_results:
                out[ev.evaluator_name].append(ev.score)
        return {k: sum(v) / len(v) for k, v in out.items() if v}

    def render_console(self, signals: list[TuningSignal]) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        if not signals:
            console.print("[green]No tuning recommendations — system looks healthy.[/green]")
            return
        table = Table(title="Tuning Recommendations", title_style="bold")
        for col in ("Severity", "Type", "Component", "Issue", "Score", "Action"):
            table.add_column(col, no_wrap=False)
        sev_color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "white"}
        for s in signals:
            table.add_row(
                f"[{sev_color[s.severity]}]{s.severity}[/{sev_color[s.severity]}]",
                s.tuning_type,
                s.component,
                s.issue_type,
                f"{s.current_score:.2f} → {s.target_score:.2f}",
                s.specific_action[:200],
            )
        console.print(table)
