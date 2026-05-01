"""agent-eval — generic agent system evaluation.

Quickstart:
    from agent_eval import AgentEval
    evaluator = AgentEval.from_mcp("http://localhost:8001/mcp", langsmith_project="my-project")
    report = evaluator.evaluate()
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.backends import get_backend
from agent_eval.config import AgentEvalConfig, get_config, set_config
from agent_eval.dataset.auto_generator import AutoDatasetGenerator
from agent_eval.dataset.schema import EvalDataset
from agent_eval.dataset.user_guide import UserWizard
from agent_eval.discovery.graph_inspector import GraphInspector
from agent_eval.discovery.langsmith_inspector import LangSmithInspector
from agent_eval.discovery.mcp_inspector import MCPInspector
from agent_eval.discovery.memory_detector import MemoryDetector
from agent_eval.discovery.registry import ComponentRegistry
from agent_eval.evaluators import (
    AnswerFaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    CitationHallucinationEvaluator,
    CompletenessEvaluator,
    CostEfficiencyEvaluator,
    CycleDetectionEvaluator,
    ErrorRateEvaluator,
    ErrorRecoveryEvaluator,
    FormatComplianceEvaluator,
    HarmfulContentEvaluator,
    InstructionFollowingEvaluator,
    IntentResolutionEvaluator,
    KeywordCoverageEvaluator,
    LatencyEvaluator,
    MCPServerHealthEvaluator,
    MemoryCostEvaluator,
    MemoryRetrievalPrecisionEvaluator,
    MemoryRetrievalRecallEvaluator,
    NodeF1Evaluator,
    ObservationHallucinationEvaluator,
    PIILeakageEvaluator,
    PlanningHallucinationEvaluator,
    ReasoningHallucinationEvaluator,
    RedundancyEvaluator,
    StepSuccessRateEvaluator,
    StreamingLatencyEvaluator,
    StructuralSimilarityEvaluator,
    TaskSuccessEvaluator,
    TokenEfficiencyEvaluator,
    ToolF1Evaluator,
    ToolPerformanceEvaluator,
    ToolSelectionEvaluator,
)
from agent_eval.monitor.online_evaluator import OnlineEvaluator
from agent_eval.runners.base import AgentRunner, RunResult
from agent_eval.runners.http_runner import HttpAgentRunner
from agent_eval.runners.langgraph_runner import LangGraphRunner
from agent_eval.runners.langsmith_replay import LangSmithReplayRunner
from agent_eval.reporters import (
    ComponentScore,
    DimensionScores,
    EvaluationReport,
    Issue,
    PerExampleResult,
    SystemOverview,
    ToolCall,
    TuningAdvisor,
    render_html_report,
    render_json_report,
)
from agent_eval.reporters.report import status_for_score

__version__ = "1.0.0"


class AgentEval:
    """Primary user-facing API."""

    def __init__(
        self,
        registry: ComponentRegistry,
        langsmith_project: str | None = None,
        mcp_url: str | None = None,
        graph: Any | None = None,
        config: AgentEvalConfig | None = None,
    ):
        self.registry = registry
        self.langsmith_project = langsmith_project
        self.mcp_url = mcp_url
        self.graph = graph
        if config is not None:
            set_config(config)
        self.config = get_config()
        self._client = None

    # -------------------------------------------------------------------- ctors

    @classmethod
    def from_mcp(
        cls,
        mcp_url: str,
        langsmith_project: str,
        langsmith_api_key: str | None = None,
        config: AgentEvalConfig | None = None,
    ) -> AgentEval:
        if langsmith_api_key:
            import os

            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        empty = ComponentRegistry(discovery_method="mcp")
        return cls(empty, langsmith_project=langsmith_project, mcp_url=mcp_url, config=config)

    @classmethod
    def from_langgraph(cls, graph: Any, langsmith_project: str, config: AgentEvalConfig | None = None) -> AgentEval:
        empty = ComponentRegistry(discovery_method="graph")
        return cls(empty, langsmith_project=langsmith_project, graph=graph, config=config)

    @classmethod
    def from_registry(cls, registry_path: str) -> AgentEval:
        registry = ComponentRegistry.model_validate_json(Path(registry_path).read_text())
        return cls(registry)

    # ---------------------------------------------------------------- discover

    def discover(self) -> ComponentRegistry:
        registry = self._discover_components_sync()
        self.registry = registry
        return registry

    def _discover_components_sync(self) -> ComponentRegistry:
        if self.mcp_url:
            try:
                reg = asyncio.run(MCPInspector(self.mcp_url).inspect())
                if reg.total_components > 0:
                    self._merge_memory(reg)
                    return reg
            except Exception:
                pass
        if self.graph is not None:
            try:
                reg = GraphInspector(self.graph).inspect()
                if reg.total_components > 0:
                    self._merge_memory(reg)
                    return reg
            except Exception:
                pass
        if self.langsmith_project:
            try:
                client = self._lc_client()
                reg = LangSmithInspector(client, self.langsmith_project).mine_components(n_traces=100)
                if reg.total_components > 0:
                    self._merge_memory(reg)
                    return reg
            except Exception:
                pass
        return ComponentRegistry(discovery_method="manual")

    def _merge_memory(self, reg: ComponentRegistry) -> None:
        if not self.langsmith_project:
            return
        try:
            client = self._lc_client()
            backends = MemoryDetector(client, self.langsmith_project).detect(n_traces=100)
            if backends:
                reg.memory_backends = backends
        except Exception:
            pass

    def _lc_client(self):
        if self._client is None:
            self._client = self.config.get_langsmith_client()
        return self._client

    # ----------------------------------------------------------------- dataset

    def generate_dataset(self, mode: str = "auto", n_examples: int = 50, template: str | None = None) -> EvalDataset:
        if mode == "template":
            if not template:
                raise ValueError("mode='template' requires --template name")
            return EvalDataset.from_template(template)
        if mode == "manual":
            return UserWizard(self.registry).run()
        if mode == "auto":
            try:
                client = self._lc_client()
            except Exception:
                client = None
            gen = AutoDatasetGenerator(self.registry, langsmith_client=client)
            return asyncio.run(gen.generate(n_examples=n_examples, project_name=self.langsmith_project))
        raise ValueError(f"unknown mode '{mode}' — use 'auto', 'template', or 'manual'")

    # ---------------------------------------------------------------- evaluate

    def evaluate(
        self,
        dataset: EvalDataset | None = None,
        dimensions: list[str] | None = None,
        backend: str = "native",
        baseline_report: EvaluationReport | None = None,
        n_examples: int | None = None,
        runner: AgentRunner | None = None,
        traces: list[Any] | None = None,
    ) -> EvaluationReport:
        """Run the evaluation suite.

        Args:
          dataset: EvalDataset to score against. Defaults to the general_agent template.
          runner: An AgentRunner (HttpAgentRunner, LangGraphRunner, or LangSmithReplayRunner).
            When provided, the runner is invoked for each example to produce real traces.
          traces: Pre-fetched list of trace objects, parallel to dataset.examples. Use this
            when you already have the runs and don't need to invoke an agent.
          dimensions: Subset of ['output_quality','trajectory','hallucination','tool_performance',
            'system_performance','safety','memory'] or ['all'].
          backend: 'native' (default), 'deepeval', or 'ragas'.
          baseline_report: Prior EvaluationReport for trend computation.
          n_examples: Cap dataset to first N examples (smoke testing).

        Without `runner` or `traces`, every evaluator receives `trace=None` and the report
        will reflect the dataset's testability rather than the agent's behavior. A WARNING
        is printed in that case.
        """
        if dataset is None:
            dataset = self.generate_dataset(mode="template", template="general_agent")
        if n_examples and n_examples < len(dataset.examples):
            dataset = dataset.model_copy(update={"examples": dataset.examples[:n_examples]})
        if not self.registry.total_components:
            self.registry = self._discover_components_sync()
        return asyncio.run(self._evaluate_async(dataset, dimensions or ["all"], backend, baseline_report, runner, traces))

    async def _evaluate_async(
        self,
        dataset: EvalDataset,
        dimensions: list[str],
        backend_name: str,
        baseline_report: EvaluationReport | None,
        runner: AgentRunner | None = None,
        traces: list[Any] | None = None,
    ) -> EvaluationReport:
        backend = get_backend(backend_name)
        evaluators = _build_evaluator_suite(self.registry, dimensions, backend)
        start = time.perf_counter()
        per_metric_scores: dict[str, list[float]] = defaultdict(list)
        per_metric_results = []
        per_example_results: list[PerExampleResult] = []

        # 1. Get traces.
        runner_results_by_id: dict[str, RunResult] = {}
        if traces is None:
            if runner is not None:
                traces, runner_results_by_id = await self._run_dataset_with_runner_detailed(runner, dataset)
            else:
                print(
                    "[agent-eval] WARNING: no runner or traces provided. "
                    "Evaluators will receive trace=None — the report will measure "
                    "dataset testability, NOT agent behavior. Pass runner=... or traces=... "
                    "to evaluate the actual system."
                )
                traces = [None] * len(dataset.examples)

        # 2. Score every example × every evaluator, and build per-example records.
        for idx, example in enumerate(dataset.examples):
            trace = traces[idx] if idx < len(traces) else None
            ex_results: list[Any] = []
            for ev in evaluators:
                res = await ev.evaluate(example, trace)
                per_metric_scores[res.evaluator_name].append(res.score)
                per_metric_results.append(res)
                ex_results.append(res)
            run_result = runner_results_by_id.get(example.id)
            per_example_results.append(_build_per_example_result(
                example=example, trace=trace, results=ex_results,
                run_result=run_result, langsmith_project=self.langsmith_project,
            ))

        duration = time.perf_counter() - start

        # Aggregate per-metric.
        per_metric_avg = {m: (sum(v) / len(v)) for m, v in per_metric_scores.items() if v}

        # Dimension scores.
        dim = _aggregate_dimensions(per_metric_avg)

        # System overview.
        overall = _overall_score(dim)
        flagged_results = [r for r in per_metric_results if r.flagged]
        critical = [r for r in flagged_results if r.score < 0.5 or r.flag_reason == "cycle detected"]
        overview = SystemOverview(
            overall_score=overall,
            health_status=status_for_score(overall),
            pass_rate=sum(1 for r in per_metric_results if r.passed) / max(1, len(per_metric_results)),
            flag_count=len(flagged_results),
            critical_flag_count=len(critical),
        )

        # Component scores: group results by tags (component) when available; else "system".
        component_scores = _component_scores(per_metric_results, baseline_report)

        # Issues.
        issues = _issues_from_results(per_metric_results, traces, self.langsmith_project)

        # Backends used: any evaluator whose backend overrides default.
        backends_used = {ev.name: backend.name for ev in evaluators}

        # Dataset stats.
        from agent_eval.reporters.report import DatasetStats

        qt_dist: dict[str, int] = defaultdict(int)
        cx_dist: dict[str, int] = defaultdict(int)
        for ex in dataset.examples:
            qt_dist[ex.query_type] += 1
            cx_dist[ex.complexity] += 1
        ds_stats = DatasetStats(
            n_examples=len(dataset.examples),
            query_type_distribution=dict(qt_dist),
            complexity_distribution=dict(cx_dist),
        )

        report = EvaluationReport(
            system_name=self.langsmith_project or "agent-system",
            dataset_name=dataset.name,
            dataset_size=len(dataset.examples),
            evaluation_duration_seconds=duration,
            langsmith_project=self.langsmith_project,
            system_overview=overview,
            component_scores=component_scores,
            dimension_scores=dim,
            flagged_issues=issues,
            backends_used=backends_used,
            per_example_results=per_example_results,
            dataset_stats=ds_stats,
        )
        report.tuning_recommendations = [s.to_dict() for s in TuningAdvisor().analyze(report)]
        return report

    async def _run_dataset_with_runner(self, runner: AgentRunner, dataset: EvalDataset) -> list[Any]:
        """Backwards-compatible wrapper that returns just the ordered traces."""
        traces, _ = await self._run_dataset_with_runner_detailed(runner, dataset)
        return traces

    async def _run_dataset_with_runner_detailed(
        self, runner: AgentRunner, dataset: EvalDataset
    ) -> tuple[list[Any], dict[str, RunResult]]:
        """Execute the runner against the dataset. Returns (ordered_traces, run_results_by_id)."""
        from rich.console import Console

        console = Console()
        n = len(dataset.examples)
        completed = {"n": 0}

        def progress(i: int, total: int, res: RunResult) -> None:
            completed["n"] = i
            status = "ok" if res.error is None else f"err: {res.error[:60]}"
            console.print(f"  [{i}/{total}] {res.example_id[:8]} {status} ({res.latency_ms or 0:.0f}ms)", end="\r")

        console.print(f"[cyan]Running {n} examples via {runner.name} runner...[/cyan]")
        results = await runner.run_dataset(dataset, progress_callback=progress)
        try:
            await runner.aclose()
        except Exception:
            pass
        # Order results back to dataset order.
        by_id = {r.example_id: r for r in results}
        ordered_traces: list[Any] = []
        n_real = 0
        for ex in dataset.examples:
            r = by_id.get(ex.id)
            if r is None or r.trace is None:
                ordered_traces.append(None)
            else:
                ordered_traces.append(r.trace)
                if not getattr(r.trace, "_is_synthetic", False):
                    n_real += 1
        console.print(
            f"[cyan]Runner done — real LangSmith traces: {n_real}/{n}, "
            f"synthetic shims: {n - n_real - sum(1 for t in ordered_traces if t is None)}, "
            f"failures: {sum(1 for t in ordered_traces if t is None)}.[/cyan]"
        )
        return ordered_traces, by_id

    # ----------------------------------------------------------- tuning / save

    def get_tuning_signals(self, report: EvaluationReport):
        return TuningAdvisor().analyze(report)

    def save_report(self, report: EvaluationReport, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = render_json_report(report, out / f"report_{ts}.json")
        html_path = render_html_report(report, out / f"report_{ts}.html")
        (out / "report.json").write_text(report.to_json())
        return {"json": json_path, "html": html_path, "latest_json": out / "report.json"}

    # ------------------------------------------------------------------ ci/monitor

    def ci_gate(
        self,
        current_report: EvaluationReport,
        baseline_report: EvaluationReport | None = None,
        thresholds_path: str | Path | None = None,
    ) -> int:
        from agent_eval.ci.gate import ci_gate as _ci_gate

        return _ci_gate(current_report, baseline_report, thresholds_path)

    def monitor(self, sampling_rate: float = 0.1, evaluators: list[Any] | None = None) -> OnlineEvaluator:
        if evaluators is None:
            evaluators = [
                TaskSuccessEvaluator(),
                AnswerRelevanceEvaluator(),
                CycleDetectionEvaluator(),
                HarmfulContentEvaluator(),
            ]
        return OnlineEvaluator(self.registry, evaluators=evaluators, sampling_rate=sampling_rate, client=self._lc_client())


# ---------------------------------------------------------------------------- helpers


def _build_evaluator_suite(registry: ComponentRegistry, dimensions: list[str], backend: Any) -> list[Any]:
    selected = set(dimensions)
    want_all = "all" in selected

    def keep(name: str) -> bool:
        return want_all or name in selected

    evs: list[Any] = []
    if keep("output_quality"):
        evs += [
            TaskSuccessEvaluator(backend=backend),
            AnswerFaithfulnessEvaluator(backend=backend),
            AnswerRelevanceEvaluator(backend=backend),
            CompletenessEvaluator(backend=backend),
            FormatComplianceEvaluator(backend=backend),
            KeywordCoverageEvaluator(backend=backend),
        ]
    if keep("trajectory"):
        evs += [
            ToolSelectionEvaluator(backend=backend),
            ToolF1Evaluator(backend=backend),
            NodeF1Evaluator(backend=backend),
            StructuralSimilarityEvaluator(backend=backend),
            IntentResolutionEvaluator(backend=backend),
            StepSuccessRateEvaluator(backend=backend),
            RedundancyEvaluator(backend=backend),
            ErrorRecoveryEvaluator(backend=backend),
            CycleDetectionEvaluator(backend=backend),
        ]
    if keep("hallucination"):
        evs += [
            PlanningHallucinationEvaluator(backend=backend),
            ObservationHallucinationEvaluator(backend=backend),
            CitationHallucinationEvaluator(backend=backend),
            ReasoningHallucinationEvaluator(backend=backend),
        ]
    if keep("tool_performance"):
        evs += [
            ToolPerformanceEvaluator(backend=backend),
            MCPServerHealthEvaluator(mcp_servers=registry.mcp_servers, backend=backend),
        ]
    if keep("system_performance"):
        evs += [
            LatencyEvaluator(backend=backend),
            StreamingLatencyEvaluator(backend=backend),
            TokenEfficiencyEvaluator(backend=backend),
            ErrorRateEvaluator(backend=backend),
            CostEfficiencyEvaluator(backend=backend),
        ]
    if keep("safety"):
        evs += [
            HarmfulContentEvaluator(backend=backend),
            PIILeakageEvaluator(backend=backend),
            InstructionFollowingEvaluator(backend=backend),
        ]
    if keep("memory") and registry.memory_backends:
        evs += [
            MemoryRetrievalRecallEvaluator(backend=backend),
            MemoryRetrievalPrecisionEvaluator(backend=backend),
            MemoryCostEvaluator(backend=backend),
        ]
    return evs


def _aggregate_dimensions(per_metric_avg: dict[str, float]) -> DimensionScores:
    def avg(*keys: str) -> float:
        vals = [per_metric_avg[k] for k in keys if k in per_metric_avg]
        return sum(vals) / len(vals) if vals else 0.0

    output_q = avg("task_success_rate", "answer_faithfulness", "answer_relevance", "completeness", "format_compliance", "keyword_coverage")
    trajectory_q = avg("tool_selection_accuracy", "tool_f1", "node_f1", "ssi", "intent_resolution", "step_success_rate", "redundancy_rate", "error_recovery_rate", "cycle_detected")
    hallu_score = avg("hallucination_planning", "hallucination_observation", "hallucination_citation", "hallucination_reasoning")
    hallucination_risk = max(0.0, 1.0 - hallu_score) if hallu_score else 0.0
    tool_p = avg("tool_success_rate", "tool_result_quality", "argument_correctness", "mcp_server_availability", "cost_per_tool")
    sys_p = avg("end_to_end_latency", "time_to_first_audio_byte", "token_efficiency", "error_rate", "cost_per_query_usd")
    safety = avg("harmful_content_rate", "pii_leakage_rate", "instruction_compliance", "response_consistency")
    memory_q = avg("memory_retrieval_recall", "memory_retrieval_precision", "memory_cost_per_query")
    return DimensionScores(
        output_quality=output_q,
        trajectory_quality=trajectory_q,
        hallucination_risk=hallucination_risk,
        tool_performance=tool_p,
        system_performance=sys_p,
        safety=safety,
        memory_quality=memory_q,
    )


def _overall_score(d: DimensionScores) -> float:
    weights = {
        "output_quality": 0.30,
        "trajectory_quality": 0.15,
        "hallucination_safety": 0.15,  # 1 - hallucination_risk
        "tool_performance": 0.10,
        "system_performance": 0.10,
        "safety": 0.10,
        "memory_quality": 0.10,
    }
    return (
        weights["output_quality"] * d.output_quality
        + weights["trajectory_quality"] * d.trajectory_quality
        + weights["hallucination_safety"] * (1.0 - d.hallucination_risk)
        + weights["tool_performance"] * d.tool_performance
        + weights["system_performance"] * d.system_performance
        + weights["safety"] * d.safety
        + weights["memory_quality"] * (d.memory_quality if d.memory_quality > 0 else 1.0)
    )


def _component_scores(results: list[Any], baseline_report: EvaluationReport | None) -> list[ComponentScore]:
    by_component: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        by_component[r.component_name].append(r)
    out = []
    for i, (name, rs) in enumerate(sorted(by_component.items(), key=lambda kv: -sum(r.score for r in kv[1]) / max(1, len(kv[1])))):
        avg_score = sum(r.score for r in rs) / max(1, len(rs))
        trend = None
        if baseline_report:
            for prev in baseline_report.component_scores:
                if prev.component_name == name:
                    trend = avg_score - prev.overall_score
                    break
        out.append(ComponentScore(component_name=name, overall_score=avg_score, evaluator_results=rs, rank=i + 1, trend=trend))
    return out


def _issues_from_results(results: list[Any], traces: list[Any], project: str | None) -> list[Issue]:
    issues = []
    for r in results:
        if not r.flagged:
            continue
        sev = "critical" if r.score < 0.3 else "high" if r.score < 0.5 else "medium"
        issues.append(
            Issue(
                severity=sev,
                component=r.component_name,
                metric=r.evaluator_name,
                score=r.score,
                description=r.flag_reason or f"{r.evaluator_name} below threshold",
            )
        )
    issues.sort(key=lambda i: ("critical", "high", "medium", "low").index(i.severity))
    return issues


# ---------------------------------------------------------------- per-example


def _build_per_example_result(
    example: Any,
    trace: Any,
    results: list[Any],
    run_result: Any | None,
    langsmith_project: str | None,
) -> PerExampleResult:
    """Build a PerExampleResult from one example + its evaluator results + (optional) RunResult."""
    actual_output = _extract_final_answer(trace)
    n = max(1, len(results))
    score = sum(r.score for r in results) / n
    pass_count = sum(1 for r in results if r.passed)
    pass_rate = pass_count / n
    flagged_count = sum(1 for r in results if r.flagged)
    critical_count = sum(1 for r in results if r.flagged and (r.score < 0.3 or r.flag_reason == "cycle detected"))
    is_overall_pass = pass_rate >= 0.7 and critical_count == 0

    run_id: str | None = None
    deep_link: str | None = None
    is_synth = bool(getattr(trace, "_is_synthetic", False))
    if trace is not None and not is_synth:
        rid_raw = getattr(trace, "id", None)
        if rid_raw:
            run_id = str(rid_raw)
            deep_link = _langsmith_deep_link(run_id, langsmith_project)
    elif run_result is not None and run_result.metadata:
        rid_raw = run_result.metadata.get("run_id")
        if rid_raw:
            run_id = str(rid_raw)
            deep_link = _langsmith_deep_link(run_id, langsmith_project)

    tool_calls = _extract_tool_calls_for_report(trace)

    runner_latency = run_result.latency_ms if run_result is not None else None
    runner_error = run_result.error if run_result is not None else None

    query = ""
    if isinstance(example.input, dict):
        query = str(example.input.get("query") or example.input.get("question") or example.input)
    else:
        query = str(example.input)

    return PerExampleResult(
        example_id=example.id,
        query=query,
        actual_output=actual_output,
        expected_output=example.reference_answer,
        expected_keywords=example.expected_answer_keywords,
        query_type=example.query_type,
        complexity=example.complexity,
        score=score,
        pass_rate=pass_rate,
        flagged_count=flagged_count,
        critical_count=critical_count,
        is_overall_pass=is_overall_pass,
        langsmith_run_id=run_id,
        langsmith_run_url=deep_link,
        tool_calls=tool_calls,
        evaluator_results=results,
        runner_latency_ms=runner_latency,
        runner_error=runner_error,
        trace_was_synthetic=is_synth,
    )


def _extract_final_answer(trace: Any) -> str:
    if trace is None:
        return ""
    outputs = getattr(trace, "outputs", None) or {}
    if isinstance(outputs, dict):
        for k in ("answer", "final_answer", "summary", "research_findings", "output", "response"):
            v = outputs.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


def _extract_tool_calls_for_report(trace: Any) -> list[ToolCall]:
    if trace is None:
        return []
    out: list[ToolCall] = []

    def walk(r: Any):
        if (getattr(r, "run_type", None) or "") == "tool":
            inputs = getattr(r, "inputs", None) or {}
            outputs = getattr(r, "outputs", None) or {}
            err = getattr(r, "error", None)
            s, e = getattr(r, "start_time", None), getattr(r, "end_time", None)
            latency = None
            try:
                if s and e and hasattr(s, "timestamp"):
                    latency = (e - s).total_seconds() * 1000.0
            except Exception:
                latency = None
            out.append(ToolCall(
                name=getattr(r, "name", "unknown") or "unknown",
                inputs=_truncate_for_display(inputs) if isinstance(inputs, dict) else {"raw": str(inputs)[:300]},
                outputs_preview=_stringify_short(outputs, max_chars=600),
                success=not bool(err),
                error=(str(err)[:300] if err else None),
                latency_ms=latency,
            ))
        for c in getattr(r, "child_runs", None) or []:
            walk(c)

    for c in getattr(trace, "child_runs", None) or []:
        walk(c)
    # Stable sort by start_time when present.
    return out


def _truncate_for_display(obj: dict[str, Any], max_str: int = 200) -> dict[str, Any]:
    """Make a shallow display-friendly copy of a dict, truncating long string values."""
    out: dict[str, Any] = {}
    for k, v in (obj or {}).items():
        if isinstance(v, str):
            out[k] = v if len(v) <= max_str else v[:max_str] + "…"
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = [_one_short(item) for item in v[:5]]
            if len(v) > 5:
                out[k].append(f"… +{len(v) - 5} more")
        elif isinstance(v, dict):
            out[k] = {kk: _one_short(vv) for kk, vv in list(v.items())[:6]}
        else:
            out[k] = str(v)[:max_str]
    return out


def _one_short(v: Any, max_str: int = 120) -> Any:
    if isinstance(v, str):
        return v if len(v) <= max_str else v[:max_str] + "…"
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)[:max_str]


def _stringify_short(obj: Any, max_chars: int = 600) -> str:
    import json

    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj if len(obj) <= max_chars else obj[:max_chars] + "…"
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def _langsmith_deep_link(run_id: str, project: str | None) -> str:
    """Build a LangSmith URL for a given run id.

    Override base via env: `LANGSMITH_DEEP_LINK_BASE` (default https://smith.langchain.com).
    The simple `/r/<run_id>` form works for any logged-in user with access to the project.
    """
    import os

    base = os.getenv("LANGSMITH_DEEP_LINK_BASE", "https://smith.langchain.com").rstrip("/")
    return f"{base}/r/{run_id}"


__all__ = [
    "AgentEval",
    "AgentEvalConfig",
    "AgentRunner",
    "RunResult",
    "HttpAgentRunner",
    "LangGraphRunner",
    "LangSmithReplayRunner",
    "__version__",
]
