"""LangGraphRunner — invoke a CompiledGraph in-process and capture the trace.

Uses LangChain's `RunCollectorCallbackHandler` to capture the full run tree
locally without depending on LangSmith ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from agent_eval.dataset.schema import EvalExample
from agent_eval.runners.base import AgentRunner, RunResult
from agent_eval.runners.synthetic import build_synthetic_trace


@dataclass
class LangGraphRunner(AgentRunner):
    """Args:

    graph: A LangGraph CompiledGraph (or anything with `.ainvoke` / `.invoke`).
    input_builder: `(example) -> dict` to shape the graph input. Default: `example.input`.
    output_extractor: `(graph_state) -> dict` to extract the final answer. Default: pass through.
    config_builder: `(example) -> dict` for the `config` argument (e.g. thread_id).
    max_concurrency: Cap concurrent invocations.
    """

    graph: Any = None
    input_builder: Callable[[EvalExample], Any] | None = None
    output_extractor: Callable[[Any], dict[str, Any]] | None = None
    config_builder: Callable[[EvalExample], dict[str, Any]] | None = None
    max_concurrency: int = 5

    name: str = "langgraph"
    _user_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        AgentRunner.__init__(self, max_concurrency=self.max_concurrency)
        if self.graph is None:
            raise ValueError("LangGraphRunner requires graph")

    async def run_one(self, example: EvalExample) -> RunResult:
        try:
            from langchain_core.tracers import RunCollectorCallbackHandler  # type: ignore
        except Exception:
            RunCollectorCallbackHandler = None  # type: ignore

        body = self.input_builder(example) if self.input_builder else example.input
        config = self.config_builder(example) if self.config_builder else {}
        config = dict(config or {})
        config.setdefault("metadata", {}).update({"eval_example_id": example.id, "eval_source": "agent-eval"})
        collector = None
        if RunCollectorCallbackHandler is not None:
            collector = RunCollectorCallbackHandler()
            config.setdefault("callbacks", []).append(collector)

        t_start = datetime.now(timezone.utc)
        try:
            if hasattr(self.graph, "ainvoke"):
                result = await self.graph.ainvoke(body, config=config)
            else:
                result = self.graph.invoke(body, config=config)
        except Exception as e:
            return RunResult(example_id=example.id, output=None, trace=None, error=f"{type(e).__name__}: {e}")
        t_end = datetime.now(timezone.utc)
        latency_ms = (t_end - t_start).total_seconds() * 1000.0

        output = self.output_extractor(result) if self.output_extractor else result
        if not isinstance(output, dict):
            output = {"output": output}

        trace = None
        if collector and collector.traced_runs:
            trace = collector.traced_runs[0]
        if trace is None:
            trace = build_synthetic_trace(output, inputs=body if isinstance(body, dict) else {"input": body},
                                           started_at=t_start, finished_at=t_end)
        return RunResult(example_id=example.id, output=output, trace=trace, latency_ms=latency_ms)
