"""Runners — execute the agent against a dataset, return trace objects.

A runner is the bridge between an EvalDataset and an EvaluationReport.
Three implementations:

  HttpAgentRunner       — POST each example to the agent's HTTP endpoint
  LangGraphRunner       — invoke a CompiledGraph in-process
  LangSmithReplayRunner — match dataset inputs to existing production traces

All return a uniform `(output_dict, trace_object)` per example. When a real
LangSmith trace cannot be recovered, the runner returns a `SyntheticTrace`
shim so the output-/safety-side evaluators still work.
"""

from agent_eval.runners.base import AgentRunner, RunResult
from agent_eval.runners.http_runner import HttpAgentRunner
from agent_eval.runners.langgraph_runner import LangGraphRunner
from agent_eval.runners.langsmith_replay import LangSmithReplayRunner
from agent_eval.runners.synthetic import SyntheticTrace, build_synthetic_trace

__all__ = [
    "AgentRunner",
    "RunResult",
    "HttpAgentRunner",
    "LangGraphRunner",
    "LangSmithReplayRunner",
    "SyntheticTrace",
    "build_synthetic_trace",
]
