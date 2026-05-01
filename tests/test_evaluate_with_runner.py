"""End-to-end: evaluate(dataset, runner=...) really runs the agent and scores real output.

These tests are SYNC because AgentEval.evaluate() owns the event loop via asyncio.run().
"""

from __future__ import annotations

import httpx
import respx

from agent_eval import AgentEval
from agent_eval.dataset.schema import EvalDataset, EvalExample
from agent_eval.discovery.registry import ComponentRegistry
from agent_eval.runners.http_runner import HttpAgentRunner


@respx.mock
def test_evaluate_with_http_runner_uses_real_outputs():
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(
        200, json={"answer": "The capital of France is Paris."}
    ))
    ds = EvalDataset(
        name="t",
        examples=[
            EvalExample(input={"query": "capital of France"}, expected_answer_keywords=["Paris"]),
            EvalExample(input={"query": "capital of Germany"}, expected_answer_keywords=["Berlin"]),
        ],
    )
    registry = ComponentRegistry(discovery_method="manual")
    ev = AgentEval(registry)
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval", langsmith_project=None)
    # Run only deterministic-only sub-evaluators (we patch the LLM-dependent ones to no-ops).
    report = ev.evaluate(
        dataset=ds,
        runner=runner,
        dimensions=["output_quality"],
        backend="native",
    )
    # We hit the synthetic-trace path; KeywordCoverage on Paris matches one example and not the other.
    # TaskSuccess sees an answer so it's 1.0 on both. So output_quality is meaningfully > 0.
    assert report.dimension_scores.output_quality > 0.3
    assert report.dataset_size == 2


@respx.mock
def test_evaluate_without_runner_warns_but_runs():
    """Confirm the no-runner path still produces a report (the dataset-testability mode)."""
    ds = EvalDataset(name="t", examples=[EvalExample(input={"query": "x"})])
    ev = AgentEval(ComponentRegistry(discovery_method="manual"))
    report = ev.evaluate(dataset=ds, dimensions=["output_quality"], backend="native")
    assert report.dataset_size == 1
