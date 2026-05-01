"""Tests for AgentRunner implementations."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from agent_eval.dataset.schema import EvalDataset, EvalExample
from agent_eval.runners.http_runner import HttpAgentRunner
from agent_eval.runners.langgraph_runner import LangGraphRunner
from agent_eval.runners.langsmith_replay import LangSmithReplayRunner
from agent_eval.runners.synthetic import SyntheticTrace, build_synthetic_trace


# ---------------------------------------------------------------------------- synthetic


def test_synthetic_trace_from_string():
    t = build_synthetic_trace("Paris is the capital of France.")
    assert t.outputs["answer"] == "Paris is the capital of France."
    assert t.run_type == "chain"
    assert t._is_synthetic is True


def test_synthetic_trace_from_dict_with_tool_calls():
    t = build_synthetic_trace({
        "answer": "Paris.",
        "tool_calls": [{"name": "web_search", "inputs": {"q": "capital France"}, "outputs": {"results": ["Paris"]}}],
    })
    assert t.outputs["answer"] == "Paris."
    assert len(t.child_runs) == 1
    assert t.child_runs[0].name == "web_search"
    assert t.child_runs[0].run_type == "tool"


# ---------------------------------------------------------------------------- HTTP


@pytest.mark.asyncio
@respx.mock
async def test_http_runner_basic():
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(
        200, json={"answer": "Paris", "langsmith_run_id": "abc123"}
    ))
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval", langsmith_project=None)
    ex = EvalExample(input={"query": "capital of France"})
    res = await runner.run_one(ex)
    await runner.aclose()
    assert res.error is None
    assert res.output["answer"] == "Paris"
    assert res.metadata["run_id"] == "abc123"
    # No real LangSmith available -> falls back to synthetic trace.
    assert getattr(res.trace, "_is_synthetic", False) is True


@pytest.mark.asyncio
@respx.mock
async def test_http_runner_handles_500():
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(500, json={"error": "boom"}))
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval")
    ex = EvalExample(input={"query": "?"})
    res = await runner.run_one(ex)
    await runner.aclose()
    assert res.error == "http_500"
    assert res.trace is not None  # synthetic trace from the 500 body


@pytest.mark.asyncio
@respx.mock
async def test_http_runner_run_dataset_concurrency():
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        return httpx.Response(200, json={"answer": f"resp-{call_count['n']}"})

    respx.post("http://api.test/eval").mock(side_effect=handler)
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval", max_concurrency=3)
    ds = EvalDataset(
        name="t",
        examples=[EvalExample(input={"query": f"q{i}"}) for i in range(5)],
    )
    results = await runner.run_dataset(ds)
    await runner.aclose()
    assert len(results) == 5
    assert call_count["n"] == 5
    assert all(r.error is None for r in results)


@pytest.mark.asyncio
@respx.mock
async def test_http_runner_extracts_run_id_from_header():
    respx.post("http://api.test/eval").mock(return_value=httpx.Response(
        200,
        json={"answer": "ok"},
        headers={"X-Langsmith-Run-Id": "header-run-456"},
    ))
    runner = HttpAgentRunner(endpoint_url="http://api.test/eval")
    res = await runner.run_one(EvalExample(input={"query": "?"}))
    await runner.aclose()
    assert res.metadata["run_id"] == "header-run-456"


# ---------------------------------------------------------------------------- LangGraph


class _FakeGraph:
    """Mimics the minimal CompiledGraph surface (`.invoke`)."""

    def __init__(self, response):
        self.response = response
        self.last_config = None

    def invoke(self, body, config=None):
        self.last_config = config
        return self.response


@pytest.mark.asyncio
async def test_langgraph_runner_invokes_graph():
    graph = _FakeGraph(response={"answer": "Paris"})
    runner = LangGraphRunner(graph=graph)
    res = await runner.run_one(EvalExample(input={"query": "capital?"}))
    assert res.output == {"answer": "Paris"}
    assert res.error is None
    # Metadata should include eval_example_id.
    assert graph.last_config["metadata"]["eval_example_id"] == res.example_id


@pytest.mark.asyncio
async def test_langgraph_runner_handles_exception():
    class BadGraph:
        def invoke(self, body, config=None):
            raise RuntimeError("graph blew up")

    runner = LangGraphRunner(graph=BadGraph())
    res = await runner.run_one(EvalExample(input={"query": "?"}))
    assert res.error is not None
    assert "graph blew up" in res.error


# ---------------------------------------------------------------------------- LangSmith replay


class _FakeRun:
    def __init__(self, id, inputs=None, outputs=None, metadata=None, error=None):
        self.id = id
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.error = error
        self.extra = {"metadata": metadata or {}}


class _FakeLsClient:
    def __init__(self, runs):
        self._runs = runs
        self.read_run_calls = []

    def list_runs(self, **kwargs):
        return iter(self._runs)

    def read_run(self, run_id, load_child_runs=False):
        self.read_run_calls.append(run_id)
        for r in self._runs:
            if str(r.id) == str(run_id):
                return r
        raise LookupError(run_id)


@pytest.mark.asyncio
async def test_replay_runner_metadata_match():
    runs = [
        _FakeRun(id="r1", inputs={"query": "Q1"}, outputs={"answer": "A1"}, metadata={"eval_example_id": "ex1"}),
        _FakeRun(id="r2", inputs={"query": "Q2"}, outputs={"answer": "A2"}),
    ]
    runner = LangSmithReplayRunner(project_name="p", client=_FakeLsClient(runs))
    res = await runner.run_one(EvalExample(id="ex1", input={"query": "Q1"}))
    assert res.error is None
    assert res.output == {"answer": "A1"}


@pytest.mark.asyncio
async def test_replay_runner_substring_match():
    runs = [
        _FakeRun(id="r1", inputs={"query": "What is the capital of France?"}, outputs={"answer": "Paris"}),
    ]
    runner = LangSmithReplayRunner(project_name="p", client=_FakeLsClient(runs))
    res = await runner.run_one(EvalExample(input={"query": "What is the capital of France?"}))
    assert res.error is None
    assert res.output == {"answer": "Paris"}


@pytest.mark.asyncio
async def test_replay_runner_no_match():
    runs = [_FakeRun(id="r1", inputs={"query": "completely different"}, outputs={"answer": "x"})]
    runner = LangSmithReplayRunner(project_name="p", client=_FakeLsClient(runs), embedding_threshold=0.99)
    res = await runner.run_one(EvalExample(input={"query": "what is the capital of France?"}))
    assert res.error is not None
    assert "no matching trace" in res.error
