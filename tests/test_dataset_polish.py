"""Tests for dataset polish: harvester dedup + auto-gen validation."""

from __future__ import annotations

from agent_eval.dataset.auto_generator import AutoDatasetGenerator
from agent_eval.dataset.schema import EvalDataset, EvalExample
from agent_eval.dataset.trace_harvester import TraceHarvester, _to_query_text
from agent_eval.discovery.registry import ComponentRegistry, ToolInfo


def test_to_query_text():
    assert _to_query_text({"query": "Hello"}) == "Hello"
    assert _to_query_text({"question": "Why?"}) == "Why?"
    assert _to_query_text("raw string") == "raw string"


def test_harvester_dedup_exact_hash():
    h = TraceHarvester(client=None)
    examples = [
        EvalExample(input={"query": "x"}),
        EvalExample(input={"query": "x"}),  # exact dup
        EvalExample(input={"query": "y"}),
    ]
    out = h.deduplicate(examples, embedding_threshold=0.0)
    assert len(out) == 2


def test_validate_dataset_flags_low_diversity():
    registry = ComponentRegistry(discovery_method="manual")
    gen = AutoDatasetGenerator(registry)
    ds = EvalDataset(name="t", examples=[
        EvalExample(input={"query": f"q{i}"}, query_type="general", complexity="simple")
        for i in range(20)
    ])
    report = gen.validate_dataset(ds)
    assert not report.passed
    assert any("query_type" in i for i in report.issues)
    assert any("complexity" in i for i in report.issues)


def test_validate_dataset_flags_missing_capability_coverage():
    registry = ComponentRegistry(tools=[ToolInfo(name="some_tool")], discovery_method="manual")
    gen = AutoDatasetGenerator(registry)
    ds = EvalDataset(name="t", examples=[
        EvalExample(input={"query": "q1"}, query_type="general"),
        EvalExample(input={"query": "q2"}, query_type="adversarial"),
    ])
    report = gen.validate_dataset(ds)
    assert any("some_tool" in i for i in report.issues)


def test_cost_estimate_includes_breakdown():
    registry = ComponentRegistry(discovery_method="manual")
    gen = AutoDatasetGenerator(registry)
    ds = EvalDataset(name="t", examples=[EvalExample(input={"query": f"q{i}"}) for i in range(5)])
    est = gen.estimate_evaluation_cost(ds)
    assert est.n_examples == 5
    assert "input_usd" in est.breakdown
    assert "model" in est.breakdown
    assert est.estimated_usd > 0


def test_cost_estimate_increases_with_memory_backend():
    from agent_eval.discovery.registry import MemoryBackendInfo

    base = ComponentRegistry(discovery_method="manual")
    with_mem = ComponentRegistry(memory_backends=[MemoryBackendInfo(type="chroma")], discovery_method="manual")
    ds = EvalDataset(name="t", examples=[EvalExample(input={"query": f"q{i}"}) for i in range(5)])
    base_cost = AutoDatasetGenerator(base).estimate_evaluation_cost(ds).estimated_usd
    mem_cost = AutoDatasetGenerator(with_mem).estimate_evaluation_cost(ds).estimated_usd
    assert mem_cost > base_cost
