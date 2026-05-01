"""Tests for the memory evaluators (full v2 suite)."""

from __future__ import annotations

import pytest

from agent_eval.dataset.schema import EvalExample
from agent_eval.evaluators.memory import (
    CrossSessionContinuityEvaluator,
    FactInjection,
    MemoryRetrievalRecallEvaluator,
    MemoryStalenessEvaluator,
    MemoryWriteQualityEvaluator,
    default_fact_harness,
    _classify_memory_op,
    _looks_like_durable_fact,
)


def test_classify_memory_op():
    assert _classify_memory_op("ChromaVectorStore.search") == "read"
    assert _classify_memory_op("MemoryStore.add") == "write"
    assert _classify_memory_op("MemoryStore.delete") == "delete"
    assert _classify_memory_op("Other.do") == "unknown"


def test_durable_fact_detection():
    assert _looks_like_durable_fact("My favorite color is teal.")
    assert _looks_like_durable_fact("I live in Vancouver.")
    assert _looks_like_durable_fact("My birthday is in May.")
    assert not _looks_like_durable_fact("What's the weather?")


@pytest.mark.asyncio
async def test_recall_no_memory_runs():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r", outputs={"answer": "ok"})
    res = await MemoryRetrievalRecallEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.passed
    assert "no memory ops" in res.details["reason"]


@pytest.mark.asyncio
async def test_recall_with_fact_harness():
    """Fact-harness mode: a runner that stores facts and recalls the matching one per query."""
    storage: dict[str, str] = {}
    last_stored = {"text": ""}  # track the most recent store, to associate with the next query

    async def runner(payload):
        if payload["op"] == "store":
            storage[payload["id"]] = payload["text"]
            last_stored["text"] = payload["text"]
            return None
        if payload["op"] == "query":
            # Match query to the most recently stored fact (1:1 store/query interleaving).
            return {"answer": f"Yes, {last_stored['text']}"}
        return None

    ev = MemoryRetrievalRecallEvaluator(fact_harness=runner)
    res = await ev.evaluate(EvalExample(input={"query": "?"}), None)
    assert res.score == 1.0
    assert res.details["mode"] == "fact_harness"


@pytest.mark.asyncio
async def test_write_quality_over_storing_detected():
    from tests.conftest import FakeRun

    # 10 writes (each contributes 1 message via inputs), 0 other messages.
    # write_ratio = 10 / 10 = 1.0 → over-storing (>0.8).
    children = [FakeRun(name="MemoryStore.add", run_type="tool", inputs={"text": f"chunk {i}"}) for i in range(10)]
    trace = FakeRun(name="r", children=children)
    res = await MemoryWriteQualityEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score < 0.7, f"expected over-storing penalty, got score={res.score}, details={res.details}"
    assert any("over-storing" in n for n in res.details["notes"])


@pytest.mark.asyncio
async def test_write_quality_under_storing_detected():
    from tests.conftest import FakeRun

    # Multiple durable-fact messages, no writes -> under-storing.
    children = [
        FakeRun(name="user_message", run_type="chain", outputs={"text": "My favorite color is teal."}),
        FakeRun(name="user_message", run_type="chain", outputs={"text": "I live in Vancouver."}),
        FakeRun(name="user_message", run_type="chain", outputs={"text": "My daughter is named Maya."}),
    ]
    trace = FakeRun(name="r", children=children)
    res = await MemoryWriteQualityEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score < 0.7
    assert any("under-storing" in n for n in res.details["notes"])


@pytest.mark.asyncio
async def test_staleness_no_reads():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r")
    res = await MemoryStalenessEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_cross_session_heuristic_no_history():
    from tests.conftest import FakeRun

    trace = FakeRun(name="r")
    res = await CrossSessionContinuityEvaluator().evaluate(EvalExample(input={"query": "?"}), trace)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_cross_session_with_session_runner():
    storage: dict[str, str] = {}

    async def runner(payload):
        sid = payload["session_id"]
        text = payload["input"]
        # Heuristic: questions are "queries", everything else is a "store".
        if text.strip().endswith("?"):
            # Query mode — pretend memory crosses sessions.
            for v in storage.values():
                if "teal" in v:
                    return {"answer": "Your favorite color is teal."}
            return {"answer": "I don't know."}
        # Store mode — remember the statement under this session id.
        storage[sid] = text.lower()
        return {"answer": "Got it."}

    ev = CrossSessionContinuityEvaluator(session_runner=runner)
    res = await ev.evaluate(EvalExample(input={"query": "?"}), None)
    # default_fact_harness()[0] = teal; should recall across sessions.
    assert res.passed, f"expected recall to succeed, got score={res.score}, details={res.details}"


def test_default_harness_facts_are_distinct():
    facts = default_fact_harness()
    ids = {f.fact_id for f in facts}
    assert len(ids) == len(facts)
    assert all(isinstance(f, FactInjection) for f in facts)
