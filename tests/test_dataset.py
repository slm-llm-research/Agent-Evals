"""Tests for dataset schema, templates, and harvester."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.dataset.schema import EvalDataset, EvalExample
from agent_eval.dataset.trace_harvester import TraceHarvester


@pytest.mark.parametrize("template", ["search_agent", "research_agent", "voice_agent", "general_agent"])
def test_template_loads_and_has_min_examples(template):
    ds = EvalDataset.from_template(template)
    assert len(ds.examples) >= 10 or template == "general_agent"
    # all examples should have an input
    for ex in ds.examples:
        assert ex.input
        assert "query" in ex.input


def test_dataset_split():
    ds = EvalDataset.from_template("search_agent")
    train, test = ds.split(train_ratio=0.7)
    assert len(train.examples) + len(test.examples) >= len(ds.examples) - 1
    assert len(train.examples) > 0 and len(test.examples) > 0


def test_dataset_filter():
    ds = EvalDataset.from_template("search_agent")
    adversarial = ds.filter(query_type="adversarial")
    assert all(e.query_type == "adversarial" for e in adversarial.examples)


def test_dataset_save_and_load(tmp_path):
    ds = EvalDataset.from_template("voice_agent")
    p = tmp_path / "ds.json"
    ds.save(p)
    loaded = EvalDataset.load(p)
    assert loaded.name == ds.name
    assert len(loaded.examples) == len(ds.examples)


def test_dataset_from_template_invalid():
    with pytest.raises(ValueError):
        EvalDataset.from_template("nonexistent_template")


def test_trace_harvester_dedup():
    harvester = TraceHarvester(client=None)
    ex1 = EvalExample(input={"query": "what is X"})
    ex2 = EvalExample(input={"query": "what is X"})
    ex3 = EvalExample(input={"query": "what is Y"})
    out = harvester.deduplicate([ex1, ex2, ex3])
    assert len(out) == 2
