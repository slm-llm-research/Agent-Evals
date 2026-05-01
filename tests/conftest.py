"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest


class FakeRun:
    """Mimics a LangSmith run object with the attributes we use."""

    def __init__(
        self,
        name: str,
        run_type: str = "chain",
        inputs: dict | None = None,
        outputs: dict | None = None,
        error: Any = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        children: list[FakeRun] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ):
        self.name = name
        self.run_type = run_type
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.error = error
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time = end_time or self.start_time + timedelta(milliseconds=100)
        self.child_runs = children or []
        self.id = f"fake-{id(self):x}"
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


@pytest.fixture
def fake_trace_simple():
    return FakeRun(
        name="root",
        outputs={"answer": "The capital of France is Paris."},
        children=[
            FakeRun(
                name="web_search",
                run_type="tool",
                inputs={"query": "capital of France"},
                outputs={"results": ["Paris is the capital of France."]},
            ),
        ],
    )


@pytest.fixture
def fake_trace_research():
    """Multi-step research trace."""
    return FakeRun(
        name="research_agent",
        run_type="agent",
        outputs={"final_answer": "US emissions: X. China: Y. India: Z."},
        children=[
            FakeRun(name="plan", run_type="chain", outputs={"plan": "search US, China, India"}),
            FakeRun(name="web_search", run_type="tool", inputs={"q": "US emissions 2024"}, outputs={"results": ["US emitted X tons CO2 in 2024."]}),
            FakeRun(name="web_search", run_type="tool", inputs={"q": "China emissions 2024"}, outputs={"results": ["China emitted Y tons CO2 in 2024."]}),
            FakeRun(name="web_search", run_type="tool", inputs={"q": "India emissions 2024"}, outputs={"results": ["India emitted Z tons CO2 in 2024."]}),
            FakeRun(name="summarize", run_type="tool", inputs={"text": "..."}, outputs={"summary": "..."}),
        ],
    )
