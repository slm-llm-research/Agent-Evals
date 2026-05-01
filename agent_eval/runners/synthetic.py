"""Synthetic trace shim — wraps an agent response when no LangSmith run is available.

Lets the output-/safety-/format-side evaluators score the response even when
we couldn't recover the actual LangSmith trace (e.g., agent ran offline,
LangSmith ingestion lag exceeded our wait, or the agent doesn't ship traces).

Trajectory / tool / system evaluators will degrade to "no trace" mode and
either skip or return their default-1.0 score; the report should make this
obvious via the `details.reason` field on each result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class SyntheticTrace:
    """Duck-typed to match the subset of langsmith.Run we read in evaluators."""

    def __init__(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any] | None = None,
        error: Any = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        name: str = "synthetic_root",
    ):
        self.id = "synthetic"
        self.name = name
        self.run_type = "chain"
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.error = error
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time = end_time or self.start_time
        self.child_runs: list[Any] = []
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self._is_synthetic = True

        for tc in tool_calls or []:
            self.child_runs.append(_SyntheticToolRun(tc, parent_start=self.start_time))


class _SyntheticToolRun:
    def __init__(self, payload: dict[str, Any], parent_start: datetime):
        self.id = "synthetic-tool"
        self.name = payload.get("name", "unknown_tool")
        self.run_type = "tool"
        self.inputs = payload.get("inputs") or payload.get("input") or {}
        self.outputs = payload.get("outputs") or payload.get("output") or {}
        if isinstance(self.outputs, str):
            self.outputs = {"output": self.outputs}
        self.error = payload.get("error")
        self.start_time = payload.get("start_time") or parent_start
        self.end_time = payload.get("end_time") or self.start_time
        self.child_runs: list[Any] = []


def build_synthetic_trace(
    response: dict[str, Any] | str,
    inputs: dict[str, Any] | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SyntheticTrace:
    """Best-effort: wrap any response shape as a SyntheticTrace.

    If `response` is a dict and contains a 'tool_calls' / 'tool_trace' / 'steps' key, those
    are surfaced as synthetic tool runs so trajectory evaluators can also score.
    """
    if isinstance(response, str):
        outputs = {"answer": response}
    elif isinstance(response, dict):
        outputs = dict(response)
    else:
        outputs = {"output": str(response)}

    if tool_calls is None and isinstance(response, dict):
        for k in ("tool_calls", "tool_trace", "steps", "tools_used"):
            v = response.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                tool_calls = v
                break

    return SyntheticTrace(
        outputs=outputs,
        inputs=inputs,
        start_time=started_at,
        end_time=finished_at,
        tool_calls=tool_calls,
    )
