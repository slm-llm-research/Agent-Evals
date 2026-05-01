"""LangSmith trace inspector — FALLBACK 2.

Mines production traces to infer:
  - node names + types
  - tool calls with sample-inferred input/output schemas
  - MCP server endpoints (HTTP-style tool calls)
  - transition matrix: node_a → node_b call counts
  - per-component stats (success rate, p95 latency, error_types breakdown, avg tokens)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent_eval.discovery.registry import (
    AgentNodeInfo,
    ComponentRegistry,
    MCPServerInfo,
    ToolInfo,
)


class ComponentStats(BaseModel):
    call_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = Field(default_factory=list)
    error_types: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = 0

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.call_count) if self.call_count else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.call_count if self.call_count else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = max(0, int(0.95 * len(sorted_l)) - 1)
        return sorted_l[idx]

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.call_count if self.call_count else 0.0


class TransitionMatrix(BaseModel):
    """node_a -> node_b -> count."""

    transitions: dict[str, dict[str, int]] = Field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def add(self, src: str, dst: str) -> None:
        self.transitions.setdefault(src, defaultdict(int))[dst] += 1

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.transitions.items()}


class LangSmithInspector:
    def __init__(self, client: Any, project_name: str):
        self.client = client
        self.project_name = project_name
        self._stats: dict[str, ComponentStats] = defaultdict(ComponentStats)

    def mine_components(self, n_traces: int = 100, lookback_days: int = 7) -> ComponentRegistry:
        traces = self._fetch_traces(n_traces, lookback_days)
        agents: dict[str, AgentNodeInfo] = {}
        tools: dict[str, ToolInfo] = {}
        mcp_servers: dict[str, MCPServerInfo] = {}
        transitions = TransitionMatrix()

        for trace in traces:
            self._walk_trace(trace, agents, tools, mcp_servers, transitions, parent_node=None)

        # Backfill stats into agent records.
        for name, info in agents.items():
            stats = self._stats.get(name)
            if stats and stats.call_count:
                info.avg_latency_ms = stats.avg_latency_ms
                info.call_count = stats.call_count

        registry = ComponentRegistry(
            agents=list(agents.values()),
            tools=list(tools.values()),
            mcp_servers=list(mcp_servers.values()),
            discovery_method="trace",
        )
        # Stash transitions on the registry as an extra field (not in the model — store on instance).
        registry._transition_matrix = transitions  # type: ignore[attr-defined]
        return registry

    def get_component_stats(self, component_name: str, n_traces: int = 100) -> dict[str, Any]:
        if not self._stats:
            self.mine_components(n_traces)
        s = self._stats.get(component_name)
        if not s:
            return {"call_count": 0}
        return {
            "call_count": s.call_count,
            "success_rate": round(s.success_rate, 3),
            "avg_latency_ms": round(s.avg_latency_ms, 2),
            "p95_latency_ms": round(s.p95_latency_ms, 2),
            "error_types": dict(s.error_types),
            "avg_tokens": round(s.avg_tokens, 2),
        }

    def detect_mcp_servers(self, traces: list[Any] | None = None) -> list[MCPServerInfo]:
        traces = traces if traces is not None else self._fetch_traces(100)
        servers: dict[str, MCPServerInfo] = {}
        for trace in traces:
            for run in self._iter_runs(trace):
                name = run.get("name") or ""
                lname = name.lower()
                # Heuristics:
                #   - run name starts with "mcp." or contains "mcp_"
                #   - tool with HTTP-style endpoint URL in inputs.metadata
                if lname.startswith("mcp.") or "mcp" in lname or "mcp_" in lname:
                    if name not in servers:
                        servers[name] = MCPServerInfo(name=name, url="", tools=[name], is_reachable=True)
                inputs = run.get("inputs") or {}
                if isinstance(inputs, dict):
                    url = inputs.get("url") or inputs.get("endpoint")
                    if isinstance(url, str) and ("/mcp" in url or "mcp." in url):
                        if url not in servers:
                            servers[url] = MCPServerInfo(name=url, url=url, tools=[], is_reachable=True)
        return list(servers.values())

    def infer_agent_structure(self, traces: list[Any] | None = None) -> dict[str, Any]:
        """Returns dict with:
          - transition_matrix (dict[src, dict[dst, count]])
          - entry_points (set of nodes that appear as roots)
          - exit_points (nodes that don't transition to anything)
        """
        traces = traces if traces is not None else self._fetch_traces(100)
        transitions = TransitionMatrix()
        roots: set[str] = set()
        all_dsts: set[str] = set()
        for trace in traces:
            self._walk_trace(trace, {}, {}, {}, transitions, parent_node=None, root_collector=roots)
            for srcs in transitions.transitions.values():
                all_dsts.update(srcs.keys())
        all_srcs = set(transitions.transitions.keys())
        return {
            "transition_matrix": transitions.to_dict(),
            "entry_points": sorted(roots),
            "exit_points": sorted(all_dsts - all_srcs),
        }

    # ------------------------------------------------------------ internal

    def _fetch_traces(self, n: int, lookback_days: int = 7) -> list[Any]:
        try:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            return list(self.client.list_runs(
                project_name=self.project_name,
                start_time=since,
                run_type="chain",
                limit=n,
            ))
        except Exception:
            return []

    def _iter_runs(self, trace: Any):
        stack = [trace]
        while stack:
            run = stack.pop()
            yield self._run_to_dict(run)
            for c in getattr(run, "child_runs", None) or []:
                stack.append(c)

    def _run_to_dict(self, run: Any) -> dict[str, Any]:
        return {
            "name": getattr(run, "name", None),
            "run_type": getattr(run, "run_type", None),
            "inputs": getattr(run, "inputs", None) or {},
            "outputs": getattr(run, "outputs", None) or {},
            "error": getattr(run, "error", None),
            "start_time": getattr(run, "start_time", None),
            "end_time": getattr(run, "end_time", None),
            "total_tokens": getattr(run, "total_tokens", None),
            "_obj": run,
        }

    def _latency_ms(self, run: dict[str, Any]) -> float | None:
        s, e = run.get("start_time"), run.get("end_time")
        if isinstance(s, datetime) and isinstance(e, datetime):
            return (e - s).total_seconds() * 1000.0
        return None

    def _walk_trace(
        self,
        trace: Any,
        agents: dict[str, AgentNodeInfo],
        tools: dict[str, ToolInfo],
        mcp_servers: dict[str, MCPServerInfo],
        transitions: TransitionMatrix,
        parent_node: str | None,
        root_collector: set[str] | None = None,
    ) -> None:
        run = self._run_to_dict(trace)
        name = run.get("name")
        if not name:
            return
        run_type = run.get("run_type") or ""
        if run_type == "tool":
            if name not in tools:
                schema = _infer_schema_from_sample(run.get("inputs") or {})
                output_schema = _infer_schema_from_sample(run.get("outputs") or {})
                tools[name] = ToolInfo(
                    name=name,
                    description="(inferred from trace)",
                    input_schema=schema,
                    output_schema=output_schema,
                    source="trace_mining",
                )
            if name.lower().startswith("mcp.") or "mcp" in name.lower():
                mcp_servers.setdefault(name, MCPServerInfo(name=name, url="", tools=[name], is_reachable=True))
        elif run_type in ("chain", "agent", "graph"):
            if name not in agents and name not in ("__start__", "__end__"):
                agents[name] = AgentNodeInfo(name=name, source="trace_mining")
            if root_collector is not None and parent_node is None:
                root_collector.add(name)
            if parent_node and parent_node != name:
                transitions.add(parent_node, name)

        # Update stats.
        s = self._stats[name]
        s.call_count += 1
        if not run.get("error"):
            s.success_count += 1
        else:
            err = run["error"]
            etype = type(err).__name__ if not isinstance(err, str) else "Error"
            s.error_types[etype] = s.error_types.get(etype, 0) + 1
        latency = self._latency_ms(run)
        if latency is not None:
            s.total_latency_ms += latency
            s.latencies.append(latency)
        if run.get("total_tokens"):
            s.total_tokens += int(run["total_tokens"])

        # Recurse.
        next_parent = name if run_type in ("chain", "agent", "graph") else parent_node
        for child in getattr(trace, "child_runs", None) or []:
            self._walk_trace(child, agents, tools, mcp_servers, transitions, parent_node=next_parent, root_collector=root_collector)


def _infer_schema_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return {}
    props = {}
    for k, v in sample.items():
        if isinstance(v, bool):
            t = "boolean"
        elif isinstance(v, int):
            t = "integer"
        elif isinstance(v, float):
            t = "number"
        elif isinstance(v, list):
            t = "array"
        elif isinstance(v, dict):
            t = "object"
        else:
            t = "string"
        props[k] = {"type": t}
    return {"type": "object", "properties": props}
