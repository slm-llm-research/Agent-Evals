"""Pydantic schema for the discovered ComponentRegistry.

This is the canonical handoff between the discovery layer and the
evaluation layer: every evaluator reads from a `ComponentRegistry`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

DiscoverySource = Literal["mcp", "graph_introspection", "trace_mining", "manual"]
AgentType = Literal[
    "orchestrator",
    "react_agent",
    "tool_node",
    "simple_node",
    "subgraph",
    "unknown",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ToolInfo(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    is_llm_based: bool = False
    estimated_cost_per_call_usd: float | None = None
    source: DiscoverySource = "manual"


class AgentNodeInfo(BaseModel):
    name: str
    agent_type: AgentType = "unknown"
    tools: list[str] = Field(default_factory=list)
    is_entry_point: bool = False
    is_exit_point: bool = False
    has_self_loop: bool = False
    calls_subgraph: bool = False
    avg_latency_ms: float | None = None
    call_count: int | None = None
    source: DiscoverySource = "manual"


class MCPServerInfo(BaseModel):
    name: str
    url: str
    tools: list[str] = Field(default_factory=list)
    is_reachable: bool = False
    health_status: str = "unknown"
    last_seen_in_trace: datetime | None = None
    server_version: str | None = None


class MemoryBackendInfo(BaseModel):
    type: str  # chroma, pinecone, weaviate, qdrant, faiss, mem0, zep, ...
    connection_url: str | None = None
    call_frequency: int = 0
    avg_retrieval_latency_ms: float | None = None
    avg_results_per_query: float | None = None
    is_long_term: bool = False
    source: DiscoverySource = "trace_mining"


class RegistryDiff(BaseModel):
    added_agents: list[str] = Field(default_factory=list)
    removed_agents: list[str] = Field(default_factory=list)
    added_tools: list[str] = Field(default_factory=list)
    removed_tools: list[str] = Field(default_factory=list)
    schema_changed_tools: list[str] = Field(default_factory=list)


class ComponentRegistry(BaseModel):
    agents: list[AgentNodeInfo] = Field(default_factory=list)
    tools: list[ToolInfo] = Field(default_factory=list)
    mcp_servers: list[MCPServerInfo] = Field(default_factory=list)
    memory_backends: list[MemoryBackendInfo] = Field(default_factory=list)
    state_schema: dict[str, Any] | None = None
    entry_point: str | None = None
    discovery_method: Literal["mcp", "graph", "trace", "hybrid", "manual"] = "manual"
    discovered_at: datetime = Field(default_factory=_utcnow)

    @property
    def total_components(self) -> int:
        return len(self.agents) + len(self.tools) + len(self.mcp_servers) + len(self.memory_backends)

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    def hash(self) -> str:
        """Stable hash of the registry — used to bind a dataset to a system version."""
        canonical = json.dumps(
            {
                "agents": sorted(a.name for a in self.agents),
                "tools": sorted(t.name for t in self.tools),
                "mcp_servers": sorted(s.name for s in self.mcp_servers),
                "memory_backends": sorted(m.type for m in self.memory_backends),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def summary(self) -> str:
        lines = [
            "ComponentRegistry",
            f"  discovery_method: {self.discovery_method}",
            f"  agents:          {len(self.agents)}  ({', '.join(a.name for a in self.agents) or '—'})",
            f"  tools:           {len(self.tools)}  ({', '.join(t.name for t in self.tools) or '—'})",
            f"  mcp_servers:     {len(self.mcp_servers)}",
            f"  memory_backends: {len(self.memory_backends)}",
            f"  entry_point:     {self.entry_point or '—'}",
            f"  hash:            {self.hash()}",
        ]
        return "\n".join(lines)

    def get_evaluable_components(self) -> list[str]:
        """Components that should run through the evaluation suite."""
        return [a.name for a in self.agents] + [t.name for t in self.tools]

    def diff(self, other: ComponentRegistry) -> RegistryDiff:
        a_now = {a.name for a in self.agents}
        a_old = {a.name for a in other.agents}
        t_now = {t.name: t.input_schema for t in self.tools}
        t_old = {t.name: t.input_schema for t in other.tools}
        schema_changed = [
            n for n in (t_now.keys() & t_old.keys()) if t_now[n] != t_old[n]
        ]
        return RegistryDiff(
            added_agents=sorted(a_now - a_old),
            removed_agents=sorted(a_old - a_now),
            added_tools=sorted(set(t_now) - set(t_old)),
            removed_tools=sorted(set(t_old) - set(t_now)),
            schema_changed_tools=sorted(schema_changed),
        )
