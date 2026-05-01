"""Tests for the ComponentRegistry."""

from __future__ import annotations

from agent_eval.discovery.registry import (
    AgentNodeInfo,
    ComponentRegistry,
    MCPServerInfo,
    MemoryBackendInfo,
    ToolInfo,
)


def _make(tools=("a", "b"), agents=("orch",)):
    return ComponentRegistry(
        agents=[AgentNodeInfo(name=a) for a in agents],
        tools=[ToolInfo(name=t, description=f"{t} desc") for t in tools],
        mcp_servers=[MCPServerInfo(name="mcp1", url="http://x", tools=list(tools))],
        memory_backends=[MemoryBackendInfo(type="chroma")],
        discovery_method="mcp",
    )


def test_total_components():
    r = _make()
    assert r.total_components == len(r.agents) + len(r.tools) + len(r.mcp_servers) + len(r.memory_backends)


def test_summary_renders():
    r = _make()
    s = r.summary()
    assert "ComponentRegistry" in s
    assert "agents" in s
    assert "hash" in s


def test_hash_stable_and_different():
    r1 = _make(tools=("a", "b"))
    r2 = _make(tools=("a", "b"))
    r3 = _make(tools=("a", "c"))
    assert r1.hash() == r2.hash()
    assert r1.hash() != r3.hash()


def test_diff_detects_added_removed_tools():
    old = _make(tools=("a", "b"))
    new = _make(tools=("b", "c"))
    diff = new.diff(old)
    assert "c" in diff.added_tools
    assert "a" in diff.removed_tools
    assert "b" not in diff.added_tools


def test_diff_detects_schema_change():
    a_old = ToolInfo(name="a", input_schema={"type": "object", "properties": {"q": {"type": "string"}}})
    a_new = ToolInfo(name="a", input_schema={"type": "object", "properties": {"q": {"type": "integer"}}})
    old = ComponentRegistry(tools=[a_old], discovery_method="mcp")
    new = ComponentRegistry(tools=[a_new], discovery_method="mcp")
    diff = new.diff(old)
    assert "a" in diff.schema_changed_tools


def test_get_evaluable_components():
    r = _make()
    names = r.get_evaluable_components()
    assert "orch" in names
    assert "a" in names and "b" in names
