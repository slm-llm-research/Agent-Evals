"""Tests for MCP inspector — registry-document path."""

from __future__ import annotations

import pytest

from agent_eval.discovery.mcp_inspector import MCPInspector


@pytest.mark.asyncio
async def test_mcp_inspector_unreachable():
    """Unreachable URL should return an empty registry, not raise."""
    insp = MCPInspector("http://127.0.0.1:1/", timeout=0.5)
    reg = await insp.inspect()
    assert reg.total_components == 0
    assert reg.discovery_method == "mcp"


@pytest.mark.asyncio
async def test_mcp_inspector_health_check_unreachable():
    insp = MCPInspector("http://127.0.0.1:1/", timeout=0.5)
    health = await insp.health_check()
    assert health["is_reachable"] is False


def test_parse_registry_document_minimal():
    insp = MCPInspector("http://x")
    body = {
        "tools": [
            {"name": "web_search", "description": "search", "input_schema": {"type": "object"}},
            {"name": "summarize_llm", "description": "uses llm to summarize"},
        ],
        "agents": [{"name": "orchestrator", "agent_type": "orchestrator", "is_entry_point": True, "tools": ["web_search"]}],
        "mcp_servers": [],
        "entry_point": "orchestrator",
    }
    reg = insp._parse_registry_document(body)
    assert len(reg.tools) == 2
    assert len(reg.agents) == 1
    assert reg.entry_point == "orchestrator"
    assert any(t.is_llm_based for t in reg.tools)  # 'llm' in description should flag it
