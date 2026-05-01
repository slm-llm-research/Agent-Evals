"""MCP inspector — PRIMARY discovery method.

Connects to a running MCP server, lists tools/resources/prompts via the
official `mcp` Python SDK when available, and falls back to plain HTTP
calls against `/registry`-style endpoints (e.g. AgentApp's
`GET /mcp/registry`) when the SDK is not installed.

The MCP path is preferred because it is framework-agnostic, requires no
source code, and the schemas are guaranteed live.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from agent_eval.discovery.registry import (
    AgentNodeInfo,
    ComponentRegistry,
    MCPServerInfo,
    ToolInfo,
)


class MCPInspector:
    def __init__(self, mcp_url: str, timeout: float = 10.0):
        self.mcp_url = mcp_url.rstrip("/")
        self.timeout = timeout

    async def health_check(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(self.mcp_url)
                latency_ms = (time.perf_counter() - start) * 1000.0
                version = None
                tool_count = None
                try:
                    body = resp.json()
                    version = body.get("server_version") or body.get("version")
                    tools = body.get("tools")
                    if isinstance(tools, list):
                        tool_count = len(tools)
                except Exception:
                    pass
                return {
                    "is_reachable": resp.status_code < 500,
                    "latency_ms": latency_ms,
                    "server_version": version,
                    "tool_count": tool_count,
                    "status_code": resp.status_code,
                }
        except Exception as e:
            return {
                "is_reachable": False,
                "latency_ms": (time.perf_counter() - start) * 1000.0,
                "server_version": None,
                "tool_count": None,
                "error": str(e),
            }

    async def inspect(self) -> ComponentRegistry:
        """Discover the agent system via MCP.

        Strategy:
        1. Try the registry-document endpoint (`{mcp_url}/registry` or `{mcp_url}` if it returns
           a registry document). This is the AgentApp convention from Instruction Set 1.
        2. Fall back to the official `mcp` SDK to call `list_tools()` / `list_resources()` /
           `list_prompts()` over HTTP transport.
        3. If neither works, return an empty registry — the caller can then fall back
           to GraphInspector / LangSmithInspector.
        """
        registry = await self._try_registry_document()
        if registry is not None and registry.total_components > 0:
            return registry

        sdk_registry = await self._try_mcp_sdk()
        if sdk_registry is not None and sdk_registry.total_components > 0:
            return sdk_registry

        return ComponentRegistry(discovery_method="mcp")

    # ---------------------------------------------------------------- helpers

    async def _try_registry_document(self) -> ComponentRegistry | None:
        candidate_urls = [f"{self.mcp_url}/registry", self.mcp_url]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in candidate_urls:
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue
                    body = resp.json()
                    if not isinstance(body, dict):
                        continue
                    parsed = self._parse_registry_document(body)
                    if parsed.total_components > 0:
                        return parsed
                except Exception:
                    continue
        return None

    def _parse_registry_document(self, body: dict[str, Any]) -> ComponentRegistry:
        tools = []
        for t in body.get("tools", []) or []:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not name:
                continue
            desc = t.get("description", "") or ""
            schema = t.get("input_schema") or t.get("inputSchema") or {}
            tools.append(
                ToolInfo(
                    name=name,
                    description=desc,
                    input_schema=schema if isinstance(schema, dict) else {},
                    output_schema=t.get("output_schema") or t.get("outputSchema"),
                    is_llm_based=_looks_llm(name, desc),
                    source="mcp",
                )
            )

        agents = []
        for a in body.get("agents", []) or []:
            if isinstance(a, str):
                agents.append(AgentNodeInfo(name=a, source="mcp"))
            elif isinstance(a, dict) and a.get("name"):
                agents.append(
                    AgentNodeInfo(
                        name=a["name"],
                        agent_type=a.get("agent_type", "unknown"),
                        tools=a.get("tools", []) or [],
                        is_entry_point=bool(a.get("is_entry_point", False)),
                        source="mcp",
                    )
                )

        mcp_servers = []
        for s in body.get("mcp_servers", []) or []:
            if not isinstance(s, dict) or not s.get("name"):
                continue
            mcp_servers.append(
                MCPServerInfo(
                    name=s["name"],
                    url=s.get("url", ""),
                    tools=s.get("tools", []) or [],
                    is_reachable=bool(s.get("is_reachable", False)),
                    server_version=s.get("server_version"),
                )
            )

        return ComponentRegistry(
            agents=agents,
            tools=tools,
            mcp_servers=mcp_servers,
            entry_point=body.get("entry_point"),
            discovery_method="mcp",
        )

    async def _try_mcp_sdk(self) -> ComponentRegistry | None:
        try:
            from mcp import ClientSession  # type: ignore
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore
        except Exception:
            return None
        try:
            async with streamablehttp_client(self.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    tools = []
                    for t in getattr(tools_resp, "tools", []):
                        name = getattr(t, "name", None)
                        if not name:
                            continue
                        desc = getattr(t, "description", "") or ""
                        schema = getattr(t, "inputSchema", None) or {}
                        tools.append(
                            ToolInfo(
                                name=name,
                                description=desc,
                                input_schema=schema if isinstance(schema, dict) else {},
                                is_llm_based=_looks_llm(name, desc),
                                source="mcp",
                            )
                        )
                    return ComponentRegistry(tools=tools, discovery_method="mcp")
        except Exception:
            return None


def _looks_llm(name: str, description: str) -> bool:
    blob = f"{name} {description}".lower()
    return any(kw in blob for kw in ("llm", "gpt", "claude", "chat", "summari", "complet"))
