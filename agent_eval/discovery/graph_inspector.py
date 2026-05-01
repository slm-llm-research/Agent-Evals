"""LangGraph inspector — FALLBACK 1.

Deep introspection of a CompiledGraph:
  - node names + types
  - edges (regular + conditional, including condition function name)
  - tools bound to each LLM/agent node
  - state schema (Pydantic JSON Schema)
  - agent type heuristics: ORCHESTRATOR / REACT_AGENT / TOOL_NODE / SIMPLE_NODE / SUBGRAPH
  - entry / exit points
  - self-loops and subgraph references
"""

from __future__ import annotations

import inspect
from typing import Any

from agent_eval.discovery.registry import (
    AgentNodeInfo,
    AgentType,
    ComponentRegistry,
    ToolInfo,
)


class GraphInspector:
    def __init__(self, graph: Any):
        self.graph = graph

    def inspect(self) -> ComponentRegistry:
        agents: list[AgentNodeInfo] = []
        tools: dict[str, ToolInfo] = {}
        entry_point: str | None = None

        nodes_dict, edges, conditional_edges = self._extract_graph_structure()
        if not nodes_dict:
            return ComponentRegistry(discovery_method="graph")

        # Entry point — first node whose source is __start__.
        for src, dst in edges:
            if src == "__start__":
                entry_point = dst
                break

        # Self-loops + subgraph detection.
        self_looping = {src for src, dst in edges if src == dst}
        for src, branches in conditional_edges.items():
            if any(d == src for d in branches.values()):
                self_looping.add(src)
        all_dsts = {dst for _src, dst in edges} | {d for branches in conditional_edges.values() for d in branches.values()}
        all_srcs = {src for src, _dst in edges} | set(conditional_edges.keys())

        for name, node in nodes_dict.items():
            if name in ("__start__", "__end__"):
                continue
            agent_type = self._detect_agent_type(name, node)
            bound = self._extract_bound_tools(node)
            for t in bound:
                tools.setdefault(t.name, t)
            calls_subgraph = self._references_subgraph(node)
            is_entry = name == entry_point
            is_exit = name not in all_srcs or name == "__end__"
            agents.append(
                AgentNodeInfo(
                    name=name,
                    agent_type=agent_type,
                    tools=[t.name for t in bound],
                    is_entry_point=is_entry,
                    is_exit_point=is_exit,
                    has_self_loop=name in self_looping,
                    calls_subgraph=calls_subgraph,
                    source="graph_introspection",
                )
            )

        # State schema.
        state_schema: dict[str, Any] | None = None
        for cand in ("schema", "input_schema", "state_schema"):
            sc = getattr(self.graph, cand, None)
            if sc is not None and hasattr(sc, "model_json_schema"):
                try:
                    state_schema = sc.model_json_schema()
                    break
                except Exception:
                    continue

        # Fallback entry point if we didn't find one in __start__ edges.
        if entry_point is None:
            entry_point = getattr(self.graph, "entry_point", None) or getattr(self.graph, "_entry_point", None)

        return ComponentRegistry(
            agents=agents,
            tools=list(tools.values()),
            entry_point=entry_point,
            state_schema=state_schema,
            discovery_method="graph",
        )

    # ------------------------------------------------------------ extraction

    def _extract_graph_structure(self) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, dict[str, str]]]:
        """Return (nodes_dict, edges, conditional_edges).

        Nodes_dict maps name -> the node's runnable.
        Edges is a list of (src, dst).
        Conditional_edges is src -> {condition_label: dst} (label is the condition fn __name__).
        """
        nodes: dict[str, Any] = {}
        edges: list[tuple[str, str]] = []
        cond: dict[str, dict[str, str]] = {}

        # Try the modern .get_graph() drawn graph first.
        try:
            drawn = self.graph.get_graph()
            for n in drawn.nodes.values():
                if n.id not in ("__start__", "__end__"):
                    nodes[n.id] = getattr(n, "data", None) or n
            for e in drawn.edges:
                edges.append((e.source, e.target))
        except Exception:
            pass

        # Walk the underlying StateGraph for richer data (conditional fn names).
        builder = getattr(self.graph, "_builder", None) or getattr(self.graph, "builder", None)
        if builder is not None:
            try:
                # builder.nodes is dict[name, NodeSpec] in modern langgraph.
                bnodes = getattr(builder, "nodes", None) or {}
                for name, spec in bnodes.items():
                    if name not in nodes:
                        runnable = getattr(spec, "runnable", None) or spec
                        nodes[name] = runnable
            except Exception:
                pass
            try:
                # builder.edges is set of (src, dst) tuples.
                for e in getattr(builder, "edges", set()) or set():
                    if isinstance(e, tuple) and len(e) == 2:
                        edges.append(e)
            except Exception:
                pass
            try:
                # builder.branches (conditional edges): dict[src, dict[branch_name, Branch]]
                branches = getattr(builder, "branches", None) or {}
                for src, brs in branches.items():
                    cond[src] = {}
                    for branch_name, branch in brs.items():
                        # Branch.path is the condition function in modern langgraph.
                        path_fn = getattr(branch, "path", None) or getattr(branch, "condition", None) or getattr(branch, "func", None)
                        label = getattr(path_fn, "__name__", str(branch_name)) if path_fn else str(branch_name)
                        # We don't always know the destination upfront — use branch_name as a placeholder destination key.
                        cond[src][label] = branch_name
            except Exception:
                pass

        # Also try .nodes attribute if present.
        if not nodes:
            try:
                nodes = dict(getattr(self.graph, "nodes", {}) or {})
            except Exception:
                nodes = {}

        return nodes, list(set(edges)), cond

    # ------------------------------------------------------- agent type

    def _detect_agent_type(self, name: str, node: Any) -> AgentType:
        cls_name = type(node).__name__.lower()
        nm = name.lower()
        # Most specific first.
        if "subgraph" in cls_name or hasattr(node, "_builder") or hasattr(node, "get_graph"):
            return "subgraph"
        if "tool" in cls_name and "node" in cls_name:
            return "tool_node"
        if "react" in cls_name or "react" in nm or "agentnode" in cls_name:
            return "react_agent"
        if any(kw in nm for kw in ("orchestrator", "router", "supervisor", "dispatcher")):
            return "orchestrator"
        # Bound tools → agentic
        if hasattr(node, "tools") or hasattr(node, "_tools"):
            return "react_agent"
        return "simple_node"

    def _references_subgraph(self, node: Any) -> bool:
        # If the node has its own .get_graph()/_builder, it's a subgraph.
        return hasattr(node, "_builder") or hasattr(node, "get_graph")

    # ------------------------------------------------------- bound tools

    def _extract_bound_tools(self, node: Any) -> list[ToolInfo]:
        out: list[ToolInfo] = []
        seen: set[str] = set()

        # 1. Direct .tools / ._tools attribute.
        for attr in ("tools", "_tools"):
            tools_attr = getattr(node, attr, None)
            if tools_attr:
                for t in tools_attr:
                    info = self._tool_to_info(t)
                    if info and info.name not in seen:
                        out.append(info)
                        seen.add(info.name)

        # 2. Wrapped runnable (LangChain ToolCallingLLM exposes `.bound.kwargs.tools`).
        for runnable_attr in ("runnable", "bound", "_bound"):
            runnable = getattr(node, runnable_attr, None)
            if runnable is None:
                continue
            for path in (
                ("tools",),
                ("kwargs", "tools"),
                ("bound", "kwargs", "tools"),
            ):
                v = runnable
                ok = True
                for p in path:
                    if isinstance(v, dict):
                        v = v.get(p)
                    else:
                        v = getattr(v, p, None)
                    if v is None:
                        ok = False
                        break
                if ok and v:
                    for t in v if isinstance(v, list) else [v]:
                        info = self._tool_to_info(t)
                        if info and info.name not in seen:
                            out.append(info)
                            seen.add(info.name)

        # 3. Inspect callable signature parameters that look like tool registries.
        if not out and callable(node):
            try:
                sig = inspect.signature(node)
                for p in sig.parameters.values():
                    ann = p.annotation
                    if ann is not inspect._empty and "tool" in str(ann).lower():
                        # Best-effort fallback — don't invent tool info we can't verify.
                        pass
            except (ValueError, TypeError):
                pass

        return out

    def _tool_to_info(self, t: Any) -> ToolInfo | None:
        if isinstance(t, dict):
            # Already a JSON-schema'd tool definition.
            name = t.get("name") or (t.get("function") or {}).get("name")
            desc = t.get("description") or (t.get("function") or {}).get("description") or ""
            schema = t.get("input_schema") or t.get("parameters") or (t.get("function") or {}).get("parameters") or {}
            if not name:
                return None
            return ToolInfo(name=name, description=desc, input_schema=schema if isinstance(schema, dict) else {}, source="graph_introspection")

        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if not name:
            return None
        desc = getattr(t, "description", "") or getattr(t, "__doc__", "") or ""
        schema: dict[str, Any] = {}
        try:
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
            elif hasattr(t, "args"):
                schema = {"type": "object", "properties": dict(t.args) if isinstance(t.args, dict) else {}}
        except Exception:
            schema = {}
        return ToolInfo(
            name=name,
            description=desc.strip() if isinstance(desc, str) else "",
            input_schema=schema,
            is_llm_based="llm" in name.lower() or "gpt" in name.lower() or "summari" in name.lower(),
            source="graph_introspection",
        )
