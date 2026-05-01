"""Tests for GraphInspector deep introspection (using fake graph objects)."""

from __future__ import annotations

from agent_eval.discovery.graph_inspector import GraphInspector


class _FakeNode:
    """Pretend LangChain runnable with bound tools."""

    def __init__(self, tools=None):
        self.tools = tools or []


class _FakeTool:
    def __init__(self, name, description="", schema=None):
        self.name = name
        self.description = description
        if schema:
            class _Schema:
                @staticmethod
                def model_json_schema():
                    return schema
            self.args_schema = _Schema


class _FakeDrawnNode:
    def __init__(self, id, data=None):
        self.id = id
        self.data = data


class _FakeDrawnEdge:
    def __init__(self, source, target):
        self.source = source
        self.target = target


class _FakeDrawnGraph:
    def __init__(self, nodes_dict, edges):
        self.nodes = nodes_dict  # dict[id, FakeDrawnNode]
        self.edges = edges  # list of FakeDrawnEdge


class _FakeCompiledGraph:
    def __init__(self, drawn):
        self._drawn = drawn

    def get_graph(self):
        return self._drawn


def test_graph_inspector_extracts_nodes_and_tools():
    tool_search = _FakeTool(name="web_search", description="search the web",
                              schema={"type": "object", "properties": {"q": {"type": "string"}}})
    nodes = {
        "__start__": _FakeDrawnNode("__start__"),
        "__end__": _FakeDrawnNode("__end__"),
        "orchestrator": _FakeDrawnNode("orchestrator", data=_FakeNode(tools=[tool_search])),
        "search_agent": _FakeDrawnNode("search_agent", data=_FakeNode(tools=[tool_search])),
    }
    edges = [
        _FakeDrawnEdge("__start__", "orchestrator"),
        _FakeDrawnEdge("orchestrator", "search_agent"),
        _FakeDrawnEdge("search_agent", "__end__"),
    ]
    graph = _FakeCompiledGraph(_FakeDrawnGraph(nodes, edges))
    reg = GraphInspector(graph).inspect()
    agent_names = {a.name for a in reg.agents}
    assert "orchestrator" in agent_names
    assert "search_agent" in agent_names
    assert "__start__" not in agent_names
    tool_names = {t.name for t in reg.tools}
    assert "web_search" in tool_names
    assert reg.entry_point == "orchestrator"


def test_graph_inspector_detects_orchestrator_type():
    nodes = {
        "router_node": _FakeDrawnNode("router_node", data=_FakeNode()),
    }
    graph = _FakeCompiledGraph(_FakeDrawnGraph(nodes, []))
    reg = GraphInspector(graph).inspect()
    assert any(a.agent_type == "orchestrator" for a in reg.agents)


def test_graph_inspector_self_loop_detected():
    nodes = {
        "loop_node": _FakeDrawnNode("loop_node", data=_FakeNode()),
        "__start__": _FakeDrawnNode("__start__"),
    }
    edges = [
        _FakeDrawnEdge("__start__", "loop_node"),
        _FakeDrawnEdge("loop_node", "loop_node"),  # self-loop
    ]
    graph = _FakeCompiledGraph(_FakeDrawnGraph(nodes, edges))
    reg = GraphInspector(graph).inspect()
    assert any(a.has_self_loop for a in reg.agents if a.name == "loop_node")


def test_graph_inspector_empty_graph():
    graph = _FakeCompiledGraph(_FakeDrawnGraph({}, []))
    reg = GraphInspector(graph).inspect()
    assert reg.total_components == 0
