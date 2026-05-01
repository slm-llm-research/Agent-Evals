"""Trajectory evaluators — including v2 structural metrics (Tool F1, Node F1, SSI)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import networkx as nx

from agent_eval.evaluators.base import BaseEvaluator, EvaluatorResult
from agent_eval.judges.rubric_judge import IntentResolutionJudge, ToolSelectionJudge


# ---------------------------------------------------------------------------- helpers


def _extract_tool_sequence(trace: Any | None) -> list[str]:
    if trace is None:
        return []
    out: list[str] = []
    stack = [trace]
    while stack:
        r = stack.pop()
        if (getattr(r, "run_type", None) or "") == "tool":
            n = getattr(r, "name", None)
            if n:
                out.append(n)
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out


def _extract_node_sequence(trace: Any | None) -> list[str]:
    if trace is None:
        return []
    out: list[str] = []
    stack = [trace]
    while stack:
        r = stack.pop()
        rt = getattr(r, "run_type", None) or ""
        if rt in ("chain", "agent", "graph"):
            n = getattr(r, "name", None)
            if n and n not in ("__start__", "__end__"):
                out.append(n)
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out


def _trace_to_graph(trace: Any | None) -> nx.DiGraph:
    g = nx.DiGraph()
    if trace is None:
        return g
    nodes = _extract_node_sequence(trace)
    for i, n in enumerate(nodes):
        g.add_node(n)
        if i > 0:
            g.add_edge(nodes[i - 1], n)
    return g


def _expected_graph(adj: dict[str, list[str]] | None) -> nx.DiGraph:
    g = nx.DiGraph()
    if not adj:
        return g
    for src, dsts in adj.items():
        g.add_node(src)
        for d in dsts:
            g.add_edge(src, d)
    return g


def _f1(actual: list[str], expected: list[str]) -> tuple[float, float, float]:
    a, e = set(actual), set(expected)
    if not a and not e:
        return 1.0, 1.0, 1.0
    if not a or not e:
        return 0.0, 0.0, 0.0
    tp = len(a & e)
    p = tp / len(a)
    r = tp / len(e)
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------------------- evaluators


class ToolSelectionEvaluator(BaseEvaluator):
    name = "tool_selection_accuracy"

    async def _evaluate_native(self, example, trace=None):
        actual = _extract_tool_sequence(trace)
        expected = list(example.expected_tool_sequence or [])
        if expected:
            _p, _r, f1 = _f1(actual, expected)
            score = f1
            details = {"actual": actual, "expected": expected, "f1": f1}
        else:
            judge = self.judge or ToolSelectionJudge()
            res = await judge.judge(query=str(example.input), tool_sequence=actual, available_tools=actual)
            score = res.score
            details = {"actual": actual, "judge_verdict": res.verdict}
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details=details, flagged=score < 0.5,
        )


class ToolF1Evaluator(BaseEvaluator):
    name = "tool_f1"

    async def _evaluate_native(self, example, trace=None):
        actual = _extract_tool_sequence(trace)
        expected = list(example.expected_tool_sequence or [])
        if not expected:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="system",
                score=1.0, passed=True, threshold=self.threshold,
                details={"reason": "no expected_tool_sequence — skipped"},
            )
        p, r, f1 = _f1(actual, expected)
        # Bonus for matching ORDER as well.
        order_bonus = _order_bonus(actual, expected)
        score = 0.8 * f1 + 0.2 * order_bonus
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"precision": p, "recall": r, "f1": f1, "order_bonus": order_bonus, "actual": actual, "expected": expected},
            flagged=score < 0.5,
        )


def _order_bonus(actual: list[str], expected: list[str]) -> float:
    """LCS-based order similarity in [0,1]."""
    if not actual or not expected:
        return 0.0
    n, m = len(actual), len(expected)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if actual[i] == expected[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[n][m]
    return lcs / max(n, m)


class NodeF1Evaluator(BaseEvaluator):
    name = "node_f1"

    async def _evaluate_native(self, example, trace=None):
        actual = _extract_node_sequence(trace)
        expected_nodes: list[str] = []
        if example.expected_task_graph:
            seen = set()
            for src, dsts in example.expected_task_graph.items():
                if src not in seen:
                    expected_nodes.append(src); seen.add(src)
                for d in dsts:
                    if d not in seen:
                        expected_nodes.append(d); seen.add(d)
        if not expected_nodes:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="system",
                score=1.0, passed=True, threshold=self.threshold,
                details={"reason": "no expected_task_graph — skipped"},
            )
        p, r, f1 = _f1(actual, expected_nodes)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=f1, passed=f1 >= self.threshold, threshold=self.threshold,
            details={"precision": p, "recall": r, "actual": actual, "expected": expected_nodes},
            flagged=f1 < 0.5,
        )


class StructuralSimilarityEvaluator(BaseEvaluator):
    """SSI = 1 - normalized_graph_edit_distance(actual_graph, expected_graph)."""

    name = "ssi"

    async def _evaluate_native(self, example, trace=None):
        if not example.expected_task_graph:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="system",
                score=1.0, passed=True, threshold=self.threshold,
                details={"reason": "no expected_task_graph — skipped"},
            )
        actual = _trace_to_graph(trace)
        expected = _expected_graph(example.expected_task_graph)
        if actual.number_of_nodes() == 0 and expected.number_of_nodes() == 0:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"note": "both graphs empty"})

        # Plain graph edit distance is exponential; use the optimize-edit-paths bound for speed.
        try:
            ged = nx.graph_edit_distance(actual, expected, timeout=2.0)
            if ged is None:
                ged = float(abs(actual.number_of_nodes() - expected.number_of_nodes()) + abs(actual.number_of_edges() - expected.number_of_edges()))
        except Exception:
            ged = float(abs(actual.number_of_nodes() - expected.number_of_nodes()) + abs(actual.number_of_edges() - expected.number_of_edges()))
        denom = float(max(actual.number_of_nodes() + actual.number_of_edges() + expected.number_of_nodes() + expected.number_of_edges(), 1))
        ssi = max(0.0, 1.0 - (ged / denom))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=ssi, passed=ssi >= self.threshold, threshold=self.threshold,
            details={"graph_edit_distance": float(ged), "actual_nodes": list(actual.nodes), "expected_nodes": list(expected.nodes)},
            flagged=ssi < 0.5,
        )


class IntentResolutionEvaluator(BaseEvaluator):
    """NEW v2 — does the agent's plan address the user's underlying goal?"""

    name = "intent_resolution"

    async def _evaluate_native(self, example, trace=None):
        # The "plan" is a best-effort surface: tool sequence + node sequence, plus the answer.
        nodes = _extract_node_sequence(trace)
        tools = _extract_tool_sequence(trace)
        plan_summary = f"nodes={nodes}; tools={tools}"
        judge = self.judge or IntentResolutionJudge()
        query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
        res = await judge.judge(query=str(query), plan=plan_summary)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=res.score, passed=res.score >= self.threshold, threshold=self.threshold,
            details={"plan": plan_summary, "verdict": res.verdict, "reasoning": res.reasoning[:500]},
            flagged=res.score < 0.5,
        )


class StepSuccessRateEvaluator(BaseEvaluator):
    name = "step_success_rate"

    async def _evaluate_native(self, example, trace=None):
        total, ok = 0, 0
        if trace is not None:
            stack = [trace]
            while stack:
                r = stack.pop()
                if (getattr(r, "run_type", None) or "") == "tool":
                    total += 1
                    if not getattr(r, "error", None):
                        ok += 1
                for c in getattr(r, "child_runs", None) or []:
                    stack.append(c)
        score = (ok / total) if total else 1.0
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"total_steps": total, "successful_steps": ok},
            flagged=score < 0.5,
        )


class RedundancyEvaluator(BaseEvaluator):
    name = "redundancy_rate"

    async def _evaluate_native(self, example, trace=None):
        seen, dups, total = set(), 0, 0
        if trace is not None:
            stack = [trace]
            while stack:
                r = stack.pop()
                if (getattr(r, "run_type", None) or "") == "tool":
                    total += 1
                    inputs = getattr(r, "inputs", None) or {}
                    key = f"{getattr(r,'name','?')}|{_hash(inputs)}"
                    if key in seen:
                        dups += 1
                    seen.add(key)
                for c in getattr(r, "child_runs", None) or []:
                    stack.append(c)
        score = 1.0 - (dups / total) if total else 1.0
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"total_calls": total, "duplicate_calls": dups},
            flagged=score < 0.5,
        )


class ErrorRecoveryEvaluator(BaseEvaluator):
    """Did the agent retry-with-modification after a failed tool call?"""

    name = "error_recovery_rate"

    async def _evaluate_native(self, example, trace=None):
        runs = []
        if trace is not None:
            stack = [trace]
            while stack:
                r = stack.pop()
                runs.append(r)
                for c in getattr(r, "child_runs", None) or []:
                    stack.append(c)
        # Linearize tool runs by start_time when possible.
        tool_runs = sorted(
            [r for r in runs if (getattr(r, "run_type", None) or "") == "tool"],
            key=lambda r: getattr(r, "start_time", 0) or 0,
        )
        recoveries, errors = 0, 0
        for i, run in enumerate(tool_runs):
            if getattr(run, "error", None):
                errors += 1
                # Look at the next tool call: did the agent change inputs?
                if i + 1 < len(tool_runs):
                    nxt = tool_runs[i + 1]
                    if not getattr(nxt, "error", None):
                        if _hash(getattr(nxt, "inputs", {})) != _hash(getattr(run, "inputs", {})):
                            recoveries += 1
        if errors == 0:
            score = 1.0
        else:
            score = recoveries / errors
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"errors": errors, "recoveries": recoveries},
            flagged=score < 0.5 and errors > 0,
        )


class CycleDetectionEvaluator(BaseEvaluator):
    """Binary: did the agent get stuck in a loop?"""

    name = "cycle_detected"

    async def _evaluate_native(self, example, trace=None):
        seen: dict[str, int] = {}
        runs = []
        if trace is not None:
            stack = [trace]
            while stack:
                r = stack.pop()
                runs.append(r)
                for c in getattr(r, "child_runs", None) or []:
                    stack.append(c)
        cycle = False
        offending = None
        for r in runs:
            if (getattr(r, "run_type", None) or "") in ("chain", "agent", "tool"):
                key = f"{getattr(r,'name','?')}|{_hash(getattr(r, 'inputs', {}))}"
                seen[key] = seen.get(key, 0) + 1
                if seen[key] >= 3 and not cycle:
                    cycle = True
                    offending = key
        score = 0.0 if cycle else 1.0
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=not cycle, threshold=self.threshold,
            details={"cycle": cycle, "offending": offending},
            flagged=cycle,
            flag_reason="cycle detected" if cycle else None,
        )


def _hash(obj: Any) -> str:
    try:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(obj).encode()).hexdigest()[:16]
