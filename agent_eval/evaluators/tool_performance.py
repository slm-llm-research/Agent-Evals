"""Tool performance evaluators."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from agent_eval.config import get_config
from agent_eval.evaluators.base import BaseEvaluator, EvaluatorResult


def _iter_tool_runs(trace: Any | None):
    if trace is None:
        return
    stack = [trace]
    while stack:
        r = stack.pop()
        if (getattr(r, "run_type", None) or "") == "tool":
            yield r
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)


def _latency_ms(run: Any) -> float | None:
    s, e = getattr(run, "start_time", None), getattr(run, "end_time", None)
    if isinstance(s, datetime) and isinstance(e, datetime):
        return (e - s).total_seconds() * 1000.0
    return None


class ToolPerformanceEvaluator(BaseEvaluator):
    """Per-tool aggregate score across a trace: success rate, P95 latency, error mix.

    Surfaces a single 0-1 score (= mean per-tool success rate) plus a per-tool
    breakdown in details.
    """

    name = "tool_success_rate"

    async def _evaluate_native(self, example, trace=None):
        per_tool: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "ok": 0, "latencies": [], "errors": defaultdict(int)})
        for run in _iter_tool_runs(trace):
            name = getattr(run, "name", "unknown")
            stats = per_tool[name]
            stats["calls"] += 1
            err = getattr(run, "error", None)
            if not err:
                stats["ok"] += 1
            else:
                stats["errors"][type(err).__name__ if not isinstance(err, str) else "Error"] += 1
            lat = _latency_ms(run)
            if lat is not None:
                stats["latencies"].append(lat)
        if not per_tool:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="system",
                score=1.0, passed=True, threshold=self.threshold,
                details={"reason": "no tool calls observed"},
            )
        breakdown = {}
        successes = []
        for tool, s in per_tool.items():
            sr = (s["ok"] / s["calls"]) if s["calls"] else 0.0
            lats = sorted(s["latencies"])
            p95 = lats[max(0, int(0.95 * len(lats)) - 1)] if lats else 0.0
            breakdown[tool] = {
                "calls": s["calls"],
                "success_rate": sr,
                "p95_latency_ms": p95,
                "avg_latency_ms": (sum(lats) / len(lats)) if lats else 0.0,
                "errors": dict(s["errors"]),
            }
            successes.append(sr)
        score = sum(successes) / len(successes)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"per_tool": breakdown},
            flagged=score < 0.5,
        )


class ToolResultQualityEvaluator(BaseEvaluator):
    """LLM judge on a sample of tool outputs."""

    name = "tool_result_quality"

    async def _evaluate_native(self, example, trace=None):
        from agent_eval.judges.rubric_judge import AnswerQualityJudge

        runs = list(_iter_tool_runs(trace))
        if not runs:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no tool calls"})
        judge = self.judge or AnswerQualityJudge()
        scores = []
        sampled = runs[:5]  # cap sample to control cost
        for r in sampled:
            outputs = getattr(r, "outputs", None) or {}
            output_str = str(outputs)[:1500]
            query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
            res = await judge.judge(query=str(query), answer=output_str)
            scores.append(res.score)
        avg = sum(scores) / len(scores) if scores else 0.0
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=avg, passed=avg >= self.threshold, threshold=self.threshold,
            details={"sample_size": len(sampled), "individual": scores},
            flagged=avg < 0.5,
        )


class ArgumentCorrectnessEvaluator(BaseEvaluator):
    """v2 — are tool inputs properly formatted and relevant?

    Prefers DeepEval ArgumentCorrectnessMetric when backend='deepeval' is set;
    otherwise falls back to a simple per-call LLM judge.
    """

    name = "argument_correctness"

    async def _evaluate_native(self, example, trace=None):
        from agent_eval.judges.base_judge import BaseJudge

        runs = list(_iter_tool_runs(trace))
        if not runs:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no tool calls"})
        judge = self.judge or _ArgumentJudge()
        scores: list[float] = []
        for r in runs[:5]:
            res = await judge.judge(
                tool_name=getattr(r, "name", "?"),
                inputs=getattr(r, "inputs", None) or {},
                user_query=example.input.get("query") if isinstance(example.input, dict) else str(example.input),
            )
            scores.append(res.score)
        avg = sum(scores) / len(scores) if scores else 1.0
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=avg, passed=avg >= self.threshold, threshold=self.threshold,
            details={"sample_size": len(scores)},
            flagged=avg < 0.5,
        )


class _ArgumentJudge:
    """Lightweight inline judge to avoid forcing the user to wire one explicitly."""

    async def judge(self, *, tool_name: str, inputs: Any, user_query: Any):
        from agent_eval.judges.base_judge import JudgeResult
        from agent_eval.judges.rubric_judge import RubricJudge

        class _J(RubricJudge):
            def _render_prompt(self_inner, **kw):
                return (
                    f"Evaluate whether the tool inputs are well-formed and relevant.\n"
                    f"Tool: {kw['tool_name']}\n"
                    f"Inputs: {kw['inputs']}\n"
                    f"User query: {kw['user_query']}\n"
                    'Reply ONLY with JSON: {"score": <0.0-1.0>, "verdict": "...", "reasoning": "..."}'
                )

        try:
            return await _J().judge(tool_name=tool_name, inputs=inputs, user_query=user_query)
        except Exception as e:
            return JudgeResult(score=0.5, verdict="judge_unavailable", reasoning=str(e))


class MCPServerHealthEvaluator(BaseEvaluator):
    """Per detected MCP server: availability, avg response time."""

    name = "mcp_server_availability"

    def __init__(self, mcp_servers: list[Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.mcp_servers = mcp_servers or []

    async def _evaluate_native(self, example, trace=None):
        from agent_eval.discovery.mcp_inspector import MCPInspector

        if not self.mcp_servers:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no MCP servers in registry"})
        results = []
        for srv in self.mcp_servers:
            url = getattr(srv, "url", None)
            if not url:
                continue
            health = await MCPInspector(url).health_check()
            results.append({"name": getattr(srv, "name", url), **health})
        if not results:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no reachable URLs"})
        avail = sum(1 for r in results if r.get("is_reachable")) / len(results)
        avg_latency = sum(r.get("latency_ms", 0) for r in results) / len(results)
        flagged = avail < 0.95 or avg_latency > 5000
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=avail, passed=avail >= 0.95, threshold=0.95,
            details={"servers": results, "avg_latency_ms": avg_latency},
            flagged=flagged,
            flag_reason=("low availability" if avail < 0.95 else "high latency") if flagged else None,
        )


class CostPerToolEvaluator(BaseEvaluator):
    name = "cost_per_tool"

    def __init__(self, max_cost_usd: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.max_cost_usd = max_cost_usd

    async def _evaluate_native(self, example, trace=None):
        cfg = get_config()
        total_cost = 0.0
        per_tool = defaultdict(float)
        for run in _iter_tool_runs(trace):
            name = getattr(run, "name", "unknown")
            inputs = getattr(run, "inputs", None) or {}
            outputs = getattr(run, "outputs", None) or {}
            tokens_in = _approx_tokens(inputs)
            tokens_out = _approx_tokens(outputs)
            rate = cfg.cost_model.get(cfg.judge_model, {"input": 2.5, "output": 10.0})
            cost = (tokens_in / 1_000_000) * rate["input"] + (tokens_out / 1_000_000) * rate["output"]
            per_tool[name] += cost
            total_cost += cost
        score = 1.0 if total_cost <= self.max_cost_usd else max(0.0, 1.0 - (total_cost - self.max_cost_usd) / self.max_cost_usd)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=total_cost <= self.max_cost_usd, threshold=self.max_cost_usd,
            details={"total_cost_usd": round(total_cost, 4), "per_tool": {k: round(v, 4) for k, v in per_tool.items()}},
            flagged=total_cost > self.max_cost_usd,
        )


def _approx_tokens(obj: Any) -> int:
    s = str(obj)
    return max(1, len(s) // 4)
