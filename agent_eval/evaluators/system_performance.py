"""System-level performance evaluators."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent_eval.config import get_config
from agent_eval.evaluators.base import BaseEvaluator, EvaluatorResult


def _trace_latency_ms(trace: Any | None) -> float | None:
    if trace is None:
        return None
    s, e = getattr(trace, "start_time", None), getattr(trace, "end_time", None)
    if isinstance(s, datetime) and isinstance(e, datetime):
        return (e - s).total_seconds() * 1000.0
    return None


def _per_node_latency(trace: Any | None) -> dict[str, float]:
    if trace is None:
        return {}
    out: dict[str, float] = {}
    stack = [trace]
    while stack:
        r = stack.pop()
        nm = getattr(r, "name", None)
        if nm and nm not in ("__start__", "__end__"):
            s, e = getattr(r, "start_time", None), getattr(r, "end_time", None)
            if isinstance(s, datetime) and isinstance(e, datetime):
                out[nm] = out.get(nm, 0.0) + (e - s).total_seconds() * 1000.0
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out


class LatencyEvaluator(BaseEvaluator):
    name = "end_to_end_latency"

    def __init__(self, p95_target_ms: float = 30000.0, **kwargs):
        super().__init__(**kwargs)
        self.p95_target_ms = p95_target_ms

    async def _evaluate_native(self, example, trace=None):
        latency = _trace_latency_ms(trace)
        per_node = _per_node_latency(trace)
        if latency is None:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no timing on trace"})
        # Score is 1.0 at 0ms, 0.0 at 2*target.
        score = max(0.0, 1.0 - latency / (2 * self.p95_target_ms))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=latency <= self.p95_target_ms, threshold=self.p95_target_ms,
            details={"latency_ms": latency, "p95_target_ms": self.p95_target_ms, "per_node_ms": per_node},
            flagged=latency > self.p95_target_ms,
        )


class StreamingLatencyEvaluator(BaseEvaluator):
    """time_to_first_audio_byte / time_to_first_token (NEW v2).

    Reads from LangSmith feedback if attached to the trace; otherwise reports
    'not_measured' and skips with score 1.0.
    """

    name = "time_to_first_audio_byte"

    def __init__(self, p95_target_ms: float = 3000.0, **kwargs):
        super().__init__(**kwargs)
        self.p95_target_ms = p95_target_ms

    async def _evaluate_native(self, example, trace=None):
        feedback = getattr(trace, "feedback_stats", None) or getattr(trace, "feedback", None) or {}
        ttfa = None
        if isinstance(feedback, dict):
            ttfa = feedback.get("time_to_first_audio_byte_ms") or feedback.get("ttfa_ms")
        if ttfa is None:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="system",
                score=1.0, passed=True, threshold=self.threshold,
                details={"reason": "time_to_first_audio_byte not present on trace — skipped"},
            )
        score = max(0.0, 1.0 - ttfa / (2 * self.p95_target_ms))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=ttfa <= self.p95_target_ms, threshold=self.p95_target_ms,
            details={"ttfa_ms": ttfa, "p95_target_ms": self.p95_target_ms},
            flagged=ttfa > self.p95_target_ms,
        )


class TokenEfficiencyEvaluator(BaseEvaluator):
    name = "token_efficiency"

    def __init__(self, max_tokens: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    async def _evaluate_native(self, example, trace=None):
        total = 0
        per_node: dict[str, int] = {}
        if trace is not None:
            stack = [trace]
            while stack:
                r = stack.pop()
                tu = getattr(r, "total_tokens", None)
                if isinstance(tu, int):
                    total += tu
                    nm = getattr(r, "name", "?")
                    per_node[nm] = per_node.get(nm, 0) + tu
                for c in getattr(r, "child_runs", None) or []:
                    stack.append(c)
        if total == 0:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no token counts"})
        score = max(0.0, 1.0 - total / (2 * self.max_tokens))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=total <= self.max_tokens, threshold=self.max_tokens,
            details={"total_tokens": total, "per_node_tokens": per_node},
            flagged=total > self.max_tokens,
        )


class ErrorRateEvaluator(BaseEvaluator):
    """% of queries ending in error. Computed from a list of traces; for a single
    trace it returns 0/1 — call from evaluator.evaluate_dataset for a real rate."""

    name = "error_rate"

    async def _evaluate_native(self, example, trace=None):
        had_error = bool(getattr(trace, "error", None)) if trace is not None else False
        score = 0.0 if had_error else 1.0
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=not had_error, threshold=self.threshold,
            details={"error_on_this_query": had_error},
            flagged=had_error,
        )


class CostEfficiencyEvaluator(BaseEvaluator):
    name = "cost_per_query_usd"

    def __init__(self, max_usd: float = 0.50, **kwargs):
        super().__init__(**kwargs)
        self.max_usd = max_usd

    async def _evaluate_native(self, example, trace=None):
        cfg = get_config()
        rate = cfg.cost_model.get(cfg.judge_model, {"input": 2.5, "output": 10.0})
        prompt_tokens = 0
        completion_tokens = 0
        if trace is not None:
            stack = [trace]
            while stack:
                r = stack.pop()
                pt = getattr(r, "prompt_tokens", None)
                ct = getattr(r, "completion_tokens", None)
                if isinstance(pt, int):
                    prompt_tokens += pt
                if isinstance(ct, int):
                    completion_tokens += ct
                for c in getattr(r, "child_runs", None) or []:
                    stack.append(c)
        cost = (prompt_tokens / 1_000_000) * rate["input"] + (completion_tokens / 1_000_000) * rate["output"]
        score = 1.0 if cost <= self.max_usd else max(0.0, 1.0 - (cost - self.max_usd) / self.max_usd)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=cost <= self.max_usd, threshold=self.max_usd,
            details={"cost_usd": round(cost, 4), "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            flagged=cost > self.max_usd,
        )
