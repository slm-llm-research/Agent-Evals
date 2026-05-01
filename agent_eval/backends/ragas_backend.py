"""Ragas backend — wraps Ragas metrics for AgentEval.

Recommended for: faithfulness, context precision/recall, answer_relevancy.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agent_eval.backends.base import EvaluatorBackend

if TYPE_CHECKING:
    from agent_eval.dataset.schema import EvalExample
    from agent_eval.evaluators.base import EvaluatorResult


_SUPPORTED = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_relevancy",
    "answer_similarity",
    "answer_correctness",
]


class RagasBackend(EvaluatorBackend):
    name = "ragas"

    def is_available(self) -> bool:
        try:
            import ragas  # noqa: F401

            return True
        except Exception:
            return False

    def supported_metrics(self) -> list[str]:
        return list(_SUPPORTED)

    async def evaluate(
        self,
        metric_name: str,
        example: "EvalExample",
        trace: Any | None = None,
        **kwargs: Any,
    ) -> "EvaluatorResult":
        from agent_eval.evaluators.base import EvaluatorResult

        threshold = float(kwargs.get("threshold", 0.7))
        if not self.is_available():
            return EvaluatorResult(
                evaluator_name=metric_name,
                component_name=kwargs.get("component_name", "unknown"),
                score=0.0,
                passed=False,
                threshold=threshold,
                details={"backend": "ragas", "error": "ragas not installed"},
                flagged=True,
                flag_reason="ragas backend requested but library is not installed",
            )
        if metric_name not in _SUPPORTED:
            return EvaluatorResult(
                evaluator_name=metric_name,
                component_name=kwargs.get("component_name", "unknown"),
                score=0.0,
                passed=False,
                threshold=threshold,
                details={"backend": "ragas", "error": f"unsupported metric '{metric_name}'"},
                flagged=True,
                flag_reason=f"Ragas backend does not support '{metric_name}'",
            )

        start = time.perf_counter()
        try:
            score = await self._call_ragas(metric_name, example, trace)
        except Exception as e:
            return EvaluatorResult(
                evaluator_name=metric_name,
                component_name=kwargs.get("component_name", "unknown"),
                score=0.0,
                passed=False,
                threshold=threshold,
                details={"backend": "ragas", "error": str(e)},
                flagged=True,
                flag_reason=f"Ragas call failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        return EvaluatorResult(
            evaluator_name=metric_name,
            component_name=kwargs.get("component_name", "unknown"),
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"backend": "ragas"},
            flagged=score < 0.5,
            flag_reason=None if score >= 0.5 else f"ragas/{metric_name} below 0.5",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    async def _call_ragas(self, metric_name: str, example: "EvalExample", trace: Any | None) -> float:
        # Ragas API surface varies across versions; try the modern (>=0.2) path first.
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            context_relevancy,
            faithfulness,
        )

        metric_obj_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_relevancy": context_relevancy,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness,
        }
        metric_obj = metric_obj_map[metric_name]
        question = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
        answer = _get_actual_output(example, trace)
        contexts = _get_context(trace)
        ground_truth = example.reference_answer or ""
        sample = {
            "question": [str(question)],
            "answer": [str(answer)],
            "contexts": [contexts or [""]],
            "ground_truth": [str(ground_truth)],
            "reference": [str(ground_truth)],
        }
        ds = Dataset.from_dict(sample)
        result = evaluate(ds, metrics=[metric_obj])
        # result is a Result-like object; the score is keyed by metric name.
        try:
            score = float(result[metric_name])
        except Exception:
            try:
                score = float(list(result.scores[0].values())[0])
            except Exception:
                score = 0.0
        return max(0.0, min(1.0, score))


def _get_actual_output(example: "EvalExample", trace: Any | None) -> str:
    if trace is not None:
        outputs = getattr(trace, "outputs", None)
        if isinstance(outputs, dict):
            for k in ("answer", "final_answer", "summary", "output"):
                v = outputs.get(k)
                if isinstance(v, str):
                    return v
    return example.reference_answer or ""


def _get_context(trace: Any | None) -> list[str]:
    if trace is None:
        return []
    out = []
    stack = [trace]
    while stack:
        r = stack.pop()
        if (getattr(r, "run_type", None) or "") == "tool":
            o = getattr(r, "outputs", None) or {}
            if isinstance(o, dict):
                for v in o.values():
                    if isinstance(v, str):
                        out.append(v[:2000])
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out[:10]
