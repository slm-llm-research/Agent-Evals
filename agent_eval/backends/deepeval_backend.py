"""DeepEval backend — wraps DeepEval metrics for AgentEval.

Recommended for: tool_correctness, argument_correctness, task_completion,
toxicity, bias.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agent_eval.backends.base import EvaluatorBackend

if TYPE_CHECKING:
    from agent_eval.dataset.schema import EvalExample
    from agent_eval.evaluators.base import EvaluatorResult


_METRIC_MAP = {
    "task_completion": "TaskCompletionMetric",
    "tool_correctness": "ToolCorrectnessMetric",
    "argument_correctness": "ArgumentCorrectnessMetric",
    "answer_relevancy": "AnswerRelevancyMetric",
    "faithfulness": "FaithfulnessMetric",
    "hallucination": "HallucinationMetric",
    "bias": "BiasMetric",
    "toxicity": "ToxicityMetric",
}


class DeepEvalBackend(EvaluatorBackend):
    name = "deepeval"

    def is_available(self) -> bool:
        try:
            import deepeval  # noqa: F401

            return True
        except Exception:
            return False

    def supported_metrics(self) -> list[str]:
        return list(_METRIC_MAP.keys())

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
                details={"backend": "deepeval", "error": "deepeval not installed"},
                flagged=True,
                flag_reason="deepeval backend requested but library is not installed",
            )
        if metric_name not in _METRIC_MAP:
            return EvaluatorResult(
                evaluator_name=metric_name,
                component_name=kwargs.get("component_name", "unknown"),
                score=0.0,
                passed=False,
                threshold=threshold,
                details={"backend": "deepeval", "error": f"unsupported metric '{metric_name}'"},
                flagged=True,
                flag_reason=f"DeepEval backend does not support '{metric_name}'",
            )

        start = time.perf_counter()
        try:
            score, reason, model_used = await self._call_deepeval(metric_name, example, trace, threshold)
        except Exception as e:
            return EvaluatorResult(
                evaluator_name=metric_name,
                component_name=kwargs.get("component_name", "unknown"),
                score=0.0,
                passed=False,
                threshold=threshold,
                details={"backend": "deepeval", "error": str(e)},
                flagged=True,
                flag_reason=f"DeepEval call failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        return EvaluatorResult(
            evaluator_name=metric_name,
            component_name=kwargs.get("component_name", "unknown"),
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"backend": "deepeval", "reason": reason, "model": model_used},
            flagged=score < 0.5,
            flag_reason=None if score >= 0.5 else f"deepeval/{metric_name} below 0.5",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    async def _call_deepeval(
        self,
        metric_name: str,
        example: "EvalExample",
        trace: Any | None,
        threshold: float,
    ) -> tuple[float, str, str]:
        from deepeval.metrics import (  # type: ignore
            AnswerRelevancyMetric,
            BiasMetric,
            FaithfulnessMetric,
            HallucinationMetric,
            ToxicityMetric,
        )
        from deepeval.test_case import LLMTestCase  # type: ignore

        # Try to import agent metrics — they may not exist on older deepeval.
        try:
            from deepeval.metrics import (  # type: ignore
                ArgumentCorrectnessMetric,
                TaskCompletionMetric,
                ToolCorrectnessMetric,
            )
        except Exception:
            ArgumentCorrectnessMetric = TaskCompletionMetric = ToolCorrectnessMetric = None  # type: ignore

        cls_map = {
            "task_completion": TaskCompletionMetric,
            "tool_correctness": ToolCorrectnessMetric,
            "argument_correctness": ArgumentCorrectnessMetric,
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": FaithfulnessMetric,
            "hallucination": HallucinationMetric,
            "bias": BiasMetric,
            "toxicity": ToxicityMetric,
        }
        metric_cls = cls_map.get(metric_name)
        if metric_cls is None:
            raise RuntimeError(f"DeepEval class for '{metric_name}' is not available in this version.")

        actual_output = _coerce_str(_get_actual_output(example, trace))
        retrieval_context = _get_context(trace)
        test_case = LLMTestCase(
            input=_coerce_str(example.input.get("query") if isinstance(example.input, dict) else example.input),
            actual_output=actual_output,
            expected_output=example.reference_answer or "",
            retrieval_context=retrieval_context,
        )
        metric = metric_cls(threshold=threshold)
        # DeepEval is sync.
        metric.measure(test_case)
        score = float(getattr(metric, "score", 0.0))
        reason = str(getattr(metric, "reason", ""))
        model_used = str(getattr(metric, "evaluation_model", ""))
        return score, reason, model_used


def _get_actual_output(example: "EvalExample", trace: Any | None) -> Any:
    if trace is not None:
        outputs = getattr(trace, "outputs", None)
        if isinstance(outputs, dict):
            for k in ("answer", "final_answer", "summary", "output"):
                if outputs.get(k):
                    return outputs[k]
    if example.expected_output:
        return next(iter(example.expected_output.values()), "")
    return example.reference_answer or ""


def _get_context(trace: Any | None) -> list[str]:
    if trace is None:
        return []
    contexts = []
    for run in _iter(trace):
        if (run.get("run_type") or "") == "tool":
            o = run.get("outputs") or {}
            if isinstance(o, dict):
                for v in o.values():
                    if isinstance(v, str):
                        contexts.append(v[:2000])
    return contexts[:10]


def _iter(trace: Any):
    stack = [trace]
    while stack:
        r = stack.pop()
        yield {
            "run_type": getattr(r, "run_type", None),
            "outputs": getattr(r, "outputs", None) or {},
        }
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)


def _coerce_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    return str(v)
