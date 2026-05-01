"""Native backend — default implementation.

Acts as an identity backend: the per-evaluator native logic in
`agent_eval/evaluators/` is the canonical implementation, so this
backend simply marks every metric as supported and delegates back
to the evaluator's `_evaluate_native` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_eval.backends.base import EvaluatorBackend

if TYPE_CHECKING:
    from agent_eval.dataset.schema import EvalExample
    from agent_eval.evaluators.base import EvaluatorResult


class NativeBackend(EvaluatorBackend):
    name = "native"

    def is_available(self) -> bool:
        return True

    def supported_metrics(self) -> list[str]:
        return ["*"]  # supports anything — identity backend

    def supports(self, metric_name: str) -> bool:
        return True

    async def evaluate(
        self,
        metric_name: str,
        example: "EvalExample",
        trace: Any | None = None,
        **kwargs: Any,
    ) -> "EvaluatorResult":
        # The native backend is intentionally a no-op — evaluators call their own
        # `_evaluate_native` when `backend.name == "native"`.
        from agent_eval.evaluators.base import EvaluatorResult

        return EvaluatorResult(
            evaluator_name=metric_name,
            component_name=kwargs.get("component_name", "unknown"),
            score=0.0,
            passed=False,
            threshold=kwargs.get("threshold", 0.7),
            details={"note": "Native backend is identity — evaluator should call _evaluate_native directly."},
        )
