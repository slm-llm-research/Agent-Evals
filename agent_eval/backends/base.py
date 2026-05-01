"""EvaluatorBackend abstract interface.

A backend wraps an external eval library so its metrics can be invoked
through the same evaluator API as native AgentEval metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_eval.dataset.schema import EvalExample
    from agent_eval.evaluators.base import EvaluatorResult


class EvaluatorBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend's underlying library is installed."""

    @abstractmethod
    def supported_metrics(self) -> list[str]:
        """Names of metrics this backend can compute."""

    @abstractmethod
    async def evaluate(
        self,
        metric_name: str,
        example: "EvalExample",
        trace: Any | None = None,
        **kwargs: Any,
    ) -> "EvaluatorResult":
        """Run the named metric against the example. Returns a normalized EvaluatorResult."""

    def supports(self, metric_name: str) -> bool:
        return metric_name in self.supported_metrics()
