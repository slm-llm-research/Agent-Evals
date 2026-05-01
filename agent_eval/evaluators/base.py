"""Base evaluator interface + composite."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agent_eval.config import get_config

if TYPE_CHECKING:
    from agent_eval.backends.base import EvaluatorBackend
    from agent_eval.dataset.schema import EvalExample
    from agent_eval.judges.base_judge import BaseJudge


class EvaluatorResult(BaseModel):
    evaluator_name: str
    component_name: str
    score: float
    passed: bool
    threshold: float
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    flagged: bool = False
    flag_reason: str | None = None
    improvement_hints: list[str] = Field(default_factory=list)


@dataclass
class CompositeResult:
    individual_results: list[EvaluatorResult]
    overall_score: float
    weakest_component: str | None
    strongest_component: str | None
    pass_rate: float
    flag_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "weakest_component": self.weakest_component,
            "strongest_component": self.strongest_component,
            "pass_rate": self.pass_rate,
            "flag_count": self.flag_count,
            "individual_results": [r.model_dump() for r in self.individual_results],
        }


class BaseEvaluator(ABC):
    """Subclasses must define `name` and implement `_evaluate_native`."""

    name: str = "base"

    def __init__(
        self,
        threshold: float | None = None,
        judge: "BaseJudge | None" = None,
        backend: "EvaluatorBackend | None" = None,
        **_: Any,
    ) -> None:
        cfg = get_config()
        self.threshold = threshold if threshold is not None else cfg.default_threshold
        self.judge = judge
        self.backend = backend  # if set and backend.supports(self.name), backend takes over

    async def evaluate(self, example: "EvalExample", trace: Any | None = None) -> EvaluatorResult:
        start = time.perf_counter()
        try:
            if self.backend is not None and self.backend.name != "native" and self.backend.supports(self.name):
                result = await self.backend.evaluate(
                    self.name, example, trace, threshold=self.threshold, component_name=self._component_name(example)
                )
            else:
                result = await self._evaluate_native(example, trace)
        except Exception as e:
            result = EvaluatorResult(
                evaluator_name=self.name,
                component_name=self._component_name(example),
                score=0.0,
                passed=False,
                threshold=self.threshold,
                details={"error": str(e)},
                flagged=True,
                flag_reason=f"evaluator raised: {type(e).__name__}: {e}",
            )
        if not result.latency_ms:
            result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result

    @abstractmethod
    async def _evaluate_native(self, example: "EvalExample", trace: Any | None = None) -> EvaluatorResult: ...

    def _component_name(self, example: "EvalExample") -> str:
        return getattr(example, "tags", [None])[0] if getattr(example, "tags", None) else "system"


class CompositeEvaluator:
    """Run a set of evaluators with weights and aggregate to a single score."""

    def __init__(self, evaluators: list[tuple[BaseEvaluator, float]]):
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")
        total = sum(w for _, w in evaluators)
        self.evaluators = [(e, w / total) for e, w in evaluators]

    async def evaluate(self, example: "EvalExample", trace: Any | None = None) -> CompositeResult:
        results: list[EvaluatorResult] = []
        for ev, _w in self.evaluators:
            results.append(await ev.evaluate(example, trace))
        weighted = sum(r.score * w for r, (_e, w) in zip(results, self.evaluators))
        scored = sorted(results, key=lambda r: r.score)
        return CompositeResult(
            individual_results=results,
            overall_score=weighted,
            weakest_component=scored[0].evaluator_name if scored else None,
            strongest_component=scored[-1].evaluator_name if scored else None,
            pass_rate=sum(1 for r in results if r.passed) / len(results),
            flag_count=sum(1 for r in results if r.flagged),
        )
