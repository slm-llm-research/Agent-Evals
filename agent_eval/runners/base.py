"""AgentRunner — abstract interface for executing the agent against a dataset."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_eval.dataset.schema import EvalDataset, EvalExample


@dataclass
class RunResult:
    """One example's result from an AgentRunner."""

    example_id: str
    output: Any  # raw response from the agent
    trace: Any  # langsmith.Run, SyntheticTrace, or None
    error: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] | None = None


class AgentRunner(ABC):
    """Subclasses implement `run_one`. Concurrency + ordering handled here."""

    name: str = "abstract"
    max_concurrency: int = 5

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max(1, max_concurrency)

    @abstractmethod
    async def run_one(self, example: "EvalExample") -> RunResult:
        """Execute the agent on one example. Must return a RunResult."""

    async def run_dataset(self, dataset: "EvalDataset", progress_callback=None) -> list[RunResult]:
        """Run the agent against every example in the dataset, bounded concurrency."""
        sem = asyncio.Semaphore(self.max_concurrency)
        results: list[RunResult | None] = [None] * len(dataset.examples)

        async def _one(i: int, ex):
            async with sem:
                try:
                    res = await self.run_one(ex)
                except Exception as e:
                    res = RunResult(example_id=ex.id, output=None, trace=None, error=f"{type(e).__name__}: {e}")
                results[i] = res
                if progress_callback is not None:
                    try:
                        progress_callback(i + 1, len(dataset.examples), res)
                    except Exception:
                        pass

        await asyncio.gather(*(_one(i, ex) for i, ex in enumerate(dataset.examples)))
        return [r for r in results if r is not None]

    async def aclose(self) -> None:
        """Optional cleanup hook."""
