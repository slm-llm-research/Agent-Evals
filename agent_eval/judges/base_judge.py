"""Base LLM judge — zero temperature, retry, calibration hook."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from agent_eval.config import get_config


class JudgeResult(BaseModel):
    score: float
    verdict: str = ""
    reasoning: str = ""
    confidence: float = 1.0
    model_used: str = ""
    latency_ms: float = 0.0


@dataclass
class EnsembleResult:
    mean_score: float
    std_score: float
    agreement: str  # high / medium / low
    individual_results: list[JudgeResult] = field(default_factory=list)


class HumanAnnotation(BaseModel):
    example_id: str
    score: float
    notes: str = ""


@dataclass
class CalibrationReport:
    n: int
    pearson_r: float
    mae: float
    cohen_kappa: float | None = None
    is_reliable: bool = True
    notes: str = ""


def _try_traceable(name: str | None = None):
    """Optional langsmith @traceable that no-ops if langsmith is unavailable."""
    try:
        from langsmith import traceable

        return traceable(name=name) if name else traceable
    except Exception:
        def _noop(fn):
            return fn

        return _noop


traceable = _try_traceable()


class BaseJudge(ABC):
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        few_shot_examples: list[dict[str, Any]] | None = None,
        llm: Any | None = None,
    ):
        cfg = get_config()
        self.model_name = model or cfg.judge_model
        self.temperature = temperature
        self.few_shot = few_shot_examples or []
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = get_config().get_chat_model(model=self.model_name, temperature=self.temperature)
        return self._llm

    @abstractmethod
    async def judge(self, **kwargs: Any) -> JudgeResult:  # pragma: no cover
        ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _call(self, prompt: str) -> str:
        if hasattr(self.llm, "ainvoke"):
            msg = await self.llm.ainvoke(prompt)
        else:
            msg = self.llm.invoke(prompt)
        return getattr(msg, "content", str(msg))

    async def _judge_to_score(self, prompt: str) -> JudgeResult:
        start = time.perf_counter()
        text = await self._call(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0
        score, verdict, reasoning = self._parse_response(text)
        return JudgeResult(
            score=score,
            verdict=verdict,
            reasoning=reasoning,
            model_used=self.model_name,
            latency_ms=latency_ms,
        )

    def _parse_response(self, text: str) -> tuple[float, str, str]:
        """Extract score (0-1), verdict, reasoning from a free-form judge response.

        Best-effort: looks for a JSON block first, otherwise scans for a `score:` line.
        Score is clamped to [0,1].
        """
        text = (text or "").strip()
        # Try JSON
        for candidate in _candidate_json_blobs(text):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "score" in obj:
                    s = float(obj["score"])
                    if s > 1.0 and s <= 10.0:
                        s = s / 10.0
                    return (
                        max(0.0, min(1.0, s)),
                        str(obj.get("verdict", "")),
                        str(obj.get("reasoning", "")),
                    )
            except Exception:
                continue
        # Heuristic line scan
        score = 0.5
        verdict = ""
        for line in text.splitlines():
            lower = line.lower().strip()
            if lower.startswith("score:") or lower.startswith("score "):
                try:
                    raw = lower.split(":", 1)[1].strip().split()[0].rstrip(",.")
                    s = float(raw)
                    if s > 1.0 and s <= 10.0:
                        s = s / 10.0
                    score = max(0.0, min(1.0, s))
                except Exception:
                    pass
            if lower.startswith("verdict:"):
                verdict = line.split(":", 1)[1].strip()
        return score, verdict, text

    # ----------------------------------------------------------- calibration

    async def calibrate(self, ground_truth: list[HumanAnnotation], examples: list[Any]) -> CalibrationReport:
        if not ground_truth or len(ground_truth) != len(examples):
            return CalibrationReport(n=0, pearson_r=0.0, mae=0.0, is_reliable=False, notes="empty or misaligned ground truth")
        scores = []
        for ex in examples:
            res = await self.judge(**(ex if isinstance(ex, dict) else {}))
            scores.append(res.score)
        truth = [a.score for a in ground_truth]
        r = _pearson(scores, truth)
        mae = sum(abs(a - b) for a, b in zip(scores, truth)) / len(scores)
        return CalibrationReport(
            n=len(scores),
            pearson_r=r,
            mae=mae,
            is_reliable=r >= 0.7,
            notes="" if r >= 0.7 else "Pearson r < 0.7 — judge may be unreliable",
        )


@traceable
async def ensemble_judge(judges: list[BaseJudge], **kwargs: Any) -> EnsembleResult:
    results = []
    for j in judges:
        try:
            results.append(await j.judge(**kwargs))
        except Exception as e:
            results.append(JudgeResult(score=0.0, verdict="error", reasoning=str(e)))
    if not results:
        return EnsembleResult(mean_score=0.0, std_score=0.0, agreement="low")
    scores = [r.score for r in results]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = var ** 0.5
    if std < 0.10:
        agreement = "high"
    elif std < 0.25:
        agreement = "medium"
    else:
        agreement = "low"
    return EnsembleResult(mean_score=mean, std_score=std, agreement=agreement, individual_results=results)


# ---------------------------------------------------------------- helpers


def _candidate_json_blobs(text: str):
    text = text.strip()
    if text.startswith("```"):
        body = text.split("```", 2)[1]
        if body.lower().startswith("json"):
            body = body[4:]
        body = body.rsplit("```", 1)[0]
        yield body.strip()
    if text.startswith("{"):
        yield text
    # last-resort: find the first {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        yield text[start : end + 1]


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)
