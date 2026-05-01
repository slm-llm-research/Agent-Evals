"""Memory subsystem evaluators (NEW v2).

Activated only when discovery detected memory backends. Implements the full
spec set:
  - MemoryRetrievalRecallEvaluator   — fact-insertion harness OR heuristic judge
  - MemoryRetrievalPrecisionEvaluator — per-retrieval relevance fraction
  - MemoryWriteQualityEvaluator       — over-storing / under-storing detection
  - MemoryStalenessEvaluator          — newer info contradicts retrieved memories
  - CrossSessionContinuityEvaluator   — recall info from prior sessions
  - MemoryCostEvaluator               — token + API cost vs value
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent_eval.evaluators.base import BaseEvaluator, EvaluatorResult


_MEMORY_NAME_KEYWORDS = (
    "memory", "vectorstore", "vector_store", "retriever", "pinecone", "chroma",
    "weaviate", "qdrant", "mem0", "zep", "redis_index",
)

_WRITE_VERBS = re.compile(r"(add|store|insert|upsert|write|put|create|persist|save|index)", re.IGNORECASE)
_READ_VERBS = re.compile(r"(query|search|retrieve|get|fetch|read|similarity|recall)", re.IGNORECASE)
_DELETE_VERBS = re.compile(r"(delete|remove|forget|expire)", re.IGNORECASE)


class MemoryEvaluationReport(BaseModel):
    backends: list[str] = Field(default_factory=list)
    recall: float | None = None
    precision: float | None = None
    write_quality: float | None = None
    staleness: float | None = None
    cross_session_continuity: float | None = None
    cost_per_query_usd: float | None = None
    notes: list[str] = Field(default_factory=list)


@dataclass
class FactInjection:
    """One synthetic fact for the memory recall harness."""

    fact_id: str
    fact_text: str
    related_query: str
    expected_keywords: list[str]


def default_fact_harness() -> list[FactInjection]:
    """Default fact-injection set used when no domain harness is provided."""
    return [
        FactInjection("f1", "User's favorite color is teal.", "What is my favorite color?", ["teal"]),
        FactInjection("f2", "User lives in Vancouver, BC.", "Where do I live?", ["Vancouver", "BC"]),
        FactInjection("f3", "User's daughter is named Maya.", "What's my daughter's name?", ["Maya"]),
        FactInjection("f4", "User dislikes coriander.", "Should I add coriander to your dish?", ["coriander"]),
        FactInjection("f5", "User uses Postgres in production.", "What database do I use in prod?", ["Postgres"]),
    ]


def _iter_memory_runs(trace: Any | None):
    if trace is None:
        return
    stack = [trace]
    while stack:
        r = stack.pop()
        nm = (getattr(r, "name", "") or "").lower()
        if any(k in nm for k in _MEMORY_NAME_KEYWORDS):
            yield r
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)


def _classify_memory_op(name: str) -> str:
    nm = name.lower()
    if _DELETE_VERBS.search(nm):
        return "delete"
    if _READ_VERBS.search(nm):
        return "read"
    if _WRITE_VERBS.search(nm):
        return "write"
    return "unknown"


def _extract_message_history(trace: Any | None) -> list[str]:
    """Pull every textual message we can find from the trace inputs/outputs."""
    if trace is None:
        return []
    out: list[str] = []
    stack = [trace]
    while stack:
        r = stack.pop()
        for attr in ("inputs", "outputs"):
            obj = getattr(r, attr, None) or {}
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        out.append(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                out.append(item)
                            elif isinstance(item, dict) and "content" in item:
                                if isinstance(item["content"], str):
                                    out.append(item["content"])
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out


# ---------------------------------------------------------------------------- evaluators


class MemoryRetrievalRecallEvaluator(BaseEvaluator):
    """% of relevant past memories surfaced when needed.

    Modes:
      A. fact_harness=<runner_callable>: injects facts via memory write API,
         then re-queries via the runner and checks the answer mentions the fact.
      B. heuristic (default): walks the trace and asks the judge whether the retrieved
         memories cover the relevant aspects of the query.
    """

    name = "memory_retrieval_recall"

    def __init__(self, fact_harness: Any | None = None, harness_facts: list[FactInjection] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.fact_harness = fact_harness
        self.harness_facts = harness_facts or default_fact_harness()

    async def _evaluate_native(self, example, trace=None):
        if self.fact_harness is not None:
            return await self._evaluate_with_harness()
        return await self._evaluate_heuristic(example, trace)

    async def _evaluate_with_harness(self):
        """Inject N facts, query for each, score recall."""
        recall_hits = 0
        per_fact: list[dict[str, Any]] = []
        for fact in self.harness_facts:
            try:
                # Store phase
                await _maybe_async(self.fact_harness, {"op": "store", "text": fact.fact_text, "id": fact.fact_id})
                # Recall phase — query the agent and read its answer.
                response = await _maybe_async(self.fact_harness, {"op": "query", "text": fact.related_query})
                answer = _extract_answer_str(response)
                hit = any(kw.lower() in answer.lower() for kw in fact.expected_keywords)
                if hit:
                    recall_hits += 1
                per_fact.append({"fact_id": fact.fact_id, "recalled": hit, "answer": answer[:150]})
            except Exception as e:
                per_fact.append({"fact_id": fact.fact_id, "error": str(e)})
        score = recall_hits / max(1, len(self.harness_facts))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"mode": "fact_harness", "n_facts": len(self.harness_facts), "recalled": recall_hits, "per_fact": per_fact},
            flagged=score < 0.5,
        )

    async def _evaluate_heuristic(self, example, trace):
        from agent_eval.judges.rubric_judge import AnswerQualityJudge

        runs = list(_iter_memory_runs(trace))
        if not runs:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no memory ops observed"})
        judge = self.judge or AnswerQualityJudge()
        retrievals = [str(getattr(r, "outputs", None) or {})[:1000] for r in runs[:5]]
        query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
        joined = "\n---\n".join(retrievals)
        res = await judge.judge(query=str(query), answer=joined)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=res.score, passed=res.score >= self.threshold, threshold=self.threshold,
            details={"mode": "heuristic", "sample_size": len(retrievals), "verdict": res.verdict},
            flagged=res.score < 0.5,
        )


class MemoryRetrievalPrecisionEvaluator(BaseEvaluator):
    """For each retrieval, what fraction of returned memories are relevant?"""

    name = "memory_retrieval_precision"

    async def _evaluate_native(self, example, trace=None):
        from agent_eval.judges.rubric_judge import HallucinationJudge

        runs = list(_iter_memory_runs(trace))
        runs = [r for r in runs if _classify_memory_op(getattr(r, "name", "")) == "read"]
        if not runs:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no memory read ops observed"})
        judge = self.judge or HallucinationJudge()
        scores = []
        for r in runs[:5]:
            o = getattr(r, "outputs", None) or {}
            results = []
            if isinstance(o, dict):
                for v in o.values():
                    if isinstance(v, list):
                        results.extend(str(x) for x in v[:5])
                    elif isinstance(v, str):
                        results.append(v[:500])
            if not results:
                continue
            query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
            res = await judge.judge(claims=str(query), sources=results)
            scores.append(res.score)
        if not scores:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no per-retrieval items extractable"})
        avg = sum(scores) / len(scores)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=avg, passed=avg >= self.threshold, threshold=self.threshold,
            details={"sample_size": len(scores)},
            flagged=avg < 0.5,
        )


class MemoryWriteQualityEvaluator(BaseEvaluator):
    """Detect over-storing and under-storing patterns.

    Heuristics:
      - over-storing: > 0.8 of conversational messages produce a memory write.
        Penalizes the "write everything verbatim" anti-pattern.
      - under-storing: messages that mention durable, named entities ("My favorite X is...")
        without a corresponding write within K turns. Penalizes missed key facts.
      - LLM judge refines the score on a sample of write decisions.
    """

    name = "memory_write_quality"

    def __init__(self, sample_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size

    async def _evaluate_native(self, example, trace=None):
        runs = list(_iter_memory_runs(trace))
        write_runs = [r for r in runs if _classify_memory_op(getattr(r, "name", "")) == "write"]
        messages = _extract_message_history(trace)
        n_messages = max(1, len(messages))
        write_ratio = len(write_runs) / n_messages

        notes = []
        scores = []

        # Over-storing check.
        if write_ratio > 0.8:
            scores.append(0.3)
            notes.append(f"over-storing detected: {len(write_runs)} writes / {n_messages} messages = {write_ratio:.0%}")
        elif write_ratio > 0.5:
            scores.append(0.6)
            notes.append(f"high write ratio: {write_ratio:.0%}")
        else:
            scores.append(1.0)

        # Under-storing check.
        durable_signals = sum(1 for m in messages if _looks_like_durable_fact(m))
        if durable_signals > 0 and len(write_runs) == 0:
            scores.append(0.0)
            notes.append(f"under-storing: {durable_signals} durable-fact messages but 0 writes")
        elif durable_signals > 2 * max(1, len(write_runs)):
            scores.append(0.4)
            notes.append(f"likely under-storing: {durable_signals} durable signals, only {len(write_runs)} writes")
        else:
            scores.append(1.0)

        # LLM refinement on sample.
        if write_runs and self.judge is not None:
            from agent_eval.judges.rubric_judge import AnswerQualityJudge

            judge = self.judge or AnswerQualityJudge()
            judged = []
            for r in write_runs[: self.sample_size]:
                inp = str(getattr(r, "inputs", None) or {})[:800]
                judge_score = await judge.judge(
                    query="Is this a high-quality, durable fact worth storing in long-term memory?",
                    answer=inp,
                )
                judged.append(judge_score.score)
            if judged:
                scores.append(sum(judged) / len(judged))

        score = sum(scores) / len(scores)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"writes": len(write_runs), "messages": n_messages, "write_ratio": round(write_ratio, 3),
                      "durable_signals": durable_signals, "notes": notes},
            flagged=score < 0.5,
        )


class MemoryStalenessEvaluator(BaseEvaluator):
    """% of retrieved memories that are outdated or contradicted by newer info.

    For each memory read in the trace, check whether any later message in the same
    trace contradicts the retrieved content (NLI when available, content-overlap
    otherwise).
    """

    name = "memory_staleness"

    async def _evaluate_native(self, example, trace=None):
        from agent_eval.evaluators.nli import check_entailment

        runs = list(_iter_memory_runs(trace))
        reads = [r for r in runs if _classify_memory_op(getattr(r, "name", "")) == "read"]
        if not reads:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no memory reads observed"})
        all_messages = _extract_message_history(trace)
        # For each retrieved memory, is it contradicted by any later message?
        n_stale = 0
        n_total = 0
        per_check = []
        for r in reads:
            o = getattr(r, "outputs", None) or {}
            retrieved_texts = []
            if isinstance(o, dict):
                for v in o.values():
                    if isinstance(v, str):
                        retrieved_texts.append(v[:800])
                    elif isinstance(v, list):
                        retrieved_texts.extend(str(x)[:800] for x in v[:5])
            for mem_text in retrieved_texts:
                n_total += 1
                # Check against all message history (we don't have ordering reliable enough to
                # split "before" vs "after" — checking all is conservative).
                verdict = check_entailment(mem_text, all_messages or [""])
                is_stale = verdict.is_contradicted
                if is_stale:
                    n_stale += 1
                per_check.append({"memory": mem_text[:120], "stale": is_stale,
                                   "contradiction_score": round(verdict.contradiction_score, 3)})
                if len(per_check) >= 20:
                    break
            if len(per_check) >= 20:
                break
        if n_total == 0:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no retrieved memory texts to check"})
        # Score = 1 - staleness_rate. Higher is better.
        staleness_rate = n_stale / n_total
        score = 1.0 - staleness_rate
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"n_checked": n_total, "n_stale": n_stale, "staleness_rate": round(staleness_rate, 3),
                      "examples": per_check[:5]},
            flagged=score < 0.5,
        )


class CrossSessionContinuityEvaluator(BaseEvaluator):
    """Did the agent recall info from a prior session when it should have?

    Modes:
      A. session_runner=<callable(input, session_id) -> response>: drives a multi-session
         test — store fact in session A, query in session B, check recall.
      B. heuristic: walks the trace looking for queries that depend on history (markers like
         "earlier", "you said", "I told you") and checks whether memory was queried.
    """

    name = "cross_session_continuity"

    def __init__(self, session_runner: Any | None = None, **kwargs):
        super().__init__(**kwargs)
        self.session_runner = session_runner

    async def _evaluate_native(self, example, trace=None):
        if self.session_runner is not None:
            return await self._evaluate_with_runner(example)
        return await self._evaluate_heuristic(trace)

    async def _evaluate_with_runner(self, example):
        # Two-session test using one synthetic fact.
        fact = default_fact_harness()[0]
        try:
            await _maybe_async(self.session_runner, {"input": fact.fact_text, "session_id": "session_A"})
            response = await _maybe_async(self.session_runner, {"input": fact.related_query, "session_id": "session_B"})
            answer = _extract_answer_str(response)
            recalled = any(kw.lower() in answer.lower() for kw in fact.expected_keywords)
            score = 1.0 if recalled else 0.0
            return EvaluatorResult(
                evaluator_name=self.name, component_name="memory",
                score=score, passed=recalled, threshold=self.threshold,
                details={"mode": "session_runner", "fact": fact.fact_text, "recalled": recalled, "answer": answer[:200]},
                flagged=not recalled,
            )
        except Exception as e:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="memory",
                score=0.0, passed=False, threshold=self.threshold,
                details={"mode": "session_runner", "error": str(e)},
                flagged=True,
            )

    async def _evaluate_heuristic(self, trace):
        messages = _extract_message_history(trace)
        if not messages:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no message history"})
        history_markers = re.compile(r"(earlier|previously|you said|I (?:told|mentioned)|last time|that thing we (?:talked|discussed))", re.IGNORECASE)
        history_queries = sum(1 for m in messages if history_markers.search(m))
        if history_queries == 0:
            return EvaluatorResult(evaluator_name=self.name, component_name="memory",
                                    score=1.0, passed=True, threshold=self.threshold,
                                    details={"reason": "no history-dependent queries observed"})
        memory_reads = sum(1 for r in _iter_memory_runs(trace) if _classify_memory_op(getattr(r, "name", "")) == "read")
        # Heuristic: if there are history queries, we expect at least some memory reads.
        ratio = min(1.0, memory_reads / max(1, history_queries))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=ratio, passed=ratio >= self.threshold, threshold=self.threshold,
            details={"history_queries": history_queries, "memory_reads": memory_reads, "ratio": round(ratio, 3)},
            flagged=ratio < 0.5,
        )


class MemoryCostEvaluator(BaseEvaluator):
    name = "memory_cost_per_query"

    def __init__(self, max_usd: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_usd = max_usd

    async def _evaluate_native(self, example, trace=None):
        runs = list(_iter_memory_runs(trace))
        # Per-call ~1500 input + 200 output tokens at gpt-4o-mini-ish embedding/reranker rates.
        per_call_usd = (1500 / 1_000_000) * 0.15 + (200 / 1_000_000) * 0.60
        total = len(runs) * per_call_usd
        score = 1.0 if total <= self.max_usd else max(0.0, 1.0 - (total - self.max_usd) / self.max_usd)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="memory",
            score=score, passed=total <= self.max_usd, threshold=self.max_usd,
            details={"memory_calls": len(runs), "approx_cost_usd": round(total, 4)},
            flagged=total > self.max_usd,
        )


# ---------------------------------------------------------------------------- helpers


_DURABLE_PATTERNS = re.compile(
    r"(my (?:name|email|phone|address|favorite|preferred|wife|husband|partner|child|kid|son|daughter|"
    r"birthday|anniversary|company|job|role|team|manager|allergy|allerg|dietary)|"
    r"I (?:live|work|study|use|prefer|hate|love|am allergic|don't eat|don't drink))",
    re.IGNORECASE,
)


def _looks_like_durable_fact(message: str) -> bool:
    return bool(_DURABLE_PATTERNS.search(message))


def _extract_answer_str(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for k in ("answer", "final_answer", "summary", "output", "response"):
            v = response.get(k)
            if isinstance(v, str):
                return v
        return str(response)
    return str(response)


async def _maybe_async(fn: Any, arg: Any) -> Any:
    import asyncio

    if asyncio.iscoroutinefunction(fn):
        return await fn(arg)
    return fn(arg)
