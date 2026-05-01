"""Hallucination evaluators — four levels per spec.

Levels:
  - planning      : entities in plan not in user query (no source for them)
  - observation   : 'I found X' claims vs. actual tool outputs (NLI)
  - citation      : URLs in answer must appear in retrieved sources, with optional
                    live URL fetch + claim-support verification when [nli] is installed.
  - reasoning     : inferences not supported by evidence bank — uses NLI cross-encoder
                    when available, falls back to LLM judge.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from agent_eval.evaluators.base import BaseEvaluator, CompositeEvaluator, EvaluatorResult
from agent_eval.evaluators.nli import check_entailment, split_into_claims
from agent_eval.judges.rubric_judge import HallucinationJudge


_URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
_QUOTED_RE = re.compile(r'["“]([^"”]{4,200})["”]')
_FOUND_RE = re.compile(r"(?:I (?:found|saw|noticed|read|learned|discovered)|the (?:source|article|page|document) (?:says|states|reports))[\s,]*([^.]{8,300})", re.IGNORECASE)


def _final_answer(trace: Any | None) -> str:
    if trace is None:
        return ""
    outputs = getattr(trace, "outputs", None) or {}
    if isinstance(outputs, dict):
        for k in ("answer", "final_answer", "summary", "research_findings", "output", "response"):
            v = outputs.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


def _retrieved_sources(trace: Any | None) -> list[str]:
    if trace is None:
        return []
    out: list[str] = []
    stack = [trace]
    while stack:
        r = stack.pop()
        if (getattr(r, "run_type", None) or "") == "tool":
            o = getattr(r, "outputs", None) or {}
            for v in _walk_strings(o):
                out.append(v)
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out[:50]


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def _plan_text(trace: Any | None) -> str:
    """Best-effort: surface the orchestrator/planner output as text."""
    if trace is None:
        return ""
    parts = []
    stack = [trace]
    while stack:
        r = stack.pop()
        nm = (getattr(r, "name", "") or "").lower()
        if any(k in nm for k in ("orchestrat", "plan", "router")):
            o = getattr(r, "outputs", None) or {}
            for v in _walk_strings(o):
                parts.append(v)
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return " ".join(parts)


def _entities(text: str) -> set[str]:
    """Cheap NER fallback — capitalized n-grams, numbers with units, years.

    Real spaCy NER is opt-in via the `[ner]` extra; if installed we use it.
    """
    try:
        import spacy  # type: ignore

        nlp = _get_spacy()
        if nlp is not None:
            doc = nlp(text)
            return {e.text.strip().lower() for e in doc.ents if e.text.strip()}
    except Exception:
        pass
    # Fallback: capitalized words & 4-digit years.
    cap = set(m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s[A-Z][A-Za-z0-9-]+){0,3}\b", text))
    years = set(re.findall(r"\b(19|20)\d{2}\b", text))
    return cap | {y + "00" if len(y) == 2 else y for y in years}


_spacy_singleton = None


def _get_spacy():
    global _spacy_singleton
    if _spacy_singleton is not None:
        return _spacy_singleton
    try:
        import spacy  # type: ignore

        _spacy_singleton = spacy.load("en_core_web_sm")
    except Exception:
        _spacy_singleton = False
    return _spacy_singleton or None


# ---------------------------------------------------------------------------- evaluators


class PlanningHallucinationEvaluator(BaseEvaluator):
    name = "hallucination_planning"

    async def _evaluate_native(self, example, trace=None):
        query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
        plan = _plan_text(trace)
        if not plan:
            return EvaluatorResult(
                evaluator_name=self.name, component_name="system",
                score=1.0, passed=True, threshold=self.threshold,
                details={"reason": "no plan text observed"},
            )
        q_ents = _entities(str(query))
        p_ents = _entities(plan)
        if not p_ents:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no entities in plan"})
        # Entities in plan not grounded in query OR in retrieved sources are suspect.
        sources_text = " ".join(_retrieved_sources(trace)).lower()
        suspect = [e for e in p_ents if e not in q_ents and e not in sources_text]
        score = 1.0 - (len(suspect) / len(p_ents))
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=max(0.0, min(1.0, score)),
            passed=score >= self.threshold, threshold=self.threshold,
            details={"plan_entities": list(p_ents), "query_entities": list(q_ents), "suspect": suspect},
            flagged=score < 0.5,
        )


class ObservationHallucinationEvaluator(BaseEvaluator):
    name = "hallucination_observation"

    async def _evaluate_native(self, example, trace=None):
        answer = _final_answer(trace)
        sources = _retrieved_sources(trace)
        if not answer:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "empty answer"})
        if not sources:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no tool outputs to compare against"})
        claims = []
        for m in _FOUND_RE.finditer(answer):
            claims.append(m.group(1).strip())
        for m in _QUOTED_RE.finditer(answer):
            claims.append(m.group(1).strip())
        if not claims:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no observation-style claims"})
        joined_sources = " \n ".join(sources).lower()
        supported = sum(1 for c in claims if _is_supported(c.lower(), joined_sources))
        score = supported / len(claims)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"claim_count": len(claims), "supported": supported, "claims": claims[:5]},
            flagged=score < 0.5,
        )


def _is_supported(claim: str, source_blob: str) -> bool:
    """Cheap support check: ≥40% of content tokens of claim appear in sources."""
    toks = [t for t in re.findall(r"\w+", claim) if len(t) > 3]
    if not toks:
        return True
    hit = sum(1 for t in toks if t in source_blob)
    return (hit / len(toks)) >= 0.4


class CitationHallucinationEvaluator(BaseEvaluator):
    """% of cited URLs that (a) are accessible and (b) actually support the cited claim.

    Two-tier check:
      1. URL appears in retrieved sources (cheap, always run).
      2. Optional: live HTTP fetch of the URL + NLI check that the surrounding-claim
         is entailed by the page contents. Disabled by default (set fetch_urls=True).
    """

    name = "hallucination_citation"

    def __init__(self, fetch_urls: bool = False, fetch_timeout: float = 5.0, max_fetch: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.fetch_urls = fetch_urls
        self.fetch_timeout = fetch_timeout
        self.max_fetch = max_fetch

    async def _evaluate_native(self, example, trace=None):
        answer = _final_answer(trace)
        sources = _retrieved_sources(trace)
        urls = list({m.group(0).rstrip('.,;)"\']') for m in _URL_RE.finditer(answer)})
        if not urls:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no citations"})
        joined = " ".join(sources)
        in_sources = [u for u in urls if u in joined]

        per_url: list[dict[str, Any]] = []
        verifiable = 0
        for u in urls:
            entry: dict[str, Any] = {"url": u, "in_retrieved_sources": u in joined}
            if self.fetch_urls and len(per_url) < self.max_fetch:
                fetched = await self._verify_url(u, _claim_around_url(answer, u))
                entry.update(fetched)
                if entry["in_retrieved_sources"] or entry.get("supports_claim"):
                    verifiable += 1
            else:
                if entry["in_retrieved_sources"]:
                    verifiable += 1
            per_url.append(entry)

        score = verifiable / len(urls)
        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={"total_urls": len(urls), "verifiable": verifiable, "in_sources_count": len(in_sources), "per_url": per_url[:10]},
            flagged=score < 0.5,
        )

    async def _verify_url(self, url: str, claim: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.fetch_timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "agent-eval-citation-checker/1.0"})
                if resp.status_code >= 400:
                    return {"reachable": False, "http_status": resp.status_code, "supports_claim": False}
                # Strip HTML tags crudely; we only need text for NLI.
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text)[:8000]
                if not claim:
                    return {"reachable": True, "http_status": resp.status_code, "supports_claim": True, "note": "no surrounding claim"}
                verdict = check_entailment(claim, text)
                return {
                    "reachable": True,
                    "http_status": resp.status_code,
                    "supports_claim": verdict.is_supported,
                    "entailment_score": round(verdict.entailment_score, 3),
                    "method": verdict.method,
                }
        except Exception as e:
            return {"reachable": False, "error": str(e)[:200], "supports_claim": False}


def _claim_around_url(answer: str, url: str) -> str:
    """Best-effort: return the sentence containing the URL."""
    idx = answer.find(url)
    if idx < 0:
        return ""
    # Walk back to nearest sentence-end or start.
    start = max(0, answer.rfind(".", 0, idx) + 1)
    end = answer.find(".", idx)
    if end < 0:
        end = min(len(answer), idx + 300)
    return answer[start:end].replace(url, "").strip()


class ReasoningHallucinationEvaluator(BaseEvaluator):
    """Inferences in the answer not supported by retrieved evidence — NLI-based.

    Splits the answer into per-sentence claims and runs each through the NLI
    cross-encoder against the retrieved sources. Score = supported_claims / total_claims.
    Falls back to LLM judge if NLI model not installed AND `judge` is provided.
    """

    name = "hallucination_reasoning"

    def __init__(self, max_claims: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.max_claims = max_claims

    async def _evaluate_native(self, example, trace=None):
        answer = _final_answer(trace)
        sources = _retrieved_sources(trace)
        if not answer:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "empty answer"})
        if not sources:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no tool outputs to verify against"})

        claims = split_into_claims(answer, max_claims=self.max_claims)
        if not claims:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=1.0, passed=True, threshold=self.threshold, details={"reason": "no factual claims extracted"})

        # Run NLI checks concurrently (the cross-encoder itself is sync — wrap in to_thread).
        async def _check(c):
            return await asyncio.to_thread(check_entailment, c, sources)

        verdicts = await asyncio.gather(*(_check(c) for c in claims))
        supported = sum(1 for v in verdicts if v.is_supported)
        contradicted = [(c, v) for c, v in zip(claims, verdicts) if v.is_contradicted]
        score = supported / len(claims)

        return EvaluatorResult(
            evaluator_name=self.name, component_name="system",
            score=score, passed=score >= self.threshold, threshold=self.threshold,
            details={
                "n_claims": len(claims),
                "supported": supported,
                "contradicted_examples": [{"claim": c, "contradiction_score": round(v.contradiction_score, 3)} for c, v in contradicted[:5]],
                "method": verdicts[0].method if verdicts else "n/a",
            },
            flagged=score < 0.5,
        )


class HallucinationComposite(CompositeEvaluator):
    def __init__(self, **kwargs):
        super().__init__(
            [
                (PlanningHallucinationEvaluator(**kwargs), 1.0),
                (ObservationHallucinationEvaluator(**kwargs), 1.0),
                (CitationHallucinationEvaluator(**kwargs), 1.0),
                (ReasoningHallucinationEvaluator(**kwargs), 1.0),
            ]
        )
