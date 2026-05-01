"""Output quality evaluators."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from agent_eval.evaluators.base import BaseEvaluator, CompositeEvaluator, EvaluatorResult
from agent_eval.judges.rubric_judge import (
    AnswerQualityJudge,
    HallucinationJudge,
)


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
            if isinstance(o, dict):
                for v in o.values():
                    if isinstance(v, str):
                        out.append(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                out.append(item)
                            elif isinstance(item, dict):
                                for sub in item.values():
                                    if isinstance(sub, str):
                                        out.append(sub)
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)
    return out[:30]


# ---------------------------------------------------------------------------- evaluators


class TaskSuccessEvaluator(BaseEvaluator):
    name = "task_success_rate"

    async def _evaluate_native(self, example, trace=None):
        outputs = getattr(trace, "outputs", None) or {}
        error = getattr(trace, "error", None)
        ans = _final_answer(trace)
        if error:
            score = 0.0
        elif ans:
            score = 1.0
        elif isinstance(outputs, dict) and outputs:
            score = 0.5
        else:
            score = 0.0
        return EvaluatorResult(
            evaluator_name=self.name,
            component_name="system",
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={"has_answer": bool(ans), "has_error": bool(error)},
            flagged=score < 0.5,
        )


class AnswerFaithfulnessEvaluator(BaseEvaluator):
    """% of factual claims grounded in retrieved sources."""

    name = "answer_faithfulness"

    async def _evaluate_native(self, example, trace=None):
        answer = _final_answer(trace)
        sources = _retrieved_sources(trace)
        if not answer:
            return EvaluatorResult(
                evaluator_name=self.name,
                component_name="system",
                score=0.0,
                passed=False,
                threshold=self.threshold,
                details={"reason": "empty answer"},
                flagged=True,
                flag_reason="empty answer",
            )
        if not sources:
            # No sources retrieved — faithfulness is undefined; mark neutral.
            return EvaluatorResult(
                evaluator_name=self.name,
                component_name="system",
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "no sources retrieved — faithfulness undefined, defaulting to 1.0"},
            )
        judge = self.judge or HallucinationJudge()
        result = await judge.judge(claims=answer, sources=sources)
        return EvaluatorResult(
            evaluator_name=self.name,
            component_name="system",
            score=result.score,
            passed=result.score >= self.threshold,
            threshold=self.threshold,
            details={"verdict": result.verdict, "reasoning": result.reasoning[:500]},
            flagged=result.score < 0.5,
            flag_reason=None if result.score >= 0.5 else "low faithfulness",
        )


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Embedding cosine similarity between query and answer (or judge fallback)."""

    name = "answer_relevance"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = max(self.threshold, 0.65)

    async def _evaluate_native(self, example, trace=None):
        query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
        answer = _final_answer(trace)
        if not answer:
            return EvaluatorResult(
                evaluator_name=self.name,
                component_name="system",
                score=0.0,
                passed=False,
                threshold=self.threshold,
                details={"reason": "empty answer"},
                flagged=True,
            )
        score = await _embedding_similarity(str(query), answer)
        if score is None:
            score = await _judge_relevance(self.judge, str(query), answer)
        return EvaluatorResult(
            evaluator_name=self.name,
            component_name="system",
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={},
            flagged=score < 0.5,
        )


class CompletenessEvaluator(BaseEvaluator):
    """Is the answer covering all aspects of the question?"""

    name = "completeness"

    async def _evaluate_native(self, example, trace=None):
        query = example.input.get("query") if isinstance(example.input, dict) else str(example.input)
        answer = _final_answer(trace)
        if not answer:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=0.0, passed=False, threshold=self.threshold, details={"reason": "empty answer"}, flagged=True)
        judge = self.judge or AnswerQualityJudge()
        result = await judge.judge(query=str(query), answer=answer, reference=example.reference_answer)
        return EvaluatorResult(
            evaluator_name=self.name,
            component_name="system",
            score=result.score,
            passed=result.score >= self.threshold,
            threshold=self.threshold,
            details={"verdict": result.verdict, "reasoning": result.reasoning[:500]},
            flagged=result.score < 0.5,
        )


class FormatComplianceEvaluator(BaseEvaluator):
    """Deterministic format checks: word count, JSON validity, markdown for voice."""

    name = "format_compliance"

    def __init__(self, max_words: int | None = None, require_json: bool = False, no_markdown: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_words = max_words
        self.require_json = require_json
        self.no_markdown = no_markdown

    async def _evaluate_native(self, example, trace=None):
        answer = _final_answer(trace)
        details: dict[str, Any] = {}
        passed_checks = 0
        total_checks = 0
        if self.max_words is not None:
            total_checks += 1
            wc = len(answer.split())
            details["word_count"] = wc
            details["max_words"] = self.max_words
            if wc <= self.max_words:
                passed_checks += 1
        if self.require_json:
            total_checks += 1
            try:
                json.loads(answer)
                passed_checks += 1
                details["valid_json"] = True
            except Exception:
                details["valid_json"] = False
        if self.no_markdown:
            total_checks += 1
            has_md = bool(re.search(r"[*_`#]|\[.+?\]\(.+?\)", answer))
            details["has_markdown"] = has_md
            if not has_md:
                passed_checks += 1
        if total_checks == 0:
            score = 1.0
        else:
            score = passed_checks / total_checks
        return EvaluatorResult(
            evaluator_name=self.name,
            component_name="system",
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details=details,
            flagged=score < 0.5,
        )


class KeywordCoverageEvaluator(BaseEvaluator):
    """NEW v2 — fraction of expected_answer_keywords that appear in answer."""

    name = "keyword_coverage"

    async def _evaluate_native(self, example, trace=None):
        kws = example.expected_answer_keywords or []
        if not kws:
            return EvaluatorResult(
                evaluator_name=self.name,
                component_name="system",
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "no expected_answer_keywords — skipped"},
            )
        answer = _final_answer(trace).lower()
        if not answer:
            return EvaluatorResult(evaluator_name=self.name, component_name="system", score=0.0, passed=False, threshold=self.threshold, details={"reason": "empty answer"}, flagged=True)
        hit = sum(1 for k in kws if k.lower() in answer)
        score = hit / len(kws)
        return EvaluatorResult(
            evaluator_name=self.name,
            component_name="system",
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={"matched": hit, "total": len(kws), "keywords": kws},
            flagged=score < 0.5,
        )


# ---------------------------------------------------------------------------- composite


class OutputQualityComposite(CompositeEvaluator):
    def __init__(self, **kwargs):
        super().__init__(
            [
                (TaskSuccessEvaluator(**kwargs), 0.25),
                (AnswerFaithfulnessEvaluator(**kwargs), 0.20),
                (AnswerRelevanceEvaluator(**kwargs), 0.15),
                (CompletenessEvaluator(**kwargs), 0.15),
                (FormatComplianceEvaluator(**kwargs), 0.10),
                (KeywordCoverageEvaluator(**kwargs), 0.15),
            ]
        )


# ---------------------------------------------------------------------------- helpers


async def _embedding_similarity(a: str, b: str) -> float | None:
    """Cosine similarity via sentence-transformers if available."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = _get_embedding_model()
        if model is None:
            return None
        emb = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float((emb[0] * emb[1]).sum())
    except Exception:
        return None


_embedding_singleton = None


def _get_embedding_model():
    global _embedding_singleton
    if _embedding_singleton is not None:
        return _embedding_singleton
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        from agent_eval.config import get_config

        _embedding_singleton = SentenceTransformer(get_config().embedding_model)
        return _embedding_singleton
    except Exception:
        return None


async def _judge_relevance(judge, query: str, answer: str) -> float:
    """Fallback when no embedding model is installed: lexical Jaccard.

    A real LLM judge would be ideal, but since the AnswerRelevanceEvaluator does
    not own a relevance judge by default, we use a deterministic lexical overlap
    so the test suite passes without an LLM key set.
    """
    qt = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
    at = set(re.findall(r"\w+", answer.lower())) - _STOPWORDS
    if not qt:
        return 0.5
    overlap = len(qt & at) / len(qt)
    # Squash to [0.3, 1.0] — pure lexical is noisy.
    return 0.3 + 0.7 * overlap


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at",
    "for", "from", "by", "with", "as", "that", "this", "these", "those",
    "it", "its", "i", "you", "we", "they", "what", "who", "which", "how",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "have", "has", "had", "not", "no", "yes", "about",
}


def _ceil(x: float) -> int:
    return int(math.ceil(x))
