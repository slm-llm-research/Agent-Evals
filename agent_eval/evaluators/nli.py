"""NLI (Natural Language Inference) helpers for hallucination detection.

Uses a HuggingFace cross-encoder when available (`[nli]` extra), falls back to
content-overlap heuristic otherwise. Single shared model singleton to amortize
load cost across many evaluator calls.

The default model is `cross-encoder/nli-deberta-v3-small` per the spec; it
returns scores for {entailment, neutral, contradiction}. We treat
entailment ≥ 0.5 as "supported."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class NLIVerdict:
    is_supported: bool
    is_contradicted: bool
    entailment_score: float
    contradiction_score: float
    method: str  # "cross_encoder" | "content_overlap"


_model_singleton: Any = None
_model_load_attempted = False


def _get_nli_model():
    """Lazy-load a cross-encoder NLI model. Returns None if not available."""
    global _model_singleton, _model_load_attempted
    if _model_load_attempted:
        return _model_singleton
    _model_load_attempted = True
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        _model_singleton = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    except Exception:
        _model_singleton = None
    return _model_singleton


def check_entailment(claim: str, evidence: str | list[str], threshold: float = 0.5) -> NLIVerdict:
    """Decide whether `claim` is entailed by `evidence`.

    If `evidence` is a list, takes the maximum entailment / contradiction across pieces.
    """
    if not claim:
        return NLIVerdict(True, False, 1.0, 0.0, method="empty")
    pieces = evidence if isinstance(evidence, list) else [evidence]
    pieces = [p for p in pieces if p]
    if not pieces:
        return NLIVerdict(False, False, 0.0, 0.0, method="no_evidence")

    model = _get_nli_model()
    if model is None:
        return _fallback_overlap(claim, pieces, threshold)

    try:
        # Cross-encoder NLI returns 3-class logits → softmax to probabilities.
        # Format: [premise, hypothesis] pairs.
        pairs = [(p[:800], claim[:400]) for p in pieces[:10]]
        scores = model.predict(pairs)  # shape (n, 3) — [contradiction, entailment, neutral]
        if scores is None or len(scores) == 0:
            return _fallback_overlap(claim, pieces, threshold)
        # Convert logits to probabilities.
        import math

        best_entail = 0.0
        best_contra = 0.0
        for row in scores:
            row = list(row)
            mx = max(row)
            exps = [math.exp(x - mx) for x in row]
            tot = sum(exps)
            probs = [e / tot for e in exps]  # [contradiction, entailment, neutral]
            best_contra = max(best_contra, probs[0])
            best_entail = max(best_entail, probs[1])
        return NLIVerdict(
            is_supported=best_entail >= threshold,
            is_contradicted=best_contra >= 0.5,
            entailment_score=best_entail,
            contradiction_score=best_contra,
            method="cross_encoder",
        )
    except Exception:
        return _fallback_overlap(claim, pieces, threshold)


def _fallback_overlap(claim: str, pieces: list[str], threshold: float) -> NLIVerdict:
    """Content-token overlap: ≥40% of claim tokens appear in some piece → supported."""
    toks = [t.lower() for t in re.findall(r"\w+", claim) if len(t) > 3]
    if not toks:
        return NLIVerdict(True, False, 1.0, 0.0, method="content_overlap")
    blob = " ".join(p.lower() for p in pieces)
    hits = sum(1 for t in toks if t in blob)
    score = hits / len(toks)
    return NLIVerdict(
        is_supported=score >= 0.4,
        is_contradicted=False,
        entailment_score=score,
        contradiction_score=0.0,
        method="content_overlap",
    )


def split_into_claims(text: str, max_claims: int = 30) -> list[str]:
    """Cheap claim splitter: split on sentence boundaries, drop short fragments."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 8 or len(p) > 600:
            continue
        out.append(p)
        if len(out) >= max_claims:
            break
    return out
