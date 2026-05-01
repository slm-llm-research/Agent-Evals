"""Trace harvester — extract EvalExamples from LangSmith traces.

MVP: scores by completion_rate, latency, tool success rate; dedupes by
exact-input hash. Sentence-transformer cosine dedup is in NEXT_STEPS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_eval.dataset.schema import EvalExample


@dataclass
class _TraceScore:
    completion_rate: float
    latency_score: float
    tool_success_rate: float

    @property
    def overall(self) -> float:
        return (self.completion_rate + self.latency_score + self.tool_success_rate) / 3.0


class TraceHarvester:
    def __init__(self, client: Any):
        self.client = client

    def harvest_from_langsmith(
        self,
        project_name: str,
        n_traces: int = 200,
        quality_threshold: float = 0.7,
        lookback_days: int = 14,
    ) -> list[EvalExample]:
        try:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            runs = list(
                self.client.list_runs(
                    project_name=project_name,
                    start_time=since,
                    run_type="chain",
                    limit=n_traces,
                )
            )
        except Exception:
            return []

        examples: list[EvalExample] = []
        for run in runs:
            score = self._score_trace(run)
            if score.overall < quality_threshold:
                continue
            ex = self.extract_example_from_trace(run)
            if ex is not None:
                examples.append(ex)
        return self.deduplicate(examples)

    def extract_example_from_trace(self, trace: Any) -> EvalExample | None:
        inputs = getattr(trace, "inputs", None) or {}
        outputs = getattr(trace, "outputs", None) or {}
        if not inputs:
            return None
        tool_seq = []
        for run in _iter_runs(trace):
            if (run.get("run_type") or "") == "tool" and run.get("name"):
                tool_seq.append(run["name"])
        return EvalExample(
            input=dict(inputs),
            expected_output=dict(outputs) if outputs else None,
            reference_answer=_extract_answer(outputs),
            expected_tool_sequence=tool_seq,
            created_by="harvested",
            tags=["harvested", f"trace:{getattr(trace, 'id', 'unknown')}"],
        )

    def deduplicate(self, examples: list[EvalExample], embedding_threshold: float = 0.92) -> list[EvalExample]:
        """Two-pass dedup: exact-input hash, then embedding cosine similarity.

        Embedding pass only runs if `sentence-transformers` is installed and
        threshold > 0. Otherwise exact-only.
        """
        # Pass 1: exact hash.
        seen_hashes: set[str] = set()
        deduped: list[EvalExample] = []
        for ex in examples:
            key = hashlib.sha256(json.dumps(ex.input, sort_keys=True, default=str).encode()).hexdigest()
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            deduped.append(ex)

        # Pass 2: embedding cosine ≥ threshold → drop.
        if embedding_threshold <= 0:
            return deduped
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            from agent_eval.config import get_config

            model = SentenceTransformer(get_config().embedding_model)
            queries = [_to_query_text(ex.input) for ex in deduped]
            valid_idx = [i for i, q in enumerate(queries) if q]
            if len(valid_idx) < 2:
                return deduped
            embs = model.encode([queries[i] for i in valid_idx], convert_to_numpy=True, normalize_embeddings=True)
            keep = [True] * len(deduped)
            for a in range(len(valid_idx)):
                if not keep[valid_idx[a]]:
                    continue
                for b in range(a + 1, len(valid_idx)):
                    if not keep[valid_idx[b]]:
                        continue
                    sim = float((embs[a] * embs[b]).sum())
                    if sim >= embedding_threshold:
                        keep[valid_idx[b]] = False
            return [ex for i, ex in enumerate(deduped) if keep[i]]
        except Exception:
            return deduped

    # --------------------------------------------------------------- internal

    def _score_trace(self, trace: Any) -> _TraceScore:
        completion = 0.0 if getattr(trace, "error", None) else 1.0
        # Latency score: 1.0 if <10s, decays to 0 by 60s.
        s, e = getattr(trace, "start_time", None), getattr(trace, "end_time", None)
        if isinstance(s, datetime) and isinstance(e, datetime):
            secs = (e - s).total_seconds()
            latency_score = max(0.0, min(1.0, 1.0 - max(0.0, secs - 10.0) / 50.0))
        else:
            latency_score = 0.5
        tool_total = 0
        tool_ok = 0
        for run in _iter_runs(trace):
            if (run.get("run_type") or "") == "tool":
                tool_total += 1
                if not run.get("error"):
                    tool_ok += 1
        tool_score = (tool_ok / tool_total) if tool_total else 1.0
        return _TraceScore(completion, latency_score, tool_score)


def _iter_runs(trace: Any):
    stack = [trace]
    while stack:
        r = stack.pop()
        yield {
            "name": getattr(r, "name", None),
            "run_type": getattr(r, "run_type", None),
            "error": getattr(r, "error", None),
        }
        for c in getattr(r, "child_runs", None) or []:
            stack.append(c)


def _to_query_text(inputs: Any) -> str:
    if isinstance(inputs, dict):
        for k in ("query", "question", "input", "text", "prompt", "message"):
            v = inputs.get(k)
            if isinstance(v, str):
                return v
        return json.dumps(inputs, sort_keys=True, default=str)[:500]
    return str(inputs)[:500]


def _extract_answer(outputs: dict[str, Any] | None) -> str | None:
    if not isinstance(outputs, dict):
        return None
    for k in ("answer", "final_answer", "summary", "output", "response", "research_findings"):
        v = outputs.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for inner in ("text", "content", "answer"):
                if isinstance(v.get(inner), str):
                    return v[inner]
    return None
