"""LangSmithReplayRunner — match dataset examples to existing production traces.

Useful when:
  - You don't want to re-run the agent (cost / external side effects).
  - You want to evaluate against real-user production traffic.
  - You're regression-testing a deploy by replaying yesterday's traces.

Match priority:
  1. `extra.metadata.eval_example_id == example.id`
  2. Substring match between `example.input.query` and `run.inputs.query`
  3. Embedding cosine ≥0.85 (only if sentence-transformers is installed)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_eval.config import get_config
from agent_eval.dataset.schema import EvalExample
from agent_eval.runners.base import AgentRunner, RunResult


@dataclass
class LangSmithReplayRunner(AgentRunner):
    """Args:

    project_name: LangSmith project to mine.
    lookback_days: How far back to look (default 14).
    fetch_limit: Max traces to scan (default 1000 — keep modest, this paginates).
    require_no_error: Only match traces that didn't error.
    embedding_threshold: Cosine threshold for the embedding fallback (default 0.85).
    client: Optional pre-built langsmith.Client (uses get_config() default otherwise).
    """

    project_name: str = ""
    lookback_days: int = 14
    fetch_limit: int = 1000
    require_no_error: bool = True
    embedding_threshold: float = 0.85
    client: Any = None
    max_concurrency: int = 5

    name: str = "langsmith_replay"

    def __post_init__(self):
        AgentRunner.__init__(self, max_concurrency=self.max_concurrency)
        if not self.project_name:
            raise ValueError("LangSmithReplayRunner requires project_name")
        if self.client is None:
            self.client = get_config().get_langsmith_client()
        self._cache: list[Any] | None = None
        self._cache_loaded = False

    async def run_one(self, example: EvalExample) -> RunResult:
        traces = self._traces()
        target_query = _coerce_query(example.input)

        # Path 1: explicit metadata tag.
        for t in traces:
            md = ((getattr(t, "extra", None) or {}).get("metadata") or {})
            if md.get("eval_example_id") == example.id:
                return self._wrap(example, t)

        # Path 2: substring match.
        if target_query:
            target_clip = target_query[:80]
            for t in traces:
                inp_str = _coerce_query(getattr(t, "inputs", None) or {}) or ""
                if target_clip and target_clip in inp_str:
                    return self._wrap(example, t)

        # Path 3: embedding cosine (best-effort).
        emb_match = self._best_embedding_match(target_query, traces)
        if emb_match is not None:
            return self._wrap(example, emb_match)

        return RunResult(example_id=example.id, output=None, trace=None,
                         error=f"no matching trace found in project '{self.project_name}'")

    # ------------------------------------------------------------- internal

    def _traces(self) -> list[Any]:
        if self._cache_loaded:
            return self._cache or []
        try:
            since = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=since,
                run_type="chain",
                limit=self.fetch_limit,
            ))
            if self.require_no_error:
                runs = [r for r in runs if not getattr(r, "error", None)]
            self._cache = runs
        except Exception as e:
            print(f"[LangSmithReplayRunner] failed to list runs: {e}")
            self._cache = []
        self._cache_loaded = True
        return self._cache

    def _wrap(self, example: EvalExample, run: Any) -> RunResult:
        try:
            full = self.client.read_run(str(run.id), load_child_runs=True)
        except Exception:
            full = run
        outputs = getattr(full, "outputs", None) or {}
        return RunResult(
            example_id=example.id,
            output=outputs if isinstance(outputs, dict) else {"output": outputs},
            trace=full,
            metadata={"matched_run_id": str(getattr(run, "id", "?"))},
        )

    def _best_embedding_match(self, target: str | None, traces: list[Any]) -> Any | None:
        if not target or not traces:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(get_config().embedding_model)
            sources = [_coerce_query(getattr(t, "inputs", None) or {}) or "" for t in traces]
            sources_keep = [(t, s) for t, s in zip(traces, sources) if s]
            if not sources_keep:
                return None
            emb = model.encode([target] + [s for _, s in sources_keep],
                                convert_to_numpy=True, normalize_embeddings=True)
            target_v = emb[0]
            best_score = -1.0
            best_t = None
            for i, (t, _) in enumerate(sources_keep):
                score = float((target_v * emb[i + 1]).sum())
                if score > best_score:
                    best_score = score
                    best_t = t
            if best_t is not None and best_score >= self.embedding_threshold:
                return best_t
            return None
        except Exception:
            return None


def _coerce_query(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k in ("query", "input", "question", "text", "prompt", "message"):
            v = obj.get(k)
            if isinstance(v, str):
                return v
        try:
            import json

            return json.dumps(obj, sort_keys=True, default=str)[:300]
        except Exception:
            return None
    if isinstance(obj, str):
        return obj
    return None
