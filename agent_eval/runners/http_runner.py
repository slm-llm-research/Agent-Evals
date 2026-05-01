"""HttpAgentRunner — POST each example to the agent's HTTP endpoint.

Trace correlation strategy (in priority order):
  1. Response body keys: `langsmith_run_id`, `run_id`, `trace_id`.
  2. Response header: `X-Langsmith-Run-Id`.
  3. Poll LangSmith for runs in the request window matching the input
     (only when `langsmith_project` is configured).
  4. Fall back to a SyntheticTrace built from the response body.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx

from agent_eval.config import get_config
from agent_eval.dataset.schema import EvalExample
from agent_eval.runners.base import AgentRunner, RunResult
from agent_eval.runners.synthetic import build_synthetic_trace


_DEFAULT_RUN_ID_KEYS = ("langsmith_run_id", "run_id", "trace_id", "trace_run_id")


@dataclass
class HttpAgentRunner(AgentRunner):
    """POST `endpoint_url` with each example and recover the trace.

    Args:
        endpoint_url: Where to POST. The agent should accept a JSON body.
        method: HTTP verb (default POST).
        headers: Extra headers (e.g. auth).
        timeout: Per-request timeout in seconds.
        request_builder: Optional `(example) -> dict` to customize the JSON body.
            Default: pass `example.input` straight through.
        response_parser: Optional `(response_json) -> dict` to normalize the response
            into `{"answer": ..., "tool_calls": [...]}` shape. Default: pass through.
        run_id_keys: Response-body keys to check for a LangSmith run id.
        langsmith_project: If set, used to poll LangSmith when no run_id is in response.
        langsmith_wait_s: Max seconds to wait for LangSmith ingestion when polling.
        max_concurrency: Cap concurrent requests.
    """

    endpoint_url: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 60.0
    request_builder: Callable[[EvalExample], dict[str, Any]] | None = None
    response_parser: Callable[[Any], dict[str, Any]] | None = None
    run_id_keys: tuple[str, ...] = _DEFAULT_RUN_ID_KEYS
    langsmith_project: str | None = None
    langsmith_wait_s: float = 10.0
    max_concurrency: int = 5

    name: str = "http"

    def __post_init__(self):
        AgentRunner.__init__(self, max_concurrency=self.max_concurrency)
        if not self.endpoint_url:
            raise ValueError("HttpAgentRunner requires endpoint_url")
        self._client: httpx.AsyncClient | None = None
        self._lc_client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def run_one(self, example: EvalExample) -> RunResult:
        body = self.request_builder(example) if self.request_builder else example.input
        client = await self._ensure_client()
        # Tag this request so the agent can attach it to the LangSmith trace as metadata.
        request_headers = {"X-Eval-Example-Id": example.id, "X-Eval-Source": "agent-eval"}

        t_start = datetime.now(timezone.utc)
        try:
            resp = await client.request(
                self.method,
                self.endpoint_url,
                json=body if isinstance(body, dict) else {"input": body},
                headers=request_headers,
            )
        except Exception as e:
            return RunResult(example_id=example.id, output=None, trace=None, error=f"http_error: {e}")
        t_end = datetime.now(timezone.utc)
        latency_ms = (t_end - t_start).total_seconds() * 1000.0

        try:
            response_json = resp.json()
        except Exception:
            response_json = {"raw_text": resp.text}

        parsed = self.response_parser(response_json) if self.response_parser else response_json
        if not isinstance(parsed, dict):
            parsed = {"answer": str(parsed)}

        run_id = self._extract_run_id(parsed, resp.headers)
        trace = None
        if run_id:
            trace = await self._fetch_langsmith_run(run_id)
        if trace is None and self.langsmith_project:
            trace = await self._poll_langsmith_for_match(example, body, t_start, t_end)
        if trace is None:
            trace = build_synthetic_trace(parsed, inputs=body if isinstance(body, dict) else {"input": body},
                                           started_at=t_start, finished_at=t_end)

        if resp.status_code >= 500:
            return RunResult(example_id=example.id, output=parsed, trace=trace,
                             error=f"http_{resp.status_code}", latency_ms=latency_ms)
        return RunResult(example_id=example.id, output=parsed, trace=trace, latency_ms=latency_ms,
                         metadata={"run_id": run_id, "http_status": resp.status_code})

    # ------------------------------------------------------------- correlation

    def _extract_run_id(self, response_json: Any, headers: dict[str, str]) -> str | None:
        if isinstance(response_json, dict):
            for k in self.run_id_keys:
                if k in response_json and response_json[k]:
                    return str(response_json[k])
            # Sometimes nested under 'meta' / 'metadata'.
            for k in ("meta", "metadata"):
                inner = response_json.get(k) if isinstance(response_json.get(k), dict) else None
                if inner:
                    for rk in self.run_id_keys:
                        if rk in inner and inner[rk]:
                            return str(inner[rk])
        for hk in ("X-Langsmith-Run-Id", "X-Langsmith-Trace-Id", "X-Trace-Id", "X-Run-Id"):
            v = headers.get(hk) or headers.get(hk.lower())
            if v:
                return str(v)
        return None

    async def _fetch_langsmith_run(self, run_id: str) -> Any | None:
        try:
            client = self._get_langsmith()
            # Brief retry loop — LangSmith ingestion lag can be a few seconds.
            deadline = datetime.now(timezone.utc) + timedelta(seconds=self.langsmith_wait_s)
            while True:
                try:
                    return client.read_run(run_id, load_child_runs=True)
                except Exception:
                    if datetime.now(timezone.utc) >= deadline:
                        return None
                    await asyncio.sleep(1.0)
        except Exception:
            return None

    async def _poll_langsmith_for_match(
        self,
        example: EvalExample,
        body: Any,
        t_start: datetime,
        t_end: datetime,
    ) -> Any | None:
        if not self.langsmith_project:
            return None
        try:
            client = self._get_langsmith()
        except Exception:
            return None
        deadline = datetime.now(timezone.utc) + timedelta(seconds=self.langsmith_wait_s)
        target_query = _coerce_query(example.input) or _coerce_query(body)
        while datetime.now(timezone.utc) < deadline:
            try:
                runs = list(client.list_runs(
                    project_name=self.langsmith_project,
                    start_time=t_start - timedelta(seconds=2),
                    run_type="chain",
                    limit=20,
                ))
            except Exception:
                runs = []
            for r in runs:
                if _input_matches(r, target_query, example.id):
                    try:
                        # Refetch with children if available.
                        return client.read_run(str(r.id), load_child_runs=True)
                    except Exception:
                        return r
            await asyncio.sleep(1.0)
        return None

    def _get_langsmith(self):
        if self._lc_client is None:
            self._lc_client = get_config().get_langsmith_client()
        return self._lc_client


def _coerce_query(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k in ("query", "input", "question", "text", "prompt", "message"):
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        try:
            return json.dumps(obj, sort_keys=True, default=str)[:300]
        except Exception:
            return None
    if isinstance(obj, str):
        return obj
    return None


def _input_matches(run: Any, target_query: str | None, example_id: str) -> bool:
    extra = getattr(run, "extra", None) or {}
    metadata = (extra.get("metadata") if isinstance(extra, dict) else {}) or {}
    if metadata.get("eval_example_id") == example_id:
        return True
    inputs = getattr(run, "inputs", None) or {}
    inp_str = _coerce_query(inputs) or ""
    if target_query and target_query[:80] in inp_str:
        return True
    return False
