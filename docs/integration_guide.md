# Integrating an agent system with `agent-eval`

This guide covers what to add to your agent so `agent-eval` can produce **honest, deep** evaluation reports — covering output quality, trajectory, hallucination, tool performance, system performance, safety, and memory.

The package works against any agent (it's framework-agnostic), but the depth of what it can measure depends on what your agent exposes. Below: changes to make, in priority order. Code examples target the [voice-agent-system](../../voice-agent-system/) at `agents/orchestrator/main.py` but the same shape applies to any FastAPI / LangGraph / MCP-based agent.

---

## TL;DR — minimum to get real evaluations

You can already run `agent-eval evaluate --runner-url http://your-agent/api/query --dataset ds.json` today. To make the report **complete**, add three things:

1. Return `langsmith_run_id` in the `/api/query` response (10 lines).
2. Tag LangSmith runs with the `eval_example_id` from the `X-Eval-Example-Id` request header (5 lines).
3. Emit `time_to_first_audio_byte` as LangSmith feedback per run (15 lines, only for streaming/voice agents).

The rest is nice-to-have.

---

## P0 — required for full evaluation

### 1. Return `langsmith_run_id` in HTTP responses

Why: lets `HttpAgentRunner` correlate the response back to the actual LangSmith trace, which is the source of truth for tool calls, latency breakdown, errors, and token counts. Without this the runner builds a "synthetic trace" from just the response body — output quality and safety still score, but trajectory / tool / system metrics degrade to "no data."

How (voice-agent-system, `agents/orchestrator/main.py`):

```python
from langsmith.run_helpers import get_current_run_tree

@app.post("/api/query")
async def query(req: QueryRequest):
    trace_id = req.session_id or str(uuid.uuid4())
    state = create_initial_state(req.query, trace_id)
    result = await _orchestrator.ainvoke(
        state,
        config={"run_name": "orchestrator", "metadata": {"trace_id": trace_id}},
    )
    # NEW: capture the LangSmith run id of this invocation.
    run_tree = get_current_run_tree()
    langsmith_run_id = str(run_tree.id) if run_tree else None
    return {
        "answer": result.get("summary", ""),
        "trace_id": trace_id,
        "intent": result.get("intent", "general"),
        "langsmith_run_id": langsmith_run_id,   # NEW
    }
```

`HttpAgentRunner` already looks for keys in this priority order: `langsmith_run_id`, `run_id`, `trace_id`, `trace_run_id`, plus the `X-Langsmith-Run-Id` response header. Use whichever fits your shape. (Note: your existing `trace_id` is a session id, not the LangSmith run id — they are different things; don't conflate.)

If you can't easily call `get_current_run_tree()` (e.g. the `ainvoke` returns before LangSmith has flushed), use the [`@traceable` callback approach](https://docs.smith.langchain.com/observability/how_to_guides/tracing/access_run_id) to capture the run id and stash it in `result`.

### 2. Honor the `X-Eval-Example-Id` request header

Why: the runner sets `X-Eval-Example-Id: <uuid>` on every request. If your agent attaches it to the LangSmith trace metadata, `LangSmithReplayRunner` can match traces back to dataset examples *exactly* — no fuzzy matching needed. This is also how the regression detector knows which production trace corresponds to which dataset example weeks later.

```python
from fastapi import Request

@app.post("/api/query")
async def query(req: QueryRequest, request: Request):
    eval_example_id = request.headers.get("X-Eval-Example-Id")
    eval_source = request.headers.get("X-Eval-Source")  # "agent-eval"

    metadata = {"trace_id": trace_id}
    if eval_example_id:
        metadata["eval_example_id"] = eval_example_id
        metadata["eval_source"] = eval_source

    result = await _orchestrator.ainvoke(
        state,
        config={"run_name": "orchestrator", "metadata": metadata},
    )
    ...
```

This unlocks two flows:
- **Replay**: `agent-eval evaluate --replay --langsmith-project voice-agent-system` matches every example by exact id.
- **Filtering**: in LangSmith UI, filter `metadata.eval_source = "agent-eval"` to find/exclude eval traffic from production dashboards.

### 3. Emit streaming-latency feedback

Why: `time_to_first_audio_byte_p95` is a **mandatory** v2 metric for voice agents (Spec table 57, marked NEW v2). Without this feedback on the trace, `StreamingLatencyEvaluator` skips with `score=1.0` — green for the wrong reason.

Where to measure: in your voice-agent service, between when STT delivers the final transcript and when TTS begins emitting audio bytes. Since voice-agent-system already streams via SSE (`text/event-stream`), the natural measurement point is when the **first** SSE `data:` chunk is sent.

How (in the voice-agent service, not the orchestrator):

```python
import time
from langsmith import Client

ls = Client()
t_query_start = time.perf_counter()
first_audio_emitted = False

async for chunk in upstream_audio_stream():
    if not first_audio_emitted:
        ttfa_ms = (time.perf_counter() - t_query_start) * 1000.0
        # Attach to the orchestrator run via its run_id.
        ls.create_feedback(
            run_id=langsmith_run_id,
            key="time_to_first_audio_byte_ms",
            score=ttfa_ms,
        )
        first_audio_emitted = True
    yield chunk
```

`StreamingLatencyEvaluator` reads from `feedback_stats.time_to_first_audio_byte_ms` or `feedback.ttfa_ms` — either key works.

---

## P1 — significantly improves evaluation quality

### 4. Expose system prompts via `/mcp/registry`

Why: `InstructionFollowingEvaluator` checks whether the agent obeyed its stated constraints (word limits, no-competitors, refusal rules). Without seeing the system prompts, the evaluator skips — a critical safety dimension goes unmeasured.

How: the registry document at `GET /mcp/registry` should include per-agent system prompts (or a hash + URL to fetch them). Suggested schema addition:

```json
{
  "agents": [
    {
      "name": "voice_agent",
      "agent_type": "simple_node",
      "system_prompt": "You are a voice assistant. Keep responses under 150 words. Never mention competitor products.",
      "constraints": {"max_words": 150}
    }
  ]
}
```

Then in the package, when building `InstructionFollowingEvaluator`, the orchestrator hands over the constraints from the registry.

### 5. Add `/api/eval/run` — eval-only endpoint

Why: production endpoints often have side effects (TTS playback, caching, analytics, billing). For eval you want a leaner, deterministic path:
- Skip TTS audio synthesis (eval doesn't listen to audio anyway — saves $$$).
- Skip session-state writes to Redis.
- Use a deterministic seed for any sampling.
- Disable user-facing analytics and billing.

A separate endpoint also keeps eval traffic clearly identified (no `X-Eval-Example-Id`-detection scattered through production code).

```python
@app.post("/api/eval/run")
async def eval_run(req: QueryRequest, request: Request):
    """Eval-only endpoint. Skips TTS, analytics, billing. Returns full structured trace."""
    eval_example_id = request.headers.get("X-Eval-Example-Id")
    state = create_initial_state(req.query, trace_id=req.session_id or str(uuid.uuid4()))
    state["_skip_tts"] = True
    state["_eval_mode"] = True

    result = await _orchestrator.ainvoke(
        state,
        config={
            "run_name": "orchestrator_eval",
            "metadata": {"eval_example_id": eval_example_id, "eval_source": "agent-eval"},
        },
    )
    run_tree = get_current_run_tree()
    return {
        "answer": result.get("summary", ""),
        "intent": result.get("intent"),
        "tool_calls": result.get("tool_call_history", []),
        "langsmith_run_id": str(run_tree.id) if run_tree else None,
    }
```

Then point the runner at it: `--runner-url http://orchestrator:8001/api/eval/run`.

### 6. Surface `tool_calls` / `tool_trace` in the response

Why: the `SyntheticTrace` shim can build `child_runs` of type "tool" if the response includes a `tool_calls` array. That lets `ToolF1Evaluator`, `RedundancyEvaluator`, and `ToolPerformanceEvaluator` work even when the LangSmith trace isn't recovered.

Schema:
```json
{
  "answer": "...",
  "tool_calls": [
    {
      "name": "tavily_search_tool",
      "inputs": {"query": "US emissions 2024"},
      "outputs": {"results": [...]},
      "start_time": "2026-04-30T12:34:56Z",
      "end_time": "2026-04-30T12:34:58Z",
      "error": null
    },
    ...
  ]
}
```

This is also useful for client-side debugging.

### 7. Token usage / cost in response

Why: `CostEfficiencyEvaluator` and `TokenEfficiencyEvaluator` rely on per-run token counts. LangSmith captures these for `@traceable` LLM calls automatically; if you're using LangGraph + LangChain models, this is free. Verify by reading any existing run in LangSmith — `prompt_tokens` / `completion_tokens` should be populated. If not, wrap your LLM calls.

---

## P2 — nice to have

### 8. `dataset_example_id` → reproducible test mode

Add an optional `seed` field to `QueryRequest`. When set:
- Use it for any sampling temperature on judge LLMs.
- Use it for randomized tool selection.
- Ensures the same example produces the same trace across eval runs (stabilizes regression detection).

### 9. Health/readiness endpoints already exist — good

You have `/health` and `/ready` — the MCP discovery's health check uses these. Already done.

### 10. Optional: webhook for live monitoring

If you want `agent-eval monitor` to react in <60s, expose a webhook that fires when a new trace completes. Otherwise the monitor polls LangSmith every 60s by default.

---

## Verifying integration

After the P0 changes:

```bash
# 1. Discover the agent (you already do this).
agent-eval discover --mcp-url http://localhost:8001/mcp --output reg.json

# 2. Pick or build a dataset.
agent-eval dataset from-template --template voice_agent --output ds.json

# 3. Evaluate WITH the runner.
export LANGCHAIN_API_KEY=...
export OPENAI_API_KEY=...   # for LLM judges
agent-eval evaluate \
  --registry reg.json \
  --dataset ds.json \
  --output-dir ./eval_results/ \
  --runner-url http://localhost:8001/api/query \
  --langsmith-project voice-agent-system
```

What you should see:
- The progress line `Running 10 examples via http runner...` followed by per-example timings.
- A summary line like `real LangSmith traces: 10/10, synthetic shims: 0, failures: 0`. If "synthetic shims" > 0, P0.1 isn't working — `langsmith_run_id` is missing from responses.
- `output_quality`, `trajectory_quality`, `hallucination_risk`, `tool_performance`, `system_performance`, `safety`, `memory_quality` — all numbers should now reflect actual agent behavior.
- The `flag_count` should drop dramatically from 94 to a small number representing real issues.

If you still see `flag_count` near `dataset_size × 8`, the runner isn't reaching your agent or the response shape isn't recognized. Run with `agent-eval evaluate --runner-url ... --n-examples 1` to debug a single call.

---

## Per-evaluator data dependencies

Use this matrix to know what each evaluator needs and when it will skip / bail.

| Evaluator | Needs | If missing |
|---|---|---|
| `task_success_rate` | response with non-empty `answer`/`summary`/`final_answer`/`output` | scores 0 |
| `answer_faithfulness` | response + tool outputs in trace | scores 1.0 if no sources (undefined) |
| `answer_relevance` | response | scores 0 if empty |
| `keyword_coverage` | response + `expected_answer_keywords` on the example | skips if no keywords |
| `format_compliance` | response + (optional) max_words/no_markdown config | trivially passes if no constraints |
| `hallucination_planning` | trace with orchestrator/planner output | scores 1.0 if no plan observed |
| `hallucination_observation` | response + tool outputs in trace | scores 1.0 if no tool outputs |
| `hallucination_citation` | response + tool outputs (URL set) | scores 1.0 if no URLs in answer |
| `tool_f1` | trace with tool runs + `expected_tool_sequence` on example | skips if no expected sequence |
| `node_f1` / `ssi` | trace with chain/agent runs + `expected_task_graph` on example | skips if no expected graph |
| `intent_resolution` | trace (any) + LLM judge | scores via LLM |
| `cycle_detected` | trace with chain/agent runs | scores 1.0 if no cycle observed |
| `tool_success_rate` | trace with tool runs | scores 1.0 if no tool calls |
| `end_to_end_latency` | trace `start_time` + `end_time` | skips if no timing |
| `time_to_first_audio_byte` | LangSmith feedback `time_to_first_audio_byte_ms` | **skips** — see P0.3 |
| `cost_per_query_usd` | trace with `prompt_tokens`/`completion_tokens` | scores 1.0 if 0 tokens |
| `harmful_content_rate` | response + (optional) `OPENAI_API_KEY` | uses OpenAI Moderation; falls back to LLM judge |
| `pii_leakage_rate` | response + original query | deterministic regex |
| `instruction_compliance` | response + system_prompt | **skips** — see P1.4 |
| `memory_retrieval_recall/precision` | trace with memory backend runs | skips if no memory backend in registry |

---

## Where to find the runner code

- [agent-trust/agent_eval/runners/](../agent_eval/runners/) — base + 3 implementations
- [agent-trust/agent_eval/runners/http_runner.py](../agent_eval/runners/http_runner.py) — what voice-agent-system will use
- [agent-trust/agent_eval/runners/synthetic.py](../agent_eval/runners/synthetic.py) — the trace shim used as fallback
- [agent-trust/tests/test_runners.py](../tests/test_runners.py) — examples of each runner's expected response shape
