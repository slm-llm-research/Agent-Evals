# agent-eval

**Generic agent system evaluation.** Attach to any LangGraph + LangSmith application — or any agent that exposes an MCP endpoint — and get a per-component evaluation report covering output quality, reasoning trajectory, hallucinations, tool performance, system performance, safety, and memory.

Built per [Instruction Set 2 v2 — AgentEval](../files/instruction_set_2_v2_evaluation.docx). **v1.0.0 implements the full specification** — every MANDATORY metric runs natively, every Recommended metric is implemented, and the package can drive an evaluation against any agent system through a runner. See [`CHANGELOG.md`](CHANGELOG.md) for a per-section accounting and [`NEXT_STEPS.md`](NEXT_STEPS.md) for the post-v1 product roadmap.

---

## Install

```bash
pip install -e .
# or with all optional backends + extras
pip install -e ".[all]"
```

Optional extras: `mcp`, `embeddings`, `deepeval`, `ragas`, `memory`, `nli`, `ner`, `dev`.

## Three-line evaluation

```python
from agent_eval import AgentEval, HttpAgentRunner

evaluator = AgentEval.from_mcp("http://localhost:8001/mcp", langsmith_project="my-project")
report = evaluator.evaluate(runner=HttpAgentRunner(endpoint_url="http://localhost:8001/api/query"))
evaluator.save_report(report, output_dir="./eval_results/")
```

## Three-mode workflow

| Mode | What it does |
|---|---|
| `discover` | MCP-first introspection (`/mcp` endpoint), with LangGraph and LangSmith trace mining as fallbacks. Outputs `ComponentRegistry`. |
| `evaluate` | Runs the full evaluation suite. Uses an `AgentRunner` to exercise the agent (HTTP / in-process / replay) and produce real traces for every evaluator. Optional DeepEval / Ragas backends. |
| `monitor` | Attaches to a running LangSmith project and continuously evaluates production traces, writing scores back as feedback and surfacing regressions via stdout / webhook / Slack / email. |

## Runners — the bridge from dataset to trace

| Runner | When to use |
|---|---|
| `HttpAgentRunner` | Most common — POST each example to your agent's HTTP endpoint. Correlates response back to LangSmith run id (response body / header / poll). |
| `LangGraphRunner` | Evaluation runs in-process with a `CompiledGraph`. Captures the trace via `RunCollectorCallbackHandler`. |
| `LangSmithReplayRunner` | Replay against existing production traces — no new agent runs. Match by metadata tag → substring → embedding cosine. |

Without a runner, `evaluate()` warns and degrades to a "dataset testability" report. See [`docs/integration_guide.md`](docs/integration_guide.md) for the small set of changes to add to your agent (return `langsmith_run_id`, honor `X-Eval-Example-Id`, emit `time_to_first_audio_byte`) so every evaluator has the data it needs.

## CLI

```bash
# Discovery
agent-eval discover --mcp-url http://localhost:8001/mcp --output registry.json

# Dataset (4 paths)
agent-eval dataset from-template --template research_agent --output ds.json   # Path D
agent-eval dataset generate --registry registry.json --mode auto --langsmith-project my-project   # Path A
agent-eval wizard --registry registry.json --mode semi-auto --langsmith-project my-project        # Path B
agent-eval wizard --registry registry.json --mode manual                                           # Path C

# Evaluation (3 runner modes)
agent-eval evaluate --registry registry.json --dataset ds.json \
  --runner-url http://localhost:8001/api/query --langsmith-project my-project    # HTTP
agent-eval evaluate --registry registry.json --dataset ds.json \
  --graph-module my_app.graph:compiled_graph                                       # in-process
agent-eval evaluate --registry registry.json --dataset ds.json \
  --replay --langsmith-project my-project                                          # trace replay

# Judge calibration (uses the shipped 20-example sets per dimension)
agent-eval calibrate --output-html ./calibration.html

# CI
agent-eval ci-gate --report ./eval_results/report.json --baseline-report baseline.json \
  --thresholds ci-thresholds.yaml          # Auto-posts to GitHub PR if GITHUB_TOKEN + PR_NUMBER set

# Monitoring
agent-eval monitor --registry registry.json --project my-project --sampling-rate 0.1
agent-eval alerts list --unacked
agent-eval alerts ack 42

# Utilities
agent-eval cost-estimate --registry registry.json --dataset ds.json
agent-eval compare --report-1 a.json --report-2 b.json
```

## Metrics implemented

**Output quality** (mandatory + recommended): task success, faithfulness, relevance, completeness, format compliance, keyword coverage.
**Trajectory**: tool selection accuracy, **Tool F1**, **Node F1**, **SSI** (graph edit distance), **intent resolution**, step success, redundancy, error recovery, cycle detection.
**Hallucination — 4-level**: planning (NER-based entity check), observation (tool-output NLI), citation (URL fetch + claim entailment), reasoning (cross-encoder NLI evidence-bank).
**Tool performance**: success rate, P95 latency, result quality, argument correctness (DeepEval-pluggable), MCP server availability, cost per tool call.
**System performance**: end-to-end latency, **time-to-first-audio-byte** (voice), token efficiency, error rate, cost per query.
**Safety**: harmful content (OpenAI Moderation + judge fallback), PII leakage (regex), instruction following (deterministic constraint parser + judge), response consistency.
**Memory** (auto-activates on memory backend detection): retrieval recall (with optional fact-injection harness), retrieval precision, write quality (over/under-storing), staleness (NLI-checked), cross-session continuity (with optional session runner), cost per query.

## Backends

| Use case | Default | Recommended |
|---|---|---|
| Faithfulness, context precision/recall | native | Ragas |
| Tool correctness, argument correctness, task completion | native | DeepEval |
| Hallucination 4-level | native (AgentEval-specific) | — |
| Structural (Tool F1 / Node F1 / SSI) | native | — |
| Trajectory CoT, memory | native | — |
| Toxicity / bias | OpenAI Moderation | DeepEval |

## Quality of judges

`agent-eval calibrate` runs the shipped 20-example calibration sets per dimension (`answer_quality`, `tool_selection`, `hallucination`, `intent_resolution`, `safety`) against your judges and reports Pearson r, MAE, and Cohen's κ. Any judge with Pearson r < 0.7 is flagged as unreliable. Drift is tracked over a rolling 7-day window.

## License

Apache-2.0.
