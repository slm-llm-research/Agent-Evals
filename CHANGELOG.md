# Changelog

## 1.0.0 — Spec-complete release

This release implements the full v2 specification of `instruction_set_2_v2_evaluation.docx` end-to-end. Every MANDATORY metric runs natively, every Recommended metric is implemented (not stubbed), and the package can drive an evaluation against any agent system through a runner — HTTP, in-process LangGraph, or LangSmith trace replay.

### Added — Judges
- **Full TRACE-style CoT trajectory judge** with progressive evidence-bank construction, per-step relevance / correctness / efficiency scoring, observation-hallucination detection inline, and holistic efficiency / coherence / adaptivity aggregation.
- **Judge calibration framework** with shipped 20-example calibration sets per dimension (`answer_quality`, `tool_selection`, `hallucination`, `intent_resolution`, `safety`). Returns Pearson r, MAE, Cohen's κ, and a per-judge reliability flag (Pearson r ≥ 0.7).
- **`JudgeMonitor`** for rolling drift tracking — alerts on >0.15 std shift over a 7-day window.
- **Position-bias mitigation** for pairwise judges (run A>B and B>A, average).
- **HTML calibration report** (`agent_eval.judges.calibration.render_calibration_html`).

### Added — Evaluators (depth)
- **NLI-based reasoning hallucination** via cross-encoder (`cross-encoder/nli-deberta-v3-small` when `[nli]` extra installed; content-overlap fallback otherwise).
- **Citation URL fetch verification** — optional live HTTP fetch of cited URLs + NLI check that the surrounding claim is entailed by page contents.
- **Memory full suite**: `MemoryRetrievalRecall` (with optional fact-injection harness), `MemoryRetrievalPrecision`, `MemoryWriteQuality` (over/under-storing detection), `MemoryStaleness`, `CrossSessionContinuity` (with optional session-runner), `MemoryCost`.
- **InstructionFollowing constraint parser** — deterministic checks for word limits, JSON-only output, no-markdown, blocklists, must-include phrases, refuse-advice triggers, plus LLM-judge backstop.

### Added — Discovery (depth)
- **GraphInspector deep introspection**: conditional-edge detection with condition function name extraction, agent-type heuristics (orchestrator / react_agent / tool_node / simple_node / subgraph), self-loop + subgraph detection, state schema dump, multi-strategy tool extraction.
- **LangSmithInspector full mining**: transition matrix, per-component stats (success rate, P95 latency, error breakdown, avg tokens), MCP server detection from HTTP-style tool calls, agent structure inference (entry / exit points).
- **MemoryDetector with Python-import scanning** — AST-walks user source for `chromadb`, `pinecone`, `weaviate`, `qdrant`, `faiss`, `mem0`, `zep` imports.

### Added — Dataset
- **Path B (semi-auto curation TUI)** — interactive review of auto-generated candidates with accept/reject/edit/keywords actions.
- **Path C wizard polish** — 5-category guided flow (typical / edge_case / adversarial / failing / multi_turn) with rich tables, optional LLM few-shot expansion, and live cost preview.
- **Trace harvester sentence-transformer dedup** — embedding-cosine ≥0.92 dedup pass on top of exact-hash.
- **AutoDatasetGenerator validation tightening** — flags low query-type diversity, complexity imbalance, and missing capability-sweep coverage per tool.
- **Per-evaluator cost-estimate model** — granular token-cost projection per metric.

### Added — Reporters
- **HTML report** with regression view (side-by-side baseline radar + delta table), dataset-coverage charts (query-type doughnut + complexity bar), filterable / sortable raw-data table, "copy LangSmith link" buttons, sticky in-page navigation.
- **TuningAdvisor — all 6 spec signal types**: Prompt, Model (per-node downgrade suggestions), ToolConfig (per-tool tuning), Architecture (cycle / depth / state), Data (per-query-type coverage), Memory (recall / precision / write / staleness specific actions).

### Added — Monitor
- **OnlineEvaluator SQLite persistence** at `agent_eval_metrics.db` — every evaluator score timestamped + queryable.
- **24h rolling regression check** vs 7-day baseline, automatic alerts on >10% drop.
- **AlertManager full channels**: stdout, webhook (POST), Slack (incoming-webhook with rich attachments), email (SMTP via `EmailConfig.from_env()`).
- **Alert dedup** (1h rolling window per dedup_key) + SQLite-backed acknowledgement.
- **`agent-eval alerts list` / `agent-eval alerts ack <id>`** CLI commands.

### Added — CI / GH integration
- **GitHub PR comment posting** from `ci-gate` — auto-detects `GITHUB_TOKEN` + `GITHUB_REPOSITORY` + (`PR_NUMBER` env or `refs/pull/N/merge` ref) and posts a markdown summary as a PR comment.
- **`.github/workflows/test.yml`** — multi-Python pytest matrix + coverage upload + ci-gate self-smoke.

### Added — CLI
- `agent-eval calibrate [--dimension] [--output-html]` — run shipped calibration sets against the standard judges.
- `agent-eval wizard --mode [manual|semi-auto]` — interactive dataset wizard (Path B or C).
- `agent-eval alerts list [--unacked]` and `agent-eval alerts ack <id>` — manage the alerts SQLite store.
- Existing commands (`discover`, `dataset`, `evaluate`, `ci-gate`, `monitor`, `compare`, `cost-estimate`) all stable.

### Tests
- 156 passing (up from 67 in 0.3.0). 63% line coverage across 4,984 LOC. Critical paths: structural metrics, registry, CI gate, dataset templates, runners, end-to-end evaluate-with-runner, calibration framework, NLI / hallucination, memory full suite, constraint parser, persistence + alerts, GitHub PR posting, deep graph inspector, tuning advisor 6 signals.

### Migration notes
- `package_version` in saved reports bumps from 0.3.0 to 1.0.0 — old reports remain loadable.
- `AutoDatasetGenerator.llm` is now a lazy property; instantiation does not require an API key.
- `TraceHarvester.deduplicate(...)` now takes an `embedding_threshold` kwarg (default 0.92). Pass 0 to keep exact-hash-only behavior.
- New optional extras: install with `pip install -e ".[nli]"` for NLI cross-encoder, `[ner]` for spaCy entity extraction.

---

## 0.3.0 — Runners

The release that made `evaluate()` actually evaluate the agent. v0.2.0 had every evaluator wired up but no way to feed real traces in; this added the runner layer that bridges dataset → agent → trace → evaluator.

### Added
- `agent_eval.runners` package with three implementations:
  - `HttpAgentRunner` — POSTs each example to the agent's HTTP endpoint, correlates the LangSmith run id from response body or `X-Langsmith-Run-Id` header, falls back to polling.
  - `LangGraphRunner` — invokes a `CompiledGraph` in-process, captures the trace via `RunCollectorCallbackHandler`.
  - `LangSmithReplayRunner` — matches dataset examples to existing production traces by metadata tag, substring, or embedding cosine.
- `SyntheticTrace` shim — when a real LangSmith trace can't be recovered, wraps the response so output / safety / format evaluators still work.
- `AgentEval.evaluate(..., runner=runner)` and `AgentEval.evaluate(..., traces=...)` parameters.
- CLI flags `--runner-url`, `--graph-module`, `--replay`, `--max-concurrency`, `--runner-header`.
- Loud warning when `evaluate()` runs without a runner or traces.
- `docs/integration_guide.md` — what to add to the target agent so every evaluator has the data it needs.

---

## 0.2.0 — MVP (initial public build)

Built per Instruction Set 2 v2 (`AgentEval`). MVP scope; see `NEXT_STEPS.md` for deferred items at the time.

### Added
- MCP-first discovery (`AgentEval.from_mcp`) with LangGraph + LangSmith trace mining as fallbacks.
- ComponentRegistry, dataset paths A/C/D + 4 templates, judges, evaluator suite, native + DeepEval + Ragas backends, JSON + HTML reporters, CI gate, online monitor + AlertManager, full CLI.
