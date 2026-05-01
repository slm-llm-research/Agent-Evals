# Roadmap (post-v1.0.0)

`v1.0.0` ships the full `instruction_set_2_v2_evaluation.docx` specification end-to-end. This file now tracks **post-v1 work** — items that are out of scope for the spec but on the product roadmap as we ship to YC users.

For what was implemented and how, see [`CHANGELOG.md`](CHANGELOG.md). For data dependencies of each evaluator (and how each metric degrades when its data is missing), see [`docs/integration_guide.md`](docs/integration_guide.md).

---

## v1.x — depth and polish

### Test coverage
- [ ] Push `pytest --cov=agent_eval` from 63% to ≥85%. Remaining gaps: interactive wizards (Path B/C), online-monitor poll loop, Slack/email channels (require credentials in CI), full HTTP runner LangSmith-poll path.

### Calibration
- [ ] Re-curate calibration sets after dogfooding — replace any examples where multiple human raters disagree on `human_score`.
- [ ] Per-judge `JudgeMonitor` integration in OnlineEvaluator (auto-write feedback drift to alerts).
- [ ] Pairwise judges with structured position-bias swap workflow.

### Reporters
- [ ] Component drill-down page (per-component HTML with all evaluator results expanded).
- [ ] Trend-over-time view across N consecutive reports (load N report.json files and chart).

### Monitor
- [ ] LangSmith native online-eval API integration (`configure_langsmith_online_eval`) — currently a hook stub.
- [ ] Webhook signature verification on inbound triggers.

### Backends
- [ ] DeepEval + Ragas: deeper version-pinned coverage as their APIs continue to shift.
- [ ] Add LiteLLM backend so users can route judges through any provider (Bedrock, Vertex, Azure).

### Documentation
- [ ] Per-agent-framework integration recipes (Anthropic Agents SDK, AutoGen, CrewAI, OpenAI Agents SDK) — current guide is FastAPI/LangGraph-centric but the package itself is framework-agnostic.

---

## v2.x — beyond the spec

These are roadmap items that go beyond `instruction_set_2_v2_evaluation.docx`:

- Multi-tenant SaaS: hosted dashboard, per-org auth, team annotation queues.
- Web UI for browsing and comparing reports (currently single-file static HTML).
- Continuous learning: feedback from human reviewers feeds back into calibration sets automatically.
- Cross-agent benchmarking: run the same dataset against multiple agent systems and rank.
- Adversarial test generation: red-team agent that searches for failures in the target system.
- Live trace streaming: WebSocket subscriber that scores traces as they happen rather than 60s polling.

---

## Out of scope for this package

- Anything that requires holding agent system state on our side (we operate against the user's LangSmith / MCP / runner; we don't host the agent).
- Scraping, jailbreaking, or other mis-use of third-party agent APIs.
- Replacing LangSmith — we treat LangSmith (or an equivalent OTel tracer) as the trace source of truth and write feedback / scores back to it.
