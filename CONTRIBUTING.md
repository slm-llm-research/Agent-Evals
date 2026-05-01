# Contributing

```bash
git clone <repo>
cd agent-trust
pip install -e ".[dev,all]"
pytest
ruff check agent_eval/
```

## Adding a new evaluator

1. Subclass `agent_eval.evaluators.base.BaseEvaluator`.
2. Implement `async evaluate(example, trace=None) -> EvaluatorResult`.
3. Register in the relevant composite (e.g., `OutputQualityComposite`).
4. Add a unit test in `tests/test_evaluators.py`.
5. Add the metric to `appendix_metrics.md` if it's a new metric.

## Adding a new backend

1. Subclass `agent_eval.backends.base.EvaluatorBackend`.
2. Implement `is_available()`, `supported_metrics()`, `async evaluate()`.
3. Add to `BACKENDS` in `agent_eval/backends/__init__.py`.

## Adding a new dataset template

1. Drop a JSON file in `agent_eval/dataset/templates/<name>.json` matching the `EvalDataset` schema.
2. Add the name to the CLI `--template` choices in `agent_eval/cli/main.py`.

## Code style

- Ruff for linting (`line-length = 110`).
- Pydantic v2 for all data models.
- Async wherever evaluators or judges call LLMs.
- `@traceable` from langsmith on every judge and evaluator entrypoint.
