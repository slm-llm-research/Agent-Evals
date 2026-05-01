"""Pydantic schema for EvalExample and EvalDataset."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

QueryType = Literal["search", "research", "general", "adversarial", "edge_case", "voice"]
Complexity = Literal["simple", "medium", "complex"]
CreatedBy = Literal["auto", "harvested", "manual", "template"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvalExample(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    reference_answer: str | None = None
    expected_answer_keywords: list[str] = Field(default_factory=list)
    expected_task_graph: dict[str, list[str]] | None = None
    expected_intent: str | None = None
    expected_tool_sequence: list[str] = Field(default_factory=list)
    complexity: Complexity = "medium"
    query_type: QueryType = "general"
    tags: list[str] = Field(default_factory=list)
    created_by: CreatedBy = "manual"
    created_at: datetime = Field(default_factory=_utcnow)
    notes: str | None = None


class EvalDataset(BaseModel):
    name: str
    description: str = ""
    examples: list[EvalExample]
    version: str = "1"
    created_at: datetime = Field(default_factory=_utcnow)
    component_registry_hash: str | None = None
    langsmith_dataset_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvalDataset:
        if len(self.examples) < 1:
            raise ValueError("EvalDataset must contain at least 1 example")
        return self

    # ------- selection helpers

    def filter(self, query_type: QueryType | None = None, complexity: Complexity | None = None) -> EvalDataset:
        items = self.examples
        if query_type:
            items = [e for e in items if e.query_type == query_type]
        if complexity:
            items = [e for e in items if e.complexity == complexity]
        return self.model_copy(update={"examples": items})

    def split(self, train_ratio: float = 0.8, seed: int = 42) -> tuple[EvalDataset, EvalDataset]:
        import random

        idx = list(range(len(self.examples)))
        random.Random(seed).shuffle(idx)
        cut = max(1, int(len(idx) * train_ratio))
        train = [self.examples[i] for i in idx[:cut]]
        test = [self.examples[i] for i in idx[cut:]] or [self.examples[idx[-1]]]
        return (
            self.model_copy(update={"examples": train, "name": f"{self.name}-train"}),
            self.model_copy(update={"examples": test, "name": f"{self.name}-test"}),
        )

    # ------- IO

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        return cls.model_validate_json(Path(path).read_text())

    @classmethod
    def from_template(cls, template_name: str) -> EvalDataset:
        valid = {"search_agent", "research_agent", "voice_agent", "general_agent"}
        if template_name not in valid:
            raise ValueError(f"Unknown template '{template_name}'. Valid: {sorted(valid)}")
        text = resources.files("agent_eval.dataset.templates").joinpath(f"{template_name}.json").read_text()
        data = json.loads(text)
        # Fill missing 'name' / 'description' if the template only stores examples.
        if "examples" not in data:
            return cls(name=template_name, description=f"Built-in template: {template_name}", examples=[EvalExample(**d) for d in data])
        for ex in data.get("examples", []):
            ex.setdefault("created_by", "template")
        return cls.model_validate(data)

    # ------- LangSmith

    def to_langsmith(self, client: Any) -> str:
        try:
            ds = client.create_dataset(dataset_name=self.name, description=self.description)
            for ex in self.examples:
                client.create_example(
                    inputs=ex.input,
                    outputs=ex.expected_output or {},
                    dataset_id=ds.id,
                )
            return str(ds.id)
        except Exception as e:
            raise RuntimeError(f"Failed to push dataset to LangSmith: {e}") from e

    @classmethod
    def from_langsmith(cls, client: Any, dataset_id: str) -> EvalDataset:
        ds = client.read_dataset(dataset_id=dataset_id)
        examples = []
        for ex in client.list_examples(dataset_id=dataset_id):
            examples.append(
                EvalExample(
                    id=str(ex.id),
                    input=dict(ex.inputs or {}),
                    expected_output=dict(ex.outputs or {}) if ex.outputs else None,
                    created_by="harvested",
                )
            )
        return cls(
            name=ds.name,
            description=getattr(ds, "description", "") or "",
            examples=examples,
            langsmith_dataset_id=str(dataset_id),
        )
