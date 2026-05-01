"""Runtime configuration for agent-eval.

A single source of truth for model selection, default thresholds, and
LangSmith client construction. Provider-agnostic: uses LangChain's
`init_chat_model` so users can swap OpenAI / Anthropic / Google by env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvalConfig:
    """Process-wide config. Every default is overridable by env or kwargs."""

    judge_model: str = field(default_factory=lambda: os.getenv("AGENT_EVAL_JUDGE_MODEL", "gpt-4o"))
    judge_temperature: float = 0.0
    judge_provider: str | None = field(
        default_factory=lambda: os.getenv("AGENT_EVAL_JUDGE_PROVIDER", "openai")
    )

    embedding_model: str = field(
        default_factory=lambda: os.getenv("AGENT_EVAL_EMBED_MODEL", "all-MiniLM-L6-v2")
    )

    langsmith_project: str | None = field(default_factory=lambda: os.getenv("LANGCHAIN_PROJECT"))
    langsmith_api_key: str | None = field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY"))
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    )

    default_threshold: float = 0.7
    flag_threshold: float = 0.5

    request_timeout_s: float = 30.0
    max_concurrency: int = 10

    cost_model: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            # USD per 1M tokens (input, output) — coarse, edit per deployment.
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
            "claude-opus-4-7": {"input": 15.0, "output": 75.0},
        }
    )

    def get_chat_model(self, **overrides: Any):
        """Return a configured LangChain chat model. Provider-agnostic."""
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            model=overrides.pop("model", self.judge_model),
            model_provider=overrides.pop("model_provider", self.judge_provider),
            temperature=overrides.pop("temperature", self.judge_temperature),
            **overrides,
        )

    def get_langsmith_client(self):
        """Return a LangSmith client (lazy import — only required for trace access)."""
        from langsmith import Client

        kwargs: dict[str, Any] = {}
        if self.langsmith_api_key:
            kwargs["api_key"] = self.langsmith_api_key
        if self.langsmith_endpoint:
            kwargs["api_url"] = self.langsmith_endpoint
        return Client(**kwargs)


_default_config = AgentEvalConfig()


def get_config() -> AgentEvalConfig:
    return _default_config


def set_config(cfg: AgentEvalConfig) -> None:
    global _default_config
    _default_config = cfg
