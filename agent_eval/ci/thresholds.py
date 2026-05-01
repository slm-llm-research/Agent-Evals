"""Threshold YAML parsing for CI gate."""

from __future__ import annotations

from pathlib import Path

from agent_eval.reporters.regression_detector import ThresholdConfig


def load_thresholds(path: str | Path) -> ThresholdConfig:
    return ThresholdConfig.from_yaml(str(path))
