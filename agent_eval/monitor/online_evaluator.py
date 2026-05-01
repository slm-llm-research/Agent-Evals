"""OnlineEvaluator — polls LangSmith for new traces, runs evaluators, persists scores.

Per-spec features:
  - SQLite metrics persistence (`agent_eval_metrics.db`)
  - 24h rolling regression check vs 7-day baseline
  - LangSmith feedback writes
  - Configurable sampling rate
  - Alert manager integration
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent_eval.config import get_config
from agent_eval.discovery.registry import ComponentRegistry
from agent_eval.monitor.alerting import Alert, AlertManager
from agent_eval.monitor.persistence import Store


class OnlineEvaluator:
    def __init__(
        self,
        registry: ComponentRegistry,
        evaluators: list[Any] | None = None,
        sampling_rate: float = 0.1,
        alert_thresholds: dict[str, float] | None = None,
        client: Any | None = None,
        db_path: str | Path = "agent_eval_metrics.db",
        alert_manager: AlertManager | None = None,
        regression_threshold_pct: float = 10.0,
    ):
        self.registry = registry
        self.evaluators = evaluators or []
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))
        self.alert_thresholds = alert_thresholds or {}
        self.client = client or get_config().get_langsmith_client()
        self.store = Store(db_path)
        self.alert_manager = alert_manager or AlertManager()
        self.regression_threshold_pct = regression_threshold_pct
        self._stop_event = asyncio.Event()
        self._last_seen_at: datetime | None = None
        self._last_regression_check: datetime | None = None

    async def start_monitoring(self, project_name: str, poll_interval_seconds: int = 60) -> None:
        self._stop_event.clear()
        self._last_seen_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        while not self._stop_event.is_set():
            try:
                n = await self._poll_once(project_name)
                if n > 0:
                    print(f"[OnlineEvaluator] polled {n} new traces")
                # Run regression check at most once per 10 minutes.
                now = datetime.now(timezone.utc)
                if self._last_regression_check is None or (now - self._last_regression_check) > timedelta(minutes=10):
                    self._last_regression_check = now
                    await self._check_regressions(project_name)
            except Exception as e:
                print(f"[OnlineEvaluator] poll failed: {e}")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_interval_seconds)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    async def _poll_once(self, project_name: str) -> int:
        since = self._last_seen_at or datetime.now(timezone.utc) - timedelta(minutes=5)
        try:
            runs = list(self.client.list_runs(project_name=project_name, start_time=since,
                                                run_type="chain", limit=200))
        except Exception:
            runs = []
        evaluated = 0
        for run in runs:
            self._last_seen_at = max(self._last_seen_at or since, getattr(run, "start_time", None) or since)
            if random.random() > self.sampling_rate:
                continue
            await self._evaluate_one(run, project_name)
            evaluated += 1
        return evaluated

    async def _evaluate_one(self, run: Any, project_name: str) -> None:
        from agent_eval.dataset.schema import EvalExample

        inputs = getattr(run, "inputs", None) or {}
        ex = EvalExample(input=dict(inputs), created_by="harvested")
        run_id = str(getattr(run, "id", "unknown"))
        for ev in self.evaluators:
            try:
                result = await ev.evaluate(ex, run)
            except Exception:
                continue
            # Persist + write feedback + check immediate threshold.
            self.store.write_metric(
                evaluator=result.evaluator_name, score=result.score,
                project=project_name, run_id=run_id, flag_reason=result.flag_reason,
            )
            try:
                self.client.create_feedback(
                    run_id=run_id, key=result.evaluator_name, score=result.score,
                    comment=result.flag_reason or "",
                )
            except Exception:
                pass
            # Per-metric threshold alerts (immediate).
            threshold = self.alert_thresholds.get(result.evaluator_name)
            if threshold is not None and result.score < threshold:
                self.alert_manager.fire(Alert(
                    severity="high", title=f"{result.evaluator_name} below threshold",
                    body=f"Score {result.score:.3f} < {threshold} on run {run_id[:8]}",
                    metric=result.evaluator_name, component=result.component_name,
                    value=result.score, threshold=threshold,
                ))

    async def _check_regressions(self, project_name: str) -> None:
        seen_evaluators = set()
        for ev in self.evaluators:
            seen_evaluators.add(ev.name)
        for evaluator in seen_evaluators:
            regression = self.store.regression_check(
                evaluator, window_hours=24, baseline_days=7,
                threshold_pct=self.regression_threshold_pct,
            )
            if regression is None:
                continue
            self.alert_manager.fire(Alert(
                severity="high",
                title=f"Regression detected on {evaluator}",
                body=(
                    f"Last 24h mean = {regression['recent_mean']:.3f} (n={regression['n_recent']}) vs "
                    f"7-day baseline = {regression['baseline_mean']:.3f} (n={regression['n_baseline']}); "
                    f"Δ {regression['pct_change']:.1f}%."
                ),
                metric=evaluator, component="system",
                value=regression["recent_mean"], threshold=regression["baseline_mean"],
            ))

    async def check_for_regressions(self, window_hours: int = 24) -> list[dict[str, Any]]:
        out = []
        for ev in self.evaluators:
            r = self.store.regression_check(ev.name, window_hours=window_hours,
                                              threshold_pct=self.regression_threshold_pct)
            if r:
                out.append(r)
        return out

    def configure_langsmith_online_eval(self, project: str, evaluator_name: str, sampling_rate: float = 1.0) -> None:
        # LangSmith online-eval API surface varies — exposed as a hook for users to wire up.
        return None
