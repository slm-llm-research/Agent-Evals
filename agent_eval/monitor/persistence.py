"""SQLite persistence for online eval metrics + alerts.

Tables:
  metrics(ts INTEGER, project TEXT, evaluator TEXT, score REAL, run_id TEXT, flag_reason TEXT)
  alerts(id INTEGER PRIMARY KEY, ts INTEGER, severity TEXT, title TEXT, body TEXT,
         metric TEXT, component TEXT, value REAL, threshold REAL,
         dedup_key TEXT, acked INTEGER DEFAULT 0)
  baselines(metric TEXT PRIMARY KEY, mean REAL, std REAL, n INTEGER, updated_ts INTEGER)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
  ts INTEGER NOT NULL,
  project TEXT,
  evaluator TEXT NOT NULL,
  score REAL NOT NULL,
  run_id TEXT,
  flag_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_evaluator_ts ON metrics(evaluator, ts);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  severity TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT,
  metric TEXT,
  component TEXT,
  value REAL,
  threshold REAL,
  dedup_key TEXT,
  acked INTEGER DEFAULT 0,
  channels TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_dedup_ts ON alerts(dedup_key, ts);
CREATE INDEX IF NOT EXISTS idx_alerts_acked ON alerts(acked);

CREATE TABLE IF NOT EXISTS baselines (
  metric TEXT PRIMARY KEY,
  mean REAL,
  std REAL,
  n INTEGER,
  updated_ts INTEGER
);
"""


class Store:
    """SQLite-backed metrics + alerts store."""

    def __init__(self, path: str | Path = "agent_eval_metrics.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------- metrics

    def write_metric(self, evaluator: str, score: float, project: str | None = None,
                     run_id: str | None = None, flag_reason: str | None = None,
                     ts: datetime | None = None) -> None:
        ts_int = int((ts or datetime.now(timezone.utc)).timestamp())
        with self._conn() as c:
            c.execute(
                "INSERT INTO metrics(ts, project, evaluator, score, run_id, flag_reason) VALUES (?,?,?,?,?,?)",
                (ts_int, project, evaluator, float(score), run_id, flag_reason),
            )

    def read_window(self, evaluator: str, since: datetime) -> list[float]:
        ts_int = int(since.timestamp())
        with self._conn() as c:
            rows = c.execute(
                "SELECT score FROM metrics WHERE evaluator = ? AND ts >= ?", (evaluator, ts_int)
            ).fetchall()
        return [r[0] for r in rows]

    def baseline_for(self, evaluator: str, lookback_days: int = 7) -> dict[str, float]:
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        scores = self.read_window(evaluator, since)
        if not scores:
            return {"n": 0, "mean": 0.0, "std": 0.0}
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return {"n": len(scores), "mean": mean, "std": var ** 0.5}

    def regression_check(self, evaluator: str, window_hours: int = 24,
                         baseline_days: int = 7, threshold_pct: float = 10.0) -> dict[str, Any] | None:
        """Compare last `window_hours` mean vs `baseline_days` baseline. Return regression dict if drop ≥ threshold_pct."""
        now = datetime.now(timezone.utc)
        recent = self.read_window(evaluator, now - timedelta(hours=window_hours))
        baseline_scores = self.read_window(evaluator, now - timedelta(days=baseline_days))
        # Baseline excludes the recent window.
        baseline_scores = baseline_scores[: -len(recent)] if len(recent) < len(baseline_scores) else baseline_scores
        if len(recent) < 5 or len(baseline_scores) < 10:
            return None
        recent_mean = sum(recent) / len(recent)
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        if baseline_mean == 0:
            return None
        pct_change = ((recent_mean - baseline_mean) / baseline_mean) * 100
        if pct_change < -threshold_pct:
            return {
                "evaluator": evaluator,
                "recent_mean": recent_mean,
                "baseline_mean": baseline_mean,
                "pct_change": pct_change,
                "n_recent": len(recent),
                "n_baseline": len(baseline_scores),
            }
        return None

    # ----------------------------------------------------------- alerts

    def write_alert(self, *, severity: str, title: str, body: str = "",
                    metric: str | None = None, component: str | None = None,
                    value: float | None = None, threshold: float | None = None,
                    dedup_key: str | None = None, channels: str = "",
                    dedup_window_minutes: int = 60, ts: datetime | None = None) -> int | None:
        """Write an alert. Returns new id, or None if deduped."""
        ts_int = int((ts or datetime.now(timezone.utc)).timestamp())
        if dedup_key:
            with self._conn() as c:
                row = c.execute(
                    "SELECT id, ts FROM alerts WHERE dedup_key = ? AND ts >= ? ORDER BY ts DESC LIMIT 1",
                    (dedup_key, ts_int - dedup_window_minutes * 60),
                ).fetchone()
                if row is not None:
                    return None  # deduped
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO alerts(ts, severity, title, body, metric, component, value, threshold, dedup_key, channels) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts_int, severity, title, body, metric, component, value, threshold, dedup_key, channels),
            )
            return cur.lastrowid

    def list_alerts(self, only_unacked: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT id, ts, severity, title, body, metric, component, value, threshold, acked, channels FROM alerts"
        if only_unacked:
            sql += " WHERE acked = 0"
        sql += " ORDER BY ts DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(sql, (limit,)).fetchall()
        return [
            {
                "id": r[0], "ts": datetime.fromtimestamp(r[1], tz=timezone.utc).isoformat(),
                "severity": r[2], "title": r[3], "body": r[4], "metric": r[5], "component": r[6],
                "value": r[7], "threshold": r[8], "acked": bool(r[9]), "channels": r[10],
            }
            for r in rows
        ]

    def ack_alert(self, alert_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE alerts SET acked = 1 WHERE id = ?", (alert_id,))
            return cur.rowcount > 0
