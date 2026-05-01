"""Tests for SQLite persistence + alert manager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_eval.monitor.alerting import Alert, AlertManager
from agent_eval.monitor.persistence import Store


def test_store_writes_and_reads_metric(tmp_path: Path):
    s = Store(tmp_path / "m.db")
    s.write_metric("acc", 0.85, project="p1")
    s.write_metric("acc", 0.90, project="p1")
    scores = s.read_window("acc", since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert sorted(scores) == [0.85, 0.90]


def test_baseline_computation(tmp_path: Path):
    s = Store(tmp_path / "b.db")
    for v in [0.7, 0.8, 0.9, 0.85, 0.75]:
        s.write_metric("metric_x", v)
    bl = s.baseline_for("metric_x", lookback_days=7)
    assert bl["n"] == 5
    assert 0.79 < bl["mean"] < 0.81
    assert bl["std"] > 0


def test_regression_detection_triggers(tmp_path: Path):
    s = Store(tmp_path / "r.db")
    now = datetime.now(timezone.utc)
    # Baseline window (older): high scores.
    for i in range(15):
        s.write_metric("acc", 0.90, ts=now - timedelta(days=3, hours=i))
    # Recent window (last 24h): low scores.
    for i in range(8):
        s.write_metric("acc", 0.60, ts=now - timedelta(hours=i))
    reg = s.regression_check("acc", window_hours=24, baseline_days=7, threshold_pct=10.0)
    assert reg is not None
    assert reg["pct_change"] < -10
    assert reg["recent_mean"] < reg["baseline_mean"]


def test_regression_no_trigger_when_stable(tmp_path: Path):
    s = Store(tmp_path / "rs.db")
    now = datetime.now(timezone.utc)
    for i in range(15):
        s.write_metric("acc", 0.85, ts=now - timedelta(days=3, hours=i))
    for i in range(8):
        s.write_metric("acc", 0.84, ts=now - timedelta(hours=i))
    reg = s.regression_check("acc", window_hours=24, baseline_days=7, threshold_pct=10.0)
    assert reg is None


def test_alert_dedup_within_window(tmp_path: Path):
    s = Store(tmp_path / "a.db")
    id1 = s.write_alert(severity="high", title="x", dedup_key="k1", dedup_window_minutes=60)
    id2 = s.write_alert(severity="high", title="x", dedup_key="k1", dedup_window_minutes=60)
    assert id1 is not None
    assert id2 is None  # deduped


def test_alert_ack(tmp_path: Path):
    s = Store(tmp_path / "ack.db")
    id1 = s.write_alert(severity="medium", title="t", dedup_key="k_ack")
    rows = s.list_alerts(only_unacked=True)
    assert any(r["id"] == id1 for r in rows)
    assert s.ack_alert(id1) is True
    assert s.ack_alert(id1) is True  # idempotent at db layer (still 0 rows changed but doesn't error)
    rows = s.list_alerts(only_unacked=True)
    assert not any(r["id"] == id1 for r in rows)


def test_alert_manager_stdout_does_not_raise(tmp_path: Path, capsys):
    am = AlertManager(channels=["stdout"], store_path=tmp_path / "am.db")
    fired = am.fire(Alert(severity="high", title="boom", body="reason", metric="m1", component="c1"))
    assert fired is True
    out = capsys.readouterr().out
    assert "BOOM" not in out  # title isn't uppercased
    assert "boom" in out
    assert "high" in out.lower()


def test_alert_manager_dedup(tmp_path: Path):
    am = AlertManager(channels=["stdout"], store_path=tmp_path / "amd.db")
    a = Alert(severity="high", title="dup", metric="m", component="c", dedup_key="k1")
    assert am.fire(a) is True
    assert am.fire(a) is False  # deduped
