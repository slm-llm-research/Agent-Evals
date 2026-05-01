"""Alert manager — stdout, webhook (POST), Slack (webhook), email (SMTP).

With deduplication (1-hour rolling window per dedup_key) and SQLite-backed
persistence + acknowledgement.
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Literal

import httpx

from agent_eval.monitor.persistence import Store

Severity = Literal["critical", "high", "medium", "low"]


@dataclass
class Alert:
    severity: Severity
    title: str
    body: str = ""
    metric: str | None = None
    component: str | None = None
    value: float | None = None
    threshold: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dedup_key: str | None = None  # if None, computed from (metric, component, severity)

    def __post_init__(self):
        if self.dedup_key is None:
            self.dedup_key = f"{self.metric}|{self.component}|{self.severity}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "title": self.title,
            "body": self.body,
            "metric": self.metric,
            "component": self.component,
            "value": self.value,
            "threshold": self.threshold,
            "dedup_key": self.dedup_key,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmailConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    use_tls: bool = True

    @classmethod
    def from_env(cls) -> EmailConfig:
        return cls(
            smtp_host=os.getenv("AGENT_EVAL_SMTP_HOST", ""),
            smtp_port=int(os.getenv("AGENT_EVAL_SMTP_PORT", "587")),
            smtp_user=os.getenv("AGENT_EVAL_SMTP_USER", ""),
            smtp_password=os.getenv("AGENT_EVAL_SMTP_PASSWORD", ""),
            from_addr=os.getenv("AGENT_EVAL_SMTP_FROM", ""),
            to_addrs=[a.strip() for a in os.getenv("AGENT_EVAL_SMTP_TO", "").split(",") if a.strip()],
            use_tls=os.getenv("AGENT_EVAL_SMTP_TLS", "true").lower() == "true",
        )


class AlertManager:
    """Multi-channel alert dispatcher with dedup + persistence."""

    def __init__(
        self,
        channels: list[str] | None = None,
        webhook_url: str | None = None,
        slack_webhook_url: str | None = None,
        email_config: EmailConfig | None = None,
        store_path: str | Path = "agent_eval_metrics.db",
        dedup_window_minutes: int = 60,
    ):
        self.channels = channels or ["stdout"]
        self.webhook_url = webhook_url
        self.slack_webhook_url = slack_webhook_url or os.getenv("AGENT_EVAL_SLACK_WEBHOOK")
        self.email_config = email_config or (EmailConfig.from_env() if "email" in self.channels else None)
        self.store = Store(store_path)
        self.dedup_window_minutes = dedup_window_minutes

    def fire(self, alert: Alert) -> bool:
        """Persist + dispatch. Returns False if deduped."""
        alert_id = self.store.write_alert(
            severity=alert.severity, title=alert.title, body=alert.body,
            metric=alert.metric, component=alert.component, value=alert.value,
            threshold=alert.threshold, dedup_key=alert.dedup_key,
            channels=",".join(self.channels), dedup_window_minutes=self.dedup_window_minutes,
            ts=alert.timestamp,
        )
        if alert_id is None:
            return False
        for ch in self.channels:
            try:
                if ch == "stdout":
                    self._stdout(alert)
                elif ch == "webhook":
                    self._webhook(alert)
                elif ch == "slack":
                    self._slack(alert)
                elif ch == "email":
                    self._email(alert)
            except Exception as e:
                print(f"[AlertManager] channel '{ch}' failed: {e}")
        return True

    # ----------------------------------------------------- channels

    def _stdout(self, alert: Alert) -> None:
        prefix = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "•"}.get(alert.severity, "•")
        print(f"{prefix} [{alert.severity.upper()}] {alert.title}")
        print(f"   metric={alert.metric} component={alert.component} value={alert.value} threshold={alert.threshold}")
        if alert.body:
            print(f"   {alert.body}")

    def _webhook(self, alert: Alert) -> None:
        if not self.webhook_url:
            return
        httpx.post(self.webhook_url, json=alert.to_dict(), timeout=5.0)

    def _slack(self, alert: Alert) -> None:
        if not self.slack_webhook_url:
            return
        emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "🔵"}.get(alert.severity, "•")
        payload = {
            "text": f"{emoji} *agent-eval* — {alert.title}",
            "attachments": [{
                "color": {"critical": "#c62828", "high": "#f9a825", "medium": "#1976d2", "low": "#9e9e9e"}.get(alert.severity, "#9e9e9e"),
                "fields": [
                    {"title": "Severity", "value": alert.severity, "short": True},
                    {"title": "Metric", "value": alert.metric or "—", "short": True},
                    {"title": "Component", "value": alert.component or "—", "short": True},
                    {"title": "Value / Threshold", "value": f"{alert.value} / {alert.threshold}", "short": True},
                ],
                "text": alert.body or "",
                "ts": int(alert.timestamp.timestamp()),
            }],
        }
        httpx.post(self.slack_webhook_url, json=payload, timeout=5.0)

    def _email(self, alert: Alert) -> None:
        cfg = self.email_config
        if not cfg or not cfg.smtp_host or not cfg.to_addrs:
            return
        msg = MIMEText(
            f"Severity: {alert.severity}\n"
            f"Metric: {alert.metric}\n"
            f"Component: {alert.component}\n"
            f"Value: {alert.value} (threshold: {alert.threshold})\n"
            f"Time: {alert.timestamp.isoformat()}\n\n"
            f"{alert.body}"
        )
        msg["Subject"] = f"[agent-eval][{alert.severity.upper()}] {alert.title}"
        msg["From"] = cfg.from_addr or cfg.smtp_user
        msg["To"] = ", ".join(cfg.to_addrs)
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10) as s:
            if cfg.use_tls:
                s.starttls()
            if cfg.smtp_user:
                s.login(cfg.smtp_user, cfg.smtp_password)
            s.sendmail(msg["From"], cfg.to_addrs, msg.as_string())

    # ----------------------------------------------------- listing / ack

    def list_alerts(self, only_unacked: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_alerts(only_unacked=only_unacked, limit=limit)

    def ack(self, alert_id: int) -> bool:
        return self.store.ack_alert(alert_id)
