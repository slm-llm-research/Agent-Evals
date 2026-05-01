"""CI gate command — closes the loop between evaluation and deploy control.

Usage from Python:
    from agent_eval.ci import ci_gate
    exit_code = ci_gate(current_report, baseline_report=baseline, thresholds_path="ci.yaml")

CLI: `agent-eval ci-gate --report .../report.json --baseline-report .../baseline.json --thresholds ci.yaml`

Optionally posts the markdown summary as a GitHub PR comment when:
  - `GITHUB_TOKEN` env var is set
  - `GITHUB_REPOSITORY` env var is set (e.g., 'org/repo')
  - `PR_NUMBER` env var is set (or `GITHUB_REF` parses to refs/pull/N/merge)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx

from agent_eval.ci.thresholds import load_thresholds
from agent_eval.reporters.regression_detector import RegressionDetector, ThresholdConfig
from agent_eval.reporters.report import EvaluationReport


def ci_gate(
    current_report: EvaluationReport,
    baseline_report: EvaluationReport | None = None,
    thresholds_path: str | Path | None = None,
    output_summary_path: str | Path | None = None,
    post_to_github: bool = True,
) -> int:
    """Returns exit code: 0 = pass, 1 = block deploy."""
    thresholds = load_thresholds(thresholds_path) if thresholds_path else ThresholdConfig()
    detector = RegressionDetector()
    regressions = detector.detect(current_report, baseline_report, thresholds) if baseline_report else []
    violations = detector.check_absolute_thresholds(current_report, thresholds)
    formatted = detector.ci_format(regressions, violations)
    print(formatted["summary"])
    if output_summary_path:
        Path(output_summary_path).write_text(formatted["summary"])
    if post_to_github:
        try:
            posted = post_to_github_pr(formatted["summary"])
            if posted:
                print(f"Posted CI gate summary to PR #{posted}")
        except Exception as e:
            print(f"[ci-gate] PR comment posting failed: {e}")
    return int(formatted["exit_code"])


def post_to_github_pr(summary_md: str, pr_number: int | None = None,
                       repo: str | None = None, token: str | None = None) -> int | None:
    """Post `summary_md` as a comment on the PR. Returns the PR number on success, None otherwise.

    Required env (or args):
      - GITHUB_TOKEN: a token with `pull-requests: write`. In GitHub Actions this is
        automatically provided as `${{ secrets.GITHUB_TOKEN }}` if you set permissions.
      - GITHUB_REPOSITORY: 'org/repo'.
      - PR_NUMBER: explicit PR number, or we'll try to parse from GITHUB_REF.
    """
    token = token or os.getenv("GITHUB_TOKEN")
    repo = repo or os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    if pr_number is None:
        env_pr = os.getenv("PR_NUMBER") or os.getenv("GITHUB_PR_NUMBER")
        if env_pr and env_pr.isdigit():
            pr_number = int(env_pr)
        else:
            ref = os.getenv("GITHUB_REF", "")
            m = re.match(r"refs/pull/(\d+)/", ref)
            if m:
                pr_number = int(m.group(1))
    if pr_number is None:
        return None

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {"body": _wrap_summary(summary_md)}
    resp = httpx.post(url, headers=headers, json=body, timeout=10.0)
    if resp.status_code in (200, 201):
        return pr_number
    raise RuntimeError(f"GitHub PR comment failed: {resp.status_code} {resp.text[:200]}")


def _wrap_summary(summary_md: str) -> str:
    return (
        "<!-- agent-eval ci-gate -->\n"
        "## 🤖 agent-eval CI gate\n\n"
        + summary_md
        + "\n\n"
        "<sub>Posted by [agent-eval](https://github.com/agent-trust/agent-eval) ci-gate.</sub>"
    )
