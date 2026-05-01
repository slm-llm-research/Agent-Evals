"""HTML reporter — single-file dashboard.

Layout (top → bottom):
  1. Header: system name + run metadata
  2. **Recommendations** — Tuning Advisor signals at the top, severity-sorted
  3. Overview: gauge + dimension radar
  4. **Metrics explained** — every headline + dimension metric with a plain-English
     description and the score
  5. **Queries evaluated** — per-example drill-down with:
       - filter (all / passed / failed)
       - LangSmith deep-link button per query
       - <details> expandable card showing actual output, tool calls (success / error / latency),
         per-evaluator scores
  6. Top issues (severity table)
  7. Components ranking
  8. Dataset coverage charts (query types + complexity)
  9. Regression view (if baseline provided)
 10. Raw data table (sortable, filterable)
 11. Backends used

Self-contained: opens via file:// without a server. CDN-loaded Chart.js.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from jinja2 import Template

from agent_eval.reporters.report import EvaluationReport, status_for_score


# ---------------------------------------------------------------------------- explanations


HEADLINE_EXPLANATIONS = {
    "overall_score": (
        "Weighted aggregate across all 7 dimensions. Output quality 30%, hallucination-safety 15%, "
        "trajectory 15%, then 10% each for tool/system/safety/memory."
    ),
    "health_status": (
        "Bucket of overall_score: ≥0.85 excellent · 0.70-0.84 good · 0.50-0.69 needs improvement · <0.50 critical."
    ),
    "pass_rate": (
        "Fraction of (example × evaluator) pairs that passed their threshold (default 0.7). Higher is better."
    ),
    "flag_count": (
        "Number of evaluator results that scored below 0.5 — likely real issues to investigate."
    ),
    "critical_flag_count": (
        "Subset of flags that are critical: score < 0.3 OR a cycle was detected. Halt new features and fix these first."
    ),
}


DIMENSION_EXPLANATIONS = {
    "output_quality": (
        "How good is the agent's final answer? Aggregates: task success, faithfulness (sources support claims), "
        "relevance (answer addresses query), completeness, format compliance, keyword coverage."
    ),
    "trajectory_quality": (
        "How efficient and correct is the agent's reasoning path? Aggregates: tool selection accuracy, Tool F1, "
        "Node F1, structural similarity, intent resolution, step success, redundancy, error recovery, cycle detection."
    ),
    "hallucination_risk": (
        "Risk that the agent fabricated facts. **Higher = MORE risk** (inverted from the supporting evaluators). "
        "Aggregates 4 levels: planning (entities), observation (vs tool outputs), citation (URLs), reasoning (NLI)."
    ),
    "tool_performance": (
        "How reliable and useful were the tools? Aggregates: per-tool success rate, P95 latency, result quality, "
        "argument correctness, MCP server availability, cost per call."
    ),
    "system_performance": (
        "End-to-end runtime characteristics. Aggregates: P95 latency, time-to-first-audio-byte (voice), "
        "token efficiency, error rate, cost per query."
    ),
    "safety": (
        "Output safety. Aggregates: harmful content (OpenAI Moderation), PII leakage (regex), "
        "instruction-following compliance, response consistency across re-runs."
    ),
    "memory_quality": (
        "Active only if a memory backend (vector store / Mem0 / Zep) is detected. Aggregates: retrieval recall, "
        "retrieval precision, write quality (over/under-storing), staleness, cross-session continuity, cost. "
        "A score of 0 means no memory backend was detected — not a failure."
    ),
}


# ---------------------------------------------------------------------------- template


_HTML_TEMPLATE = Template(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>agent-eval report — {{ r.system_name }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 32px; color: #222; max-width: 1200px; }
  h1 { margin: 0 0 4px 0; }
  h2 { margin-top: 36px; padding-bottom: 6px; border-bottom: 1px solid #eee; }
  h3 { margin-top: 18px; font-size: 15px; }
  .muted { color: #777; font-size: 12px; }
  .badge { display: inline-block; padding: 3px 9px; border-radius: 12px; font-weight: 600; font-size: 12px; color: white; }
  .badge.excellent { background: #2e7d32; }
  .badge.good { background: #689f38; }
  .badge.needs_improvement { background: #f9a825; color: #222; }
  .badge.critical { background: #c62828; }
  .badge.passed { background: #2e7d32; }
  .badge.failed { background: #c62828; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .card { border: 1px solid #eee; border-radius: 8px; padding: 16px; background: #fff; }
  .gauge { font-size: 48px; font-weight: 700; }
  .small { font-size: 12px; color: #555; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #f1f1f1; vertical-align: top; }
  th { background: #fafafa; font-weight: 600; cursor: pointer; user-select: none; }
  th.asc::after { content: " ▲"; color: #999; }
  th.desc::after { content: " ▼"; color: #999; }
  tr.flagged td { background: #fff3e0; }
  tr.critical td { background: #ffebee; }
  details { margin: 8px 0; }
  details > summary { cursor: pointer; padding: 4px 0; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  pre { background: #fafafa; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px; line-height: 1.4; max-height: 320px; }
  .pill { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 8px; margin-right: 4px; background: #e0e0e0; color: #333; }
  .pill.critical { background: #c62828; color: white; }
  .pill.high { background: #f9a825; color: #222; }
  .pill.medium { background: #1976d2; color: white; }
  .pill.low { background: #9e9e9e; color: white; }
  .delta-pos { color: #2e7d32; font-weight: 600; }
  .delta-neg { color: #c62828; font-weight: 600; }
  .copy-btn { font-size: 11px; padding: 2px 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; border-radius: 4px; }
  .copy-btn:hover { background: #f0f0f0; }
  .nav { position: sticky; top: 0; background: white; padding: 8px 0; border-bottom: 1px solid #eee; margin-bottom: 16px; z-index: 10; }
  .nav a { margin-right: 12px; color: #1976d2; text-decoration: none; font-size: 13px; }
  .nav a:hover { text-decoration: underline; }
  input.filter { padding: 4px 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 4px; width: 240px; }
  .filter-bar { display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
  .filter-btn { padding: 4px 12px; border: 1px solid #ccc; background: white; cursor: pointer; border-radius: 4px; font-size: 13px; }
  .filter-btn.active { background: #1976d2; color: white; border-color: #1976d2; }
  .query-card { border: 1px solid #eee; border-radius: 8px; margin: 8px 0; padding: 0; background: #fff; }
  .query-header { padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .query-header:hover { background: #fafafa; }
  .query-header.passed { border-left: 4px solid #2e7d32; }
  .query-header.failed { border-left: 4px solid #c62828; }
  .query-text { flex: 1; min-width: 300px; font-weight: 500; }
  .query-body { padding: 0 16px 16px 16px; border-top: 1px solid #f0f0f0; background: #fafbfc; display: none; }
  .query-card.expanded .query-body { display: block; }
  .meta { font-size: 12px; color: #555; }
  .meta strong { color: #333; }
  .tool-table td { font-size: 12px; padding: 4px 8px; }
  .ok { color: #2e7d32; font-weight: 600; }
  .err { color: #c62828; font-weight: 600; }
  .recommendation { background: #fff7e6; border: 1px solid #ffe082; padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
  .recommendation.critical { background: #ffebee; border-color: #ef9a9a; }
  .recommendation.high { background: #fff8e1; border-color: #ffd54f; }
  .empty { color: #999; font-style: italic; padding: 12px 0; }
  .kbd { font-family: monospace; background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
</style>
</head>
<body>
  <h1>agent-eval — {{ r.system_name }}</h1>
  <div class="muted">
    Report ID <code>{{ r.report_id }}</code> · Generated {{ r.generated_at.strftime('%Y-%m-%d %H:%M UTC') }}
    · Dataset <code>{{ r.dataset_name }}</code> ({{ r.dataset_size }} examples)
    · Duration {{ '%.1f'|format(r.evaluation_duration_seconds) }}s
    · Package v{{ r.package_version }}
    {% if r.langsmith_project %} · LangSmith project <code>{{ r.langsmith_project }}</code>{% endif %}
  </div>

  <div class="nav">
    <a href="#recommendations">⚡ Recommendations</a>
    <a href="#overview">Overview</a>
    <a href="#metrics">Metrics</a>
    <a href="#queries">Queries</a>
    <a href="#issues">Issues</a>
    <a href="#components">Components</a>
    <a href="#dataset">Dataset</a>
    {% if baseline %}<a href="#regression">Regression</a>{% endif %}
    <a href="#raw">Raw data</a>
  </div>

  <!-- ════════════════════════════ RECOMMENDATIONS (TOP) ════════════════════════════ -->
  <h2 id="recommendations">⚡ Recommendations</h2>
  <div class="muted">Prioritized by severity. Each card explains the issue and the next concrete action.</div>
  {% if r.tuning_recommendations %}
    {% for s in r.tuning_recommendations %}
      <div class="recommendation {{ s.severity }}">
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
          <span class="pill {{ s.severity }}">{{ s.severity }}</span>
          <strong>{{ s.component }}</strong>
          <span class="muted">— {{ s.issue_type }}</span>
          <span style="margin-left: auto;" class="muted">
            score {{ '%.2f'|format(s.current_score) }} → target {{ '%.2f'|format(s.target_score) }}
            · effort: {{ s.effort_estimate }} · type: <code>{{ s.tuning_type }}</code>
          </span>
        </div>
        <div style="margin-top: 8px;">{{ s.specific_action }}</div>
        {% if s.example_fix %}
          <details><summary class="muted">example fix</summary>
            <pre>{{ s.example_fix }}</pre>
          </details>
        {% endif %}
      </div>
    {% endfor %}
  {% else %}
    <div class="empty">No tuning recommendations — system is healthy.</div>
  {% endif %}

  <!-- ════════════════════════════ OVERVIEW ════════════════════════════ -->
  <h2 id="overview">📊 Overview</h2>
  <div class="grid">
    <div class="card">
      <div class="small">Overall score</div>
      <div class="gauge">{{ '%.2f'|format(r.system_overview.overall_score) }}</div>
      <div><span class="badge {{ r.system_overview.health_status }}">{{ r.system_overview.health_status|replace('_',' ')|upper }}</span></div>
      <div class="small" style="margin-top: 12px;">
        <strong>{{ '%.0f'|format(r.system_overview.pass_rate * 100) }}%</strong> pass rate ·
        <strong>{{ r.system_overview.flag_count }}</strong> flag{{ '' if r.system_overview.flag_count == 1 else 's' }}
        ({{ r.system_overview.critical_flag_count }} critical)
      </div>
    </div>
    <div class="card">
      <div class="small">Dimension radar</div>
      <canvas id="dimensionRadar" width="350" height="280"></canvas>
    </div>
  </div>

  <!-- ════════════════════════════ METRICS EXPLAINED ════════════════════════════ -->
  <h2 id="metrics">📈 Metrics</h2>

  <h3>Headline metrics</h3>
  <table>
    <thead><tr><th style="width: 200px;">Metric</th><th style="width: 100px;">Score</th><th>What it means</th></tr></thead>
    <tbody>
      <tr><td><strong>Overall</strong></td><td>{{ '%.3f'|format(r.system_overview.overall_score) }}</td><td>{{ headline_expl.overall_score }}</td></tr>
      <tr><td><strong>Status</strong></td><td><span class="badge {{ r.system_overview.health_status }}">{{ r.system_overview.health_status|replace('_',' ')|upper }}</span></td><td>{{ headline_expl.health_status }}</td></tr>
      <tr><td><strong>Pass rate</strong></td><td>{{ '%.0f'|format(r.system_overview.pass_rate * 100) }}%</td><td>{{ headline_expl.pass_rate }}</td></tr>
      <tr{% if r.system_overview.flag_count > 0 %} class="flagged"{% endif %}><td><strong>Flags</strong></td><td>{{ r.system_overview.flag_count }}</td><td>{{ headline_expl.flag_count }}</td></tr>
      <tr{% if r.system_overview.critical_flag_count > 0 %} class="critical"{% endif %}><td><strong>Critical flags</strong></td><td>{{ r.system_overview.critical_flag_count }}</td><td>{{ headline_expl.critical_flag_count }}</td></tr>
    </tbody>
  </table>

  <h3>Dimension scores</h3>
  <table>
    <thead><tr><th style="width: 200px;">Dimension</th><th style="width: 100px;">Score</th><th>What it measures</th></tr></thead>
    <tbody>
    {% for k, v in dimension_pairs %}
      <tr {% if (k != 'hallucination_risk' and v < 0.5) or (k == 'hallucination_risk' and v > 0.25) %}class="critical"{% elif (k != 'hallucination_risk' and v < 0.7) or (k == 'hallucination_risk' and v > 0.10) %}class="flagged"{% endif %}>
        <td><strong>{{ k }}</strong></td>
        <td>{{ '%.3f'|format(v) }}</td>
        <td>{{ dimension_expl.get(k, '—') }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <!-- ════════════════════════════ QUERIES EVALUATED ════════════════════════════ -->
  <h2 id="queries">📋 Queries evaluated</h2>
  <div class="muted">{{ r.dataset_size }} queries ran against the agent. Click any card to expand actual output, tool calls, and per-evaluator scores.</div>

  <div class="filter-bar">
    <button class="filter-btn active" data-filter="all" onclick="filterQueries('all', this)">All ({{ r.per_example_results|length }})</button>
    <button class="filter-btn" data-filter="passed" onclick="filterQueries('passed', this)">✅ Passed ({{ n_passed }})</button>
    <button class="filter-btn" data-filter="failed" onclick="filterQueries('failed', this)">❌ Failed ({{ n_failed }})</button>
    <input class="filter" id="query-text-filter" placeholder="Filter by query text…" oninput="filterByText()">
  </div>

  {% if r.per_example_results %}
    <div id="queries-container">
    {% for ex in r.per_example_results %}
      <div class="query-card" data-status="{{ 'passed' if ex.is_overall_pass else 'failed' }}">
        <div class="query-header {{ 'passed' if ex.is_overall_pass else 'failed' }}" onclick="toggleQuery(this)">
          <span class="badge {{ 'passed' if ex.is_overall_pass else 'failed' }}">{{ '✓ PASS' if ex.is_overall_pass else '✗ FAIL' }}</span>
          <span class="query-text">{{ ex.query }}</span>
          <span class="pill">{{ ex.query_type }}</span>
          <span class="pill">{{ ex.complexity }}</span>
          <span class="muted" title="Mean evaluator score">score {{ '%.2f'|format(ex.score) }}</span>
          <span class="muted" title="Fraction of evaluators that passed">pass {{ '%.0f'|format(ex.pass_rate * 100) }}%</span>
          {% if ex.flagged_count > 0 %}<span class="pill {{ 'critical' if ex.critical_count > 0 else 'high' }}">{{ ex.flagged_count }} flag{{ '' if ex.flagged_count == 1 else 's' }}</span>{% endif %}
          {% if ex.langsmith_run_url %}
            <a href="{{ ex.langsmith_run_url }}" target="_blank" onclick="event.stopPropagation()" style="margin-left: 8px;" title="Open trace in LangSmith">🔗 trace</a>
          {% elif ex.trace_was_synthetic %}
            <span class="muted" title="No real LangSmith trace; agent did not return langsmith_run_id">(synthetic trace)</span>
          {% endif %}
        </div>
        <div class="query-body">
          <h3>Actual output</h3>
          {% if ex.actual_output %}
            <pre>{{ ex.actual_output }}</pre>
          {% else %}
            <div class="empty">(no output captured)</div>
          {% endif %}

          {% if ex.expected_keywords %}
          <h3>Expected keywords</h3>
          <div class="meta">
            {% for kw in ex.expected_keywords %}
              {% if kw.lower() in (ex.actual_output or '')|lower %}
                <span class="pill" style="background: #c8e6c9; color: #1b5e20;">✓ {{ kw }}</span>
              {% else %}
                <span class="pill" style="background: #ffcdd2; color: #b71c1c;">✗ {{ kw }}</span>
              {% endif %}
            {% endfor %}
          </div>
          {% endif %}

          {% if ex.expected_output %}
          <h3>Reference answer</h3>
          <pre>{{ ex.expected_output }}</pre>
          {% endif %}

          <h3>Tool calls ({{ ex.tool_calls|length }})</h3>
          {% if ex.tool_calls %}
          <table class="tool-table">
            <thead><tr><th>#</th><th>Tool</th><th>Status</th><th>Latency</th><th>Inputs</th><th>Outputs (preview)</th></tr></thead>
            <tbody>
            {% for tc in ex.tool_calls %}
              <tr {% if not tc.success %}class="critical"{% endif %}>
                <td>{{ loop.index }}</td>
                <td><code>{{ tc.name }}</code></td>
                <td>{% if tc.success %}<span class="ok">✓ ok</span>{% else %}<span class="err">✗ {{ tc.error or 'error' }}</span>{% endif %}</td>
                <td>{{ '%.0f'|format(tc.latency_ms) if tc.latency_ms else '—' }}ms</td>
                <td><pre style="margin: 0; max-height: 100px;">{{ tc.inputs|tojson(indent=2) }}</pre></td>
                <td><pre style="margin: 0; max-height: 100px;">{{ tc.outputs_preview }}</pre></td>
              </tr>
            {% endfor %}
            </tbody>
          </table>
          {% else %}
            <div class="empty">No tool calls observed for this query.</div>
          {% endif %}

          <h3>Per-evaluator scores</h3>
          <table class="tool-table">
            <thead><tr><th>Metric</th><th>Score</th><th>Pass</th><th>Threshold</th><th>Latency</th><th>Reason / details</th></tr></thead>
            <tbody>
            {% for ev in ex.evaluator_results %}
              <tr {% if ev.flagged %}class="{{ 'critical' if ev.score < 0.3 else 'flagged' }}"{% endif %}>
                <td><code>{{ ev.evaluator_name }}</code></td>
                <td>{{ '%.3f'|format(ev.score) }}</td>
                <td>{% if ev.passed %}<span class="ok">✓</span>{% else %}<span class="err">✗</span>{% endif %}</td>
                <td>{{ '%.2f'|format(ev.threshold) }}</td>
                <td>{{ '%.0f'|format(ev.latency_ms) }}ms</td>
                <td>{{ ev.flag_reason or (ev.details.get('reason') if ev.details else '') or '' }}</td>
              </tr>
            {% endfor %}
            </tbody>
          </table>

          <div class="meta" style="margin-top: 10px;">
            <strong>Example ID:</strong> <code>{{ ex.example_id }}</code>
            {% if ex.langsmith_run_id %} · <strong>LangSmith run:</strong> <code>{{ ex.langsmith_run_id }}</code>{% endif %}
            {% if ex.runner_latency_ms %} · <strong>Runner latency:</strong> {{ '%.0f'|format(ex.runner_latency_ms) }}ms{% endif %}
            {% if ex.runner_error %} · <strong style="color: #c62828;">Runner error:</strong> {{ ex.runner_error }}{% endif %}
          </div>
        </div>
      </div>
    {% endfor %}
    </div>
  {% else %}
    <div class="empty">No per-example data — was this report generated without a runner?</div>
  {% endif %}

  <!-- ════════════════════════════ ISSUES ════════════════════════════ -->
  <h2 id="issues">⚠️ Top issues</h2>
  {% if r.flagged_issues %}
  <table>
    <thead><tr><th>Severity</th><th>Component</th><th>Metric</th><th>Score</th><th>Description</th></tr></thead>
    <tbody>
    {% for issue in r.flagged_issues[:30] %}
      <tr class="{{ 'critical' if issue.severity == 'critical' else 'flagged' if issue.severity in ['high','medium'] else '' }}">
        <td><span class="pill {{ issue.severity }}">{{ issue.severity }}</span></td>
        <td>{{ issue.component }}</td>
        <td><code>{{ issue.metric }}</code></td>
        <td>{{ '%.2f'|format(issue.score) }}</td>
        <td>{{ issue.description }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">No flagged issues.</div>
  {% endif %}

  <!-- ════════════════════════════ COMPONENTS ════════════════════════════ -->
  <h2 id="components">📦 Components</h2>
  <table id="components-table">
    <thead><tr><th data-sort="num">#</th><th data-sort="text">Component</th><th data-sort="text">Type</th><th data-sort="num">Score</th><th>Status</th><th data-sort="num">Trend</th></tr></thead>
    <tbody>
    {% for c in r.component_scores %}
      <tr>
        <td>{{ c.rank }}</td>
        <td><strong>{{ c.component_name }}</strong></td>
        <td>{{ c.component_type }}</td>
        <td>{{ '%.2f'|format(c.overall_score) }}</td>
        <td><span class="badge {{ status_for(c.overall_score) }}">{{ status_for(c.overall_score)|replace('_',' ')|upper }}</span></td>
        <td>
          {% if c.trend is not none %}
            <span class="{{ 'delta-pos' if c.trend >= 0 else 'delta-neg' }}">{{ '+%.2f'|format(c.trend) if c.trend >= 0 else '%.2f'|format(c.trend) }}</span>
          {% else %}—{% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <!-- ════════════════════════════ DATASET ════════════════════════════ -->
  <h2 id="dataset">📊 Dataset coverage</h2>
  <div class="grid">
    <div class="card">
      <div class="small">Query types</div>
      <canvas id="queryTypePie" width="350" height="280"></canvas>
    </div>
    <div class="card">
      <div class="small">Complexity</div>
      <canvas id="complexityBar" width="350" height="280"></canvas>
    </div>
  </div>

  {% if baseline %}
  <h2 id="regression">↻ Regression vs baseline</h2>
  <div class="grid">
    <div class="card">
      <div class="small">Overall delta</div>
      <div class="gauge {{ 'delta-pos' if delta_overall >= 0 else 'delta-neg' }}">{{ '+%.3f'|format(delta_overall) if delta_overall >= 0 else '%.3f'|format(delta_overall) }}</div>
      <div class="small">{{ baseline.system_name }} ({{ baseline.generated_at.strftime('%Y-%m-%d') }}) → current</div>
    </div>
    <div class="card">
      <div class="small">Side-by-side dimensions</div>
      <canvas id="regressionRadar" width="350" height="280"></canvas>
    </div>
  </div>
  <table>
    <thead><tr><th>Dimension</th><th>Baseline</th><th>Current</th><th>Δ%</th></tr></thead>
    <tbody>
    {% for k, v in regression_changes.items() %}
      <tr>
        <td><code>{{ k }}</code></td>
        <td>{{ '%.3f'|format(v.baseline) }}</td>
        <td>{{ '%.3f'|format(v.current) }}</td>
        <td><span class="{{ 'delta-pos' if v.pct_change >= 0 else 'delta-neg' }}">{{ '+%.1f'|format(v.pct_change) if v.pct_change >= 0 else '%.1f'|format(v.pct_change) }}%</span></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <!-- ════════════════════════════ RAW DATA ════════════════════════════ -->
  <h2 id="raw">📚 Raw data</h2>
  <input class="filter" id="raw-filter" placeholder="Filter by component, metric, reason…" oninput="filterRaw()">
  <table id="raw-table">
    <thead><tr><th data-sort="text">Component</th><th data-sort="text">Metric</th><th data-sort="num">Score</th><th>Pass</th><th data-sort="num">Latency</th><th>Flag reason</th></tr></thead>
    <tbody>
    {% for c in r.component_scores %}
      {% for ev in c.evaluator_results %}
      <tr class="{{ 'flagged' if ev.flagged else '' }}">
        <td>{{ c.component_name }}</td>
        <td><code>{{ ev.evaluator_name }}</code></td>
        <td>{{ '%.3f'|format(ev.score) }}</td>
        <td>{{ '✓' if ev.passed else '✗' }}</td>
        <td>{{ '%.1f'|format(ev.latency_ms) }}ms</td>
        <td>{{ ev.flag_reason or '—' }}</td>
      </tr>
      {% endfor %}
    {% endfor %}
    </tbody>
  </table>

  <h2 id="backends">🔧 Backends used</h2>
  <ul class="small">
  {% for evaluator, backend in r.backends_used.items() %}
    <li><code>{{ evaluator }}</code> via <strong>{{ backend }}</strong></li>
  {% else %}
    <li>—</li>
  {% endfor %}
  </ul>

<script>
new Chart(document.getElementById('dimensionRadar'), {
  type: 'radar',
  data: {
    labels: ['output','trajectory','hallu_risk','tool','system','safety','memory'],
    datasets: [{
      label: '{{ r.system_name }}',
      data: [
        {{ r.dimension_scores.output_quality }},
        {{ r.dimension_scores.trajectory_quality }},
        {{ r.dimension_scores.hallucination_risk }},
        {{ r.dimension_scores.tool_performance }},
        {{ r.dimension_scores.system_performance }},
        {{ r.dimension_scores.safety }},
        {{ r.dimension_scores.memory_quality }}
      ],
      fill: true,
      backgroundColor: 'rgba(70,130,180,0.2)',
      borderColor: 'rgb(70,130,180)',
      pointBackgroundColor: 'rgb(70,130,180)',
    }]
  },
  options: { scales: { r: { suggestedMin: 0, suggestedMax: 1 } } }
});

new Chart(document.getElementById('queryTypePie'), {
  type: 'doughnut',
  data: {
    labels: {{ query_type_labels|tojson }},
    datasets: [{ data: {{ query_type_counts|tojson }}, backgroundColor: ['#1976d2','#689f38','#fbc02d','#c62828','#7b1fa2','#00838f'] }]
  }
});

new Chart(document.getElementById('complexityBar'), {
  type: 'bar',
  data: {
    labels: {{ complexity_labels|tojson }},
    datasets: [{ data: {{ complexity_counts|tojson }}, backgroundColor: '#689f38' }]
  },
  options: { plugins: { legend: { display: false } } }
});

{% if baseline %}
new Chart(document.getElementById('regressionRadar'), {
  type: 'radar',
  data: {
    labels: ['output','trajectory','hallu_risk','tool','system','safety','memory'],
    datasets: [
      { label: 'baseline', borderColor: '#999', backgroundColor: 'rgba(150,150,150,0.1)',
        data: [{{ baseline.dimension_scores.output_quality }}, {{ baseline.dimension_scores.trajectory_quality }}, {{ baseline.dimension_scores.hallucination_risk }}, {{ baseline.dimension_scores.tool_performance }}, {{ baseline.dimension_scores.system_performance }}, {{ baseline.dimension_scores.safety }}, {{ baseline.dimension_scores.memory_quality }}] },
      { label: 'current', borderColor: '#1976d2', backgroundColor: 'rgba(25,118,210,0.2)',
        data: [{{ r.dimension_scores.output_quality }}, {{ r.dimension_scores.trajectory_quality }}, {{ r.dimension_scores.hallucination_risk }}, {{ r.dimension_scores.tool_performance }}, {{ r.dimension_scores.system_performance }}, {{ r.dimension_scores.safety }}, {{ r.dimension_scores.memory_quality }}] }
    ]
  },
  options: { scales: { r: { suggestedMin: 0, suggestedMax: 1 } } }
});
{% endif %}

// Sortable
function attachSort(tableId) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const headers = table.querySelectorAll('th[data-sort]');
  headers.forEach((h, idx) => {
    h.addEventListener('click', () => {
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const sortType = h.dataset.sort;
      const asc = !h.classList.contains('asc');
      headers.forEach(hh => hh.classList.remove('asc', 'desc'));
      h.classList.add(asc ? 'asc' : 'desc');
      rows.sort((a, b) => {
        const av = a.children[idx].textContent.trim();
        const bv = b.children[idx].textContent.trim();
        if (sortType === 'num') return (parseFloat(av) - parseFloat(bv)) * (asc ? 1 : -1);
        return av.localeCompare(bv) * (asc ? 1 : -1);
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
}
attachSort('components-table');
attachSort('raw-table');

function filterRaw() {
  const q = document.getElementById('raw-filter').value.toLowerCase();
  const tbody = document.querySelector('#raw-table tbody');
  Array.from(tbody.querySelectorAll('tr')).forEach(r => {
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}

// Query card expand/collapse
function toggleQuery(header) {
  header.parentElement.classList.toggle('expanded');
}

// Filter queries by status
let _activeStatusFilter = 'all';
function filterQueries(status, btn) {
  _activeStatusFilter = status;
  document.querySelectorAll('.filter-bar .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyQueryFilters();
}
function filterByText() { applyQueryFilters(); }
function applyQueryFilters() {
  const txt = (document.getElementById('query-text-filter').value || '').toLowerCase();
  document.querySelectorAll('.query-card').forEach(card => {
    const matchStatus = _activeStatusFilter === 'all' || card.dataset.status === _activeStatusFilter;
    const matchText = !txt || card.textContent.toLowerCase().includes(txt);
    card.style.display = (matchStatus && matchText) ? '' : 'none';
  });
}
</script>
</body>
</html>
""")


# ---------------------------------------------------------------------------- entry


def render_html_report(
    report: EvaluationReport,
    output_path: str | Path,
    baseline: EvaluationReport | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    qt_dist = report.dataset_stats.query_type_distribution or {}
    cx_dist = report.dataset_stats.complexity_distribution or {}
    if not qt_dist:
        qt_dist = {"unknown": report.dataset_size}
    if not cx_dist:
        cx_dist = {"unknown": report.dataset_size}

    regression_changes: dict[str, dict[str, float]] = {}
    delta_overall = 0.0
    if baseline is not None:
        cur = report.dimension_scores.model_dump()
        base = baseline.dimension_scores.model_dump()
        for metric, c in cur.items():
            b = float(base.get(metric, c))
            pct = ((c - b) / b * 100) if b else 0.0
            regression_changes[metric] = {"baseline": b, "current": c, "pct_change": round(pct, 2)}
        delta_overall = report.system_overview.overall_score - baseline.system_overview.overall_score

    n_passed = sum(1 for ex in report.per_example_results if ex.is_overall_pass)
    n_failed = len(report.per_example_results) - n_passed

    html = _HTML_TEMPLATE.render(
        r=report,
        baseline=baseline,
        delta_overall=delta_overall,
        regression_changes=regression_changes,
        status_for=status_for_score,
        query_type_labels=list(qt_dist.keys()),
        query_type_counts=list(qt_dist.values()),
        complexity_labels=list(cx_dist.keys()),
        complexity_counts=list(cx_dist.values()),
        headline_expl=HEADLINE_EXPLANATIONS,
        dimension_expl=DIMENSION_EXPLANATIONS,
        dimension_pairs=list(report.dimension_scores.model_dump().items()),
        n_passed=n_passed,
        n_failed=n_failed,
    )
    path.write_text(html, encoding="utf-8")
    return path


def deep_link_for_run(langsmith_url: str, run_id: str, project: str | None = None, org: str | None = None) -> str:
    """Build a LangSmith deep-link for a run id."""
    base = langsmith_url.rstrip("/").replace("api.smith.langchain.com", "smith.langchain.com")
    if org and project:
        return f"{base}/o/{org}/projects/p/{project}/r/{run_id}"
    return f"{base}/r/{run_id}"
