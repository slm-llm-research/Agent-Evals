"""Tests for the deeper discovery layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_eval.discovery.langsmith_inspector import LangSmithInspector
from agent_eval.discovery.memory_detector import MemoryDetector


def test_langsmith_get_component_stats_no_runs():
    class _C:
        def list_runs(self, **_):
            return iter([])

    insp = LangSmithInspector(_C(), "p")
    stats = insp.get_component_stats("anything")
    assert stats["call_count"] == 0


def test_langsmith_transition_matrix():
    """Mine an in-memory trace and check that transitions are recorded."""
    from tests.conftest import FakeRun

    parent = FakeRun(
        name="orchestrator", run_type="chain",
        children=[
            FakeRun(name="search_agent", run_type="agent",
                    children=[FakeRun(name="web_search", run_type="tool", inputs={"q": "x"})]),
            FakeRun(name="format_response", run_type="chain"),
        ],
    )

    class _C:
        def list_runs(self, **_):
            return iter([parent])

    insp = LangSmithInspector(_C(), "p")
    structure = insp.infer_agent_structure()
    assert "transition_matrix" in structure
    assert "orchestrator" in structure["transition_matrix"]
    transitions = structure["transition_matrix"]["orchestrator"]
    assert "search_agent" in transitions
    assert "format_response" in transitions


def test_memory_detector_source_imports(tmp_path: Path):
    """Drop a fake user file with vector-store imports and confirm detection."""
    f = tmp_path / "user_app.py"
    f.write_text(
        "import chromadb\n"
        "from pinecone import Pinecone\n"
        "from mem0 import Memory\n"
        "import os\n"
    )
    backends = MemoryDetector.detect_from_source([tmp_path])
    types = {b.type for b in backends}
    assert "chroma" in types
    assert "pinecone" in types
    assert "mem0" in types


def test_memory_detector_source_skips_venv(tmp_path: Path):
    """Make sure we don't scan venv / site-packages."""
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "should_not_be_scanned.py").write_text("import chromadb")
    real = tmp_path / "real.py"
    real.write_text("import os")
    backends = MemoryDetector.detect_from_source([tmp_path])
    assert all(b.type != "chroma" for b in backends)


def test_memory_detector_source_no_imports_returns_empty(tmp_path: Path):
    f = tmp_path / "no_mem.py"
    f.write_text("import os\nimport sys\n")
    backends = MemoryDetector.detect_from_source([tmp_path])
    assert backends == []
