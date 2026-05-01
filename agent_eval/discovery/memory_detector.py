"""Memory subsystem detector — finds vector stores, Mem0, Zep, etc.

Three detection paths:
  1. LangSmith trace span names matching `*VectorStore*`, `*Retriever*`, `*Memory*`,
     `*Pinecone*`, `*Chroma*`, `*Weaviate*`, `*Qdrant*`, `*Mem0*`, `*Zep*`.
  2. Mem0 / Zep API call patterns in trace inputs.
  3. Source-code import scanning: greps user Python source for
     `chromadb`, `pinecone`, `weaviate`, `qdrant`, `faiss`, `mem0`, `zep` imports.
     Useful for systems whose memory subsystem hasn't been exercised yet in traces.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.discovery.registry import MemoryBackendInfo

_BACKEND_PATTERNS = {
    "chroma": re.compile(r"chroma", re.IGNORECASE),
    "pinecone": re.compile(r"pinecone", re.IGNORECASE),
    "weaviate": re.compile(r"weaviate", re.IGNORECASE),
    "qdrant": re.compile(r"qdrant", re.IGNORECASE),
    "faiss": re.compile(r"\bfaiss\b", re.IGNORECASE),
    "mem0": re.compile(r"mem0", re.IGNORECASE),
    "zep": re.compile(r"\bzep\b", re.IGNORECASE),
    "redis": re.compile(r"redis.*(vector|index|embedding)", re.IGNORECASE),
    "generic_vector_store": re.compile(r"(vector\s*store|retriever)", re.IGNORECASE),
}

_LONG_TERM_HINTS = {"mem0", "zep"}


class MemoryDetector:
    def __init__(self, langsmith_client: Any | None = None, project_name: str | None = None):
        self.client = langsmith_client
        self.project_name = project_name

    def detect_from_traces(self, traces: list[Any]) -> list[MemoryBackendInfo]:
        seen: dict[str, MemoryBackendInfo] = {}
        for trace in traces:
            for run in _iter_runs(trace):
                name = run.get("name") or ""
                blob = f"{name} {run.get('run_type') or ''}"
                inputs = run.get("inputs") or {}
                if isinstance(inputs, dict):
                    blob += " " + " ".join(str(v) for v in inputs.values() if isinstance(v, str))
                for backend, pat in _BACKEND_PATTERNS.items():
                    if pat.search(blob):
                        info = seen.setdefault(
                            backend,
                            MemoryBackendInfo(
                                type=backend,
                                call_frequency=0,
                                is_long_term=backend in _LONG_TERM_HINTS,
                                source="trace_mining",
                            ),
                        )
                        info.call_frequency += 1
                        latency = _latency_ms(run)
                        if latency is not None:
                            prev = info.avg_retrieval_latency_ms or 0.0
                            info.avg_retrieval_latency_ms = (prev * (info.call_frequency - 1) + latency) / info.call_frequency
                        break
        return list(seen.values())

    def detect_memory_operations(self, traces: list[Any]) -> dict[str, int]:
        """Counts of store/retrieve/update/delete operations across traces."""
        counts = {"store": 0, "retrieve": 0, "update": 0, "delete": 0}
        store_re = re.compile(r"(add|store|insert|upsert|write|put|create)", re.IGNORECASE)
        retrieve_re = re.compile(r"(query|search|retrieve|get|fetch|read|similarity)", re.IGNORECASE)
        update_re = re.compile(r"update", re.IGNORECASE)
        delete_re = re.compile(r"delete|remove", re.IGNORECASE)
        for trace in traces:
            for run in _iter_runs(trace):
                name = run.get("name") or ""
                blob = f"{name}"
                is_memoryish = any(p.search(name) for p in _BACKEND_PATTERNS.values())
                if not is_memoryish:
                    continue
                if delete_re.search(blob):
                    counts["delete"] += 1
                elif update_re.search(blob):
                    counts["update"] += 1
                elif retrieve_re.search(blob):
                    counts["retrieve"] += 1
                elif store_re.search(blob):
                    counts["store"] += 1
        return counts

    def detect(self, n_traces: int = 100, source_paths: list[Path] | None = None) -> list[MemoryBackendInfo]:
        """Run all detection paths and merge results."""
        results: dict[str, MemoryBackendInfo] = {}
        # Path 1+2: trace-based.
        if self.client is not None and self.project_name is not None:
            try:
                runs = list(
                    self.client.list_runs(project_name=self.project_name, run_type="chain", limit=n_traces)
                )
            except Exception:
                runs = []
            for backend in self.detect_from_traces(runs):
                results[backend.type] = backend
        # Path 3: source-import.
        if source_paths:
            for backend in self.detect_from_source(source_paths):
                if backend.type not in results:
                    results[backend.type] = backend
                else:
                    # Mark source as combined.
                    results[backend.type].source = "trace_mining"
        return list(results.values())

    @staticmethod
    def detect_from_source(paths: list[Path]) -> list[MemoryBackendInfo]:
        """Scan Python source for memory-backend imports.

        Returns one MemoryBackendInfo per detected backend type. `call_frequency`
        reflects the count of import statements found (a rough proxy for usage).
        """
        backend_imports = {
            "chroma": ("chromadb",),
            "pinecone": ("pinecone",),
            "weaviate": ("weaviate",),
            "qdrant": ("qdrant_client", "qdrant"),
            "faiss": ("faiss",),
            "mem0": ("mem0", "mem0ai"),
            "zep": ("zep_python", "zep"),
        }
        counts: dict[str, int] = {}

        def _scan_file(path: Path):
            try:
                src = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(src, filename=str(path))
            except Exception:
                return
            for node in ast.walk(tree):
                mod_names: list[str] = []
                if isinstance(node, ast.Import):
                    mod_names.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mod_names.append(node.module)
                for mn in mod_names:
                    head = mn.split(".", 1)[0]
                    for backend, prefixes in backend_imports.items():
                        if head in prefixes:
                            counts[backend] = counts.get(backend, 0) + 1

        for p in paths:
            p = Path(p)
            if p.is_dir():
                for f in p.rglob("*.py"):
                    # Skip virtualenvs and our own tests.
                    if any(part in ("venv", ".venv", "site-packages", "__pycache__", "tests", "test") for part in f.parts):
                        continue
                    _scan_file(f)
            elif p.suffix == ".py":
                _scan_file(p)

        out: list[MemoryBackendInfo] = []
        for backend, count in counts.items():
            out.append(MemoryBackendInfo(
                type=backend,
                call_frequency=count,
                is_long_term=backend in _LONG_TERM_HINTS,
                source="manual",  # source-import is closer to manual/static analysis than trace mining
            ))
        return out


def _iter_runs(trace: Any):
    stack = [trace]
    while stack:
        run = stack.pop()
        yield {
            "name": getattr(run, "name", None),
            "run_type": getattr(run, "run_type", None),
            "inputs": getattr(run, "inputs", None) or {},
            "start_time": getattr(run, "start_time", None),
            "end_time": getattr(run, "end_time", None),
        }
        for c in getattr(run, "child_runs", None) or []:
            stack.append(c)


def _latency_ms(run: dict[str, Any]) -> float | None:
    s, e = run.get("start_time"), run.get("end_time")
    if isinstance(s, datetime) and isinstance(e, datetime):
        return (e - s).total_seconds() * 1000.0
    return None
