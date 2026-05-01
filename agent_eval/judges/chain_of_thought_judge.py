"""Chain-of-thought trajectory judge — TRACE evidence-bank methodology.

Implements the spec's TRACE approach:
  1. Walk the trajectory step-by-step.
  2. Maintain a progressive evidence bank seeded by tool outputs.
  3. For each reasoning step, run an LLM judge to evaluate:
       - relevance to the user query
       - correctness given accumulated evidence
       - efficiency vs. shorter alternatives
  4. Detect hallucinations by comparing stated observations to actual tool outputs.
  5. Detect redundancy by hashing (node, input).
  6. Emit per-step + holistic scores: efficiency, coherence, adaptivity.

The evidence bank caps at `evidence_bank_size` to keep prompts bounded; old
evidence is discarded LRU.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, Field

from agent_eval.judges.base_judge import BaseJudge, JudgeResult, traceable


class TrajectoryStep(BaseModel):
    node: str
    input: Any = None
    output: Any = None
    tool_calls: list[str] = Field(default_factory=list)
    error: str | None = None
    is_tool: bool = False
    is_reasoning: bool = False


class StepScore(BaseModel):
    step_index: int
    node: str
    relevance: float = 0.5
    correctness: float = 0.5
    efficiency: float = 0.5
    notes: str = ""


class TrajectoryJudgeResult(BaseModel):
    overall_efficiency: float = 0.5
    overall_coherence: float = 0.5
    overall_adaptivity: float = 0.5
    per_step_scores: list[StepScore] = Field(default_factory=list)
    hallucinations_detected: list[dict[str, Any]] = Field(default_factory=list)
    redundant_calls: list[dict[str, Any]] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    evidence_bank_final: list[str] = Field(default_factory=list)

    @property
    def aggregate_score(self) -> float:
        return (self.overall_efficiency + self.overall_coherence + self.overall_adaptivity) / 3.0

    def to_judge_result(self) -> JudgeResult:
        return JudgeResult(
            score=self.aggregate_score,
            verdict="trajectory_evaluated",
            reasoning="; ".join(self.improvement_suggestions[:3]) or f"steps={len(self.per_step_scores)}",
        )


_FOUND_PATTERN = re.compile(
    r"(?:I (?:found|saw|noticed|read|learned|discovered|observed)|the (?:source|article|page|document|tool) (?:says|states|reports|returns?))[\s,:]*([^.]{8,400})",
    re.IGNORECASE,
)


class TrajectoryJudge(BaseJudge):
    """Per-step CoT judge with progressive evidence bank."""

    def __init__(self, evidence_bank_size: int = 10, **kwargs: Any):
        super().__init__(**kwargs)
        self.evidence_bank_size = evidence_bank_size

    @traceable
    async def judge(self, *, query: str, trajectory: list[TrajectoryStep] | list[dict[str, Any]], **_: Any) -> JudgeResult:
        return (await self.judge_trajectory(query=query, trajectory=trajectory)).to_judge_result()

    async def judge_trajectory(
        self,
        *,
        query: str,
        trajectory: list[TrajectoryStep] | list[dict[str, Any]],
    ) -> TrajectoryJudgeResult:
        steps = [s if isinstance(s, TrajectoryStep) else TrajectoryStep(**s) for s in trajectory]
        if not steps:
            return TrajectoryJudgeResult(improvement_suggestions=["empty trajectory"])

        # 1. Detect redundancy locally — deterministic, cheap.
        redundancy = _detect_redundancy(steps)

        # 2. Walk the trajectory, building the evidence bank as we go.
        evidence_bank: OrderedDict[str, str] = OrderedDict()  # claim_hash -> claim_text
        per_step: list[StepScore] = []
        hallucinations: list[dict[str, Any]] = []

        for i, step in enumerate(steps):
            if step.is_tool or _looks_like_tool(step):
                # Add the tool output to the evidence bank (the tool is the source of truth).
                _add_to_evidence(evidence_bank, _stringify(step.output), self.evidence_bank_size)
                # Tool steps get scored deterministically based on success.
                per_step.append(
                    StepScore(
                        step_index=i, node=step.node,
                        relevance=1.0 if not step.error else 0.0,
                        correctness=1.0 if not step.error else 0.0,
                        efficiency=1.0 if i not in {r["step_index"] for r in redundancy} else 0.4,
                        notes="tool_call_failed" if step.error else "tool_call_ok",
                    )
                )
                continue

            # Reasoning step — score with LLM + extract claims for hallucination check.
            reasoning_text = _stringify(step.output)
            stated_claims = [m.group(1).strip() for m in _FOUND_PATTERN.finditer(reasoning_text)]
            for claim in stated_claims:
                if not _claim_supported_by_bank(claim, evidence_bank):
                    hallucinations.append({
                        "step_index": i,
                        "node": step.node,
                        "claim": claim[:200],
                        "type": "observation_unsupported_by_evidence_bank",
                    })

            # Score this step with the LLM, given query + accumulated evidence.
            try:
                step_score = await self._score_one_step(i, step, query, evidence_bank)
            except Exception:
                step_score = StepScore(step_index=i, node=step.node, notes="judge_error")
            per_step.append(step_score)

            # Also add the reasoning output as derivative evidence (lower priority — would be evicted first).
            _add_to_evidence(evidence_bank, reasoning_text, self.evidence_bank_size)

        # 3. Aggregate holistic scores.
        if per_step:
            efficiency = sum(s.efficiency for s in per_step) / len(per_step)
            coherence = sum(s.correctness for s in per_step) / len(per_step)
            # Adaptivity: did the agent recover from the errors / dead-ends it hit?
            adaptivity = self._adaptivity(steps)
        else:
            efficiency = coherence = adaptivity = 0.5

        suggestions = self._suggestions(per_step, redundancy, hallucinations)

        return TrajectoryJudgeResult(
            overall_efficiency=efficiency,
            overall_coherence=coherence,
            overall_adaptivity=adaptivity,
            per_step_scores=per_step,
            hallucinations_detected=hallucinations,
            redundant_calls=redundancy,
            improvement_suggestions=suggestions,
            evidence_bank_final=list(evidence_bank.values()),
        )

    # -------------------------------------------------------------------- impl

    async def _score_one_step(
        self,
        i: int,
        step: TrajectoryStep,
        query: str,
        evidence_bank: OrderedDict[str, str],
    ) -> StepScore:
        evidence_text = "\n".join(f"  - {v[:200]}" for v in list(evidence_bank.values())[-self.evidence_bank_size:])
        prompt = (
            "You are evaluating one reasoning step of an autonomous agent.\n\n"
            f"User query: {query}\n"
            f"Step index: {i}\n"
            f"Node name: {step.node}\n"
            f"Step output: {_stringify(step.output)[:1000]}\n\n"
            f"Evidence accumulated from prior tool calls:\n{evidence_text or '  (none)'}\n\n"
            "Score this step on three dimensions, each 0.0-1.0:\n"
            "  - relevance: does this step move toward answering the user's query?\n"
            "  - correctness: are the claims/inferences in this step supported by the evidence above (or by general knowledge if no evidence applies)?\n"
            "  - efficiency: is this step necessary, or could it be skipped/merged?\n\n"
            'Reply ONLY with JSON: {"relevance": <float>, "correctness": <float>, "efficiency": <float>, "notes": "<≤20 words>"}'
        )
        text = await self._call(prompt)
        obj = _parse_step_json(text)
        return StepScore(
            step_index=i,
            node=step.node,
            relevance=float(obj.get("relevance", 0.5)),
            correctness=float(obj.get("correctness", 0.5)),
            efficiency=float(obj.get("efficiency", 0.5)),
            notes=str(obj.get("notes", ""))[:200],
        )

    def _adaptivity(self, steps: list[TrajectoryStep]) -> float:
        errors = sum(1 for s in steps if s.error)
        if errors == 0:
            return 1.0
        # Did any error get followed by a non-error step with different input?
        recovered = 0
        for i, s in enumerate(steps):
            if not s.error:
                continue
            for nxt in steps[i + 1 : i + 4]:
                if not nxt.error and _hash(nxt.input) != _hash(s.input):
                    recovered += 1
                    break
        return min(1.0, recovered / errors)

    def _suggestions(
        self,
        per_step: list[StepScore],
        redundancy: list[dict[str, Any]],
        hallucinations: list[dict[str, Any]],
    ) -> list[str]:
        out: list[str] = []
        weak = [s for s in per_step if s.correctness < 0.5]
        if weak:
            out.append(f"Tighten reasoning at step(s) {[s.step_index for s in weak[:3]]} — judge flagged low correctness.")
        if redundancy:
            out.append(f"Remove redundant tool calls at step(s) {[r['step_index'] for r in redundancy[:3]]}.")
        if hallucinations:
            out.append(f"Reduce unsupported observation-style claims (found {len(hallucinations)}).")
        if not out:
            out.append("Trajectory looks healthy.")
        return out


# ---------------------------------------------------------------- helpers


def _detect_redundancy(steps: list[TrajectoryStep]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    out = []
    for i, s in enumerate(steps):
        if not (s.is_tool or _looks_like_tool(s)):
            continue
        key = f"{s.node}|{_hash(s.input)}"
        if key in seen:
            out.append({"step_index": i, "node": s.node, "duplicate_of": seen[key]})
        else:
            seen[key] = i
    return out


def _looks_like_tool(step: TrajectoryStep) -> bool:
    return (step.tool_calls and len(step.tool_calls) > 0) or "tool" in step.node.lower()


def _stringify(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)[:2000]
    except Exception:
        return str(obj)[:2000]


def _hash(obj: Any) -> str:
    try:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(obj).encode()).hexdigest()[:16]


def _add_to_evidence(bank: OrderedDict[str, str], claim: str, max_size: int) -> None:
    if not claim:
        return
    h = _hash(claim)
    if h in bank:
        bank.move_to_end(h)
        return
    bank[h] = claim
    while len(bank) > max_size:
        bank.popitem(last=False)


def _claim_supported_by_bank(claim: str, bank: OrderedDict[str, str]) -> bool:
    """Cheap content-overlap check: does ≥40% of claim's content tokens appear in any bank entry?"""
    toks = [t.lower() for t in re.findall(r"\w+", claim) if len(t) > 3]
    if not toks:
        return True
    for src in bank.values():
        src_lower = src.lower()
        hits = sum(1 for t in toks if t in src_lower)
        if hits / len(toks) >= 0.4:
            return True
    return False


def _parse_step_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    if "{" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}
