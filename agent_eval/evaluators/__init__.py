"""Evaluator suite — output quality, trajectory, hallucination, tool, system, safety, memory."""

from agent_eval.evaluators.base import (
    BaseEvaluator,
    CompositeEvaluator,
    CompositeResult,
    EvaluatorResult,
)
from agent_eval.evaluators.hallucination import (
    CitationHallucinationEvaluator,
    HallucinationComposite,
    ObservationHallucinationEvaluator,
    PlanningHallucinationEvaluator,
    ReasoningHallucinationEvaluator,
)
from agent_eval.evaluators.memory import (
    CrossSessionContinuityEvaluator,
    FactInjection,
    MemoryCostEvaluator,
    MemoryEvaluationReport,
    MemoryRetrievalPrecisionEvaluator,
    MemoryRetrievalRecallEvaluator,
    MemoryStalenessEvaluator,
    MemoryWriteQualityEvaluator,
    default_fact_harness,
)
from agent_eval.evaluators.output_quality import (
    AnswerFaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    CompletenessEvaluator,
    FormatComplianceEvaluator,
    KeywordCoverageEvaluator,
    OutputQualityComposite,
    TaskSuccessEvaluator,
)
from agent_eval.evaluators.safety import (
    ConsistencyEvaluator,
    HarmfulContentEvaluator,
    InstructionFollowingEvaluator,
    PIILeakageEvaluator,
)
from agent_eval.evaluators.system_performance import (
    CostEfficiencyEvaluator,
    ErrorRateEvaluator,
    LatencyEvaluator,
    StreamingLatencyEvaluator,
    TokenEfficiencyEvaluator,
)
from agent_eval.evaluators.tool_performance import (
    ArgumentCorrectnessEvaluator,
    CostPerToolEvaluator,
    MCPServerHealthEvaluator,
    ToolPerformanceEvaluator,
    ToolResultQualityEvaluator,
)
from agent_eval.evaluators.trajectory import (
    CycleDetectionEvaluator,
    ErrorRecoveryEvaluator,
    IntentResolutionEvaluator,
    NodeF1Evaluator,
    RedundancyEvaluator,
    StepSuccessRateEvaluator,
    StructuralSimilarityEvaluator,
    ToolF1Evaluator,
    ToolSelectionEvaluator,
)

__all__ = [
    "BaseEvaluator",
    "EvaluatorResult",
    "CompositeEvaluator",
    "CompositeResult",
    # output
    "TaskSuccessEvaluator",
    "AnswerFaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "CompletenessEvaluator",
    "FormatComplianceEvaluator",
    "KeywordCoverageEvaluator",
    "OutputQualityComposite",
    # trajectory
    "ToolSelectionEvaluator",
    "ToolF1Evaluator",
    "NodeF1Evaluator",
    "StructuralSimilarityEvaluator",
    "IntentResolutionEvaluator",
    "StepSuccessRateEvaluator",
    "RedundancyEvaluator",
    "ErrorRecoveryEvaluator",
    "CycleDetectionEvaluator",
    # hallucination
    "PlanningHallucinationEvaluator",
    "ObservationHallucinationEvaluator",
    "CitationHallucinationEvaluator",
    "ReasoningHallucinationEvaluator",
    "HallucinationComposite",
    # tool
    "ToolPerformanceEvaluator",
    "ToolResultQualityEvaluator",
    "ArgumentCorrectnessEvaluator",
    "MCPServerHealthEvaluator",
    "CostPerToolEvaluator",
    # system
    "LatencyEvaluator",
    "StreamingLatencyEvaluator",
    "TokenEfficiencyEvaluator",
    "ErrorRateEvaluator",
    "CostEfficiencyEvaluator",
    # safety
    "HarmfulContentEvaluator",
    "PIILeakageEvaluator",
    "InstructionFollowingEvaluator",
    "ConsistencyEvaluator",
    # memory
    "MemoryRetrievalRecallEvaluator",
    "MemoryRetrievalPrecisionEvaluator",
    "MemoryWriteQualityEvaluator",
    "MemoryStalenessEvaluator",
    "CrossSessionContinuityEvaluator",
    "MemoryCostEvaluator",
    "MemoryEvaluationReport",
    "FactInjection",
    "default_fact_harness",
]
