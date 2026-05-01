"""External evaluator backends — DeepEval, Ragas, native (default)."""

from agent_eval.backends.base import EvaluatorBackend
from agent_eval.backends.deepeval_backend import DeepEvalBackend
from agent_eval.backends.native_backend import NativeBackend
from agent_eval.backends.ragas_backend import RagasBackend


def get_backend(name: str) -> EvaluatorBackend:
    name = (name or "native").lower()
    if name == "native":
        return NativeBackend()
    if name == "deepeval":
        return DeepEvalBackend()
    if name == "ragas":
        return RagasBackend()
    raise ValueError(f"Unknown backend '{name}'. Valid: native, deepeval, ragas.")


BACKENDS = {
    "native": NativeBackend,
    "deepeval": DeepEvalBackend,
    "ragas": RagasBackend,
}

__all__ = ["EvaluatorBackend", "NativeBackend", "DeepEvalBackend", "RagasBackend", "get_backend", "BACKENDS"]
