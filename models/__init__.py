"""Model harnesses for DLLM hallucination experiments."""

from models.dream_harness_native import DreamHarnessNative
from models.dream_harness import DreamHarness
from models.llada_harness import DemaskingOrder, LLaDAHarness


def create_harness(model_id: str, **kwargs):
    model_type = str(model_id).lower()
    if "dream" in model_type:
        return DreamHarnessNative(model_id=model_id, **kwargs)
    return LLaDAHarness(model_id=model_id, **kwargs)


__all__ = [
    "LLaDAHarness",
    "DreamHarness",
    "DreamHarnessNative",
    "DemaskingOrder",
    "create_harness",
]
