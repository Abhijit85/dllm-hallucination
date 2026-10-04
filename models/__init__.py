"""Model harnesses for OSCAR hallucination experiments."""

from models.llada_harness import DemaskingOrder, LLaDAHarness


def create_harness(model_id: str, **kwargs):
    model_type = str(model_id).lower()
    if "dream" in model_type:
        from models.dream_harness_native import DreamHarnessNative

        return DreamHarnessNative(model_id=model_id, **kwargs)
    return LLaDAHarness(model_id=model_id, **kwargs)


def __getattr__(name: str):
    if name == "DreamHarnessNative":
        from models.dream_harness_native import DreamHarnessNative

        return DreamHarnessNative
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "DemaskingOrder",
    "DreamHarnessNative",
    "LLaDAHarness",
    "create_harness",
]
