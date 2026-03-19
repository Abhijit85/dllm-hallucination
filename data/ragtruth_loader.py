"""
RAGTruth data loader.

Dataset: https://arxiv.org/abs/2401.00396
HuggingFace: wanderkid/RAGTruth

Each sample has:
  - source_info   : retrieved passage(s) used as context
  - response      : the LLM-generated response
  - labels        : word-level hallucination spans + intensity
  - task_type     : 'QA' | 'Summary' | 'Data2txt'
  - llm_name      : which model generated the response
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    load_dataset = None


TaskType = Literal["QA", "Summary", "Data2txt"]
HallucinationKind = Literal["evident_conflict", "subtle_conflict", "unverifiable"]


@dataclass
class HallucinationSpan:
    start: int                   # char offset in response
    end: int                     # char offset in response
    text: str
    kind: HallucinationKind
    intensity: float             # 0-1, provided by RAGTruth annotations


@dataclass
class RAGTruthSample:
    sample_id: str
    task_type: TaskType
    llm_name: str
    source_info: str             # retrieved context (the "ground truth" passage)
    response: str                # the model-generated response to evaluate
    spans: list[HallucinationSpan] = field(default_factory=list)

    # ---------- derived helpers ----------

    @property
    def has_hallucination(self) -> bool:
        return len(self.spans) > 0

    def token_labels(self, tokenizer) -> list[int]:
        """
        Return a binary label per token: 1 = hallucinated, 0 = grounded.
        Uses char-to-token alignment from the tokenizer's offset mapping.
        """
        enc = tokenizer(
            self.response,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"]
        labels = [0] * len(offsets)

        for span in self.spans:
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start < span.end and tok_end > span.start:
                    labels[i] = 1

        return labels


def _parse_spans(raw_labels: list[dict]) -> list[HallucinationSpan]:
    spans = []
    for lbl in raw_labels:
        spans.append(
            HallucinationSpan(
                start=lbl["start"],
                end=lbl["end"],
                text=lbl["text"],
                kind=lbl.get("hallucination_type", "evident_conflict"),
                intensity=float(lbl.get("intensity", 1.0)),
            )
        )
    return spans


def load_ragtruth(
    split: str = "test",
    task_type: TaskType | None = None,
    llm_filter: str | None = None,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> list[RAGTruthSample]:
    """
    Load RAGTruth from HuggingFace hub.

    Args:
        split:       'train' | 'validation' | 'test'
        task_type:   filter to one task type, or None for all
        llm_filter:  e.g. 'gpt-4', 'llama-2-13b-chat', or None for all
        max_samples: cap for quick iteration
        cache_dir:   local HF cache directory
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load RAGTruth from HuggingFace. "
            "Install project dependencies with `pip install -e .`."
        )

    ds = load_dataset(
        "wanderkid/RAGTruth",
        split=split,
        cache_dir=cache_dir,
    )

    samples: list[RAGTruthSample] = []
    for row in ds:
        if task_type and row["task_type"] != task_type:
            continue
        if llm_filter and llm_filter.lower() not in row["llm_name"].lower():
            continue

        raw_labels = json.loads(row["labels"]) if isinstance(row["labels"], str) else row["labels"]

        samples.append(
            RAGTruthSample(
                sample_id=str(row["id"]),
                task_type=row["task_type"],
                llm_name=row["llm_name"],
                source_info=row["source_info"],
                response=row["response"],
                spans=_parse_spans(raw_labels or []),
            )
        )

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def load_ragtruth_from_jsonl(path: str | Path) -> list[RAGTruthSample]:
    """Fallback: load from a locally downloaded JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            samples.append(
                RAGTruthSample(
                    sample_id=str(row["id"]),
                    task_type=row["task_type"],
                    llm_name=row.get("llm_name", "unknown"),
                    source_info=row["source_info"],
                    response=row["response"],
                    spans=_parse_spans(row.get("labels") or []),
                )
            )
    return samples


def iter_batches(
    samples: list[RAGTruthSample],
    batch_size: int = 8,
) -> Iterator[list[RAGTruthSample]]:
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


# ---------- quick sanity check ----------
if __name__ == "__main__":
    samples = load_ragtruth(split="test", max_samples=20)
    print(f"Loaded {len(samples)} samples")
    s = samples[0]
    print(f"  task={s.task_type}  llm={s.llm_name}  hallucinated={s.has_hallucination}")
    print(f"  response[:80]: {s.response[:80]!r}")
    if s.spans:
        print(f"  first span: {s.spans[0]}")
