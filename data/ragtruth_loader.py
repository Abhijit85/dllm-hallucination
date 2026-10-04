"""
RAGTruth data loader.

Paper  : https://arxiv.org/abs/2401.00396
GitHub : https://github.com/ParticleMedia/RAGTruth

Primary layout (what you have locally)
---------------------------------------
RAGTruth/
  dataset/
    response.jsonl      <- LLM responses + word-level hallucination labels
    source_info.jsonl   <- retrieved passages, joined to response via source_id

response.jsonl schema
---------------------
{
  "id":          "1472",
  "source_id":   "11316",
  "model":       "mistral-7B-instruct",
  "temperature": 0.925,
  "response":    "The Palestinian Authority ...",
  "labels": [
    {
      "start":      219,
      "end":        229,
      "text":       "Gaza Strip",
      "label_type": "Evident Baseless Info",
      "meta":       "HIGH INTRO OF NEW INFO\n..."
    }
  ],
  "split":    "train",
  "quality":  "good"
}

source_info.jsonl schema
------------------------
{
  "source_id":   "11316",
  "task_type":   "QA",
  "source":      "MARCO",
  "source_info": {
    "question": "how to prepare beets ...",
    "passages": "passage 1: ..."
  }
}

Fallback layouts also supported
--------------------------------
- Single merged JSONL (already joined)
- HuggingFace hub  (wanderkid/RAGTruth)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TaskType = Literal["QA", "Summary", "Data2txt"]
HallucinationKind = Literal[
    "Evident Conflict",
    "Subtle Conflict",
    "Evident Baseless Info",
    "evident_conflict",
    "subtle_conflict",
    "unverifiable",
]


@dataclass
class HallucinationSpan:
    start: int
    end: int
    text: str
    kind: str
    meta: str = ""
    intensity: float = 1.0


@dataclass
class RAGTruthSample:
    sample_id: str
    source_id: str
    task_type: TaskType
    llm_name: str
    temperature: float
    prompt: str
    source_info: str
    response: str
    split: str
    quality: str
    spans: list[HallucinationSpan] = field(default_factory=list)

    @property
    def has_hallucination(self) -> bool:
        return len(self.spans) > 0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of response characters that are hallucinated."""
        if not self.response:
            return 0.0
        hall_chars = sum(span.end - span.start for span in self.spans)
        return hall_chars / len(self.response)

    def token_labels(self, tokenizer) -> list[int]:
        """
        Binary label per token: 1 = hallucinated, 0 = grounded.
        Uses char-to-token alignment via offset mapping.
        """
        try:
            enc = tokenizer(
                self.response,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        except (NotImplementedError, ValueError, TypeError):
            enc = tokenizer(
                self.response,
                add_special_tokens=False,
            )
        offsets = enc.get("offset_mapping")
        if offsets is None:
            # Some remote-code tokenizers do not expose native offsets. Fall back
            # to prefix-decoding each tokenized prefix, which preserves exact
            # character boundaries for byte/BPE tokenizers used in this repo.
            token_ids = enc.get("input_ids", [])
            offsets = []
            prev_text = ""
            for i in range(len(token_ids)):
                prefix_text = tokenizer.decode(
                    token_ids[: i + 1],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                offsets.append((len(prev_text), len(prefix_text)))
                prev_text = prefix_text
        labels = [0] * len(offsets)
        for span in self.spans:
            for i, (tok_s, tok_e) in enumerate(offsets):
                if tok_s < span.end and tok_e > span.start:
                    labels[i] = 1
        return labels


def _flatten_source_info(raw: dict | str, task_type: str) -> str:
    """
    Convert source_info field (dict or str) to a plain text context string.
    """
    if isinstance(raw, str):
        return raw.strip()

    if task_type == "QA":
        question = raw.get("question", "")
        passages = raw.get("passages", "")
        return f"Question: {question}\n\nPassages:\n{passages}".strip()

    if task_type == "Data2txt":
        return _flatten_yelp(raw)

    return json.dumps(raw, ensure_ascii=False)


def _flatten_yelp(raw: dict) -> str:
    """
    Flatten a Yelp business dict into natural-language prose.
    """
    parts = []

    name = raw.get("name", "")
    if name:
        parts.append(f"Business name: {name}.")

    addr_parts = [raw.get("address", ""), raw.get("city", ""), raw.get("state", "")]
    addr = ", ".join(part for part in addr_parts if part)
    if addr.strip(", "):
        parts.append(f"Location: {addr}.")

    stars = raw.get("stars")
    if stars is not None:
        parts.append(f"Rating: {stars} stars.")

    review_count = raw.get("review_count")
    if review_count is not None:
        parts.append(f"Number of reviews: {review_count}.")

    categories = raw.get("categories", "")
    if categories:
        parts.append(f"Categories: {categories}.")

    is_open = raw.get("is_open")
    if is_open is not None:
        parts.append(f"Currently open: {'yes' if is_open else 'no'}.")

    attributes = raw.get("attributes", {})
    if isinstance(attributes, dict):
        price = attributes.get("RestaurantsPriceRange2")
        if price:
            price_map = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}
            parts.append(f"Price range: {price_map.get(str(price), str(price))}.")
        wifi = attributes.get("WiFi", "")
        if wifi and str(wifi).lower() not in ("none", "no", "u'no'", "'no'", "false"):
            parts.append(f"WiFi available: {wifi}.")

    hours = raw.get("hours", {})
    if isinstance(hours, dict) and hours:
        hours_str = "; ".join(f"{day} {hours_val}" for day, hours_val in list(hours.items())[:4])
        parts.append(f"Hours: {hours_str}.")

    reviews = raw.get("reviews", [])
    if isinstance(reviews, list) and reviews:
        review_text = " ".join(str(review) for review in reviews[:3])
        parts.append(f"Customer reviews: {review_text}")
    elif isinstance(reviews, str) and reviews:
        parts.append(f"Customer reviews: {reviews}")

    return " ".join(parts) if parts else json.dumps(raw, ensure_ascii=False)


def _parse_labels(raw: list[dict] | str | None) -> list[HallucinationSpan]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    spans = []
    for label in raw:
        kind = label.get(
            "label_type",
            label.get("hallucination_type", "unknown"),
        )
        kind = str(kind).strip().lower().replace(" ", "_")
        spans.append(
            HallucinationSpan(
                start=int(label.get("start", 0)),
                end=int(label.get("end", 0)),
                text=label.get("text", ""),
                kind=kind,
                meta=label.get("meta", ""),
                intensity=float(label.get("intensity", 1.0)),
            )
        )
    return spans


def load_from_two_files(
    response_path: str | Path,
    source_info_path: str | Path,
    split: str | None = None,
    task_type: TaskType | None = None,
    llm_filter: str | None = None,
    max_samples: int | None = None,
) -> list[RAGTruthSample]:
    """
    Load RAGTruth from the two canonical files.
    """
    response_path = Path(response_path)
    source_info_path = Path(source_info_path)

    if not response_path.exists():
        raise FileNotFoundError(f"response.jsonl not found: {response_path}")
    if not source_info_path.exists():
        raise FileNotFoundError(f"source_info.jsonl not found: {source_info_path}")

    print(f"  -> Reading source_info: {source_info_path}")
    source_map: dict[str, dict] = {}
    with open(source_info_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id", row.get("id", "")))
            source_map[source_id] = row

    print(f"  -> {len(source_map):,} source entries loaded")

    print(f"  -> Reading responses: {response_path}")
    samples: list[RAGTruthSample] = []
    missing_source = 0

    with open(response_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            row_split = row.get("split", "")
            if split and row_split != split:
                continue

            source_id = str(row.get("source_id", ""))
            source_row = source_map.get(source_id)
            if source_row is None:
                missing_source += 1
                continue

            row_task = source_row.get("task_type", "")
            if task_type and row_task != task_type:
                continue

            model_name = row.get("model", row.get("llm_name", "unknown"))
            if llm_filter and llm_filter.lower() not in model_name.lower():
                continue

            flat_source = _flatten_source_info(
                source_row.get("source_info", ""),
                row_task,
            )

            samples.append(
                RAGTruthSample(
                    sample_id=str(row.get("id", "")),
                    source_id=source_id,
                    task_type=row_task,
                    llm_name=model_name,
                    temperature=float(row.get("temperature", 0.0)),
                    prompt=str(source_row.get("prompt", "")),
                    source_info=flat_source,
                    response=row.get("response", ""),
                    split=row_split,
                    quality=row.get("quality", ""),
                    spans=_parse_labels(row.get("labels")),
                )
            )

            if max_samples and len(samples) >= max_samples:
                break

    if missing_source:
        print(
            f"  ! {missing_source} responses skipped "
            "(source_id not found in source_info.jsonl)"
        )

    return samples


def load_ragtruth(
    split: str | None = "test",
    task_type: TaskType | None = None,
    llm_filter: str | None = None,
    max_samples: int | None = None,
    data_path: str | None = None,
    cache_dir: str | None = None,
) -> list[RAGTruthSample]:
    """
    Smart loader that auto-detects the local layout.
    """
    if data_path is not None:
        path = Path(data_path)

        if path.is_dir():
            response_file = path / "response.jsonl"
            source_info_file = path / "source_info.jsonl"
            if response_file.exists() and source_info_file.exists():
                print(f"  -> Layout: response.jsonl + source_info.jsonl  [{path}]")
                return load_from_two_files(
                    response_file,
                    source_info_file,
                    split=split,
                    task_type=task_type,
                    llm_filter=llm_filter,
                    max_samples=max_samples,
                )

            for filename in (
                f"{split}.jsonl",
                f"ragtruth_{split}.jsonl",
                "data.jsonl",
            ):
                candidate = path / filename
                if candidate.exists():
                    print(f"  -> Layout: merged JSONL  [{candidate}]")
                    return _load_merged_jsonl(
                        candidate,
                        task_type,
                        llm_filter,
                        max_samples,
                    )

            raise FileNotFoundError(
                f"\nCould not find RAGTruth files in: {path}\n"
                f"Expected either:\n"
                f"  {path}/response.jsonl  +  {path}/source_info.jsonl\n"
                f"or\n"
                f"  {path}/{split}.jsonl\n"
            )

        if path.is_file():
            print(f"  -> Layout: single merged JSONL  [{path}]")
            return _load_merged_jsonl(path, task_type, llm_filter, max_samples)

        raise FileNotFoundError(f"data_path does not exist: {path}")

    return _load_from_hub(split, task_type, llm_filter, max_samples, cache_dir)


def _load_merged_jsonl(
    path: Path,
    task_type: str | None,
    llm_filter: str | None,
    max_samples: int | None,
) -> list[RAGTruthSample]:
    """Load a pre-merged JSONL where each row already has source_info."""
    samples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            row_task = row.get("task_type", "")
            if task_type and row_task != task_type:
                continue

            model_name = row.get("model", row.get("llm_name", "unknown"))
            if llm_filter and llm_filter.lower() not in model_name.lower():
                continue

            raw_source = row.get("source_info", "")
            if isinstance(raw_source, (dict, list)):
                flat_source = _flatten_source_info(raw_source, row_task)
            else:
                flat_source = str(raw_source)

            samples.append(
                RAGTruthSample(
                    sample_id=str(row.get("id", row.get("sample_id", ""))),
                    source_id=str(row.get("source_id", "")),
                    task_type=row_task,
                    llm_name=model_name,
                    temperature=float(row.get("temperature", 0.0)),
                    prompt=str(row.get("prompt", "")),
                    source_info=flat_source,
                    response=row.get("response", ""),
                    split=row.get("split", ""),
                    quality=row.get("quality", ""),
                    spans=_parse_labels(row.get("labels")),
                )
            )

            if max_samples and len(samples) >= max_samples:
                break
    return samples


def _load_from_hub(
    split: str | None,
    task_type: str | None,
    llm_filter: str | None,
    max_samples: int | None,
    cache_dir: str | None,
) -> list[RAGTruthSample]:
    from datasets import load_dataset

    print(f"  -> Downloading from HuggingFace hub (wanderkid/RAGTruth, split={split})")
    dataset = load_dataset(
        "wanderkid/RAGTruth",
        split=split or "test",
        cache_dir=cache_dir,
    )
    samples = []
    for row in dataset:
        row = dict(row)
        row_task = row.get("task_type", "")
        if task_type and row_task != task_type:
            continue
        model_name = row.get("llm_name", row.get("model", "unknown"))
        if llm_filter and llm_filter.lower() not in model_name.lower():
            continue
        raw_source = row.get("source_info", "")
        flat_source = (
            _flatten_source_info(raw_source, row_task)
            if isinstance(raw_source, dict)
            else str(raw_source)
        )
        samples.append(
            RAGTruthSample(
                sample_id=str(row.get("id", "")),
                source_id=str(row.get("source_id", "")),
                task_type=row_task,
                llm_name=model_name,
                temperature=float(row.get("temperature", 0.0)),
                prompt=str(row.get("prompt", "")),
                source_info=flat_source,
                response=row.get("response", ""),
                split=row.get("split", split or ""),
                quality=row.get("quality", ""),
                spans=_parse_labels(row.get("labels")),
            )
        )
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def iter_batches(
    samples: list[RAGTruthSample],
    batch_size: int = 8,
) -> Iterator[list[RAGTruthSample]]:
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


if __name__ == "__main__":
    import sys
    import textwrap

    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    split = sys.argv[2] if len(sys.argv) > 2 else "test"

    samples = load_ragtruth(
        data_path=data_path,
        split=split,
        task_type="QA",
        max_samples=5,
    )

    print(f"\nLoaded {len(samples)} samples\n")
    for sample in samples:
        print(
            f"  id={sample.sample_id}  source_id={sample.source_id}  "
            f"model={sample.llm_name}  task={sample.task_type}  "
            f"hallucinated={sample.has_hallucination}  spans={len(sample.spans)}"
        )
        print(f"    source : {textwrap.shorten(sample.source_info, 80)!r}")
        print(f"    response: {textwrap.shorten(sample.response, 80)!r}")
        if sample.spans:
            span = sample.spans[0]
            print(f"    span[0]: {span.text!r}  ({span.kind})")
        print()
