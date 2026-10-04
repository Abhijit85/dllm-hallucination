"""
OSCAR rebuttal judge runner for Experiments 1 and 2.

Turnkey wrapper around the unified judge prompt and parser with:
- GPT-4o or Anthropic second-judge backends
- temperature=0 requests
- retry/backoff
- disk-backed caching
- concurrent execution
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from oscar_eval import FLUENCY_PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE, parse_label

PROMPT_VERSION = "factual-v1"


class JudgeClient:
    """provider in {'openai', 'anthropic'}. Returns raw strings for parse_label()."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        max_retries: int = 6,
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        if provider == "openai":
            from openai import OpenAI

            self.client = OpenAI()
            self.model = model or "gpt-4o"
        elif provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic()
            self.model = model or "claude-opus-4-1"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _once(self, prompt: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=4,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        response = self.client.messages.create(
            model=self.model,
            temperature=0,
            max_tokens=4,
            messages=[{"role": "user", "content": prompt}],
        )
        first = response.content[0]
        return getattr(first, "text", "") or ""

    def __call__(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                return self._once(prompt)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError("Unreachable retry loop exit.")


class MockJudge:
    """Offline judge for testing cache/resume behavior."""

    provider = "mock"
    model = "mock"

    def __call__(self, prompt: str) -> str:
        candidate = prompt.split("CANDIDATE:")[-1].strip().lower()
        reference = prompt.split("REFERENCE:")[-1].split("CANDIDATE:")[0].strip().lower()
        if not candidate or candidate in {"i don't know", "idk"}:
            return "INCORRECT"
        if reference and reference in candidate:
            return "CORRECT" if candidate == reference else "PARTIAL"
        return "INCORRECT"


def shuffled_records(records: Iterable[dict], seed: int = 0) -> list[dict]:
    """Method-blind submission order for API calls and cache population."""
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def _key(rec: dict, model: str, kind: str) -> str:
    digest = hashlib.sha256(
        f"{PROMPT_VERSION}|{kind}|{model}|{rec['question']}|{rec.get('reference', '')}|{rec['candidate']}".encode()
    ).hexdigest()[:16]
    return f"{rec['id']}::{digest}"


def _load_cache(path: str | os.PathLike[str] | None) -> dict[str, str]:
    cache: dict[str, str] = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                cache[record["key"]] = record["label"]
    return cache


def judge_records(
    records: Iterable[dict],
    client,
    kind: str = "factual",
    cache_path: str | os.PathLike[str] | None = None,
    workers: int = 8,
    shuffle_seed: int = 0,
) -> dict[str, str]:
    """
    Judge records of the form {id, question, reference, candidate}.

    kind='factual' uses the factuality prompt; kind='fluency' uses the fluency prompt.
    Results are resumable through a JSONL cache file.
    """
    if kind not in {"factual", "fluency"}:
        raise ValueError("kind must be 'factual' or 'fluency'.")

    template = JUDGE_PROMPT_TEMPLATE if kind == "factual" else FLUENCY_PROMPT_TEMPLATE
    cache = _load_cache(cache_path)
    results: dict[str, str] = {}
    pending: list[tuple[dict, str]] = []
    for rec in records:
        cache_key = _key(rec, client.model, kind)
        if cache_key in cache:
            results[rec["id"]] = cache[cache_key]
        else:
            pending.append((rec, cache_key))

    pending = [(rec, key) for rec, key in shuffled_records(pending, seed=shuffle_seed)]

    cache_file = None
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache_file = open(cache_path, "a", encoding="utf-8")

    def work(item: tuple[dict, str]) -> tuple[str, str, str]:
        rec, cache_key = item
        if kind == "factual":
            prompt = template.format(**rec)
        else:
            prompt = template.format(candidate=rec["candidate"])
        return rec["id"], cache_key, parse_label(client(prompt))

    try:
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(work, item) for item in pending]
                for future in as_completed(futures):
                    rec_id, cache_key, label = future.result()
                    results[rec_id] = label
                    if cache_file:
                        cache_file.write(json.dumps({"key": cache_key, "label": label}) + "\n")
                        cache_file.flush()
    finally:
        if cache_file:
            cache_file.close()
    return results


def judge_all_methods(
    base_records: Iterable[dict],
    method_outputs: dict[str, dict[str, float]],
    client,
    cache_dir: str | os.PathLike[str] = "judge_cache",
) -> dict[str, str]:
    """
    Judge the shared base answers used for a detection table.

    method_outputs are passed through only to make the calling contract explicit.
    """
    del method_outputs
    os.makedirs(cache_dir, exist_ok=True)
    return judge_records(
        base_records,
        client,
        kind="factual",
        cache_path=os.path.join(cache_dir, f"base_{client.model}.jsonl"),
    )


if __name__ == "__main__":
    records = [
        {"id": "q1", "question": "Capital of France?", "reference": "Paris", "candidate": "Paris"},
        {"id": "q2", "question": "Capital of France?", "reference": "Paris", "candidate": "Paris, France"},
        {"id": "q3", "question": "Capital of France?", "reference": "Paris", "candidate": "Lyon"},
        {"id": "q4", "question": "Capital of France?", "reference": "Paris", "candidate": "I don't know"},
    ]
    cache_path = "/tmp/_judge_test.jsonl"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    judge = MockJudge()
    print("run 1:", judge_records(records, judge, cache_path=cache_path))
    with open(cache_path, encoding="utf-8") as handle:
        print("cache lines:", len(handle.readlines()))
    print("run 2 (resume, no recompute):", judge_records(records, judge, cache_path=cache_path))
    os.remove(cache_path)
