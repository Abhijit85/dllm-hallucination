#!/usr/bin/env python3
"""
C1 — Fair AUROC: Re-evaluate ALL baselines under LLM-as-Judge
=============================================================
PAPER-FATAL: Table 1 compares OSCAR(Judge)=86.5 vs DynHD(EM)=84.2,
but under EM, OSCAR=76.4 — BELOW DynHD. Must add Judge column for all methods.

This script:
1. Loads existing raw_results.jsonl from each method's results directory
2. Sends (question, gold_answer, generated_answer) to GPT-4o for Judge labels
3. Recomputes AUROC under Judge labels for every method
4. Outputs a revised Table 1 with both EM and Judge columns for all methods

Requirements: pip install openai
Cost estimate: ~500 samples × 9 methods × 2 models ≈ ~9000 API calls ≈ $15-25

Usage:
    # Set your API key
    export OPENAI_API_KEY="sk-..."

    # Run on all existing results
    python scripts/c1_fair_auroc_judge.py \
        --results_base results/ \
        --output results/c1_fair_auroc/ \
        --model gpt-4o \
        --max_concurrent 20
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


# ── Judge prompt (matches Appendix A of paper) ───────────────────────────────

JUDGE_PROMPT = """You are evaluating whether an AI-generated answer is factually correct.
The answer does not need to match the reference exactly---it should be
semantically equivalent and contain the key facts. Rate as CORRECT if
the core factual content matches, INCORRECT if it contains factual errors,
PARTIAL if partially correct. Respond with only the label."""

JUDGE_USER_TEMPLATE = """Question: {question}
Reference Answer: {gold}
AI Answer: {generated}

Label:"""


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    sample_id: str
    question: str
    gold: str
    generated: str
    em_label: int
    judge_label: int
    judge_raw: str
    entropy_score: float


# ── Judge backends ───────────────────────────────────────────────────────────

DEFAULT_LLAMA_70B_PATH = os.environ.get(
    "LLAMA_70B_PATH",
    "meta-llama/Llama-3.3-70B-Instruct",  # HF hub ID; override with local path via env var
)


def _parse_judge_label(raw: str) -> int:
    raw_upper = raw.strip().upper()
    if "CORRECT" in raw_upper and "INCORRECT" not in raw_upper:
        return 1
    if "INCORRECT" in raw_upper:
        return 0
    if "PARTIAL" in raw_upper:
        return 0
    return 0


def resolve_local_model_path(model_name: str) -> str:
    if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        return DEFAULT_LLAMA_70B_PATH
    return model_name


class LocalHFJudge:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = resolve_local_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()
        self.input_device = next(self.model.parameters()).device

    def judge_one(self, question: str, gold: str, generated: str) -> tuple[int, str]:
        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold=gold,
            generated=generated,
        )
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {
            k: v.to(self.input_device)
            for k, v in inputs.items()
        }
        output = self.model.generate(
            **inputs,
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return _parse_judge_label(raw), raw


# ── Async GPT-4o Judge ───────────────────────────────────────────────────────

async def judge_one(client, question, gold, generated, model="gpt-4o",
                    semaphore=None, retries=3):
    """Call GPT-4o to judge correctness. Returns (label_int, raw_text)."""
    if semaphore:
        await semaphore.acquire()
    try:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                            question=question, gold=gold, generated=generated
                        )},
                    ],
                    max_tokens=10,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                return _parse_judge_label(raw), raw
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  Judge failed after {retries} retries: {e}")
                    return 0, f"ERROR: {e}"
    finally:
        if semaphore:
            semaphore.release()


async def judge_batch(client, records, model="gpt-4o", max_concurrent=20):
    """Judge a batch of records concurrently."""
    sem = asyncio.Semaphore(max_concurrent)
    tasks = []
    for rec in records:
        tasks.append(judge_one(
            client,
            rec["question"],
            rec["gold"],
            rec["generated"],
            model=model,
            semaphore=sem,
        ))
    return await asyncio.gather(*tasks)


def judge_batch_local(local_judge, records):
    """Judge a batch of records sequentially with a local HF model."""
    results = []
    for rec in records:
        results.append(local_judge.judge_one(
            rec["question"],
            rec["gold"],
            rec["generated"],
        ))
    return results


# ── Load existing results ────────────────────────────────────────────────────

def find_result_dirs(base_path):
    """Auto-discover results directories with raw_results.jsonl."""
    base = Path(base_path)
    dirs = {}
    for jsonl in sorted(base.rglob("raw_results.jsonl")):
        dirs[jsonl.parent.name] = jsonl.parent
    return dirs


def load_records(results_dir):
    """Load raw_results.jsonl and extract fields needed for judging."""
    jsonl_path = Path(results_dir) / "raw_results.jsonl"
    if not jsonl_path.exists():
        return []

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = rec.get("sample_id", rec.get("id", ""))
            question = rec.get("question", rec.get("query", ""))

            gold_raw = rec.get("gold_answers", rec.get("gold", rec.get("reference", "")))
            if isinstance(gold_raw, list):
                gold = " | ".join(str(g) for g in gold_raw[:3])
            else:
                gold = str(gold_raw)

            generated = rec.get("generated_before", rec.get("generated",
                        rec.get("prediction", rec.get("output", ""))))

            em = rec.get("em_correct", rec.get("is_correct", rec.get("correct", None)))
            if em is None:
                candidates = gold_raw if isinstance(gold_raw, list) else [gold_raw]
                generated_lower = str(generated).lower()
                em = int(any(str(g).lower().strip() in generated_lower for g in candidates if g))

            entropy = rec.get("mean_cross_entropy", rec.get("cross_chain_entropy",
                      rec.get("mean_entropy", rec.get("entropy_score", 0.0))))

            records.append({
                "sample_id": str(sample_id),
                "question": str(question),
                "gold": gold,
                "generated": str(generated),
                "em_label": int(em),
                "entropy_score": float(entropy),
            })

    return records


# ── AUROC computation ────────────────────────────────────────────────────────

def compute_auroc(labels, scores):
    """Compute AUROC. labels: 1=correct, 0=incorrect. scores: higher=more uncertain."""
    from sklearn.metrics import roc_auc_score

    labels = np.array(labels)
    scores = np.array(scores)
    if len(set(labels)) < 2:
        return float("nan")

    hall_labels = 1 - labels
    return roc_auc_score(hall_labels, scores)


# ── Main pipeline ────────────────────────────────────────────────────────────

async def run_c1(args):
    client = None
    local_judge = None
    if args.judge_backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --judge_backend openai")
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is required for --judge_backend openai")
        client = AsyncOpenAI(api_key=api_key)
    else:
        print(f"Loading local judge model: {args.model}")
        local_judge = LocalHFJudge(args.model)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = find_result_dirs(args.results_base)
    print(f"Found {len(result_dirs)} result directories:")
    for name, path in result_dirs.items():
        print(f"  {name}: {path}")
    print()

    if args.results_dirs:
        result_dirs = {Path(d).name: Path(d) for d in args.results_dirs if Path(d).exists()}

    all_results = {}

    for dir_name, dir_path in result_dirs.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dir_name}")
        print(f"{'='*60}")

        records = load_records(dir_path)
        if not records:
            print(f"  No records found in {dir_path}")
            continue

        print(f"  Loaded {len(records)} records")

        cache_path = out_dir / f"judge_cache_{dir_name}.jsonl"
        cached = {}
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    c = json.loads(line)
                    cached[c["sample_id"]] = c
            print(f"  Found {len(cached)} cached judge results")

        to_judge = [r for r in records if r["sample_id"] not in cached]
        print(f"  Need to judge {len(to_judge)} new records")

        if to_judge:
            if args.judge_backend == "openai":
                judge_results = await judge_batch(
                    client, to_judge, model=args.model,
                    max_concurrent=args.max_concurrent
                )
            else:
                judge_results = judge_batch_local(local_judge, to_judge)

            with open(cache_path, "a") as f:
                for rec, (label, raw) in zip(to_judge, judge_results):
                    entry = {
                        "sample_id": rec["sample_id"],
                        "judge_label": label,
                        "judge_raw": raw,
                    }
                    cached[rec["sample_id"]] = entry
                    f.write(json.dumps(entry) + "\n")

        em_labels = []
        judge_labels = []
        scores = []
        for rec in records:
            em_labels.append(rec["em_label"])
            jc = cached.get(rec["sample_id"], {})
            judge_labels.append(jc.get("judge_label", rec["em_label"]))
            scores.append(rec["entropy_score"])

        auroc_em = compute_auroc(em_labels, scores)
        auroc_judge = compute_auroc(judge_labels, scores)

        n_correct_em = sum(em_labels)
        n_correct_judge = sum(judge_labels)

        result = {
            "dir_name": dir_name,
            "n_samples": len(records),
            "n_correct_em": n_correct_em,
            "n_correct_judge": n_correct_judge,
            "auroc_em": auroc_em,
            "auroc_judge": auroc_judge,
            "delta": auroc_judge - auroc_em if not (
                np.isnan(auroc_em) or np.isnan(auroc_judge)) else None,
        }
        all_results[dir_name] = result

        print(f"  AUROC(EM)    = {auroc_em:.3f}")
        print(f"  AUROC(Judge) = {auroc_judge:.3f}")
        if result["delta"] is not None:
            print(f"  Delta        = {result['delta']:.3f}")
        print(f"  Correct: EM={n_correct_em}, Judge={n_correct_judge}")

    summary_path = out_dir / "c1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    latex_path = out_dir / "table1_revised.tex"
    with open(latex_path, "w") as f:
        f.write("% Table 1 (revised) — AUROC with both EM and Judge for ALL methods\n")
        f.write("% Generated by c1_fair_auroc_judge.py\n\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{AUROC(\\%) — fair comparison under both EM and Judge evaluation.}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}llcc@{}}\n\\toprule\n")
        f.write("Method & Train? & AUROC (EM) & AUROC (Judge) \\\\\n\\midrule\n")
        for name, res in sorted(all_results.items()):
            auroc_em = f"{res['auroc_em']*100:.1f}" if not np.isnan(res["auroc_em"]) else "—"
            auroc_j = f"{res['auroc_judge']*100:.1f}" if not np.isnan(res["auroc_judge"]) else "—"
            f.write(f"{name} & — & {auroc_em} & {auroc_j} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}/")
    print("  c1_summary.json      — raw numbers")
    print("  table1_revised.tex   — LaTeX table")
    print("  judge_cache_*.jsonl  — cached judge results (reusable)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="C1: Fair AUROC under LLM-as-Judge")
    parser.add_argument("--results_base", default="results/",
                        help="Base directory to search for raw_results.jsonl")
    parser.add_argument("--results_dirs", nargs="*", default=None,
                        help="Specific result directories to process")
    parser.add_argument("--output", default="results/c1_fair_auroc/")
    parser.add_argument("--model", default="gpt-4o",
                        help="Judge model (gpt-4o recommended)")
    parser.add_argument(
        "--judge_backend",
        choices=["openai", "hf"],
        default="openai",
        help="Use OpenAI API or a local Hugging Face causal LM as the judge",
    )
    parser.add_argument("--max_concurrent", type=int, default=20,
                        help="Max concurrent API calls")
    args = parser.parse_args()
    asyncio.run(run_c1(args))


if __name__ == "__main__":
    main()
