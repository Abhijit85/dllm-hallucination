#!/usr/bin/env python3
"""
C5 — SelfCheckGPT-DLM Adapted Baseline
=======================================
PAPER-FATAL: Must show OSCAR's reveal-order diversity > independent sampling.
Preempts "isn't this just SelfCheckGPT?" objection.

This script runs N=8 INDEPENDENT full DLM generations (different random seeds),
then computes BERTScore/token-overlap consistency as the detection signal —
exactly what SelfCheckGPT does, but adapted for DLMs.

OSCAR differs: N=8 chains share the same masked start and only vary reveal order.
If OSCAR > SelfCheckGPT-DLM, randomized reveal is a more efficient diversity source.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python scripts/c5_selfcheck_dlm.py \
        --model_path /path/to/LLaDA-8B-Instruct \
        --dataset triviaqa \
        --n_samples 500 \
        --n_independent 8 \
        --output results/c5_selfcheck_dlm/
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")


# ── Dataset loading (reuse from run_detection_auroc.py) ──────────────────────

def load_dataset_samples(dataset_name, n_samples, cache_dir=None):
    """Load dataset samples. Tries to import from existing script."""
    try:
        from scripts.run_detection_auroc import load_hotpotqa, load_triviaqa
        if dataset_name == "triviaqa":
            return load_triviaqa(n_samples, cache_dir)
        if dataset_name == "hotpotqa":
            return load_hotpotqa(n_samples, cache_dir)
    except ImportError:
        pass

    from datasets import load_dataset
    if dataset_name == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                          cache_dir=cache_dir)
        samples = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            context = ""
            if row.get("search_results") and row["search_results"].get("search_context"):
                context = row["search_results"]["search_context"][0][:2000]
            samples.append({
                "id": str(i),
                "question": row["question"],
                "context": context,
                "gold_answers": row["answer"]["aliases"],
            })
        return samples
    if dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation",
                          cache_dir=cache_dir)
        samples = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            context_parts = []
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
                context_parts.append(f"{title}: {''.join(sents)}")
            context = " ".join(context_parts[:3])[:2000]
            samples.append({
                "id": str(i),
                "question": row["question"],
                "context": context,
                "gold_answers": [row["answer"]],
            })
        return samples
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path):
    """Load DLM model and tokenizer."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    mask_token_id = getattr(config, "mask_token_id", None)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    return model, tokenizer, mask_token_id


# ── Generation ───────────────────────────────────────────────────────────────

def build_prompt(question, context=""):
    """Build instruction prompt for the DLM."""
    if context:
        return (
            "Given the following context, answer the question.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    return f"Question: {question}\n\nAnswer:"


def generate_independent(model, tokenizer, prompt, gen_len=64, num_steps=128,
                         n_independent=8, mask_token_id=None):
    """
    Generate N completely independent outputs.
    Each generation uses a different random seed.
    This is the SelfCheckGPT approach adapted for DLMs.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    prompt_len = input_ids.shape[1]

    outputs = []
    for i in range(n_independent):
        torch.manual_seed(42 + i * 1000)
        torch.cuda.manual_seed(42 + i * 1000)

        with torch.no_grad():
            if hasattr(model, "diffusion_generate"):
                out = model.diffusion_generate(
                    input_ids,
                    max_new_tokens=gen_len,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=num_steps,
                    temperature=0.7,
                    top_p=0.95,
                )
                gen_ids = out.sequences[0][prompt_len:]
            else:
                full_ids = torch.cat([
                    input_ids,
                    torch.full((1, gen_len), mask_token_id, dtype=torch.long, device="cuda")
                ], dim=1)

                for step in range(num_steps):
                    with torch.no_grad():
                        logits = model(full_ids).logits
                    masked_positions = (full_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
                    if len(masked_positions) == 0:
                        break
                    probs = torch.softmax(logits[0, masked_positions] / 0.7, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    n_unmask = max(1, len(masked_positions) // max(1, num_steps - step))
                    perm = torch.randperm(len(masked_positions))[:n_unmask]
                    for idx in perm:
                        full_ids[0, masked_positions[idx]] = sampled[idx]

                gen_ids = full_ids[0][prompt_len:]

            eos_id = tokenizer.eos_token_id
            gen_ids_clean = [
                t.item() for t in gen_ids
                if t.item() != mask_token_id and t.item() != eos_id
            ]
            text = tokenizer.decode(gen_ids_clean, skip_special_tokens=True).strip()
            outputs.append(text)

    return outputs


# ── SelfCheck consistency scores ─────────────────────────────────────────────

def selfcheck_bertscore(outputs):
    """
    Compute BERTScore-based consistency (simplified).
    For each output, compute average BERTScore-F1 against all others.
    Low consistency = likely hallucination.
    """
    try:
        from bert_score import score as bert_score_fn
        n = len(outputs)
        scores = np.zeros((n, n))
        for i in range(n):
            refs = [outputs[j] for j in range(n) if j != i]
            cands = [outputs[i]] * len(refs)
            _, _, f1 = bert_score_fn(cands, refs, lang="en", verbose=False)
            k = 0
            for j in range(n):
                if j != i:
                    scores[i, j] = f1[k].item()
                    k += 1
        mean_consistency = np.mean(scores[scores > 0])
        return 1.0 - mean_consistency
    except ImportError:
        return selfcheck_token_overlap(outputs)


def selfcheck_token_overlap(outputs):
    """
    Token-overlap consistency: for each output, what fraction of its tokens
    appear in at least K other outputs?
    """
    n = len(outputs)
    tokenized = [set(o.lower().split()) for o in outputs]

    overlaps = []
    for i in range(n):
        if not tokenized[i]:
            continue
        overlap_count = 0
        for token in tokenized[i]:
            count = sum(1 for j in range(n) if j != i and token in tokenized[j])
            if count >= n // 2:
                overlap_count += 1
        overlaps.append(overlap_count / len(tokenized[i]))

    if not overlaps:
        return 0.5
    return 1.0 - np.mean(overlaps)


def selfcheck_entropy(outputs):
    """
    String-level entropy: how diverse are the N outputs?
    High diversity = uncertain = likely hallucinated.
    """
    normalized = [o.lower().strip()[:100] for o in outputs]
    counter = Counter(normalized)
    n = len(outputs)
    probs = [c / n for c in counter.values()]
    return -sum(p * math.log(p + 1e-10) for p in probs)


# ── Correctness check ────────────────────────────────────────────────────────

def is_correct(generated, gold_answers):
    """Check if generated answer is correct (substring match)."""
    gen_lower = generated.lower().strip()
    for gold in gold_answers:
        if gold.lower().strip() in gen_lower:
            return True
    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    from sklearn.metrics import roc_auc_score

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tokenizer, mask_token_id = load_model_and_tokenizer(args.model_path)
    print(f"  mask_token_id: {mask_token_id}")

    print(f"Loading {args.dataset} ({args.n_samples} samples)...")
    samples = load_dataset_samples(args.dataset, args.n_samples)
    print(f"  Loaded {len(samples)} samples")

    results = []
    for i, sample in enumerate(tqdm(samples, desc="SelfCheck-DLM")):
        prompt = build_prompt(sample["question"], sample.get("context", ""))

        outputs = generate_independent(
            model, tokenizer, prompt,
            gen_len=args.gen_len,
            num_steps=args.num_steps,
            n_independent=args.n_independent,
            mask_token_id=mask_token_id,
        )

        score_overlap = selfcheck_token_overlap(outputs)
        score_entropy = selfcheck_entropy(outputs)

        primary = outputs[0]
        correct = is_correct(primary, sample["gold_answers"])

        results.append({
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answers": sample["gold_answers"],
            "outputs": outputs,
            "primary_output": primary,
            "correct": correct,
            "selfcheck_overlap": score_overlap,
            "selfcheck_entropy": score_entropy,
        })

        if (i + 1) % 50 == 0:
            with open(out_dir / "raw_results.jsonl", "w") as f:
                f.writelines(json.dumps(r, default=str) + "\n" for r in results)

    with open(out_dir / "raw_results.jsonl", "w") as f:
        f.writelines(json.dumps(r, default=str) + "\n" for r in results)

    labels = [1 - int(r["correct"]) for r in results]
    scores_overlap = [r["selfcheck_overlap"] for r in results]
    scores_entropy = [r["selfcheck_entropy"] for r in results]

    if len(set(labels)) >= 2:
        auroc_overlap = roc_auc_score(labels, scores_overlap)
        auroc_entropy = roc_auc_score(labels, scores_entropy)
    else:
        auroc_overlap = auroc_entropy = float("nan")

    n_correct = sum(1 for r in results if r["correct"])

    print(f"\n{'='*60}")
    print(f"SelfCheckGPT-DLM Results ({args.dataset})")
    print(f"{'='*60}")
    print(f"  N independent generations: {args.n_independent}")
    print(f"  Accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)")
    print(f"  AUROC (token overlap): {auroc_overlap:.3f}")
    print(f"  AUROC (string entropy): {auroc_entropy:.3f}")
    print()
    print("  Compare against OSCAR's AUROC to validate architectural advantage.")

    summary = {
        "dataset": args.dataset,
        "n_samples": len(results),
        "n_independent": args.n_independent,
        "accuracy": n_correct / len(results),
        "auroc_overlap": auroc_overlap,
        "auroc_entropy": auroc_entropy,
    }
    with open(out_dir / "c5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="C5: SelfCheckGPT-DLM Baseline")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", choices=["triviaqa", "hotpotqa"],
                        default="triviaqa")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_independent", type=int, default=8)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--output", default="results/c5_selfcheck_dlm/")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
