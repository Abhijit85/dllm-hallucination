#!/usr/bin/env python3
"""
Re-runs H1 analysis correctly on a fresh set of samples.

WHY A FRESH RUN IS NEEDED
--------------------------
The raw_results.jsonl from the full experiment stores per-sample aggregate
metrics but not per-token entropy arrays. To compute the correct H1, we need
per-token entropy paired with LLaDA's own generated token IDs.

This script runs a lightweight pass (no refinement, just N paths + entropy)
on a subset of samples and computes the corrected H1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from data.ragtruth_loader import load_ragtruth
from eval.h1_analysis import aggregate_h1_results, compute_h1_correct
from models import create_harness
from strategies.parallel_remask import compute_disagreement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_paths", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--output", default="results/h1_reanalysis.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.n_samples} samples (all task types)...")
    samples = load_ragtruth(
        data_path=args.data_path,
        split="test",
        max_samples=args.n_samples,
    )
    print(
        f"  {len(samples)} loaded  |  "
        f"hallucinated: {sum(1 for sample in samples if sample.has_hallucination)}"
    )

    print(f"\nLoading model: {args.model_path}")
    harness = create_harness(model_id=args.model_path)

    h1_results = []
    per_sample = []

    for sample in tqdm(samples, desc="H1 reanalysis"):
        try:
            result = harness.run_parallel_paths(
                prompt=sample.response[:50],
                source_info=sample.source_info,
                n_paths=args.n_paths,
                gen_len=args.gen_len,
                num_steps=args.num_steps,
                learned_paths=1,
                base_seed=args.seed,
            )

            report = compute_disagreement(result, top_k_percent=0.20)
            prompt_len = len(result.prompt_tokens)
            entropy = report.token_entropy[prompt_len:]
            generated_ids = result.majority_vote()[prompt_len:]

            h1 = compute_h1_correct(
                entropy=entropy,
                generated_ids=generated_ids,
                source_info=sample.source_info,
                tokenizer=harness.tokenizer,
            )

            if h1 is not None:
                h1_results.append(h1)
                per_sample.append(
                    {
                        "sample_id": sample.sample_id,
                        "has_hallucination": sample.has_hallucination,
                        "task_type": sample.task_type,
                        "rho_bigram": h1.spearman_rho_bigram,
                        "rho_unigram": h1.spearman_rho_unigram,
                        "entropy_gap": h1.entropy_gap,
                        "mean_ent_unf": h1.mean_entropy_unfaithful,
                        "mean_ent_fai": h1.mean_entropy_faithful,
                        "n_unfaithful_bi": h1.n_unfaithful_bigram,
                        "n_tokens": h1.n_tokens,
                    }
                )
        except Exception as exc:
            print(f"  [skip] {sample.sample_id}: {exc}")

    agg = aggregate_h1_results(h1_results)

    print("\n" + "=" * 54)
    print("  Corrected H1 Results")
    print("=" * 54)
    print(f"  Samples with valid H1 : {agg['n_samples']}")
    print(f"  Mean Spearman rho (bigram)  : {agg['mean_rho_bigram']:+.4f}")
    print(f"  Mean Spearman rho (unigram) : {agg['mean_rho_unigram']:+.4f}")
    print(f"  Mean entropy gap            : {agg['mean_entropy_gap']:+.4f}")
    print("    (ungrounded - grounded tokens; positive = H1 supported)")
    print(f"  % samples with rho > 0      : {agg['pct_positive_rho'] * 100:.1f}%")
    print(
        f"  Mean unfaithful token frac  : "
        f"{agg['mean_unfaithful_frac'] * 100:.1f}%"
    )

    if agg["mean_rho_bigram"] > 0:
        print("\n  H1 SUPPORTED: entropy positively correlates with")
        print("  source-ungrounded tokens in LLaDA's own output.")
    elif agg["mean_entropy_gap"] > 0:
        print("\n  H1 PARTIAL: entropy gap is positive even if overall rho is weak.")
        print("  High-entropy tokens have higher mean entropy at ungrounded positions.")
    else:
        print("\n  H1 WEAK: entropy does not predict source grounding in LLaDA's output.")

    out = {
        "aggregate": agg,
        "per_sample": per_sample,
        "config": vars(args),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
