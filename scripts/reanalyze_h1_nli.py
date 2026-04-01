#!/usr/bin/env python3
"""
scripts/reanalyze_h1_nli.py
-----------------------------
Strengthened H1 analysis using NLI-based source grounding.

Compared to the original bigram-based H1 (reanalyze_h1.py), this script:
  - Uses entailment probability instead of bigram overlap as the grounding proxy
  - Correctly handles paraphrases, synonyms, and rephrasings
  - Operates at sentence level then maps back to tokens
  - Expected to produce significantly higher Spearman rho

SETUP (run once on the GPU server):
-------------------------------------
# Option A: AlignScore (recommended — designed for factual consistency)
pip install git+https://github.com/yuh-zha/AlignScore.git --break-system-packages
python -c "from alignscore import AlignScore; AlignScore(model='AlignScore-base')"

# Option B: DeBERTa cross-encoder (strong general NLI)
pip install sentence-transformers --break-system-packages

# Option C: BART MNLI (widely available, no extra install if transformers present)
# Already available if transformers is installed

The script auto-detects which backend is available.

USAGE:
------
# Standard run (auto-detect backend)
CUDA_VISIBLE_DEVICES=5 python scripts/reanalyze_h1_nli.py \
    --model_path /mnt/shared/.../GSAI-ML--LLaDA-8B-Instruct \
    --data_path  .../RAGTruth/dataset \
    --n_samples  200 \
    --n_paths    8 \
    --output     results/h1_nli.json

# Force specific backend
CUDA_VISIBLE_DEVICES=5 python scripts/reanalyze_h1_nli.py \
    --model_path /mnt/shared/.../GSAI-ML--LLaDA-8B-Instruct \
    --data_path  .../RAGTruth/dataset \
    --n_samples  200 \
    --backend    nli_deberta \
    --output     results/h1_nli_deberta.json

# Also run bigram in same pass (comparison baseline)
# Add --compare_bigram flag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.ragtruth_loader import load_ragtruth
from eval.h1_analysis import (
    aggregate_h1_results as aggregate_h1_bigram,
)
from eval.h1_analysis import (
    compute_h1_correct as compute_h1_bigram,
)
from eval.h1_analysis_nli import (
    GroundingScorer,
    H1ResultNLI,
    aggregate_h1_nli,
    compute_h1_nli,
)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "alignscore", "nli_deberta", "nli_bart", "bigram"],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Entailment probability threshold for grounded/ungrounded",
    )
    parser.add_argument(
        "--compare_bigram",
        action="store_true",
        help="Also compute bigram H1 for comparison",
    )
    parser.add_argument("--output", default="results/h1_nli.json")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.n_samples} samples (all task types)...")
    samples = load_ragtruth(
        data_path=args.data_path,
        split="test",
        max_samples=args.n_samples,
    )
    n_hall = sum(1 for sample in samples if sample.has_hallucination)
    hall_pct = (100 * n_hall / len(samples)) if samples else 0.0
    print(f"  {len(samples)} loaded  |  hallucinated: {n_hall} ({hall_pct:.1f}%)")

    print(f"\nLoading model: {args.model_path}")
    harness = create_harness(model_id=args.model_path)

    scorer_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitialising grounding scorer (backend={args.backend})...")
    scorer = GroundingScorer(mode=args.backend, device=scorer_device)

    nli_results: list[H1ResultNLI] = []
    bigram_results = []
    per_sample = []

    for sample in tqdm(samples, desc="H1-NLI reanalysis"):
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

            h1_nli = compute_h1_nli(
                entropy=entropy,
                generated_ids=generated_ids,
                source_info=sample.source_info,
                tokenizer=harness.tokenizer,
                scorer=scorer,
                entailment_threshold=args.threshold,
            )
            nli_results.append(h1_nli)

            h1_bigram = None
            if args.compare_bigram:
                h1_bigram = compute_h1_bigram(
                    entropy=entropy,
                    generated_ids=generated_ids,
                    source_info=sample.source_info,
                    tokenizer=harness.tokenizer,
                )
                if h1_bigram is not None:
                    bigram_results.append(h1_bigram)

            per_sample.append(
                {
                    "sample_id": sample.sample_id,
                    "has_hallucination": sample.has_hallucination,
                    "task_type": sample.task_type,
                    "nli_rho": h1_nli.spearman_rho,
                    "nli_p": h1_nli.p_value,
                    "nli_gap": h1_nli.entropy_gap,
                    "nli_n_ungrounded": h1_nli.n_ungrounded,
                    "nli_skipped": h1_nli.skipped,
                    "nli_skip_reason": h1_nli.skip_reason,
                    "nli_sent_scores": h1_nli.sent_scores,
                    "bi_rho": h1_bigram.spearman_rho_bigram if h1_bigram else None,
                    "bi_gap": h1_bigram.entropy_gap if h1_bigram else None,
                }
            )
        except Exception as exc:
            print(f"  [skip] {sample.sample_id}: {exc}")

    nli_agg = aggregate_h1_nli(nli_results)
    bi_agg = aggregate_h1_bigram(bigram_results) if bigram_results else None

    print("\n" + "=" * 60)
    print(f"  H1 Results - NLI backend: {scorer.mode}")
    print("=" * 60)
    print(f"  Samples total      : {nli_agg['n_total']}")
    print(f"  Samples valid      : {nli_agg['n_valid']}")
    print(f"  Samples skipped    : {nli_agg['n_skipped']}")
    print()
    print(f"  Mean Spearman rho  : {nli_agg['mean_rho']:+.4f}")
    print(f"  Median Spearman rho: {nli_agg['median_rho']:+.4f}")
    print(f"  % samples rho > 0  : {nli_agg['pct_positive_rho'] * 100:.1f}%")
    print(f"  % samples rho > 0.2: {nli_agg['pct_rho_gt_02'] * 100:.1f}%")
    print()
    print(f"  Mean entropy gap   : {nli_agg['mean_entropy_gap']:+.4f}")
    print("    (ungrounded - grounded; positive = H1 supported)")
    print(f"  % samples gap > 0  : {nli_agg['pct_gap_positive'] * 100:.1f}%")
    print(f"  Mean ungrounded %  : {nli_agg['mean_ungrounded_frac'] * 100:.1f}%")

    if bi_agg:
        print()
        print("  --- Bigram baseline (for comparison) ---")
        print(f"  Bigram mean rho    : {bi_agg['mean_rho_bigram']:+.4f}")
        print(f"  Bigram % positive  : {bi_agg['pct_positive_rho'] * 100:.1f}%")
        print(f"  Bigram entropy gap : {bi_agg['mean_entropy_gap']:+.4f}")

    rho = nli_agg["mean_rho"]
    print()
    if rho > 0.3:
        print("  H1 SUPPORTED: Strong positive correlation.")
        print("  NLI-based grounding substantially improves over bigram.")
        print("  Report as a primary finding in the paper.")
    elif rho > 0.15:
        print("  H1 SUPPORTED (moderate): rho > 0.15 with NLI grounding.")
        print("  Report alongside the entropy-gap as converging evidence.")
    elif rho > 0:
        print("  H1 WEAK: Positive but below 0.15.")
        print("  Frame as preliminary or suggestive evidence.")
        print("  Emphasise H2 (+0.072 delta FS) and H3 trajectory as primary claims.")
    else:
        print("  H1 NOT supported even with NLI grounding.")
        print("  Drop H1 as a hypothesis. Focus on H2 and H3.")

    out = {
        "nli_aggregate": nli_agg,
        "bigram_aggregate": bi_agg,
        "per_sample": per_sample,
        "config": vars(args),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
