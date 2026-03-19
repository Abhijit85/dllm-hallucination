"""
Main experiment runner.

Usage
-----
python run_experiment.py \
    --n_paths 8 \
    --num_steps 64 \
    --entropy_threshold 0.5 \
    --refine_steps 16 \
    --max_samples 200 \
    --task_type QA \
    --output_dir results/

What this script measures
--------------------------
For each RAGTruth sample:

1. Run N parallel denoising paths (1 learned + N-1 random)
2. Compute token-level entropy → disagreement report
3. Run threshold ablation to find best entropy cutoff
4. Apply random-remask refinement on high-entropy positions
5. Evaluate:
   - Token F1 vs. RAGTruth labels (before & after refinement)
   - FactScore (source grounding proxy)
   - Spearman ρ: does entropy predict hallucination?
   - Refinement delta: does remasking help?

Key hypothesis tests
---------------------
H1: Random demasking paths produce higher entropy at hallucinated tokens
    than at grounded tokens → validated if mean(ρ) >> 0

H2: Re-denoising high-entropy positions reduces hallucination rate
    → validated if mean(refinement_delta) > 0

H3: Hybrid paths (learned then random) offer a Pareto-optimal trade-off
    between coherence and hallucination reduction
    → compare H=0.5 vs H=0.0 (all learned) vs H=1.0 (all random)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from data.ragtruth_loader import load_ragtruth, iter_batches, RAGTruthSample
from models.llada_harness import LLaDAHarness, DemaskingOrder
from strategies.parallel_remask import (
    compute_disagreement,
    random_remask_and_refine,
    ablate_threshold,
)
from eval.metrics import (
    token_level_f1,
    fact_score,
    disagreement_hallucination_correlation,
    refinement_delta,
    aggregate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_paths",           type=int,   default=8)
    p.add_argument("--num_steps",         type=int,   default=64)
    p.add_argument("--gen_len",           type=int,   default=128)
    p.add_argument("--entropy_threshold", type=float, default=0.5)
    p.add_argument("--refine_steps",      type=int,   default=16)
    p.add_argument("--max_samples",       type=int,   default=200)
    p.add_argument("--task_type",         type=str,   default=None,
                   choices=["QA", "Summary", "Data2txt", None])
    p.add_argument("--split",             type=str,   default="test")
    p.add_argument("--output_dir",        type=str,   default="results/")
    p.add_argument("--model_id",          type=str,   default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--ablate_thresholds", action="store_true",
                   help="Run threshold sweep on first 20 samples before main loop")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


def run_single_sample(
    sample: RAGTruthSample,
    harness: LLaDAHarness,
    args: argparse.Namespace,
) -> dict:
    """Full pipeline for one sample. Returns a metrics dict."""

    # ── 1. parallel denoising ──────────────────────────────────────────────────
    result = harness.run_parallel_paths(
        prompt=sample.response[:50],       # use first 50 chars as query proxy
        source_info=sample.source_info,
        n_paths=args.n_paths,
        gen_len=args.gen_len,
        num_steps=args.num_steps,
        learned_paths=1,
        base_seed=args.seed,
    )

    # ── 2. disagreement ────────────────────────────────────────────────────────
    report = compute_disagreement(
        result,
        entropy_threshold=args.entropy_threshold,
        track_trajectory=True,
    )

    # ── 3. refinement ──────────────────────────────────────────────────────────
    refinement = random_remask_and_refine(
        harness=harness,
        result=result,
        report=report,
        source_info=sample.source_info,
        refine_steps=args.refine_steps,
        refine_order=DemaskingOrder.LEARNED,
    )

    prompt_len = len(result.prompt_tokens)

    # ── 4. decode ──────────────────────────────────────────────────────────────
    original_text = harness.decode(refinement.original_tokens[prompt_len:])
    refined_text  = harness.decode(refinement.refined_tokens[prompt_len:])

    # ── 5. metrics ─────────────────────────────────────────────────────────────
    tf1 = token_level_f1(
        pred_tokens=refinement.refined_tokens,
        sample=sample,
        tokenizer=harness.tokenizer,
        prompt_len=prompt_len,
    )

    corr = disagreement_hallucination_correlation(
        report=report,
        sample=sample,
        tokenizer=harness.tokenizer,
        prompt_len=prompt_len,
    )

    rdelta = refinement_delta(
        original_text=original_text,
        refined_text=refined_text,
        source_info=sample.source_info,
        refinement_result=refinement,
        prompt_len=prompt_len,
    )

    return {
        "sample_id":         sample.sample_id,
        "task_type":         sample.task_type,
        "llm_name":          sample.llm_name,
        "has_hallucination": sample.has_hallucination,
        "n_hall_spans":      len(sample.spans),
        "n_flagged":         len(report.high_entropy_positions),
        "n_remasked":        refinement.n_remasked,
        "token_f1":          tf1.f1,
        "token_precision":   tf1.precision,
        "token_recall":      tf1.recall,
        "auc_roc":           tf1.auc_roc,
        "fact_score_before": rdelta.fact_score_before,
        "fact_score_after":  rdelta.fact_score_after,
        "refinement_delta":  rdelta.delta,
        "change_rate":       rdelta.change_rate,
        "spearman_rho":      corr.spearman_rho,
        "spearman_p":        corr.p_value,
        # Trajectory entropy (flattened for JSON)
        "entropy_trajectory": (
            report.path_entropy_trajectory.tolist()
            if report.path_entropy_trajectory is not None else None
        ),
        "original_text": original_text,
        "refined_text":  refined_text,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RAGTruth ({args.split}, task={args.task_type}, max={args.max_samples})")
    samples = load_ragtruth(
        split=args.split,
        task_type=args.task_type,
        max_samples=args.max_samples,
    )
    print(f"  {len(samples)} samples loaded")

    print(f"Loading {args.model_id}")
    harness = LLaDAHarness(model_id=args.model_id)

    # ── optional threshold ablation on warm-up samples ─────────────────────────
    if args.ablate_thresholds:
        print("\n── Threshold ablation (first 20 samples) ──")
        warmup = samples[:20]
        all_ablations = []
        for s in tqdm(warmup, desc="ablation"):
            r = harness.run_parallel_paths(
                prompt=s.response[:50],
                source_info=s.source_info,
                n_paths=args.n_paths,
                gen_len=args.gen_len,
                num_steps=args.num_steps,
            )
            gt_pos = s.token_labels(harness.tokenizer)
            gt_hall = [i for i, l in enumerate(gt_pos) if l == 1]
            ablations = ablate_threshold(r, gt_hall)
            all_ablations.append([a._asdict() for a in ablations])

        with open(out_dir / "threshold_ablation.json", "w") as f:
            json.dump(all_ablations, f, indent=2)
        print(f"  Saved to {out_dir}/threshold_ablation.json")

    # ── main experiment loop ───────────────────────────────────────────────────
    records = []
    for sample in tqdm(samples, desc="experiment"):
        try:
            rec = run_single_sample(sample, harness, args)
            records.append(rec)
        except Exception as e:
            print(f"  [skip] sample {sample.sample_id}: {e}")
            continue

        # Stream results to disk incrementally
        with open(out_dir / "raw_results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ── aggregate ──────────────────────────────────────────────────────────────
    agg = aggregate(records)
    print("\n── Aggregate Results ──────────────────────────────")
    print(f"  Samples evaluated:          {agg.n_samples}")
    print(f"  Hallucinated sample frac:   {agg.hallucinated_sample_fraction:.3f}")
    print(f"  Mean token F1:              {agg.mean_token_f1:.4f}")
    print(f"  Mean FactScore (before):    {agg.mean_fact_score_before:.4f}")
    print(f"  Mean FactScore (after):     {agg.mean_fact_score_after:.4f}")
    print(f"  Mean refinement delta:      {agg.mean_refinement_delta:+.4f}")
    print(f"  Mean Spearman ρ:            {agg.mean_spearman_rho:.4f}")
    print(f"  Mean token change rate:     {agg.mean_change_rate:.4f}")

    with open(out_dir / "aggregate.json", "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(agg), f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
