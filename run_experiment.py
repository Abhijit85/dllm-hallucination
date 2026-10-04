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
2. Compute token-level entropy -> disagreement report
3. Run threshold ablation to find best entropy cutoff
4. Apply random-remask refinement on high-entropy positions
5. Evaluate:
   - Token F1 vs. RAGTruth labels (before & after refinement)
   - FactScore (source grounding proxy)
   - Spearman rho: does entropy predict hallucination?
   - Refinement delta: does remasking help?

Key hypothesis tests
---------------------
H1: Random demasking paths produce higher entropy at hallucinated tokens
    than at grounded tokens -> validated if mean(rho) >> 0

H2: Re-denoising high-entropy positions reduces hallucination rate
    -> validated if mean(refinement_delta) > 0

H3: Hybrid paths (learned then random) offer a Pareto-optimal trade-off
    between coherence and hallucination reduction
    -> compare H=0.5 vs H=0.0 (all learned) vs H=1.0 (all random)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.ragtruth_loader import RAGTruthSample, load_ragtruth
from eval.metrics import (
    aggregate,
    disagreement_hallucination_correlation,
    refinement_delta,
    token_level_f1,
)
from models import DemaskingOrder, LLaDAHarness, create_harness
from strategies.parallel_remask import (
    ablate_threshold,
    compute_disagreement,
    random_remask_and_refine,
)


def parse_args():
    p = argparse.ArgumentParser()
    # model + data
    p.add_argument("--model_id",          type=str,   default="GSAI-ML/LLaDA-8B-Instruct",
                   help="HF model id or local path to LLaDA-8B-Instruct")
    p.add_argument("--data_path",         type=str,   default=None,
                   help="Local RAGTruth dir (with response.jsonl + source_info.jsonl) or JSONL file")
    # generation
    p.add_argument("--n_paths",           type=int,   default=8)
    p.add_argument("--num_steps",         type=int,   default=64)
    p.add_argument("--gen_len",           type=int,   default=128)
    # disagreement flagging
    p.add_argument("--top_k_percent",     type=float, default=0.20,
                   help="Flag top K%% highest-entropy tokens (default: 0.20 = top 20%%)")
    p.add_argument("--entropy_threshold", type=float, default=0.8,
                   help="Absolute entropy threshold (used only when --top_k_percent 0)")
    # refinement
    p.add_argument("--steps_per_token",   type=float, default=0.5,
                   help="Auto-scale refine steps: max(min_refine_steps, n_remasked * this)")
    p.add_argument("--min_refine_steps",  type=int,   default=16)
    p.add_argument("--refine_min_mean_entropy", type=float, default=1.65,
                   help="Skip refinement unless mean generation entropy reaches this value")
    # dataset
    p.add_argument("--max_samples",       type=int,   default=500)
    p.add_argument("--task_type",         type=str,   default=None,
                   choices=["QA", "Summary", "Data2txt"])
    p.add_argument("--split",             type=str,   default="test")
    # output
    p.add_argument("--output_dir",        type=str,   default="results/")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--ablate_thresholds", action="store_true",
                   help="Sweep entropy thresholds on first 20 samples before main loop")
    return p.parse_args()


def run_single_sample(
    sample: RAGTruthSample,
    harness: LLaDAHarness,
    args: argparse.Namespace,
) -> dict:
    """Full pipeline for one sample. Returns a metrics dict."""
    prompt_text = sample.prompt or sample.response[:50]
    prompt_source = "" if sample.prompt else sample.source_info

    # -- 1. parallel denoising -------------------------------------------------
    result = harness.run_parallel_paths(
        prompt=prompt_text,
        source_info=prompt_source,
        n_paths=args.n_paths,
        gen_len=args.gen_len,
        num_steps=args.num_steps,
        learned_paths=1,
        base_seed=args.seed,
    )

    # -- 2. disagreement -------------------------------------------------------
    report = compute_disagreement(
        result,
        top_k_percent=getattr(args, "top_k_percent", 0.20),
        entropy_threshold=getattr(args, "entropy_threshold", 0.8),
        track_trajectory=True,
    )

    # -- 3. refinement ---------------------------------------------------------
    prompt_len = len(result.prompt_tokens)
    gen_ent = report.token_entropy[prompt_len:]
    should_refine = len(gen_ent) > 0 and float(gen_ent.mean()) >= getattr(args, "refine_min_mean_entropy", 1.65)
    refinement = random_remask_and_refine(
        harness=harness,
        result=result,
        report=report,
        source_info=sample.source_info,
        refine_steps=None,
        steps_per_token=getattr(args, "steps_per_token", 0.5),
        min_refine_steps=getattr(args, "min_refine_steps", 16),
        refine_order=DemaskingOrder.LEARNED,
        max_remask=None if should_refine else 0,
    )

    # -- 4. decode -------------------------------------------------------------
    original_text = harness.decode(refinement.original_tokens[prompt_len:])
    refined_text  = harness.decode(refinement.refined_tokens[prompt_len:])
    chain_outputs = [
        harness.decode(path.final_tokens[prompt_len:])
        for path in result.paths
        if path.final_tokens is not None
    ]

    # -- 5. metrics ------------------------------------------------------------
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

    # -- 6. entropy summaries over the generation region ----------------------
    gen_len_actual = max(1, len(gen_ent))
    top_k = max(1, int(0.20 * gen_len_actual))
    top20_ent = float(torch.topk(gen_ent, k=top_k).values.mean())
    per_position_entropy = gen_ent.detach().cpu().tolist()

    gt_labels = sample.token_labels(harness.tokenizer)
    min_label_len = min(len(per_position_entropy), len(gt_labels))
    per_position_hallucinated = gt_labels[:min_label_len] if min_label_len > 0 else None

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
        "token_change_rate": rdelta.change_rate,
        "change_rate":       rdelta.change_rate,
        "n_paths":           args.n_paths,
        "gen_len":           gen_len_actual,
        "spearman_rho":      corr.spearman_rho,
        "spearman_skipped":  corr.skipped,
        "rho_skipped":       corr.skipped,
        "spearman_p":        corr.p_value,
        "n_flagged_pct":     len(report.high_entropy_positions) / gen_len_actual,
        "flagged_frac":      len(report.high_entropy_positions) / gen_len_actual,
        "mean_entropy":      float(gen_ent.mean()),
        "mean_cross_entropy": float(gen_ent.mean()),
        "top20_mean_entropy": top20_ent,
        "top20_entropy":     top20_ent,
        "max_entropy":       float(gen_ent.max()),
        "chain_outputs":     chain_outputs,
        "per_position_entropy": per_position_entropy,
        "per_position_hallucinated": per_position_hallucinated,
        "hallucination_mask": per_position_hallucinated,
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
        data_path=getattr(args, "data_path", None),
    )
    print(f"  {len(samples)} samples loaded")

    print(f"Loading {args.model_id}")
    harness = create_harness(model_id=args.model_id)

    # -- optional threshold ablation on warm-up samples ------------------------
    if args.ablate_thresholds:
        print("\n-- Threshold ablation (first 20 samples) --")
        warmup = samples[:20]
        all_ablations = []
        for s in tqdm(warmup, desc="ablation"):
            prompt_text = s.prompt or s.response[:50]
            prompt_source = "" if s.prompt else s.source_info
            r = harness.run_parallel_paths(
                prompt=prompt_text,
                source_info=prompt_source,
                n_paths=args.n_paths,
                gen_len=args.gen_len,
                num_steps=args.num_steps,
            )
            gt_pos = s.token_labels(harness.tokenizer)
            gt_hall = [i for i, lbl in enumerate(gt_pos) if lbl == 1]
            ablations = ablate_threshold(r, gt_hall)
            all_ablations.append([a._asdict() for a in ablations])

        with open(out_dir / "threshold_ablation.json", "w") as f:
            json.dump(all_ablations, f, indent=2)
        print(f"  Saved to {out_dir}/threshold_ablation.json")

    # -- main experiment loop --------------------------------------------------
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

        # Explicitly free GPU memory between samples
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- aggregate -------------------------------------------------------------
    agg = aggregate(records)
    print("\n-- Aggregate Results ------------------------------")
    print(f"  Samples evaluated:          {agg.n_samples}")
    print(f"  Hallucinated sample frac:   {agg.hallucinated_sample_fraction:.3f}")
    print(f"  Mean token F1:              {agg.mean_token_f1:.4f}")
    print(f"  Mean FactScore (before):    {agg.mean_fact_score_before:.4f}")
    print(f"  Mean FactScore (after):     {agg.mean_fact_score_after:.4f}")
    print(f"  Mean refinement delta:      {agg.mean_refinement_delta:+.4f}")
    print(f"  Mean Spearman rho:          {agg.mean_spearman_rho:.4f}  (computed on {agg.n_rho_computed}/{agg.n_samples} hallucinated samples)")
    print(f"  Mean token change rate:     {agg.mean_change_rate:.4f}")

    with open(out_dir / "aggregate.json", "w") as f:
        import dataclasses
        agg_dict = dataclasses.asdict(agg)
        # Embed run config into aggregate for easy comparison across runs
        agg_dict["n_paths"]         = args.n_paths
        agg_dict["num_steps"]       = args.num_steps
        agg_dict["gen_len"]         = args.gen_len
        agg_dict["top_k_percent"]   = getattr(args, "top_k_percent", 0.20)
        agg_dict["total_time_min"]  = round(sum(
            r.get("elapsed_s", 0) for r in records if "elapsed_s" in r
        ) / 60, 2)
        json.dump(agg_dict, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
