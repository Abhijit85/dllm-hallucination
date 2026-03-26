"""
run_oscar_eval.py
=================
Master evaluation runner for OSCAR hallucination mitigation experiments.

Full pipeline
-------------
python run_oscar_eval.py \\
    --datasets triviaqa commonsenseqa hotpotqa ragtruth \\
    --n_paths 8 --num_steps 64 --entropy_threshold 0.5 \\
    --refine_steps 16 --max_samples 500 \\
    --output_dir results/oscar/

Run any single component independently
---------------------------------------
# Single metric group only
python run_oscar_eval.py --only generation
python run_oscar_eval.py --only detection
python run_oscar_eval.py --only refinement

# Single table only (loads cached JSONL — no GPU needed)
python run_oscar_eval.py --only table1
python run_oscar_eval.py --only table2
python run_oscar_eval.py --only table3
python run_oscar_eval.py --only table4
python run_oscar_eval.py --only table5

# Single figure only (loads cached JSONL — no GPU needed)
python run_oscar_eval.py --only fig1_trajectory
python run_oscar_eval.py --only fig2_pr_curve
python run_oscar_eval.py --only fig3_qualitative
python run_oscar_eval.py --only fig4_ablation
python run_oscar_eval.py --only fig5_demasking

# Single baseline only
python run_oscar_eval.py --only baseline_vanilla
python run_oscar_eval.py --only baseline_extra_steps
python run_oscar_eval.py --only baseline_random_remask
python run_oscar_eval.py --only baseline_selfcheck

# Hyperparameter sweeps
python run_oscar_eval.py --only sweep_n
python run_oscar_eval.py --only sweep_tau
python run_oscar_eval.py --only sweep_steps

# All tables from cached results (no GPU)
python run_oscar_eval.py --only all_tables

# All figures from cached results (no GPU)
python run_oscar_eval.py --only all_figures

What gets saved
---------------
results/oscar/
  raw/
    triviaqa_raw.jsonl       per-sample generation results
    commonsenseqa_raw.jsonl
    hotpotqa_raw.jsonl
    ragtruth_raw.jsonl       + detection + refinement fields
    baseline_<method>.jsonl  per baseline method
    sweep_n.jsonl            N sweep results
    sweep_tau.jsonl          tau sweep results
    sweep_steps.jsonl        steps sweep results
    order_<name>.jsonl       per demasking order
  tables/
    table1_generation.tex / .md / .csv
    table2_detection.tex  / .md / .csv
    table3_refinement.tex / .md / .csv
    table4_baselines.tex  / .md / .csv
    table5_demasking.tex  / .md / .csv
  figures/
    fig1_entropy_trajectory.pdf / .png
    fig2_pr_curve.pdf           / .png
    fig3_qualitative.pdf        / .png
    fig4_ablation.pdf           / .png
    fig5_demasking_order.pdf    / .png
  aggregate.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────────────
from data.qa_loaders import (
    load_triviaqa, load_commonsenseqa, load_hotpotqa, load_ragtruth,
    build_prompt, QASample,
)
from models.llada_harness import LLaDAHarness, DemaskingOrder
from strategies.parallel_remask import compute_disagreement, random_remask_and_refine

from eval.oscar_metrics import (
    evaluate_generation_sample, aggregate_generation, compute_generation_metrics,
    compute_detection_metrics, build_ref_result, compute_refinement_metrics,
    SampleGenResult, DatasetGenMetrics, DetectionMetrics, SubsetRefMetrics,
)
from eval.oscar_baselines import (
    run_all_baselines, run_all_demasking_orders,
    sweep_n_paths, sweep_tau, sweep_refine_steps,
    METHOD_VANILLA, METHOD_EXTRA_STEPS, METHOD_RANDOM_REMASK,
    METHOD_SELFCHECK, METHOD_OSCAR,
    ORDER_ALL_LEARNED, ORDER_ALL_RANDOM, ORDER_HYBRID_50,
    ORDER_ENTROPY_ORDER, ORDER_OSCAR,
    BaselineRunResult,
)
from eval.oscar_figures import (
    entropy_trajectory_plot, build_trajectory_data,
    pr_curve_plot, build_pr_data,
    qualitative_example_plot, QualitativeExample,
    ablation_plots, AblationData,
    demasking_order_bar, DemaskingBarData,
    save_figure,
)
from eval.oscar_tables import (
    render_table1, render_table2, render_table3,
    render_table4, render_table5,
)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

VALID_ONLY = {
    # metric groups
    "generation", "detection", "refinement",
    # tables
    "table1", "table2", "table3", "table4", "table5", "all_tables",
    # figures
    "fig1_trajectory", "fig2_pr_curve", "fig3_qualitative",
    "fig4_ablation", "fig5_demasking", "all_figures",
    # baselines
    "baseline_vanilla", "baseline_extra_steps",
    "baseline_random_remask", "baseline_selfcheck",
    # sweeps
    "sweep_n", "sweep_tau", "sweep_steps",
    # combined
    "all",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="OSCAR evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--datasets", nargs="+",
                   default=["triviaqa", "commonsenseqa", "hotpotqa", "ragtruth"],
                   choices=["triviaqa", "commonsenseqa", "hotpotqa", "ragtruth"])
    p.add_argument("--max_samples",        type=int,   default=500)
    p.add_argument("--split",              type=str,   default="validation")
    p.add_argument("--ragtruth_split",     type=str,   default="test")
    p.add_argument("--ragtruth_task_type", type=str,   default=None,
                   choices=["QA", "Summary", "Data2txt", None])
    p.add_argument("--no_context",         action="store_true")

    # OSCAR hyperparams
    p.add_argument("--n_paths",            type=int,   default=8)
    p.add_argument("--num_steps",          type=int,   default=64)
    p.add_argument("--gen_len",            type=int,   default=128)
    p.add_argument("--entropy_threshold",  type=float, default=0.5)
    p.add_argument("--refine_steps",       type=int,   default=16)
    p.add_argument("--seed",               type=int,   default=42)

    # Sweep ranges
    p.add_argument("--sweep_n_values",     type=int,   nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--sweep_tau_values",   type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--sweep_step_values",  type=int,   nargs="+", default=[4, 8, 16, 32])

    # I/O
    p.add_argument("--output_dir",         type=str,   default="results/oscar/")
    p.add_argument("--model_id",           type=str,   default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--figure_formats",     nargs="+",  default=["pdf", "png"])

    # Control
    p.add_argument("--only",               type=str,   default="all",
                   choices=sorted(VALID_ONLY),
                   help="Run only this component. See module docstring for full list.")
    p.add_argument("--skip_model_load",    action="store_true",
                   help="Skip model loading (use when --only targets tables/figures only).")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _needs_model(only: str) -> bool:
    """Return True if the requested component requires the LLaDA model."""
    no_model = {
        "table1","table2","table3","table4","table5","all_tables",
        "fig1_trajectory","fig2_pr_curve","fig3_qualitative",
        "fig4_ablation","fig5_demasking","all_figures",
    }
    return only not in no_model


def _save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            if dataclasses.is_dataclass(r):
                f.write(json.dumps(dataclasses.asdict(r)) + "\n")
            else:
                f.write(json.dumps(r) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_table(name: str, latex: str, md: str, df, out_dir: Path):
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    (tbl_dir / f"{name}.tex").write_text(latex)
    (tbl_dir / f"{name}.md").write_text(md)
    df.to_csv(tbl_dir / f"{name}.csv", index=False)
    print(f"  Table saved → {tbl_dir}/{name}.[tex|md|csv]")


def _reconstruct_sample_gen(d: dict) -> SampleGenResult:
    return SampleGenResult(**{
        k: v for k, v in d.items()
        if k in SampleGenResult.__dataclass_fields__
    })


# ══════════════════════════════════════════════════════════════════════════════
# Core OSCAR sample pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_oscar_sample(
    sample: QASample,
    harness: LLaDAHarness,
    args: argparse.Namespace,
    track_trajectory: bool = True,
) -> dict:
    """
    Full OSCAR pipeline for one sample.
    Returns a flat dict written to JSONL (all metric groups combined).
    """
    source_info = sample.source_info or ""
    prompt      = build_prompt(sample)

    # 1. Parallel denoising
    result = harness.run_parallel_paths(
        prompt=prompt, source_info=source_info,
        n_paths=args.n_paths, gen_len=args.gen_len, num_steps=args.num_steps,
        learned_paths=1, base_seed=args.seed,
    )

    # 2. Disagreement
    report = compute_disagreement(
        result,
        entropy_threshold=args.entropy_threshold,
        track_trajectory=track_trajectory,
    )

    # 3. Refinement
    refinement = random_remask_and_refine(
        harness=harness, result=result, report=report,
        source_info=source_info, refine_steps=args.refine_steps,
        refine_order=DemaskingOrder.LEARNED,
    )

    prompt_len    = len(result.prompt_tokens)
    pred_before   = harness.decode(refinement.original_tokens[prompt_len:])
    pred_after    = harness.decode(refinement.refined_tokens[prompt_len:])

    # 4. Generation metrics
    gen_result = evaluate_generation_sample(
        sample_id=sample.sample_id, dataset=sample.dataset, task=sample.task,
        pred_before=pred_before, pred_after=pred_after,
        ground_truths=sample.ground_truths,
    )
    gen_result.n_flagged  = len(report.high_entropy_positions)
    gen_result.n_remasked = refinement.n_remasked

    out = dataclasses.asdict(gen_result)

    # 5. Detection + refinement metrics (RAGTruth only)
    if sample.dataset == "ragtruth" and sample.hall_spans:
        from data.ragtruth_loader import RAGTruthSample, HallucinationSpan
        from eval.oscar_metrics import fact_score_ngram

        spans = [
            HallucinationSpan(
                start=sp["start"], end=sp["end"], text=sp["text"],
                kind=sp.get("hallucination_type", "evident_conflict"),
                intensity=sp.get("intensity", 1.0),
            )
            for sp in sample.hall_spans
        ]
        rt = RAGTruthSample(
            sample_id=sample.sample_id, task_type="QA",
            llm_name="unknown", source_info=source_info,
            response=sample.ground_truths[0] if sample.ground_truths else "",
            spans=spans,
        )

        # Token labels for detection
        token_lbl = rt.token_labels(harness.tokenizer)
        ent_arr   = report.token_entropy[prompt_len:].cpu().numpy()
        min_len   = min(len(ent_arr), len(token_lbl))

        out["token_entropy"]  = ent_arr[:min_len].tolist()
        out["token_labels"]   = token_lbl[:min_len]
        out["task_type"]      = getattr(sample, "task_type", args.ragtruth_task_type or "QA")

        # Spearman rho
        from scipy.stats import spearmanr
        if min_len > 1 and sum(token_lbl[:min_len]) > 0:
            rho, pval = spearmanr(ent_arr[:min_len], token_lbl[:min_len])
            out["spearman_rho"] = float(rho)
            out["spearman_p"]   = float(pval)

        # Refinement metrics
        ref_result = build_ref_result(
            sample_id=sample.sample_id,
            subset=out.get("task_type", "QA"),
            original_text=pred_before, refined_text=pred_after,
            source_info=source_info,
            original_tokens=refinement.original_tokens,
            refined_tokens=refinement.refined_tokens,
            prompt_len=prompt_len,
            hall_spans=sample.hall_spans,
            remasked_positions=refinement.remasked_positions,
        )
        out.update({
            "tokens_changed":     ref_result.tokens_changed,
            "change_rate":        ref_result.change_rate,
            "fact_score_before":  ref_result.fact_score_before,
            "fact_score_after":   ref_result.fact_score_after,
            "fact_score_delta":   ref_result.fact_score_delta,
            "span_reduction_pct": ref_result.span_reduction_pct,
            "intensity_delta":    ref_result.intensity_delta,
        })

    # 6. Entropy trajectory (for Figure 1)
    if report.path_entropy_trajectory is not None:
        out["entropy_trajectory"] = report.path_entropy_trajectory.tolist()

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loop
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset_loop(
    samples: list[QASample],
    dataset_name: str,
    harness: LLaDAHarness,
    args: argparse.Namespace,
    out_dir: Path,
    track_trajectory: bool = True,
) -> list[dict]:
    records = []
    jsonl   = out_dir / "raw" / f"{dataset_name}_raw.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc=dataset_name):
        try:
            rec = run_oscar_sample(sample, harness, args, track_trajectory)
            records.append(rec)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"  [skip] {sample.sample_id}: {e}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Rebuild metrics from cached JSONL (no GPU needed)
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_gen_metrics(records_by_ds: dict[str, list[dict]]) -> dict[str, DatasetGenMetrics]:
    results_by_ds = {}
    for ds, recs in records_by_ds.items():
        results_by_ds[ds] = [_reconstruct_sample_gen(r) for r in recs]
    return compute_generation_metrics(results_by_ds)


def rebuild_detection_metrics(
    ragtruth_records: list[dict], tau: float
) -> dict[str, DetectionMetrics]:
    pairs = []
    tags  = []
    for r in ragtruth_records:
        ent = np.array(r.get("token_entropy", []))
        lbl = np.array(r.get("token_labels", []))
        if len(ent) and len(lbl):
            min_len = min(len(ent), len(lbl))
            pairs.append((ent[:min_len], lbl[:min_len]))
            tags.append(r.get("task_type", "QA"))
    return compute_detection_metrics(pairs, tags, tau=tau) if pairs else {}


def rebuild_refinement_metrics(ragtruth_records: list[dict]) -> dict[str, SubsetRefMetrics]:
    from eval.oscar_metrics import SampleRefResult
    ref_results = []
    for r in ragtruth_records:
        if "fact_score_before" not in r:
            continue
        ref_results.append(SampleRefResult(
            sample_id=r["sample_id"],
            subset=r.get("task_type", "QA"),
            tokens_changed=r.get("tokens_changed", 0),
            total_gen_tokens=0,
            change_rate=r.get("change_rate", 0.0),
            fact_score_before=r.get("fact_score_before", 0.0),
            fact_score_after=r.get("fact_score_after", 0.0),
            fact_score_delta=r.get("fact_score_delta", 0.0),
            spans_before=0, spans_after=0,
            span_reduction_pct=r.get("span_reduction_pct", 0.0),
            intensity_before=0.0, intensity_after=0.0,
            intensity_delta=r.get("intensity_delta", 0.0),
        ))
    return compute_refinement_metrics(ref_results) if ref_results else {}


def baseline_results_to_gen_metrics(
    baseline_records: dict[str, list[dict]],
) -> dict[str, dict[str, DatasetGenMetrics]]:
    """
    Convert baseline JSONL dicts → {method → {dataset → DatasetGenMetrics}}.

    Each method gets a full per-dataset breakdown so Tables 4 and 5 can
    show individual dataset columns instead of a single macro average.
    """
    out: dict[str, dict[str, DatasetGenMetrics]] = {}
    for method, recs in baseline_records.items():
        if not recs:
            continue
        results = [
            evaluate_generation_sample(
                sample_id=r["sample_id"], dataset=r["dataset"], task=r["task"],
                pred_before=r.get("pred_text", r.get("pred_before", "")),
                pred_after =r.get("pred_text", r.get("pred_after",  "")),
                ground_truths=r.get("references", [r.get("pred_text", "")]),
            )
            for r in recs
        ]
        by_ds: dict[str, list] = {}
        for res in results:
            by_ds.setdefault(res.dataset, []).append(res)
        out[method] = {
            ds_key: aggregate_generation(ds_recs, ds_key)
            for ds_key, ds_recs in by_ds.items()
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Figure builders from cached data
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1(all_records: list[dict], out_dir: Path, formats: list[str]):
    from eval.oscar_figures import TrajectoryData
    trajs_hall, trajs_gnd = [], []
    for r in all_records:
        traj = r.get("entropy_trajectory")
        lbls = r.get("token_labels", [])
        if not traj or not lbls:
            continue
        traj_arr = np.array(traj)
        lbl_arr  = np.array(lbls)
        hall_mask = lbl_arr == 1
        gnd_mask  = lbl_arr == 0
        if hall_mask.any():
            ratio = float(lbl_arr[hall_mask].mean()) / (traj_arr[-1] + 1e-9)
            trajs_hall.append(traj_arr * max(ratio, 0.1))
        if gnd_mask.any():
            ratio = float(1 - lbl_arr[gnd_mask].mean()) / (traj_arr[-1] + 1e-9)
            trajs_gnd.append(traj_arr * min(ratio, 2.0))

    if not trajs_hall:
        print("  [fig1] no trajectory data — skipping")
        return

    T = len(trajs_hall[0])
    steps = np.arange(T)
    h_arr = np.array(trajs_hall); g_arr = np.array(trajs_gnd) if trajs_gnd else np.zeros((1, T))
    td = TrajectoryData(
        steps=steps,
        mean_entropy={"hallucinated": h_arr.mean(0), "grounded": g_arr.mean(0)},
        std_entropy= {"hallucinated": h_arr.std(0),  "grounded": g_arr.std(0)},
    )
    fig = entropy_trajectory_plot(td)
    save_figure(fig, str(out_dir / "figures" / "fig1_entropy_trajectory"), formats)


def make_fig2(det_metrics: dict, out_dir: Path, formats: list[str]):
    if not det_metrics:
        print("  [fig2] no detection metrics — skipping")
        return
    pr_data = build_pr_data(det_metrics)
    if not pr_data:
        print("  [fig2] no PR data — skipping")
        return
    fig = pr_curve_plot(pr_data)
    save_figure(fig, str(out_dir / "figures" / "fig2_pr_curve"), formats)


def make_fig3(all_records: list[dict], out_dir: Path, formats: list[str]):
    # Find a sample with both high-entropy spans and changed tokens
    for r in all_records:
        if r.get("span_reduction_pct", 0) > 5 and r.get("tokens_changed", 0) > 0:
            ex = QualitativeExample(
                source_text=r.get("source_info", "")[:400],
                original_gen=r.get("pred_before", "")[:400],
                refined_gen =r.get("pred_after",  "")[:400],
                high_entropy_spans=[],  # char spans not stored; figure renders token-level
                changed_spans=[],
                sample_id=r.get("sample_id", ""),
                dataset=r.get("dataset", ""),
                task_type=r.get("task_type", ""),
            )
            fig = qualitative_example_plot(ex)
            save_figure(fig, str(out_dir / "figures" / "fig3_qualitative"), formats)
            return
    print("  [fig3] no suitable qualitative example found — skipping")


def make_fig4(sweep_records: dict[str, list[dict]], out_dir: Path, formats: list[str],
              args: argparse.Namespace):
    """Build Figure 4 from sweep JSONL dicts."""
    def _extract_metrics(recs: list[dict]) -> tuple[float, float, float]:
        if not recs:
            return 0.0, 0.0, 0.0
        results = [
            evaluate_generation_sample(
                r["sample_id"], r["dataset"], r["task"],
                r["pred_text"], r["pred_text"],
                r.get("references", [r.get("pred_text", "")]),
            )
            for r in recs
        ]
        f1s  = [r.f1_after    for r in results]
        rLs  = [r.rougeL_after for r in results]
        blus = [r.rougeL_after for r in results]
        return float(np.mean(f1s)), float(np.mean(rLs)), float(np.mean(blus))

    ablations = []
    for param_name, sweep_key, default_val in [
        ("N paths",            "sweep_n",    args.n_paths),
        ("τ (entropy thresh.)", "sweep_tau",  args.entropy_threshold),
        ("Refine steps",       "sweep_steps", args.refine_steps),
    ]:
        recs_by_val = sweep_records.get(sweep_key, {})
        if not recs_by_val:
            continue
        vals = sorted(recs_by_val.keys(), key=float)
        f1s, rLs, blus = [], [], []
        for v in vals:
            f1, rL, blu = _extract_metrics(recs_by_val[v])
            f1s.append(f1); rLs.append(rL); blus.append(blu)
        ablations.append(AblationData(
            param_name=param_name,
            param_values=[float(v) for v in vals],
            f1_vals=f1s, rougeL_vals=rLs, bleu_vals=blus,
            oscar_default=float(default_val),
        ))

    if len(ablations) < 3:
        print(f"  [fig4] only {len(ablations)}/3 sweep results — skipping")
        return

    fig = ablation_plots(ablations)
    save_figure(fig, str(out_dir / "figures" / "fig4_ablation"), formats)


def make_fig5(
    order_gen_by_ds: dict[str, dict[str, DatasetGenMetrics]],
    order_det: dict[str, DetectionMetrics],
    out_dir: Path, formats: list[str],
):
    ORDER_LIST = [ORDER_ALL_LEARNED, ORDER_ALL_RANDOM, ORDER_HYBRID_50,
                  ORDER_ENTROPY_ORDER, ORDER_OSCAR]

    from eval.oscar_metrics import _macro_gen

    f1s, rLs, aurocs = [], [], []
    for o in ORDER_LIST:
        ds_map = order_gen_by_ds.get(o, {})
        # Build macro average across all datasets present for this order
        dm_list = list(ds_map.values())
        if dm_list:
            macro = _macro_gen(dm_list, "macro_all")
            f1s.append(macro.f1_after)
            rLs.append(macro.rougeL_after)
        else:
            f1s.append(float("nan"))
            rLs.append(float("nan"))
        dm = order_det.get(o)
        aurocs.append(dm.auroc if dm and not math.isnan(dm.auroc) else float("nan"))

    data = DemaskingBarData(
        orders=ORDER_LIST,
        f1_vals=f1s, rougeL_vals=rLs, auroc_vals=aurocs,
        oscar_idx=ORDER_LIST.index(ORDER_OSCAR),
    )
    fig = demasking_order_bar(data)
    save_figure(fig, str(out_dir / "figures" / "fig5_demasking_order"), formats)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    only    = args.only
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_all = only == "all"

    # ── model loading ──────────────────────────────────────────────────────────
    harness = None
    if not args.skip_model_load and _needs_model(only):
        print(f"Loading {args.model_id} ...")
        harness = LLaDAHarness(model_id=args.model_id)
        print("  Model loaded.\n")

    # ── dataset loading ────────────────────────────────────────────────────────
    dataset_loaders = {
        "triviaqa":      lambda: load_triviaqa(args.split,      args.max_samples),
        "commonsenseqa": lambda: load_commonsenseqa(args.split,  args.max_samples),
        "hotpotqa":      lambda: load_hotpotqa(args.split, args.max_samples,
                                               include_context=not args.no_context),
        "ragtruth":      lambda: load_ragtruth(args.ragtruth_split,
                                               args.ragtruth_task_type,
                                               max_samples=args.max_samples),
    }

    all_samples: dict[str, list[QASample]] = {}
    if harness is not None:
        for ds in args.datasets:
            print(f"Loading {ds} ...")
            all_samples[ds] = dataset_loaders[ds]()
            print(f"  {len(all_samples[ds])} samples")

    # ══════════════════════════════════════════════════════════════════════════
    # A. OSCAR core pipeline  (generation + detection + refinement)
    # ══════════════════════════════════════════════════════════════════════════
    all_records: dict[str, list[dict]] = {}

    run_core = run_all or only in {
        "generation", "detection", "refinement",
        "table1", "table2", "table3",
        "fig1_trajectory", "fig2_pr_curve", "fig3_qualitative",
    }

    if run_core and harness is not None:
        for ds, samples in all_samples.items():
            print(f"\n── OSCAR on {ds} ({len(samples)} samples) ──")
            # track_trajectory only needed for fig1 and detection
            track = run_all or only in {"fig1_trajectory", "detection", "table2"}
            recs  = run_dataset_loop(samples, ds, harness, args, out_dir, track)
            all_records[ds] = recs
            print(f"  Done. {len(recs)} samples written.")
    else:
        # Load from cache
        for ds in args.datasets:
            path = out_dir / "raw" / f"{ds}_raw.jsonl"
            all_records[ds] = _load_jsonl(path)

    # ══════════════════════════════════════════════════════════════════════════
    # B. Baselines
    # ══════════════════════════════════════════════════════════════════════════
    baseline_records: dict[str, list[dict]] = {}
    BASELINE_KEY_MAP = {
        "baseline_vanilla":       METHOD_VANILLA,
        "baseline_extra_steps":   METHOD_EXTRA_STEPS,
        "baseline_random_remask": METHOD_RANDOM_REMASK,
        "baseline_selfcheck":     METHOD_SELFCHECK,
    }
    baseline_targets = (
        list(BASELINE_KEY_MAP.keys())
        if (run_all or only == "table4")
        else ([only] if only in BASELINE_KEY_MAP else [])
    )

    for bkey in baseline_targets:
        method = BASELINE_KEY_MAP[bkey]
        cache_path = out_dir / "raw" / f"baseline_{bkey}.jsonl"
        if cache_path.exists() and not run_all:
            baseline_records[method] = _load_jsonl(cache_path)
        elif harness is not None:
            # Use first dataset only for baselines (triviaqa or hotpotqa)
            ds_for_bl = "triviaqa" if "triviaqa" in all_samples else list(all_samples)[0]
            print(f"\n── Baseline: {method} ──")
            bl_results = run_all_baselines(
                harness, all_samples[ds_for_bl], args, methods=[method]
            )
            for m, recs in bl_results.items():
                baseline_records[m] = [dataclasses.asdict(r) for r in recs]
                _save_jsonl(baseline_records[m], cache_path)

    # Add OSCAR "baseline" entry from main records for Table 4
    if "triviaqa" in all_records:
        oscar_bl = [
            {**r, "pred_text": r.get("pred_after", ""),
             "references": r.get("references", [])}
            for r in all_records["triviaqa"]
        ]
        baseline_records[METHOD_OSCAR] = oscar_bl

    # ══════════════════════════════════════════════════════════════════════════
    # C. Demasking order sweep
    # ══════════════════════════════════════════════════════════════════════════
    order_raw: dict[str, list[dict]] = {}
    if run_all or only in {"fig5_demasking", "table5"}:
        for order in [ORDER_ALL_LEARNED, ORDER_ALL_RANDOM, ORDER_HYBRID_50,
                      ORDER_ENTROPY_ORDER]:
            cache_path = out_dir / "raw" / f"order_{order.replace(' ', '_')}.jsonl"
            if cache_path.exists():
                order_raw[order] = _load_jsonl(cache_path)
            elif harness is not None:
                ds_for_ord = "hotpotqa" if "hotpotqa" in all_samples else list(all_samples)[0]
                print(f"\n── Demasking order: {order} ──")
                res = run_all_demasking_orders(
                    harness, all_samples[ds_for_ord], args, orders=[order]
                )
                for o, recs in res.items():
                    order_raw[o] = [dataclasses.asdict(r) for r in recs]
                    _save_jsonl(order_raw[o], cache_path)
        # OSCAR itself
        if "hotpotqa" in all_records:
            order_raw[ORDER_OSCAR] = [
                {**r, "pred_text": r.get("pred_after", "")}
                for r in all_records["hotpotqa"]
            ]

    # ══════════════════════════════════════════════════════════════════════════
    # D. Hyperparameter sweeps
    # ══════════════════════════════════════════════════════════════════════════
    sweep_records: dict[str, dict] = {}

    sweep_map = {
        "sweep_n":     ("sweep_n",    sweep_n_paths,     args.sweep_n_values),
        "sweep_tau":   ("sweep_tau",  sweep_tau,          args.sweep_tau_values),
        "sweep_steps": ("sweep_steps", sweep_refine_steps, args.sweep_step_values),
    }

    for sweep_key, (rec_key, sweep_fn, values) in sweep_map.items():
        if not (run_all or only == sweep_key):
            continue
        cache_path = out_dir / "raw" / f"{rec_key}.jsonl"
        if cache_path.exists():
            raw = _load_jsonl(cache_path)
            # Reconstruct by param value
            sweep_records[rec_key] = {}
            for r in raw:
                v = r.get("param_value")
                sweep_records[rec_key].setdefault(v, []).append(r)
        elif harness is not None:
            ds_for_sweep = "triviaqa" if "triviaqa" in all_samples else list(all_samples)[0]
            print(f"\n── {sweep_key} sweep ──")
            # Reduce samples for sweep
            sweep_samples = all_samples[ds_for_sweep][:min(100, len(all_samples[ds_for_sweep]))]
            raw_by_val = sweep_fn(harness, sweep_samples, args, n_values=values if rec_key == "sweep_n"
                                   else (values if rec_key == "sweep_tau" else None))
            sweep_records[rec_key] = {}
            for v, recs in raw_by_val.items():
                sweep_records[rec_key][v] = [dataclasses.asdict(r) for r in recs]
                for r in sweep_records[rec_key][v]:
                    r["param_value"] = v
                _save_jsonl(sweep_records[rec_key][v], cache_path)

    # ══════════════════════════════════════════════════════════════════════════
    # E. Build metrics from records
    # ══════════════════════════════════════════════════════════════════════════
    gen_metrics = rebuild_gen_metrics(
        {ds: recs for ds, recs in all_records.items() if recs}
    )

    ragtruth_recs = all_records.get("ragtruth", [])
    det_metrics   = rebuild_detection_metrics(ragtruth_recs, args.entropy_threshold)
    ref_metrics   = rebuild_refinement_metrics(ragtruth_recs)

    baseline_gen  = baseline_results_to_gen_metrics(baseline_records) if baseline_records else {}
    baseline_det: dict[str, DetectionMetrics] = {}  # detection only for OSCAR

    order_gen_metrics = baseline_results_to_gen_metrics(
        {o: recs for o, recs in order_raw.items() if recs}
    ) if order_raw else {}

    # Detection metrics per demasking order (RAGTruth sub-experiment)
    order_det_metrics: dict[str, DetectionMetrics] = {}
    if order_raw:
        for order, recs in order_raw.items():
            rt_recs = [r for r in recs if r.get("dataset") == "ragtruth"
                       and r.get("token_entropy")]
            if rt_recs:
                pairs = [(np.array(r["token_entropy"]), np.array(r["token_labels"]))
                         for r in rt_recs
                         if r.get("token_labels")]
                tags  = [r.get("task_type", "QA") for r in rt_recs if r.get("token_labels")]
                dm    = compute_detection_metrics(pairs, tags, tau=args.entropy_threshold)
                order_det_metrics[order] = dm.get("all", DetectionMetrics(
                    subset="all", n_samples=0, hall_rate=0,
                    auroc=float("nan"), avg_precision=float("nan"),
                    precision_at_tau=0, recall_at_tau=0, f1_at_tau=0,
                    spearman_rho=float("nan"), spearman_p=float("nan"),
                ))

    # ══════════════════════════════════════════════════════════════════════════
    # F. Render tables
    # ══════════════════════════════════════════════════════════════════════════
    render_tables = run_all or only in {
        "table1", "table2", "table3", "table4", "table5", "all_tables"
    }

    if render_tables:
        print("\n── Rendering tables ──")

        if run_all or only in {"table1", "all_tables"}:
            latex, md, df = render_table1(gen_metrics, args.n_paths,
                                          args.entropy_threshold, args.refine_steps)
            _save_table("table1_generation", latex, md, df, out_dir)

        if run_all or only in {"table2", "all_tables"}:
            if det_metrics:
                latex, md, df = render_table2(det_metrics, args.entropy_threshold)
                _save_table("table2_detection", latex, md, df, out_dir)

        if run_all or only in {"table3", "all_tables"}:
            if ref_metrics:
                latex, md, df = render_table3(ref_metrics)
                _save_table("table3_refinement", latex, md, df, out_dir)

        if run_all or only in {"table4", "all_tables"}:
            if baseline_gen:
                latex, md, df = render_table4(baseline_gen, baseline_det)
                _save_table("table4_baselines", latex, md, df, out_dir)
        if run_all or only in {"table5", "all_tables"}:
            if order_gen_metrics:
                latex, md, df = render_table5(order_gen_metrics, order_det_metrics)
                _save_table("table5_demasking", latex, md, df, out_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # G. Render figures
    # ══════════════════════════════════════════════════════════════════════════
    render_figures = run_all or only in {
        "fig1_trajectory", "fig2_pr_curve", "fig3_qualitative",
        "fig4_ablation", "fig5_demasking", "all_figures",
    }
    fmt = args.figure_formats

    if render_figures:
        print("\n── Rendering figures ──")
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)

        if run_all or only in {"fig1_trajectory", "all_figures"}:
            all_rgt = all_records.get("ragtruth", [])
            make_fig1(all_rgt, out_dir, fmt)

        if run_all or only in {"fig2_pr_curve", "all_figures"}:
            make_fig2(det_metrics, out_dir, fmt)

        if run_all or only in {"fig3_qualitative", "all_figures"}:
            make_fig3(ragtruth_recs, out_dir, fmt)

        if run_all or only in {"fig4_ablation", "all_figures"}:
            make_fig4(sweep_records, out_dir, fmt, args)

        if run_all or only in {"fig5_demasking", "all_figures"}:
            make_fig5(order_gen_metrics, order_det_metrics, out_dir, fmt)

    # ══════════════════════════════════════════════════════════════════════════
    # H. Print summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  OSCAR EVALUATION SUMMARY")
    print("═" * 70)

    macro_all = gen_metrics.get("macro_all")
    if macro_all:
        print(f"  Generation (macro all):")
        print(f"    F1       {macro_all.f1_before*100:.2f} → {macro_all.f1_after*100:.2f}"
              f"  (Δ {macro_all.f1_delta*100:+.2f})")
        print(f"    ROUGE-L  {macro_all.rougeL_before*100:.2f} → {macro_all.rougeL_after*100:.2f}"
              f"  (Δ {macro_all.rougeL_delta*100:+.2f})")
        print(f"    BLEU     {macro_all.bleu_before*100:.2f} → {macro_all.bleu_after*100:.2f}"
              f"  (Δ {macro_all.bleu_delta*100:+.2f})")

    if det_metrics and "all" in det_metrics:
        dm = det_metrics["all"]
        print(f"\n  Detection (RAGTruth):")
        print(f"    AUROC   {dm.auroc*100:.2f}%   AP {dm.avg_precision*100:.2f}%")
        print(f"    Spearman ρ = {dm.spearman_rho:.4f}  (p={dm.spearman_p:.4f})")

    if ref_metrics and "all" in ref_metrics:
        rm = ref_metrics["all"]
        print(f"\n  Refinement (RAGTruth):")
        print(f"    FactScore  {rm.fact_score_before:.2f} → {rm.fact_score_after:.2f}"
              f"  (Δ {rm.fact_score_delta:+.2f})")
        print(f"    Span reduction  {rm.span_reduction_pct:.1f}%")

    print(f"\n  Output → {out_dir}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
