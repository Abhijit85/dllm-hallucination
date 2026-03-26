"""
eval/oscar_baselines.py
=======================
Baseline methods and ablation sweeps for OSCAR evaluation.

Table 4 — Baselines (comparison against OSCAR)
-----------------------------------------------
  METHOD_VANILLA      : LLaDA 1 path, learned order (standard generation)
  METHOD_EXTRA_STEPS  : LLaDA 1 path, 2× denoising steps (compute-matched)
  METHOD_RANDOM_REMASK: Random position remasking — no entropy guidance
  METHOD_SELFCHECK    : SelfCheckGPT-style: flag tokens where N paths disagree
                        by majority vote (binary, not entropy-weighted)
  METHOD_OSCAR        : Ours — entropy-guided remasking + learned refinement

Table 5 — Demasking order comparison
--------------------------------------
  ORDER_ALL_LEARNED   : 1 learned path only
  ORDER_ALL_RANDOM    : N random paths
  ORDER_HYBRID_50     : 50% learned, 50% random
  ORDER_ENTROPY       : entropy-ordered unmasking
  ORDER_OSCAR         : 1 learned + (N-1) random  ← OSCAR default

Table 6 — Hyperparameter ablation
-----------------------------------
  sweep_n_paths(harness, sample, n_values)   : N ∈ {2,4,8,16}
  sweep_tau(harness, sample, tau_values)     : τ ∈ {0.3,0.4,0.5,0.6,0.7}
  sweep_refine_steps(harness, sample, steps) : steps ∈ {4,8,16,32}

Standalone usage
----------------
  # Run a single baseline on a list of samples:
  from eval.oscar_baselines import run_baseline, METHOD_SELFCHECK
  results = run_baseline(METHOD_SELFCHECK, harness, samples, args)

  # Run full ablation sweep:
  from eval.oscar_baselines import sweep_n_paths
  sweep_results = sweep_n_paths(harness, samples, n_values=[2,4,8,16], args=args)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

# Method name constants
METHOD_VANILLA       = "Vanilla LLaDA (1-path)"
METHOD_EXTRA_STEPS   = "Extra steps (2×T, no remask)"
METHOD_RANDOM_REMASK = "Random remask (no entropy)"
METHOD_SELFCHECK     = "SelfCheck-style consistency"
METHOD_OSCAR         = "OSCAR (ours)"

ORDER_ALL_LEARNED    = "All learned"
ORDER_ALL_RANDOM     = "All random"
ORDER_HYBRID_50      = "Hybrid (50/50)"
ORDER_ENTROPY_ORDER  = "Entropy-ordered"
ORDER_OSCAR          = "OSCAR (1L + N-1R)"


# ══════════════════════════════════════════════════════════════════════════════
# Shared result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BaselineRunResult:
    """Decoded outputs for one (method, sample) — feed into oscar_metrics."""
    method:      str
    sample_id:   str
    dataset:     str
    task:        str
    pred_text:   str        # single decoded generation (no before/after for baselines)
    source_info: str = ""


@dataclass
class AblationPoint:
    """One (hyperparam_value, metric_dict) data point for ablation plots."""
    param_name:  str    # "n_paths" | "tau" | "refine_steps"
    param_value: float
    f1:          float
    rougeL:      float
    bleu:        float
    auroc:       float  # nan for non-RAGTruth
    spearman_rho: float # nan for non-RAGTruth


# ══════════════════════════════════════════════════════════════════════════════
# Table 4 — individual baseline runners
# ══════════════════════════════════════════════════════════════════════════════

def run_vanilla(harness, sample, args) -> BaselineRunResult:
    """
    Vanilla LLaDA: single learned-order path, no parallel sampling, no remask.
    This is the standard LLaDA generation baseline.
    """
    from models.llada_harness import DemaskingOrder
    from data.qa_loaders import build_prompt

    prompt      = build_prompt(sample)
    source_info = sample.source_info or ""

    result = harness.run_parallel_paths(
        prompt=prompt, source_info=source_info,
        n_paths=1, gen_len=args.gen_len, num_steps=args.num_steps,
        learned_paths=1, base_seed=args.seed,
    )
    prompt_len = len(result.prompt_tokens)
    text = harness.decode(result.paths[0].final_tokens[prompt_len:])

    return BaselineRunResult(
        method=METHOD_VANILLA, sample_id=sample.sample_id,
        dataset=sample.dataset, task=sample.task,
        pred_text=text, source_info=source_info,
    )


def run_extra_steps(harness, sample, args) -> BaselineRunResult:
    """
    Extra-steps baseline: 1 learned path with 2× the denoising steps.
    Compute-matched to OSCAR (same FLOPs as running N=2 paths at T steps).
    Tests whether improvement is from more compute vs. the entropy signal.
    """
    from data.qa_loaders import build_prompt

    prompt      = build_prompt(sample)
    source_info = sample.source_info or ""

    result = harness.run_parallel_paths(
        prompt=prompt, source_info=source_info,
        n_paths=1, gen_len=args.gen_len,
        num_steps=args.num_steps * 2,   # 2× steps
        learned_paths=1, base_seed=args.seed,
    )
    prompt_len = len(result.prompt_tokens)
    text = harness.decode(result.paths[0].final_tokens[prompt_len:])

    return BaselineRunResult(
        method=METHOD_EXTRA_STEPS, sample_id=sample.sample_id,
        dataset=sample.dataset, task=sample.task,
        pred_text=text, source_info=source_info,
    )


def run_random_remask(harness, sample, args) -> BaselineRunResult:
    """
    Random remask baseline: run N paths, then remask a RANDOM subset of
    positions (same count as OSCAR would flag at τ) and re-denoise.
    This is the critical ablation: isolates whether entropy-guided selection
    matters vs. just remasking any tokens.
    """
    from models.llada_harness import DemaskingOrder
    from strategies.parallel_remask import compute_disagreement, random_remask_and_refine
    from data.qa_loaders import build_prompt

    prompt      = build_prompt(sample)
    source_info = sample.source_info or ""

    result = harness.run_parallel_paths(
        prompt=prompt, source_info=source_info,
        n_paths=args.n_paths, gen_len=args.gen_len, num_steps=args.num_steps,
        learned_paths=1, base_seed=args.seed,
    )

    # Compute disagreement report (to know how many positions OSCAR would flag)
    report = compute_disagreement(result, entropy_threshold=args.entropy_threshold)
    n_to_remask = len(report.high_entropy_positions)

    # Randomly select that many positions instead of entropy-guided selection
    prompt_len = len(result.prompt_tokens)
    gen_len    = args.gen_len
    all_gen_positions = list(range(prompt_len, prompt_len + gen_len))
    torch.manual_seed(args.seed + 9999)
    perm = torch.randperm(len(all_gen_positions))[:n_to_remask].tolist()
    random_positions = [all_gen_positions[i] for i in perm]

    # Patch the report with random positions
    from copy import copy
    rand_report = copy(report)
    rand_report.high_entropy_positions = random_positions

    refinement = random_remask_and_refine(
        harness=harness, result=result, report=rand_report,
        source_info=source_info, refine_steps=args.refine_steps,
        refine_order=DemaskingOrder.LEARNED,
    )

    text = harness.decode(refinement.refined_tokens[prompt_len:])

    return BaselineRunResult(
        method=METHOD_RANDOM_REMASK, sample_id=sample.sample_id,
        dataset=sample.dataset, task=sample.task,
        pred_text=text, source_info=source_info,
    )


def run_selfcheck_style(harness, sample, args) -> BaselineRunResult:
    """
    SelfCheckGPT-style consistency baseline.
    Flag positions where the majority-vote token differs from >= k paths.
    Uses binary disagreement (vote-based) rather than soft entropy.
    Then remask and refine identically to OSCAR.
    """
    from models.llada_harness import DemaskingOrder
    from strategies.parallel_remask import random_remask_and_refine
    from data.qa_loaders import build_prompt
    from copy import copy
    from dataclasses import replace

    prompt      = build_prompt(sample)
    source_info = sample.source_info or ""

    result = harness.run_parallel_paths(
        prompt=prompt, source_info=source_info,
        n_paths=args.n_paths, gen_len=args.gen_len, num_steps=args.num_steps,
        learned_paths=1, base_seed=args.seed,
    )

    prompt_len   = len(result.prompt_tokens)
    n_paths      = len(result.paths)
    majority     = result.majority_vote()   # (seq_len,)

    # Flag positions where >= 50% of paths disagree with majority
    import torch
    stacked   = torch.stack([p.final_tokens for p in result.paths], dim=0)  # (N, L)
    n_agree   = (stacked == majority.unsqueeze(0)).sum(dim=0).float()        # (L,)
    agree_frac = n_agree / n_paths
    flagged    = (agree_frac < 0.5).nonzero(as_tuple=True)[0].tolist()
    flagged    = [p for p in flagged if p >= prompt_len]

    # Build a fake DisagreementReport with these positions
    from strategies.parallel_remask import DisagreementReport
    sc_report = DisagreementReport(
        token_entropy=result.token_entropy(),
        high_entropy_positions=flagged,
        entropy_threshold=0.5,
        path_entropy_trajectory=None,
    )

    refinement = random_remask_and_refine(
        harness=harness, result=result, report=sc_report,
        source_info=source_info, refine_steps=args.refine_steps,
        refine_order=DemaskingOrder.LEARNED,
    )

    text = harness.decode(refinement.refined_tokens[prompt_len:])

    return BaselineRunResult(
        method=METHOD_SELFCHECK, sample_id=sample.sample_id,
        dataset=sample.dataset, task=sample.task,
        pred_text=text, source_info=source_info,
    )


# Registry: method name → runner function
BASELINE_RUNNERS: dict[str, Callable] = {
    METHOD_VANILLA:       run_vanilla,
    METHOD_EXTRA_STEPS:   run_extra_steps,
    METHOD_RANDOM_REMASK: run_random_remask,
    METHOD_SELFCHECK:     run_selfcheck_style,
}


def run_baseline(
    method: str,
    harness,
    samples: list,
    args: argparse.Namespace,
) -> list[BaselineRunResult]:
    """
    Standalone entry point: run one baseline method over a list of samples.

    Args:
        method:   one of the METHOD_* constants
        harness:  LLaDAHarness instance
        samples:  list[QASample]
        args:     argparse namespace with n_paths, gen_len, etc.

    Returns:
        list[BaselineRunResult]
    """
    if method not in BASELINE_RUNNERS:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose from: {list(BASELINE_RUNNERS)}")
    runner = BASELINE_RUNNERS[method]
    results = []
    from tqdm import tqdm
    for sample in tqdm(samples, desc=f"baseline:{method}"):
        try:
            results.append(runner(harness, sample, args))
        except Exception as e:
            print(f"  [skip] {sample.sample_id}: {e}")
    return results


def run_all_baselines(
    harness,
    samples: list,
    args: argparse.Namespace,
    methods: list[str] | None = None,
) -> dict[str, list[BaselineRunResult]]:
    """Run all baselines (or a subset) and return keyed by method name."""
    methods = methods or list(BASELINE_RUNNERS)
    return {m: run_baseline(m, harness, samples, args) for m in methods}


# ══════════════════════════════════════════════════════════════════════════════
# Table 5 — Demasking order comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_demasking_order(
    order_name: str,
    harness,
    samples: list,
    args: argparse.Namespace,
) -> list[BaselineRunResult]:
    """
    Run one demasking order configuration over a list of samples.

    Maps order_name → (learned_paths, random_paths, hybrid_paths) config
    for LLaDAHarness.run_parallel_paths().
    """
    from models.llada_harness import DemaskingOrder
    from strategies.parallel_remask import compute_disagreement, random_remask_and_refine
    from data.qa_loaders import build_prompt

    # (learned, random, hybrid) path counts for each order config
    ORDER_CONFIGS = {
        ORDER_ALL_LEARNED:   (args.n_paths, 0,                 0),
        ORDER_ALL_RANDOM:    (0,             args.n_paths,      0),
        ORDER_HYBRID_50:     (args.n_paths // 2, args.n_paths // 2, 0),
        ORDER_ENTROPY_ORDER: (0,             0,                 args.n_paths),
        ORDER_OSCAR:         (1,             args.n_paths - 1,  0),
    }
    if order_name not in ORDER_CONFIGS:
        raise ValueError(f"Unknown order '{order_name}'")

    learned, random_, hybrid = ORDER_CONFIGS[order_name]
    results = []

    from tqdm import tqdm
    for sample in tqdm(samples, desc=f"order:{order_name}"):
        try:
            prompt      = build_prompt(sample)
            source_info = sample.source_info or ""

            result = harness.run_parallel_paths(
                prompt=prompt, source_info=source_info,
                n_paths=args.n_paths, gen_len=args.gen_len, num_steps=args.num_steps,
                learned_paths=learned, random_paths=random_, hybrid_paths=hybrid,
                base_seed=args.seed,
            )
            report = compute_disagreement(result, entropy_threshold=args.entropy_threshold)
            refinement = random_remask_and_refine(
                harness=harness, result=result, report=report,
                source_info=source_info, refine_steps=args.refine_steps,
                refine_order=DemaskingOrder.LEARNED,
            )
            prompt_len = len(result.prompt_tokens)
            text = harness.decode(refinement.refined_tokens[prompt_len:])

            results.append(BaselineRunResult(
                method=order_name, sample_id=sample.sample_id,
                dataset=sample.dataset, task=sample.task,
                pred_text=text, source_info=source_info,
            ))
        except Exception as e:
            print(f"  [skip] {sample.sample_id}: {e}")

    return results


def run_all_demasking_orders(
    harness, samples: list, args: argparse.Namespace,
    orders: list[str] | None = None,
) -> dict[str, list[BaselineRunResult]]:
    """Run all demasking order configs and return keyed by order name."""
    orders = orders or [
        ORDER_ALL_LEARNED, ORDER_ALL_RANDOM,
        ORDER_HYBRID_50, ORDER_ENTROPY_ORDER, ORDER_OSCAR,
    ]
    return {o: run_demasking_order(o, harness, samples, args) for o in orders}


# ══════════════════════════════════════════════════════════════════════════════
# Table 6 — Hyperparameter sweeps
# ══════════════════════════════════════════════════════════════════════════════

def _run_oscar_with_config(
    harness, samples, args,
    n_paths=None, tau=None, refine_steps=None,
) -> list[BaselineRunResult]:
    """Run OSCAR with overridden hyperparameters."""
    from models.llada_harness import DemaskingOrder
    from strategies.parallel_remask import compute_disagreement, random_remask_and_refine
    from data.qa_loaders import build_prompt
    from tqdm import tqdm

    _n       = n_paths     or args.n_paths
    _tau     = tau         or args.entropy_threshold
    _refine  = refine_steps or args.refine_steps

    results = []
    for sample in tqdm(samples, desc=f"sweep N={_n} τ={_tau} R={_refine}"):
        try:
            prompt      = build_prompt(sample)
            source_info = sample.source_info or ""
            result      = harness.run_parallel_paths(
                prompt=prompt, source_info=source_info,
                n_paths=_n, gen_len=args.gen_len, num_steps=args.num_steps,
                learned_paths=1, base_seed=args.seed,
            )
            report = compute_disagreement(result, entropy_threshold=_tau)
            refinement = random_remask_and_refine(
                harness=harness, result=result, report=report,
                source_info=source_info, refine_steps=_refine,
                refine_order=DemaskingOrder.LEARNED,
            )
            prompt_len = len(result.prompt_tokens)
            text = harness.decode(refinement.refined_tokens[prompt_len:])
            results.append(BaselineRunResult(
                method=f"oscar_N{_n}_tau{_tau}_r{_refine}",
                sample_id=sample.sample_id, dataset=sample.dataset,
                task=sample.task, pred_text=text, source_info=source_info,
            ))
        except Exception as e:
            print(f"  [skip] {sample.sample_id}: {e}")
    return results


def sweep_n_paths(
    harness, samples: list, args,
    n_values: list[int] | None = None,
) -> dict[int, list[BaselineRunResult]]:
    """Sweep N paths. Standalone entry point for N ablation."""
    n_values = n_values or [2, 4, 8, 16]
    return {n: _run_oscar_with_config(harness, samples, args, n_paths=n)
            for n in n_values}


def sweep_tau(
    harness, samples: list, args,
    tau_values: list[float] | None = None,
) -> dict[float, list[BaselineRunResult]]:
    """Sweep entropy threshold τ. Standalone entry point for τ ablation."""
    tau_values = tau_values or [0.3, 0.4, 0.5, 0.6, 0.7]
    return {t: _run_oscar_with_config(harness, samples, args, tau=t)
            for t in tau_values}


def sweep_refine_steps(
    harness, samples: list, args,
    step_values: list[int] | None = None,
) -> dict[int, list[BaselineRunResult]]:
    """Sweep refinement step count. Standalone entry point for steps ablation."""
    step_values = step_values or [4, 8, 16, 32]
    return {s: _run_oscar_with_config(harness, samples, args, refine_steps=s)
            for s in step_values}
