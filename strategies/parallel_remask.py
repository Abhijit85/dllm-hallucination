"""
Strategy: parallel-path disagreement + random remasking.

Pipeline
--------
1. Run N paths (1 learned + N-1 random) via LLaDAHarness.run_parallel_paths()
2. Compute token-level entropy across final tokens of all paths
3. Flag high-entropy positions as uncertain (potential hallucinations)
4. Remask those positions and run a targeted re-denoising pass
   conditioned on the source passage (grounding signal)
5. Return the refined response

The random demasking hypothesis
--------------------------------
We conjecture that the learned demasking order can lock the model into
"hallucination attractors": once a hallucinated token is committed early
in the denoising process, subsequent tokens reinforce it (the model becomes
self-consistent around a wrong fact).

Random demasking breaks this by removing the causal commitment structure.
Tokens are revealed in unpredictable order so the model cannot "plan ahead"
to produce internally consistent (but factually wrong) text.

Measuring this
--------------
We track:
  - path_entropy_at_step(t): how much paths diverge across the trajectory
  - hallucination_entropy_correlation: do high-entropy tokens match RAGTruth labels?
  - remask_delta: does re-denoising high-entropy spans reduce hallucination rate?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from models.llada_harness import (
    DemaskingOrder,
    LLaDAHarness,
    ParallelPathResult,
)

# ── disagreement analysis ──────────────────────────────────────────────────────


@dataclass
class DisagreementReport:
    token_entropy: torch.Tensor  # (seq_len,) final-token entropy
    high_entropy_positions: list[int]  # positions above threshold
    entropy_threshold: float
    selection_fraction: float | None
    path_entropy_trajectory: torch.Tensor | None  # (T,) mean entropy per step


def compute_disagreement(
    result: ParallelPathResult,
    entropy_threshold: float = 0.8,
    top_k_percent: float | None = 0.20,
    selection_fraction: float | None = None,
    track_trajectory: bool = True,
) -> DisagreementReport:
    """
    Compute token-level entropy across N paths' final outputs.

    Two modes (top_k_percent takes priority when set):

    top_k_percent=0.20 (default, recommended):
        Flag the top 20% highest-entropy tokens in the generation region.
        Adapts to the entropy scale of the current model / n_paths combo.
        With only 4 paths, random disagreement inflates entropy everywhere;
        percentile-based flagging still finds the relatively uncertain
        tokens without remasking everything.

    entropy_threshold (fallback when top_k_percent=None):
        Flag tokens with H > threshold. Set higher than 0.5 for few paths:
        4 paths -> 0.8+,  8 paths -> 0.6+,  16 paths -> 0.5
    """
    if selection_fraction is not None and top_k_percent == 0.20:
        top_k_percent = selection_fraction

    ent = result.token_entropy()
    prompt_len = len(result.prompt_tokens)
    gen_ent = ent[prompt_len:]  # operate on generation region only

    if top_k_percent is not None and len(gen_ent) > 0:
        k = max(1, int(len(gen_ent) * top_k_percent))
        topk = gen_ent.topk(k=min(k, len(gen_ent)))
        high_ent_pos = [i + prompt_len for i in topk.indices.tolist()]
        realized_threshold = float(topk.values.min().item())
    elif len(gen_ent) == 0:
        high_ent_pos = []
        realized_threshold = float("nan")
    else:
        high_ent_pos = (ent > entropy_threshold).nonzero(as_tuple=True)[0].tolist()
        realized_threshold = entropy_threshold

    trajectory = None
    if track_trajectory and all(len(p.steps) > 0 for p in result.paths):
        # Entropy trajectory now read from pre-computed scalars in DenoiseStep
        # (logits are no longer stored — avoids OOM on long runs)
        num_steps = min(len(path.steps) for path in result.paths)
        traj = []
        for step_idx in range(num_steps):
            step_ents = [path.steps[step_idx].mean_entropy for path in result.paths]
            traj.append(sum(step_ents) / len(step_ents) if step_ents else 0.0)
        trajectory = torch.tensor(traj)

    return DisagreementReport(
        token_entropy=ent,
        high_entropy_positions=high_ent_pos,
        entropy_threshold=realized_threshold,
        selection_fraction=top_k_percent,
        path_entropy_trajectory=trajectory,
    )


# ── random-remasking refinement ────────────────────────────────────────────────


@dataclass
class RefinementResult:
    original_tokens: torch.Tensor  # majority vote of N paths (pre-refinement)
    refined_tokens: torch.Tensor  # after targeted remasking
    remasked_positions: list[int]  # positions that were re-denoised
    n_remasked: int


def random_remask_and_refine(
    harness: LLaDAHarness,
    result: ParallelPathResult,
    report: DisagreementReport,
    source_info: str,
    refine_steps: int | None = None,
    steps_per_token: float = 0.5,
    min_refine_steps: int = 16,
    refine_order: DemaskingOrder = DemaskingOrder.LEARNED,
    max_remask: int | None = None,
    remask_fraction: float = 1.0,
) -> RefinementResult:
    """
    Take the majority-vote output, remask high-entropy positions, and
    run a short targeted denoising pass conditioned on source_info.

    Why learned order for refinement?
        In the refinement pass we want grounding: commit verifiable tokens first.
        Random order would re-introduce variance.

    Step scaling (critical for coherent output):
        refine_steps is auto-scaled to n_remasked by default:
            steps = max(min_refine_steps, int(n_remasked * steps_per_token))
        This guarantees at least ~2 forward passes per masked token,
        giving the model enough iterations to produce coherent text.
        With 8 steps for 63 tokens (old default) each step unmasks ~8 tokens,
        far too coarse for LLaDA to produce coherent output.

    Args:
        harness:         the LLaDAHarness instance
        result:          output of run_parallel_paths()
        report:          output of compute_disagreement()
        source_info:     retrieved passage for conditioning
        harness:          the LLaDAHarness instance
        result:           output of run_parallel_paths()
        report:           output of compute_disagreement()
        source_info:      retrieved passage for conditioning
        refine_steps:     override step count (None = auto-scale)
        steps_per_token:  ratio used for auto-scaling (default 0.5)
        min_refine_steps: floor on auto-scaled step count (default 16)
        refine_order:     demasking order for refinement
        max_remask:       cap on positions to remask
        remask_fraction:  subsample flagged positions for ablation
    """
    prompt_len = len(result.prompt_tokens)
    if result.paths and result.paths[0].final_tokens is not None:
        base_tokens = result.paths[0].final_tokens.detach().clone().cpu()
    else:
        base_tokens = result.majority_vote()

    flagged = [p for p in report.high_entropy_positions if p >= prompt_len]

    if remask_fraction < 1.0:
        k = max(1, int(len(flagged) * remask_fraction))
        perm = torch.randperm(len(flagged))[:k].tolist()
        flagged = [flagged[i] for i in perm]

    if max_remask is not None:
        flagged = flagged[:max_remask]

    remasked_positions = flagged

    # Build refined x: start from majority vote, remask flagged positions
    if hasattr(harness, "remask_and_refine"):
        gen_len = len(base_tokens) - prompt_len
        if gen_len <= 0:
            return RefinementResult(
                original_tokens=base_tokens,
                refined_tokens=base_tokens,
                remasked_positions=[],
                n_remasked=0,
            )

        high_entropy_mask = torch.zeros(gen_len, dtype=torch.bool)
        for pos in remasked_positions:
            local_pos = pos - prompt_len
            if 0 <= local_pos < gen_len:
                high_entropy_mask[local_pos] = True

        if not high_entropy_mask.any():
            return RefinementResult(
                original_tokens=base_tokens,
                refined_tokens=base_tokens,
                remasked_positions=[],
                n_remasked=0,
            )

        native_refine_steps = refine_steps or max(min_refine_steps * 8, 256)
        refined_gen = harness.remask_and_refine(
            prompt_ids_2d=result.prompt_tokens.to(harness.device).unsqueeze(0),
            generated_tokens=base_tokens[prompt_len:].cpu(),
            high_entropy_mask=high_entropy_mask,
            gen_len=gen_len,
            dream_steps=native_refine_steps,
        )
        refined_tokens = torch.cat(
            [result.prompt_tokens.cpu(), refined_gen.cpu()], dim=0
        )
        return RefinementResult(
            original_tokens=base_tokens,
            refined_tokens=refined_tokens,
            remasked_positions=remasked_positions,
            n_remasked=len(remasked_positions),
        )

    refined_x = base_tokens.clone()
    for pos in remasked_positions:
        refined_x[pos] = harness.mask_token_id

    still_masked = (refined_x == harness.mask_token_id).sum().item()
    if still_masked == 0:
        return RefinementResult(
            original_tokens=base_tokens,
            refined_tokens=base_tokens,
            remasked_positions=[],
            n_remasked=0,
        )

    if refine_steps is None:
        refine_steps = max(min_refine_steps, int(still_masked * steps_per_token))

    unmask_schedule = harness._build_schedule(int(still_masked), refine_steps)

    rng_cpu = torch.Generator(device="cpu")
    rng_sample = torch.Generator(device=harness.device)
    x = refined_x.to(harness.device)
    prompt_ids = result.prompt_tokens

    for step_idx, n_unmask in enumerate(unmask_schedule):
        logits = harness._forward(x.unsqueeze(0)).squeeze(0)

        gen_x = x[prompt_len:]
        gen_logits = logits[prompt_len:]

        positions = harness._pick_positions_to_unmask(
            x=gen_x,
            logits=gen_logits,
            n_to_unmask=n_unmask,
            order=refine_order,
            step=step_idx,
            total_steps=refine_steps,
            rng_cpu=rng_cpu,
        )

        for pos in positions:
            sampled = harness._sample_token_id(gen_logits[pos], rng_sample=rng_sample)
            gen_x[pos] = sampled

        x = torch.cat([prompt_ids.to(harness.device), gen_x])

    return RefinementResult(
        original_tokens=base_tokens,
        refined_tokens=x.detach().cpu(),
        remasked_positions=remasked_positions,
        n_remasked=len(remasked_positions),
    )


# ── ablation: varying entropy threshold ───────────────────────────────────────


class ThresholdAblation(NamedTuple):
    threshold: float
    n_flagged: int
    n_hallucinated_flagged: int  # requires ground-truth labels
    precision: float  # flagged ∩ hallucinated / flagged
    recall: float  # flagged ∩ hallucinated / all hallucinated


def ablate_threshold(
    result: ParallelPathResult,
    ground_truth_hall_positions: list[int],
    thresholds: list[float] | None = None,
) -> list[ThresholdAblation]:
    """
    Sweep entropy thresholds and compute precision/recall vs. RAGTruth labels.
    Use this to pick the best threshold before running refinement.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ent = result.token_entropy()
    gt_set = set(ground_truth_hall_positions)
    total_gt = len(gt_set)
    results = []

    for thr in thresholds:
        flagged_set = set((ent > thr).nonzero(as_tuple=True)[0].tolist())
        tp = len(flagged_set & gt_set)
        prec = tp / len(flagged_set) if flagged_set else 0.0
        rec = tp / total_gt if total_gt else 0.0
        results.append(
            ThresholdAblation(
                threshold=thr,
                n_flagged=len(flagged_set),
                n_hallucinated_flagged=tp,
                precision=prec,
                recall=rec,
            )
        )

    return results
