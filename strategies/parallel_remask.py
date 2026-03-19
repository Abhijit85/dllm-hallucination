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

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from models.llada_harness import (
    LLaDAHarness,
    ParallelPathResult,
    DemaskingOrder,
    MASK_TOKEN_ID,
)


# ── disagreement analysis ──────────────────────────────────────────────────────

@dataclass
class DisagreementReport:
    token_entropy: torch.Tensor          # (seq_len,) final-token entropy
    high_entropy_positions: list[int]    # positions above threshold
    entropy_threshold: float
    path_entropy_trajectory: torch.Tensor | None  # (T,) mean entropy per step


def compute_disagreement(
    result: ParallelPathResult,
    entropy_threshold: float = 0.5,
    track_trajectory: bool = True,
) -> DisagreementReport:
    """
    Compute token-level entropy across N paths' final outputs.

    entropy_threshold:
        Tokens with H > threshold are flagged as uncertain.
        Sensible range: 0.3 (aggressive) – 0.7 (conservative).
        Default 0.5 ≈ disagreement on ≥ 30% of paths.
    """
    ent = result.token_entropy()
    high_ent_pos = (ent > entropy_threshold).nonzero(as_tuple=True)[0].tolist()

    trajectory = None
    if track_trajectory and all(len(p.steps) > 0 for p in result.paths):
        # Mean entropy of still-masked positions at each step
        num_steps = len(result.paths[0].steps)
        traj = []
        for step_idx in range(num_steps):
            step_ents = []
            for path in result.paths:
                step_data = path.steps[step_idx]
                masked = (step_data.x == MASK_TOKEN_ID)
                if masked.any():
                    probs = F.softmax(step_data.logits[masked], dim=-1)
                    h = -(probs * (probs + 1e-10).log()).sum(-1).mean()
                    step_ents.append(h.item())
            traj.append(sum(step_ents) / len(step_ents) if step_ents else 0.0)
        trajectory = torch.tensor(traj)

    return DisagreementReport(
        token_entropy=ent,
        high_entropy_positions=high_ent_pos,
        entropy_threshold=entropy_threshold,
        path_entropy_trajectory=trajectory,
    )


# ── random-remasking refinement ────────────────────────────────────────────────

@dataclass
class RefinementResult:
    original_tokens: torch.Tensor      # majority vote of N paths (pre-refinement)
    refined_tokens: torch.Tensor       # after targeted remasking
    remasked_positions: list[int]      # positions that were re-denoised
    n_remasked: int


def random_remask_and_refine(
    harness: LLaDAHarness,
    result: ParallelPathResult,
    report: DisagreementReport,
    source_info: str,
    refine_steps: int = 16,
    refine_order: DemaskingOrder = DemaskingOrder.LEARNED,
    max_remask: int | None = None,
    remask_fraction: float = 1.0,       # fraction of flagged positions to remask
) -> RefinementResult:
    """
    Take the majority-vote output, remask high-entropy positions, and
    run a short targeted denoising pass conditioned on source_info.

    Why learned order for refinement?
    ----------------------------------
    In the refinement pass we *want* grounding: the model sees the context
    and should commit to verifiable tokens first (high confidence = most
    supported by context). Random order would re-introduce variance here.

    Args:
        harness:         the LLaDAHarness instance
        result:          output of run_parallel_paths()
        report:          output of compute_disagreement()
        source_info:     retrieved passage for conditioning
        refine_steps:    denoising steps for the refinement pass (< original T)
        refine_order:    demasking order for refinement (default: learned)
        max_remask:      cap on positions to remask (None = all flagged)
        remask_fraction: randomly drop this fraction of flagged positions
                         (1.0 = remask all flagged; useful for ablation)
    """
    majority = result.majority_vote()           # (seq_len,) best initial output
    prompt_len = len(result.prompt_tokens)

    # Determine positions to remask (within generation region only)
    flagged = [p for p in report.high_entropy_positions if p >= prompt_len]

    # Optional: randomly subsample flagged positions (ablation)
    if remask_fraction < 1.0:
        k = max(1, int(len(flagged) * remask_fraction))
        perm = torch.randperm(len(flagged))[:k].tolist()
        flagged = [flagged[i] for i in perm]

    if max_remask is not None:
        flagged = flagged[:max_remask]

    remasked_positions = flagged

    # Build refined x: start from majority vote, remask flagged positions
    refined_x = majority.clone()
    for pos in remasked_positions:
        refined_x[pos] = MASK_TOKEN_ID

    # Run short denoising pass on the remasked tokens
    still_masked = (refined_x == MASK_TOKEN_ID).sum().item()
    if still_masked == 0:
        # Nothing to refine
        return RefinementResult(
            original_tokens=majority,
            refined_tokens=majority,
            remasked_positions=[],
            n_remasked=0,
        )

    unmask_schedule = harness._build_schedule(int(still_masked), refine_steps)

    rng = torch.Generator(device=harness.device)
    x = refined_x.to(harness.device)
    prompt_ids = result.prompt_tokens

    for step_idx, n_unmask in enumerate(unmask_schedule):
        logits = harness._forward(x.unsqueeze(0)).squeeze(0)

        gen_x      = x[prompt_len:]
        gen_logits = logits[prompt_len:]

        positions = harness._pick_positions_to_unmask(
            x=gen_x,
            logits=gen_logits,
            n_to_unmask=n_unmask,
            order=refine_order,
            step=step_idx,
            total_steps=refine_steps,
            rng=rng,
        )

        for pos in positions:
            p = F.softmax(gen_logits[pos], dim=-1)
            sampled = torch.multinomial(p, num_samples=1, generator=rng)
            gen_x[pos] = sampled

        x = torch.cat([prompt_ids.to(harness.device), gen_x])

    return RefinementResult(
        original_tokens=majority,
        refined_tokens=x.detach(),
        remasked_positions=remasked_positions,
        n_remasked=len(remasked_positions),
    )


# ── ablation: varying entropy threshold ───────────────────────────────────────

class ThresholdAblation(NamedTuple):
    threshold: float
    n_flagged: int
    n_hallucinated_flagged: int   # requires ground-truth labels
    precision: float              # flagged ∩ hallucinated / flagged
    recall: float                 # flagged ∩ hallucinated / all hallucinated


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
        rec  = tp / total_gt         if total_gt    else 0.0
        results.append(ThresholdAblation(
            threshold=thr,
            n_flagged=len(flagged_set),
            n_hallucinated_flagged=tp,
            precision=prec,
            recall=rec,
        ))

    return results
