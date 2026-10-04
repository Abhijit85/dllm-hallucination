"""
Evaluation metrics for DLLM hallucination experiments.

Metrics
-------
1. Token-level F1       vs. RAGTruth word-level labels
2. Response-level AUC  for hallucination detection
3. FactScore            token overlap with retrieved passage
4. Disagreement-Hallucination correlation (Spearman ρ)
5. Refinement delta     hallucination rate before vs. after remasking
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.ragtruth_loader import RAGTruthSample
from strategies.parallel_remask import DisagreementReport, RefinementResult

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# ── 1. token-level F1 ─────────────────────────────────────────────────────────


@dataclass
class TokenF1Result:
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float  # average precision (handles class imbalance)


def token_level_f1(
    pred_tokens: torch.Tensor,  # (seq_len,) predicted token ids
    sample: RAGTruthSample,
    tokenizer: PreTrainedTokenizer,
    prompt_len: int,
) -> TokenF1Result:
    """
    Compare predicted generation against RAGTruth word-level labels.

    pred_tokens: includes prompt; we slice from prompt_len onwards
    """
    gen_tokens = pred_tokens[prompt_len:]
    gt_labels = sample.token_labels(tokenizer)  # (response_len,)

    # Align lengths (pad/trim to min)
    min_len = min(len(gen_tokens), len(gt_labels))
    gt_np = np.array(gt_labels[:min_len])

    # For predicted labels: a token is flagged as hallucination if it
    # does NOT appear in the source passage. Simple n-gram proxy.
    source_tokens = set(
        tokenizer(sample.source_info, add_special_tokens=False)["input_ids"]
    )
    gen_np = np.array(
        [0 if int(t) in source_tokens else 1 for t in gen_tokens[:min_len]]
    )

    if gt_np.sum() == 0:
        # No hallucinations in this sample; skip AUC
        return TokenF1Result(
            precision=float(precision_score(gt_np, gen_np, zero_division=0)),
            recall=float(recall_score(gt_np, gen_np, zero_division=0)),
            f1=float(f1_score(gt_np, gen_np, zero_division=0)),
            auc_roc=float("nan"),
            auc_pr=float("nan"),
        )

    return TokenF1Result(
        precision=float(precision_score(gt_np, gen_np, zero_division=0)),
        recall=float(recall_score(gt_np, gen_np, zero_division=0)),
        f1=float(f1_score(gt_np, gen_np, zero_division=0)),
        auc_roc=float(roc_auc_score(gt_np, gen_np)),
        auc_pr=float(average_precision_score(gt_np, gen_np)),
    )


# ── 2. FactScore (source grounding proxy) ─────────────────────────────────────


def fact_score(
    generated_text: str,
    source_text: str,
    n: int = 2,
) -> float:
    """
    Token n-gram overlap between generated response and source passage.
    A lightweight proxy for FactScore when a full NLI model is not available.

    Returns: fraction of generated n-grams present in source (0-1).
    """

    def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        tokens = text.lower().split()
        return set(zip(*[tokens[i:] for i in range(n)]))

    gen_ngrams = ngrams(generated_text, n)
    src_ngrams = ngrams(source_text, n)
    if not gen_ngrams:
        return 0.0
    return len(gen_ngrams & src_ngrams) / len(gen_ngrams)


# ── 3. disagreement-hallucination correlation ──────────────────────────────────


@dataclass
class CorrelationResult:
    spearman_rho: float
    p_value: float
    n_tokens: int
    skipped: bool


def disagreement_hallucination_correlation(
    report: DisagreementReport,
    sample: RAGTruthSample,
    tokenizer: PreTrainedTokenizer,
    prompt_len: int,
) -> CorrelationResult:
    """
    Measure Spearman ρ between token entropy and ground-truth hallucination labels.

    High ρ validates the hypothesis that disagreement predicts hallucination.
    """
    ent = report.token_entropy[prompt_len:].cpu().numpy()
    gt = np.array(sample.token_labels(tokenizer))

    min_len = min(len(ent), len(gt))
    ent = ent[:min_len]
    gt = gt[:min_len]

    if min_len == 0 or np.all(ent == ent[0]) or np.all(gt == gt[0]):
        return CorrelationResult(
            spearman_rho=float("nan"),
            p_value=float("nan"),
            n_tokens=min_len,
            skipped=True,
        )

    rho, pval = spearmanr(ent, gt)
    return CorrelationResult(
        spearman_rho=float(rho),
        p_value=float(pval),
        n_tokens=min_len,
        skipped=False,
    )


# ── 4. refinement delta ────────────────────────────────────────────────────────


@dataclass
class RefinementDelta:
    fact_score_before: float
    fact_score_after: float
    delta: float  # after - before (positive = improvement)
    tokens_changed: int
    total_gen_tokens: int
    change_rate: float  # tokens_changed / total_gen_tokens


def refinement_delta(
    original_text: str,
    refined_text: str,
    source_info: str,
    refinement_result: RefinementResult,
    prompt_len: int,
) -> RefinementDelta:
    fs_before = fact_score(original_text, source_info)
    fs_after = fact_score(refined_text, source_info)

    orig_gen = refinement_result.original_tokens[prompt_len:].detach().cpu()
    ref_gen = refinement_result.refined_tokens[prompt_len:].detach().cpu()
    min_len = min(len(orig_gen), len(ref_gen))
    changed = int((orig_gen[:min_len] != ref_gen[:min_len]).sum().item())

    return RefinementDelta(
        fact_score_before=fs_before,
        fact_score_after=fs_after,
        delta=fs_after - fs_before,
        tokens_changed=changed,
        total_gen_tokens=len(orig_gen),
        change_rate=changed / len(orig_gen) if len(orig_gen) > 0 else 0.0,
    )


# ── 5. aggregate across dataset ───────────────────────────────────────────────


@dataclass
class AggregateMetrics:
    n_samples: int
    mean_token_f1: float
    mean_fact_score_before: float
    mean_fact_score_after: float
    mean_refinement_delta: float
    mean_spearman_rho: float
    n_rho_computed: int
    mean_change_rate: float
    hallucinated_sample_fraction: float


def aggregate(records: list[dict]) -> AggregateMetrics:
    """
    records: list of per-sample dicts with keys matching the fields above.
    """

    def safe_mean(key: str) -> float:
        vals = [
            r[key] for r in records if r.get(key) is not None and not math.isnan(r[key])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    if not records:
        return AggregateMetrics(
            n_samples=0,
            mean_token_f1=float("nan"),
            mean_fact_score_before=float("nan"),
            mean_fact_score_after=float("nan"),
            mean_refinement_delta=float("nan"),
            mean_spearman_rho=float("nan"),
            n_rho_computed=0,
            mean_change_rate=float("nan"),
            hallucinated_sample_fraction=float("nan"),
        )

    return AggregateMetrics(
        n_samples=len(records),
        mean_token_f1=safe_mean("token_f1"),
        mean_fact_score_before=safe_mean("fact_score_before"),
        mean_fact_score_after=safe_mean("fact_score_after"),
        mean_refinement_delta=safe_mean("refinement_delta"),
        mean_spearman_rho=safe_mean("spearman_rho"),
        n_rho_computed=sum(1 for r in records if not r.get("rho_skipped", False)),
        mean_change_rate=safe_mean("change_rate"),
        hallucinated_sample_fraction=sum(
            1 for r in records if r.get("has_hallucination")
        )
        / len(records),
    )
