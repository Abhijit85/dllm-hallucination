"""
Correct H1 analysis: correlate LLaDA's per-token entropy with
LLaDA's own token-level source grounding, not RAGTruth labels
for a different model's response.

The bug in the original implementation
---------------------------------------
disagreement_hallucination_correlation() correlated:
  - LLaDA's entropy over positions 0..gen_len of LLaDA's own output
  - RAGTruth hallucination labels over positions 0..N of GPT-4/Llama-2's output

These are different texts. Any resulting rho is correlation noise
between unrelated arrays.

The correct measurement
------------------------
For each token i in LLaDA's generation:
  1. grounding_label[i] = 0 if token i appears in source_info
                       = 1 if not grounded (potential hallucination)
  2. Correlate entropy[i] with grounding_label[i] via Spearman rho

This is self-consistent: both arrays describe the same generated text.

Three grounding proxies (in increasing sophistication)
-------------------------------------------------------
  A. Token unigram: is token i in the source vocabulary?
  B. Bigram window: does any bigram containing token i appear in source?
  C. Sentence NLI:  does the sentence containing token i entail from source?
     (requires a separate NLI model and is not implemented here)

We implement A and B. B is more robust to common words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import PreTrainedTokenizer


@dataclass
class H1Result:
    spearman_rho_unigram: float
    spearman_rho_bigram: float
    p_value_unigram: float
    p_value_bigram: float
    n_tokens: int
    n_unfaithful_unigram: int
    n_unfaithful_bigram: int
    mean_entropy_unfaithful: float
    mean_entropy_faithful: float
    entropy_gap: float


def _tokenize_source(source_info: str, tokenizer: PreTrainedTokenizer) -> set[int]:
    """Token IDs present in the source passage."""
    encoded = tokenizer(source_info, add_special_tokens=False)
    return set(encoded["input_ids"])


def _bigrams_in_source(
    source_info: str,
    tokenizer: PreTrainedTokenizer,
) -> set[tuple[int, int]]:
    """All consecutive token-id bigrams present in the source passage."""
    ids = tokenizer(source_info, add_special_tokens=False)["input_ids"]
    return {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}


def token_source_grounding_labels(
    generated_ids: torch.Tensor,
    source_info: str,
    tokenizer: PreTrainedTokenizer,
    mode: str = "bigram",
) -> np.ndarray:
    """
    Binary label per generated token.
    0 = grounded
    1 = ungrounded
    """
    ids = generated_ids.detach().cpu().tolist()

    if mode == "unigram":
        src_vocab = _tokenize_source(source_info, tokenizer)
        return np.array(
            [0 if token_id in src_vocab else 1 for token_id in ids], dtype=int
        )

    if mode == "bigram":
        src_bigrams = _bigrams_in_source(source_info, tokenizer)
        labels = np.ones(len(ids), dtype=int)
        for i in range(len(ids) - 1):
            bigram = (ids[i], ids[i + 1])
            if bigram in src_bigrams:
                labels[i] = 0
                labels[i + 1] = 0
        return labels

    raise ValueError(f"Unknown mode: {mode}")


def compute_h1_correct(
    entropy: torch.Tensor,
    generated_ids: torch.Tensor,
    source_info: str,
    tokenizer: PreTrainedTokenizer,
    min_unfaithful_tokens: int = 3,
) -> H1Result | None:
    """
    Correlate LLaDA entropy with LLaDA's own source-grounding labels.

    Returns None when there are too few ungrounded tokens for a stable estimate.
    """
    ent = entropy.detach().cpu().numpy()
    min_len = min(len(ent), int(generated_ids.shape[0]))
    ent = ent[:min_len]
    gen_ids = generated_ids[:min_len]

    labels_uni = token_source_grounding_labels(
        gen_ids, source_info, tokenizer, "unigram"
    )
    labels_bi = token_source_grounding_labels(gen_ids, source_info, tokenizer, "bigram")

    n_unfaithful_uni = int(labels_uni.sum())
    n_unfaithful_bi = int(labels_bi.sum())

    if n_unfaithful_bi < min_unfaithful_tokens:
        return None

    if labels_uni.sum() > 0:
        rho_uni, p_uni = spearmanr(ent, labels_uni)
    else:
        rho_uni, p_uni = float("nan"), 1.0
    rho_bi, p_bi = spearmanr(ent, labels_bi)

    mask_unfaithful = labels_bi == 1
    mask_faithful = labels_bi == 0

    mean_ent_unfaithful = (
        float(ent[mask_unfaithful].mean()) if mask_unfaithful.any() else float("nan")
    )
    mean_ent_faithful = (
        float(ent[mask_faithful].mean()) if mask_faithful.any() else float("nan")
    )

    return H1Result(
        spearman_rho_unigram=float(rho_uni),
        spearman_rho_bigram=float(rho_bi),
        p_value_unigram=float(p_uni),
        p_value_bigram=float(p_bi),
        n_tokens=min_len,
        n_unfaithful_unigram=n_unfaithful_uni,
        n_unfaithful_bigram=n_unfaithful_bi,
        mean_entropy_unfaithful=mean_ent_unfaithful,
        mean_entropy_faithful=mean_ent_faithful,
        entropy_gap=mean_ent_unfaithful - mean_ent_faithful,
    )


def aggregate_h1_results(results: list[H1Result]) -> dict:
    """Aggregate H1Result objects across samples."""

    def mean(values):
        filtered = [value for value in values if not math.isnan(value)]
        return float(np.mean(filtered)) if filtered else float("nan")

    return {
        "n_samples": len(results),
        "mean_rho_bigram": mean([result.spearman_rho_bigram for result in results]),
        "mean_rho_unigram": mean([result.spearman_rho_unigram for result in results]),
        "mean_entropy_gap": mean([result.entropy_gap for result in results]),
        "pct_positive_rho": (
            sum(1 for result in results if result.spearman_rho_bigram > 0)
            / len(results)
            if results
            else float("nan")
        ),
        "mean_unfaithful_frac": mean(
            [result.n_unfaithful_bigram / max(1, result.n_tokens) for result in results]
        ),
    }
