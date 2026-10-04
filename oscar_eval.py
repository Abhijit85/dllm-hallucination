"""
OSCAR unified-judge evaluation utilities.

Covers:
1. Judge prompt and robust label parsing.
2. AUROC against unified judge labels.
3. Per-method bootstrap confidence intervals and paired bootstrap differences.
4. Inter-rater agreement via Cohen's and Fleiss' kappa.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


JUDGE_PROMPT_TEMPLATE = """You are a strict factuality grader. You are given a QUESTION, a REFERENCE answer (the gold answer), and a CANDIDATE answer produced by an AI system. Judge ONLY whether the CANDIDATE is factually correct with respect to the QUESTION and REFERENCE.

Rules:
- The CANDIDATE need not match the REFERENCE word-for-word. It is CORRECT if it is semantically equivalent and contains the key fact(s) the REFERENCE expresses.
- A paraphrase, alias, or alternative valid surface form of the correct answer is CORRECT.
- If the CANDIDATE contains the correct fact but adds an unsupported or wrong extra claim, label PARTIAL.
- If the CANDIDATE states a wrong fact, contradicts the REFERENCE, or answers a different question, label INCORRECT.
- A refusal, empty answer, or "I don't know" is INCORRECT.
- If the QUESTION is genuinely ambiguous and the CANDIDATE gives any one valid answer, label CORRECT.
- Do not reward fluency or length. Judge facts only.

Respond with EXACTLY one word on a single line: CORRECT, PARTIAL, or INCORRECT.

QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}
"""


FLUENCY_PROMPT_TEMPLATE = """Rate ONLY the fluency and coherence of the following TEXT, ignoring whether it is factually correct. Look for grammatical errors, broken phrases, abrupt discontinuities, or repeated/garbled tokens.
Respond with EXACTLY one word: FLUENT, MINOR_ISSUES, or DISFLUENT.

TEXT:
{candidate}
"""


def call_judge(prompt: str) -> str:
    """Stub for a deterministic judge client integration."""
    raise NotImplementedError("Wire this to your GPT-4o / second-judge client.")


def parse_label(raw: str) -> str:
    """Return CORRECT, PARTIAL, INCORRECT, or UNKNOWN from a raw judge string."""
    text = (raw or "").strip().upper()
    for label in ("INCORRECT", "PARTIAL", "CORRECT"):
        if label in text:
            return label
    return "UNKNOWN"


def binarize(
    labels: Iterable[str],
    partial_positive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map judge labels to hallucinated=1 / not hallucinated=0.

    Returns:
      y: label array
      keep_mask: True for rows kept in the evaluation
    """
    y = []
    keep = []
    for label in labels:
        if label == "CORRECT":
            y.append(0)
            keep.append(True)
        elif label == "INCORRECT":
            y.append(1)
            keep.append(True)
        elif label == "PARTIAL":
            if partial_positive:
                y.append(1)
                keep.append(True)
            else:
                y.append(0)
                keep.append(False)
        else:
            y.append(0)
            keep.append(False)
    return np.asarray(y, dtype=int), np.asarray(keep, dtype=bool)


def auroc(scores: Sequence[float], y: Sequence[int]) -> float:
    """Compute AUROC for detection scores against binary labels."""
    return float(roc_auc_score(np.asarray(y, dtype=int), np.asarray(scores, dtype=float)))


def bootstrap_auroc_ci(
    scores: Sequence[float],
    y: Sequence[int],
    B: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a single method's AUROC."""
    scores_arr = np.asarray(scores, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    _validate_binary_labels(y_arr)

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if np.unique(y_arr[idx]).size < 2:
            continue
        vals.append(roc_auc_score(y_arr[idx], scores_arr[idx]))
    if not vals:
        raise ValueError("Bootstrap produced no valid resamples with both label classes.")

    vals_arr = np.asarray(vals, dtype=float)
    return (
        auroc(scores_arr, y_arr),
        float(np.percentile(vals_arr, 100 * alpha / 2)),
        float(np.percentile(vals_arr, 100 * (1 - alpha / 2))),
    )


def paired_bootstrap_diff(
    score_a: Sequence[float],
    score_b: Sequence[float],
    y: Sequence[int],
    B: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """
    Paired bootstrap on AUROC(score_a) - AUROC(score_b).

    Uses the same resampled indices for both methods on each bootstrap draw.
    Significance is determined by whether the CI excludes 0.
    """
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if a.shape != b.shape or a.shape != y_arr.shape:
        raise ValueError("score_a, score_b, and y must have the same length.")
    _validate_binary_labels(y_arr)

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if np.unique(y_arr[idx]).size < 2:
            continue
        diffs.append(roc_auc_score(y_arr[idx], a[idx]) - roc_auc_score(y_arr[idx], b[idx]))
    if not diffs:
        raise ValueError("Bootstrap produced no valid resamples with both label classes.")

    diffs_arr = np.asarray(diffs, dtype=float)
    lo = float(np.percentile(diffs_arr, 100 * alpha / 2))
    hi = float(np.percentile(diffs_arr, 100 * (1 - alpha / 2)))
    p_two_sided = float(min(1.0, 2 * min((diffs_arr <= 0).mean(), (diffs_arr >= 0).mean())))
    obs = auroc(a, y_arr) - auroc(b, y_arr)
    return {
        "diff": float(obs),
        "lo": lo,
        "hi": hi,
        "p": p_two_sided,
        "significant": bool(lo > 0 or hi < 0),
    }


def cohen(
    a: Sequence[str | int],
    b: Sequence[str | int],
    weighted: str | None = None,
) -> float:
    """Cohen's kappa for two raters."""
    return float(cohen_kappa_score(a, b, weights=weighted))


def fleiss(
    label_matrix: Sequence[Sequence[str]],
    categories: Sequence[str] = ("CORRECT", "PARTIAL", "INCORRECT"),
) -> float:
    """Fleiss' kappa for more than two raters."""
    if not label_matrix:
        raise ValueError("label_matrix must not be empty.")

    row_lengths = {len(row) for row in label_matrix}
    if len(row_lengths) != 1:
        raise ValueError("Each item must have the same number of rater labels.")
    m = row_lengths.pop()
    if m < 2:
        raise ValueError("Fleiss' kappa requires at least two ratings per item.")

    cats = list(categories)
    index = {label: j for j, label in enumerate(cats)}
    counts = np.zeros((len(label_matrix), len(cats)), dtype=float)
    for i, row in enumerate(label_matrix):
        for label in row:
            if label not in index:
                raise ValueError(f"Unknown label {label!r}; expected one of {cats}.")
            counts[i, index[label]] += 1

    P_i = ((counts**2).sum(axis=1) - m) / (m * (m - 1))
    P_bar = float(P_i.mean())
    p_j = counts.sum(axis=0) / (len(label_matrix) * m)
    P_e = float((p_j**2).sum())
    if P_e == 1.0:
        raise ValueError("Fleiss' kappa is undefined when expected agreement is 1.")
    return float((P_bar - P_e) / (1 - P_e))


def _validate_binary_labels(y: np.ndarray) -> None:
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if np.unique(y).size < 2:
        raise ValueError("y must contain both label classes for AUROC.")


__all__ = [
    "FLUENCY_PROMPT_TEMPLATE",
    "JUDGE_PROMPT_TEMPLATE",
    "auroc",
    "binarize",
    "bootstrap_auroc_ci",
    "call_judge",
    "cohen",
    "fleiss",
    "paired_bootstrap_diff",
    "parse_label",
]
