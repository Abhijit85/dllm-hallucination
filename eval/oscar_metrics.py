"""
eval/oscar_metrics.py
=====================
All evaluation metrics for OSCAR (Ours) hallucination mitigation in LLaDA.

Three metric groups, each independently runnable:

  GROUP A — Generation quality  (Tables 1 / comparison rows)
    compute_generation_metrics(results)
    → EM, Token-F1, ROUGE-L, BLEU-4 per dataset + macro avg

  GROUP B — Detection quality   (Table 2)
    compute_detection_metrics(results, ragtruth_samples)
    → AUROC, Avg-Precision, Prec@τ, Rec@τ, F1@τ, Spearman ρ

  GROUP C — Refinement analysis (Table 3)
    compute_refinement_metrics(results, ragtruth_samples)
    → FactScore Δ, tokens changed, change rate, span reduction, intensity Δ

Standalone usage
----------------
  from eval.oscar_metrics import compute_generation_metrics, GenerationMetrics
  metrics = compute_generation_metrics(results)          # list[QASampleResult]

  from eval.oscar_metrics import compute_detection_metrics
  det = compute_detection_metrics(results, rt_samples)   # RAGTruth only

  from eval.oscar_metrics import compute_refinement_metrics
  ref = compute_refinement_metrics(results, rt_samples)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ══════════════════════════════════════════════════════════════════════════════
# Text normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    """SQuAD-style normalisation: lower, strip articles, remove punct, collapse ws."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_answer(generated: str, task: str = "open") -> str:
    """Extract answer span from raw generation text."""
    generated = generated.strip()
    if task == "mcq":
        m = re.match(r"^([A-Ea-e])[\.\)\:\-\s]?\s*(.*)", generated)
        if m:
            return m.group(2).strip() or generated
        return generated
    if "Answer:" in generated:
        generated = generated.split("Answer:")[-1].strip()
    for sep in ["\n", ". ", "! ", "? "]:
        if sep in generated:
            generated = generated.split(sep)[0].strip()
            break
    return generated


def _extract_choice_letter(generated: str) -> Optional[str]:
    m = re.search(r"\b([A-Ea-e])\b", generated)
    return m.group(1).upper() if m else None


# ══════════════════════════════════════════════════════════════════════════════
# Per-metric primitives  (all accept raw strings → return float)
# ══════════════════════════════════════════════════════════════════════════════

def exact_match(prediction: str, ground_truths: list[str], task: str = "open") -> float:
    """1.0 if normalised prediction matches any GT alias."""
    pred_norm = _normalise(prediction)
    for gt in ground_truths:
        if pred_norm == _normalise(gt):
            return 1.0
    if task == "mcq":
        letter = _extract_choice_letter(prediction)
        if letter and any(_normalise(gt).startswith(letter.lower()) for gt in ground_truths):
            return 1.0
    return 0.0


def _token_f1_single(prediction: str, gt: str) -> tuple[float, float, float]:
    pred_toks = _normalise(prediction).split()
    gt_toks   = _normalise(gt).split()
    if not pred_toks and not gt_toks:
        return 1.0, 1.0, 1.0
    if not pred_toks or not gt_toks:
        return 0.0, 0.0, 0.0
    common   = Counter(pred_toks) & Counter(gt_toks)
    n_same   = sum(common.values())
    if n_same == 0:
        return 0.0, 0.0, 0.0
    prec = n_same / len(pred_toks)
    rec  = n_same / len(gt_toks)
    return prec, rec, 2 * prec * rec / (prec + rec)


def token_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float, float]:
    """Best (precision, recall, F1) over all GT aliases."""
    return max(
        (_token_f1_single(prediction, gt) for gt in ground_truths),
        key=lambda t: t[2],
    )


_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=False
)


def rouge_scores(prediction: str, ground_truths: list[str]) -> dict[str, float]:
    """Max ROUGE-1/2/L F-measure over all GT aliases."""
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for gt in ground_truths:
        s = _rouge_scorer.score(gt, prediction)
        for k in best:
            best[k] = max(best[k], s[k].fmeasure)
    return best


def corpus_bleu(
    predictions: list[str],
    references_per_sample: list[list[str]],
) -> float:
    """Corpus-level BLEU-4 (sacrebleu). Returns 0-1 fraction."""
    if not predictions:
        return 0.0
    max_refs = max(len(r) for r in references_per_sample)
    padded   = [r + [r[0]] * (max_refs - len(r)) for r in references_per_sample]
    ref_cols = [[padded[i][j] for i in range(len(padded))] for j in range(max_refs)]
    bleu     = BLEU(effective_order=True)
    return bleu.corpus_score(predictions, ref_cols).score / 100.0


def fact_score_ngram(generated: str, source: str, n: int = 2) -> float:
    """Bigram overlap of generation with source (lightweight FactScore proxy)."""
    def ngrams(text: str, n: int):
        toks = text.lower().split()
        return set(zip(*[toks[i:] for i in range(n)]))
    gen_ng = ngrams(generated, n)
    src_ng = ngrams(source, n)
    return len(gen_ng & src_ng) / len(gen_ng) if gen_ng else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A — Generation quality
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleGenResult:
    """Per-sample generation quality result (stored for corpus-level BLEU)."""
    sample_id:   str
    dataset:     str
    task:        str

    em_before:     float;  em_after:     float
    f1_before:     float;  f1_after:     float
    rouge1_before: float;  rouge1_after: float
    rouge2_before: float;  rouge2_after: float
    rougeL_before: float;  rougeL_after: float

    pred_before: str
    pred_after:  str
    references:  list[str]

    n_flagged:    int   = 0
    n_remasked:   int   = 0
    spearman_rho: float = float("nan")


def evaluate_generation_sample(
    sample_id:    str,
    dataset:      str,
    task:         str,
    pred_before:  str,
    pred_after:   str,
    ground_truths: list[str],
) -> SampleGenResult:
    """Compute per-sample generation metrics for one (before, after) pair."""
    pb = _extract_answer(pred_before, task)
    pa = _extract_answer(pred_after,  task)

    em_b  = exact_match(pb, ground_truths, task)
    em_a  = exact_match(pa, ground_truths, task)
    _, _, f1_b = token_f1(pb, ground_truths)
    _, _, f1_a = token_f1(pa, ground_truths)
    r_b   = rouge_scores(pb, ground_truths)
    r_a   = rouge_scores(pa, ground_truths)

    return SampleGenResult(
        sample_id=sample_id, dataset=dataset, task=task,
        em_before=em_b,          em_after=em_a,
        f1_before=f1_b,          f1_after=f1_a,
        rouge1_before=r_b["rouge1"], rouge1_after=r_a["rouge1"],
        rouge2_before=r_b["rouge2"], rouge2_after=r_a["rouge2"],
        rougeL_before=r_b["rougeL"], rougeL_after=r_a["rougeL"],
        pred_before=pb, pred_after=pa, references=ground_truths,
    )


@dataclass
class DatasetGenMetrics:
    dataset:  str
    n_samples: int
    em_before: float;     em_after: float;     em_delta: float
    f1_before: float;     f1_after: float;     f1_delta: float
    rouge1_before: float; rouge1_after: float; rouge1_delta: float
    rouge2_before: float; rouge2_after: float; rouge2_delta: float
    rougeL_before: float; rougeL_after: float; rougeL_delta: float
    bleu_before: float;   bleu_after: float;   bleu_delta: float


def aggregate_generation(
    results: list[SampleGenResult],
    dataset_name: str,
) -> DatasetGenMetrics:
    """Aggregate per-sample → dataset-level generation metrics."""
    def mn(vals): return float(np.mean(vals)) if vals else float("nan")

    em_b  = mn([r.em_before    for r in results])
    em_a  = mn([r.em_after     for r in results])
    f1_b  = mn([r.f1_before    for r in results])
    f1_a  = mn([r.f1_after     for r in results])
    r1_b  = mn([r.rouge1_before for r in results])
    r1_a  = mn([r.rouge1_after  for r in results])
    r2_b  = mn([r.rouge2_before for r in results])
    r2_a  = mn([r.rouge2_after  for r in results])
    rL_b  = mn([r.rougeL_before for r in results])
    rL_a  = mn([r.rougeL_after  for r in results])
    bl_b  = corpus_bleu([r.pred_before for r in results], [r.references for r in results])
    bl_a  = corpus_bleu([r.pred_after  for r in results], [r.references for r in results])

    return DatasetGenMetrics(
        dataset=dataset_name, n_samples=len(results),
        em_before=em_b,     em_after=em_a,     em_delta=em_a - em_b,
        f1_before=f1_b,     f1_after=f1_a,     f1_delta=f1_a - f1_b,
        rouge1_before=r1_b, rouge1_after=r1_a, rouge1_delta=r1_a - r1_b,
        rouge2_before=r2_b, rouge2_after=r2_a, rouge2_delta=r2_a - r2_b,
        rougeL_before=rL_b, rougeL_after=rL_a, rougeL_delta=rL_a - rL_b,
        bleu_before=bl_b,   bleu_after=bl_a,   bleu_delta=bl_a - bl_b,
    )


def compute_generation_metrics(
    results_by_dataset: dict[str, list[SampleGenResult]],
) -> dict[str, DatasetGenMetrics]:
    """
    Standalone entry point — GROUP A.

    Args:
        results_by_dataset: {dataset_name: [SampleGenResult, ...]}

    Returns:
        dict with one DatasetGenMetrics per dataset + "macro_qa" + "macro_all"
    """
    out: dict[str, DatasetGenMetrics] = {}
    for ds, results in results_by_dataset.items():
        if results:
            out[ds] = aggregate_generation(results, ds)

    # QA macro (TriviaQA + CommonsenseQA + HotpotQA)
    qa_keys = [k for k in out if k in {"triviaqa", "commonsenseqa", "hotpotqa"}]
    if qa_keys:
        out["macro_qa"] = _macro_gen([out[k] for k in qa_keys], label="macro_qa")

    # Overall macro (all datasets)
    if out:
        all_keys = [k for k in out if not k.startswith("macro")]
        out["macro_all"] = _macro_gen([out[k] for k in all_keys], label="macro_all")

    return out


def _macro_gen(metrics: list[DatasetGenMetrics], label: str) -> DatasetGenMetrics:
    def avg(attr): return float(np.mean([getattr(m, attr) for m in metrics]))
    fields = [
        "em_before","em_after","em_delta",
        "f1_before","f1_after","f1_delta",
        "rouge1_before","rouge1_after","rouge1_delta",
        "rouge2_before","rouge2_after","rouge2_delta",
        "rougeL_before","rougeL_after","rougeL_delta",
        "bleu_before","bleu_after","bleu_delta",
    ]
    return DatasetGenMetrics(
        dataset=label,
        n_samples=sum(m.n_samples for m in metrics),
        **{f: avg(f) for f in fields},
    )


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B — Detection quality  (RAGTruth only)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionMetrics:
    """Hallucination detection metrics for one RAGTruth subset."""
    subset:           str      # "QA" | "Summary" | "Data2txt" | "all"
    n_samples:        int
    hall_rate:        float    # fraction of samples with >= 1 span

    auroc:            float    # threshold-free
    avg_precision:    float    # average precision (PR-AUC proxy)

    precision_at_tau: float    # at operating threshold τ
    recall_at_tau:    float
    f1_at_tau:        float

    spearman_rho:     float    # entropy vs binary hallucination labels
    spearman_p:       float

    # Raw arrays stored for PR-curve plotting
    entropies:  np.ndarray = field(default_factory=lambda: np.array([]))
    labels:     np.ndarray = field(default_factory=lambda: np.array([]))


def compute_detection_metrics(
    entropy_label_pairs: list[tuple[np.ndarray, np.ndarray]],
    subset_tags: list[str],
    tau: float = 0.5,
) -> dict[str, DetectionMetrics]:
    """
    Standalone entry point — GROUP B.

    Args:
        entropy_label_pairs: list of (token_entropy_array, binary_label_array)
                             one per RAGTruth sample, arrays already prompt-stripped
        subset_tags:         task_type string per sample ("QA"|"Summary"|"Data2txt")
        tau:                 entropy threshold used for flagging

    Returns:
        dict keyed by subset name + "all"

    Example
    -------
    pairs = []
    tags  = []
    for sample, report in zip(ragtruth_samples, reports):
        ent = report.token_entropy[prompt_len:].cpu().numpy()
        lbl = np.array(sample.token_labels(tokenizer))
        min_len = min(len(ent), len(lbl))
        pairs.append((ent[:min_len], lbl[:min_len]))
        tags.append(sample.task_type)

    det = compute_detection_metrics(pairs, tags, tau=0.5)
    """
    from collections import defaultdict
    subsets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for (ent, lbl), tag in zip(entropy_label_pairs, subset_tags):
        subsets[tag].append((ent, lbl))
    subsets["all"] = list(entropy_label_pairs)

    out: dict[str, DetectionMetrics] = {}
    for subset, pairs in subsets.items():
        ent_cat = np.concatenate([p[0] for p in pairs])
        lbl_cat = np.concatenate([p[1] for p in pairs])

        # Sample-level hall rate
        hall_rate = float(np.mean([lbl.max() for _, lbl in pairs]))

        # Binary predictions at τ
        pred = (ent_cat > tau).astype(int)

        has_pos = lbl_cat.sum() > 0
        auroc = float(roc_auc_score(lbl_cat, ent_cat)) if has_pos else float("nan")
        ap    = float(average_precision_score(lbl_cat, ent_cat)) if has_pos else float("nan")
        prec  = float(precision_score(lbl_cat, pred, zero_division=0))
        rec   = float(recall_score(lbl_cat, pred, zero_division=0))
        f1t   = float(f1_score(lbl_cat, pred, zero_division=0))

        rho, pval = spearmanr(ent_cat, lbl_cat)

        out[subset] = DetectionMetrics(
            subset=subset,
            n_samples=len(pairs),
            hall_rate=hall_rate,
            auroc=auroc,
            avg_precision=ap,
            precision_at_tau=prec,
            recall_at_tau=rec,
            f1_at_tau=f1t,
            spearman_rho=float(rho),
            spearman_p=float(pval),
            entropies=ent_cat,
            labels=lbl_cat,
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — Refinement analysis  (RAGTruth only)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleRefResult:
    """Per-sample refinement metrics (RAGTruth)."""
    sample_id:          str
    subset:             str
    tokens_changed:     int
    total_gen_tokens:   int
    change_rate:        float
    fact_score_before:  float
    fact_score_after:   float
    fact_score_delta:   float
    spans_before:       int
    spans_after:        int      # estimated from flagged positions
    span_reduction_pct: float
    intensity_before:   float
    intensity_after:    float
    intensity_delta:    float


@dataclass
class SubsetRefMetrics:
    subset: str
    n_samples: int
    tokens_changed_mean:  float
    change_rate_mean:     float
    fact_score_before:    float
    fact_score_after:     float
    fact_score_delta:     float
    span_reduction_pct:   float
    intensity_delta_mean: float


def compute_refinement_metrics(
    sample_ref_results: list[SampleRefResult],
) -> dict[str, SubsetRefMetrics]:
    """
    Standalone entry point — GROUP C.

    Args:
        sample_ref_results: list of SampleRefResult, one per RAGTruth sample.
                            Build these in the runner using build_ref_result().

    Returns:
        dict keyed by subset name + "all"
    """
    from collections import defaultdict
    subsets: dict[str, list[SampleRefResult]] = defaultdict(list)
    for r in sample_ref_results:
        subsets[r.subset].append(r)
    subsets["all"] = list(sample_ref_results)

    out: dict[str, SubsetRefMetrics] = {}
    for subset, recs in subsets.items():
        def mn(attr): return float(np.mean([getattr(r, attr) for r in recs]))
        out[subset] = SubsetRefMetrics(
            subset=subset,
            n_samples=len(recs),
            tokens_changed_mean  = mn("tokens_changed"),
            change_rate_mean     = mn("change_rate") * 100,
            fact_score_before    = mn("fact_score_before") * 100,
            fact_score_after     = mn("fact_score_after")  * 100,
            fact_score_delta     = mn("fact_score_delta")  * 100,
            span_reduction_pct   = mn("span_reduction_pct"),
            intensity_delta_mean = mn("intensity_delta"),
        )
    return out


def build_ref_result(
    sample_id:          str,
    subset:             str,
    original_text:      str,
    refined_text:       str,
    source_info:        str,
    original_tokens,               # torch.Tensor  (seq_len,)
    refined_tokens,                # torch.Tensor  (seq_len,)
    prompt_len:         int,
    hall_spans:         list[dict],
    remasked_positions: list[int],
) -> SampleRefResult:
    """
    Build one SampleRefResult from raw pipeline outputs.
    Call this inside the experiment runner per RAGTruth sample.
    """
    import torch
    fs_b = fact_score_ngram(original_text, source_info)
    fs_a = fact_score_ngram(refined_text,  source_info)

    orig_gen = original_tokens[prompt_len:]
    ref_gen  = refined_tokens[prompt_len:]
    min_len  = min(len(orig_gen), len(ref_gen))
    changed  = int((orig_gen[:min_len] != ref_gen[:min_len]).sum().item())

    n_spans_before = len(hall_spans)
    # Estimate post-refinement span reduction: what fraction of hallucinated
    # span tokens were covered by remasked positions
    span_tokens = set()
    for sp in hall_spans:
        span_tokens.update(range(sp["start"], sp["end"]))
    remasked_in_spans = len(span_tokens & set(remasked_positions))
    span_reduction = (remasked_in_spans / len(span_tokens) * 100) if span_tokens else 0.0

    int_before = float(np.mean([sp.get("intensity", 1.0) for sp in hall_spans])) if hall_spans else 0.0
    int_after  = int_before * (1 - span_reduction / 100)  # conservative estimate

    return SampleRefResult(
        sample_id=sample_id,
        subset=subset,
        tokens_changed=changed,
        total_gen_tokens=len(orig_gen),
        change_rate=changed / max(len(orig_gen), 1),
        fact_score_before=fs_b,
        fact_score_after=fs_a,
        fact_score_delta=fs_a - fs_b,
        spans_before=n_spans_before,
        spans_after=max(0, n_spans_before - int(n_spans_before * span_reduction / 100)),
        span_reduction_pct=span_reduction,
        intensity_before=int_before,
        intensity_after=int_after,
        intensity_delta=int_after - int_before,
    )
