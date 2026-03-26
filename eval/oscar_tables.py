"""
eval/oscar_tables.py
====================
LaTeX + markdown rendering for all OSCAR paper tables.

  Table 1 — Generation quality     (EM / F1 / ROUGE-L / BLEU, 4 datasets)
  Table 2 — Detection quality      (AUROC / AP / Prec / Rec / F1 / ρ, RAGTruth)
  Table 3 — Refinement analysis    (FactScore / span reduction, RAGTruth)
  Table 4 — Baseline comparison    (OSCAR vs. 4 baselines)
  Table 5 — Demasking order        (5 order configs)

Each table function accepts the relevant dataclass objects from oscar_metrics.py
and returns a (latex_str, markdown_str, pandas_df) tuple.

Standalone usage
----------------
  from eval.oscar_tables import render_table1
  latex, md, df = render_table1(generation_metrics_dict)
  print(latex)
  df.to_csv("table1.csv")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.oscar_metrics import (
    DatasetGenMetrics,
    DetectionMetrics,
    SubsetRefMetrics,
)
from eval.oscar_baselines import (
    METHOD_VANILLA, METHOD_EXTRA_STEPS,
    METHOD_RANDOM_REMASK, METHOD_SELFCHECK, METHOD_OSCAR,
    ORDER_ALL_LEARNED, ORDER_ALL_RANDOM,
    ORDER_HYBRID_50, ORDER_ENTROPY_ORDER, ORDER_OSCAR,
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.2f}"


def _pct_delta(v: float) -> str:
    """Format a delta value with +/- sign and colour commands."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    pct = v * 100
    sign = "+" if pct >= 0 else ""
    s = f"{sign}{pct:.2f}"
    if pct > 0.05:
        return rf"\textcolor{{green!55!black}}{{\textbf{{{s}}}}}"
    elif pct < -0.05:
        return rf"\textcolor{{red!65!black}}{{{s}}}"
    return s


def _float_delta(v: float) -> str:
    """Format a non-percentage delta (e.g. Spearman rho, intensity)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else ""
    s = f"{sign}{v:.3f}"
    if v > 0.001:
        return rf"\textcolor{{green!55!black}}{{\textbf{{{s}}}}}"
    elif v < -0.001:
        return rf"\textcolor{{red!65!black}}{{{s}}}"
    return s


def _booktabs_wrap(
    label: str,
    caption: str,
    col_fmt: str,
    header_rows: list[str],
    body_rows: list[str],
    notes: str = "",
) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        *header_rows,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if notes:
        lines += [
            r"\vspace{2pt}",
            rf"\footnotesize\textit{{Note:}} {notes}",
        ]
    lines.append(r"\end{table*}")
    return "\n".join(lines)


DATASET_DISPLAY = {
    "triviaqa":      "TriviaQA",
    "commonsenseqa": "CommonsenseQA",
    "hotpotqa":      "HotpotQA",
    "ragtruth":      "RAGTruth",
    "macro_qa":      "QA macro avg",
    "macro_all":     "Overall macro avg",
}


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Generation quality
# ══════════════════════════════════════════════════════════════════════════════

def render_table1(
    gen_metrics: dict[str, DatasetGenMetrics],
    n_paths: int = 8, tau: float = 0.5, refine_steps: int = 16,
) -> tuple[str, str, pd.DataFrame]:
    """
    Render Table 1: Generation quality across all 4 datasets.
    Returns (latex, markdown, dataframe).
    """
    DATASET_ORDER = [
        "triviaqa", "commonsenseqa", "hotpotqa",
        "ragtruth",
        "macro_qa", "macro_all",
    ]
    QA_DATASETS  = {"triviaqa", "commonsenseqa", "hotpotqa"}

    rows_latex = []
    rows_df    = []

    def _row(ds_key: str, bold: bool = False) -> tuple[str, dict] | None:
        dm = gen_metrics.get(ds_key)
        if dm is None:
            return None

        na     = r"\textit{n/a}"
        em_b   = na   if ds_key == "ragtruth" else _pct(dm.em_before)
        em_a   = na   if ds_key == "ragtruth" else _pct(dm.em_after)
        em_d   = "—"  if ds_key == "ragtruth" else _pct_delta(dm.em_delta)

        name = DATASET_DISPLAY.get(ds_key, ds_key)
        if bold:
            name = rf"\textbf{{{name}}}"
            fmt = lambda v: rf"\textbf{{{v}}}"
        else:
            fmt = lambda v: v

        cells = [
            name,
            str(dm.n_samples),
            fmt(em_b),     fmt(em_a),      em_d,
            fmt(_pct(dm.f1_before)),      fmt(_pct(dm.f1_after)),      _pct_delta(dm.f1_delta),
            fmt(_pct(dm.rougeL_before)),  fmt(_pct(dm.rougeL_after)),  _pct_delta(dm.rougeL_delta),
            fmt(_pct(dm.bleu_before)),    fmt(_pct(dm.bleu_after)),     _pct_delta(dm.bleu_delta),
        ]
        latex_row = " & ".join(cells) + r" \\"

        df_row = {
            "Dataset": DATASET_DISPLAY.get(ds_key, ds_key),
            "N": dm.n_samples,
            "EM (bef.)": None if ds_key == "ragtruth" else round(dm.em_before * 100, 2),
            "EM (aft.)": None if ds_key == "ragtruth" else round(dm.em_after  * 100, 2),
            "EM Δ":      None if ds_key == "ragtruth" else round(dm.em_delta  * 100, 2),
            "F1 (bef.)":      round(dm.f1_before   * 100, 2),
            "F1 (aft.)":      round(dm.f1_after    * 100, 2),
            "F1 Δ":           round(dm.f1_delta    * 100, 2),
            "R-L (bef.)":     round(dm.rougeL_before * 100, 2),
            "R-L (aft.)":     round(dm.rougeL_after  * 100, 2),
            "R-L Δ":          round(dm.rougeL_delta  * 100, 2),
            "BLEU (bef.)":    round(dm.bleu_before * 100, 2),
            "BLEU (aft.)":    round(dm.bleu_after  * 100, 2),
            "BLEU Δ":         round(dm.bleu_delta  * 100, 2),
        }
        return latex_row, df_row

    # Group header
    rows_latex.append(r"\multicolumn{14}{l}{\textit{Short-answer QA}} \\")
    for ds in ["triviaqa", "commonsenseqa", "hotpotqa"]:
        r = _row(ds)
        if r:
            rows_latex.append(r[0]); rows_df.append(r[1])

    rows_latex.append(r"\cmidrule(lr){1-1}")
    rows_latex.append(r"\multicolumn{14}{l}{\textit{Grounded generation (RAG)}} \\")
    for ds in ["ragtruth"]:
        r = _row(ds)
        if r:
            rows_latex.append(r[0]); rows_df.append(r[1])

    rows_latex.append(r"\midrule")
    for ds in ["macro_qa", "macro_all"]:
        r = _row(ds, bold=(ds == "macro_all"))
        if r:
            rows_latex.append(r[0]); rows_df.append(r[1])

    col_fmt    = "l r " + " ".join(["rrr"] * 4)
    hdr1_cols  = r"\multicolumn{3}{c}{\textbf{EM}} & \multicolumn{3}{c}{\textbf{F1}} & \multicolumn{3}{c}{\textbf{ROUGE-L}} & \multicolumn{3}{c}{\textbf{BLEU-4}}"
    hdr_rows   = [
        rf"\textbf{{Dataset}} & \textbf{{N}} & {hdr1_cols} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
        r" & & \textit{Bef.} & \textit{Aft.} & $\Delta$ " * 4 + r"\\",
    ]

    caption = (
        f"Generation quality across all four benchmarks. "
        f"OSCAR uses $N={n_paths}$ parallel paths, $\\tau={tau}$, "
        f"{refine_steps} refinement steps. "
        r"EM is undefined for RAGTruth (long-form generation). "
        r"$\Delta$ = After $-$ Before; all metrics in \%."
    )

    latex = _booktabs_wrap(
        label="tab:generation_quality",
        caption=caption,
        col_fmt=col_fmt,
        header_rows=hdr_rows,
        body_rows=rows_latex,
        notes=(r"EM = exact match (SQuAD normalisation, best alias). "
               r"F1 = token-level. R-L = ROUGE-L. BLEU = corpus-level BLEU-4 (sacrebleu)."),
    )
    df = pd.DataFrame(rows_df)
    return latex, df.to_markdown(index=False, floatfmt=".2f"), df


# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — Detection quality
# ══════════════════════════════════════════════════════════════════════════════

def render_table2(
    det_metrics: dict[str, DetectionMetrics],
    tau: float = 0.5,
) -> tuple[str, str, pd.DataFrame]:
    """Render Table 2: Hallucination detection quality (RAGTruth)."""
    SUBSET_ORDER = ["QA", "Summary", "Data2txt", "all"]

    rows_latex = []
    rows_df    = []

    for subset in SUBSET_ORDER:
        dm = det_metrics.get(subset)
        if dm is None:
            continue
        bold = subset == "all"
        fmt  = (lambda v: rf"\textbf{{{v}}}") if bold else (lambda v: v)
        name = r"\textbf{All (macro)}" if bold else subset

        rho_s = f"{dm.spearman_rho:.3f}" if not math.isnan(dm.spearman_rho) else "—"
        p_s   = r"$<$.001" if (not math.isnan(dm.spearman_p) and dm.spearman_p < 0.001) else f"{dm.spearman_p:.3f}"

        cells = [
            name,
            fmt(str(dm.n_samples)),
            fmt(f"{dm.hall_rate * 100:.1f}"),
            fmt(f"{dm.auroc * 100:.2f}"         if not math.isnan(dm.auroc) else "—"),
            fmt(f"{dm.avg_precision * 100:.2f}"  if not math.isnan(dm.avg_precision) else "—"),
            fmt(f"{dm.precision_at_tau * 100:.2f}"),
            fmt(f"{dm.recall_at_tau * 100:.2f}"),
            fmt(f"{dm.f1_at_tau * 100:.2f}"),
            fmt(rho_s),
            p_s,
        ]
        rows_latex.append(" & ".join(cells) + r" \\")
        if bold:
            rows_latex.insert(-1, r"\midrule")

        rows_df.append({
            "Subset": subset,
            "N": dm.n_samples,
            "Hall. rate %": round(dm.hall_rate * 100, 1),
            "AUROC %": round(dm.auroc * 100, 2) if not math.isnan(dm.auroc) else None,
            "Avg Prec %": round(dm.avg_precision * 100, 2) if not math.isnan(dm.avg_precision) else None,
            f"Prec@τ={tau}": round(dm.precision_at_tau * 100, 2),
            f"Rec@τ={tau}":  round(dm.recall_at_tau  * 100, 2),
            f"F1@τ={tau}":   round(dm.f1_at_tau      * 100, 2),
            "Spearman ρ":    round(dm.spearman_rho, 3) if not math.isnan(dm.spearman_rho) else None,
            "p-value":       round(dm.spearman_p, 4)   if not math.isnan(dm.spearman_p)   else None,
        })

    col_fmt  = "l r r r r r r r r r"
    hdr_rows = [
        r"\textbf{Subset} & \textbf{N} & \textbf{Hall. rate} & "
        r"\textbf{AUROC} & \textbf{Avg Prec} & "
        rf"\textbf{{Prec@$\tau$}} & \textbf{{Rec@$\tau$}} & \textbf{{F1@$\tau$}} & "
        r"\textbf{Spearman $\rho$} & \textbf{$p$} \\",
    ]
    caption = (
        rf"Hallucination detection quality on RAGTruth test set. "
        rf"$\tau={tau}$ is the entropy threshold used for flagging. "
        r"AUROC and Avg. Precision are threshold-free. "
        r"Spearman $\rho$ measures rank correlation between per-token entropy "
        r"and binary RAGTruth hallucination labels."
    )
    latex = _booktabs_wrap(
        label="tab:detection_quality",
        caption=caption,
        col_fmt=col_fmt,
        header_rows=hdr_rows,
        body_rows=rows_latex,
    )
    df = pd.DataFrame(rows_df)
    return latex, df.to_markdown(index=False, floatfmt=".2f"), df


# ══════════════════════════════════════════════════════════════════════════════
# Table 3 — Refinement analysis
# ══════════════════════════════════════════════════════════════════════════════

def render_table3(
    ref_metrics: dict[str, SubsetRefMetrics],
) -> tuple[str, str, pd.DataFrame]:
    """Render Table 3: Refinement delta breakdown (RAGTruth)."""
    SUBSET_ORDER = ["QA", "Summary", "Data2txt", "all"]

    rows_latex = []
    rows_df    = []

    for subset in SUBSET_ORDER:
        rm = ref_metrics.get(subset)
        if rm is None:
            continue
        bold = subset == "all"
        fmt  = (lambda v: rf"\textbf{{{v}}}") if bold else (lambda v: v)
        name = r"\textbf{All (macro)}" if bold else subset

        fs_d = rm.fact_score_delta
        fs_d_s = (rf"\textcolor{{green!55!black}}{{\textbf{{+{fs_d:.2f}}}}}"
                  if fs_d > 0.05 else f"{fs_d:.2f}")
        sp_s = (rf"\textcolor{{green!55!black}}{{\textbf{{{rm.span_reduction_pct:.1f}}}}}"
                if rm.span_reduction_pct > 0 else f"{rm.span_reduction_pct:.1f}")
        int_s = (rf"\textcolor{{green!55!black}}{{\textbf{{{rm.intensity_delta_mean:.3f}}}}}"
                 if rm.intensity_delta_mean < 0 else f"{rm.intensity_delta_mean:.3f}")

        cells = [
            name,
            fmt(str(rm.n_samples)),
            fmt(f"{rm.tokens_changed_mean:.1f}"),
            fmt(f"{rm.change_rate_mean:.2f}"),
            fmt(f"{rm.fact_score_before:.2f}"),
            fmt(f"{rm.fact_score_after:.2f}"),
            fs_d_s,
            sp_s,
            int_s,
        ]
        if bold:
            rows_latex.append(r"\midrule")
        rows_latex.append(" & ".join(cells) + r" \\")

        rows_df.append({
            "Subset":           subset,
            "N":                rm.n_samples,
            "Tokens changed":   round(rm.tokens_changed_mean, 1),
            "Change rate %":    round(rm.change_rate_mean, 2),
            "FactScore bef.":   round(rm.fact_score_before, 2),
            "FactScore aft.":   round(rm.fact_score_after,  2),
            "FactScore Δ":      round(rm.fact_score_delta,  2),
            "Span reduction %": round(rm.span_reduction_pct, 1),
            "Intensity Δ":      round(rm.intensity_delta_mean, 3),
        })

    col_fmt = "l r r r r r r r r"
    hdr_rows = [
        r"\textbf{Subset} & \textbf{N} & \textbf{Toks chg.} & \textbf{Chg. rate\%} & "
        r"\textbf{FS bef.} & \textbf{FS aft.} & \textbf{FS $\Delta$} & "
        r"\textbf{Span red.\%} & \textbf{Intens. $\Delta$} \\",
    ]
    caption = (
        r"Refinement analysis on RAGTruth. "
        r"FS = FactScore (bigram overlap with source passage). "
        r"Span red. = percentage of hallucinated span tokens covered by remasked positions. "
        r"Intensity $\Delta$ = change in mean RAGTruth hallucination intensity score "
        r"(negative = less severe)."
    )
    latex = _booktabs_wrap(
        label="tab:refinement_analysis",
        caption=caption,
        col_fmt=col_fmt,
        header_rows=hdr_rows,
        body_rows=rows_latex,
    )
    df = pd.DataFrame(rows_df)
    return latex, df.to_markdown(index=False, floatfmt=".2f"), df


# ══════════════════════════════════════════════════════════════════════════════
# Table 4 — Baseline comparison  (per-dataset sub-columns)
# ══════════════════════════════════════════════════════════════════════════════

# Dataset columns shown in Table 4 and Table 5.
# Each tuple: (dataset_key, display_name, primary_metric_attr, col_header)
# Primary metric: F1 for short-answer QA; ROUGE-L for RAGTruth long-form.
_DS_COLS = [
    ("triviaqa",      "TriviaQA",      "f1_after",     "F1"),
    ("commonsenseqa", "CommQA",        "f1_after",     "F1"),
    ("hotpotqa",      "HotpotQA",      "f1_after",     "F1"),
    ("ragtruth",      "RAGTruth",      "rougeL_after",  "R-L"),
]

# Short method display names (keeps rows narrow)
_METHOD_SHORT = {
    METHOD_VANILLA:       "Vanilla LLaDA",
    METHOD_EXTRA_STEPS:   "Extra steps (2$\\times$T)",
    METHOD_RANDOM_REMASK: "Random remask",
    METHOD_SELFCHECK:     "SelfCheck-style",
    METHOD_OSCAR:         "\\textbf{OSCAR (ours)}",
}

_ORDER_SHORT = {
    ORDER_ALL_LEARNED:   "All learned",
    ORDER_ALL_RANDOM:    "All random",
    ORDER_HYBRID_50:     "Hybrid (50/50)",
    ORDER_ENTROPY_ORDER: "Entropy-ordered",
    ORDER_OSCAR:         "\\textbf{OSCAR (1L$+$(N-1)R)}",
}


def render_table4(
    baseline_gen_by_ds: dict[str, dict[str, DatasetGenMetrics]],
    baseline_det:       dict[str, DetectionMetrics],
    method_order:       list[str] | None = None,
) -> tuple[str, str, pd.DataFrame]:
    """
    Render Table 4: OSCAR vs. baselines with per-dataset sub-columns.

    Args:
        baseline_gen_by_ds: {method → {dataset_key → DatasetGenMetrics}}
                            Each inner dict must have keys matching _DS_COLS.
        baseline_det:       {method → DetectionMetrics (RAGTruth, subset='all')}
        method_order:       display order of methods (defaults to canonical order)

    Column layout (left → right)
    ----------------------------
      Method | TriviaQA F1 | CommQA F1 | HotpotQA F1 | RAGTruth R-L | AUROC | ρ | Avg F1
    """
    method_order = method_order or [
        METHOD_VANILLA, METHOD_EXTRA_STEPS,
        METHOD_RANDOM_REMASK, METHOD_SELFCHECK,
        METHOD_OSCAR,
    ]

    # n_ds dataset cols + 2 detection cols + 1 avg col = n_ds+3 data cols
    n_ds       = len(_DS_COLS)
    n_data_col = n_ds + 3          # per-ds + AUROC + ρ + Avg

    rows_latex = []
    rows_df    = []

    for method in method_order:
        is_oscar = method == METHOD_OSCAR
        fmt = (lambda v: rf"\textbf{{{v}}}") if is_oscar else (lambda v: v)

        ds_metrics = baseline_gen_by_ds.get(method, {})
        dm         = baseline_det.get(method)

        # Per-dataset primary metric cells
        ds_vals = []
        for ds_key, _, attr, _ in _DS_COLS:
            gm = ds_metrics.get(ds_key)
            if gm:
                v = getattr(gm, attr, None)
                ds_vals.append(fmt(f"{v * 100:.2f}") if v is not None else "—")
            else:
                ds_vals.append("—")

        # Detection cells (RAGTruth only — undefined for single-path methods)
        auroc_s = (fmt(f"{dm.auroc * 100:.2f}")
                   if dm and not math.isnan(dm.auroc) else "—")
        rho_s   = (fmt(f"{dm.spearman_rho:.3f}")
                   if dm and not math.isnan(getattr(dm, "spearman_rho", float("nan")))
                   else "—")

        # Macro-avg F1 across QA datasets (exclude RAGTruth from avg F1)
        qa_f1s = [
            getattr(ds_metrics[k], "f1_after", None)
            for k, _, attr, _ in _DS_COLS
            if k != "ragtruth" and k in ds_metrics and attr == "f1_after"
        ]
        avg_f1_s = fmt(f"{float(np.mean(qa_f1s)) * 100:.2f}") if qa_f1s else "—"

        name = _METHOD_SHORT.get(method, method)
        cells = [name] + ds_vals + [auroc_s, rho_s, avg_f1_s]
        if is_oscar:
            rows_latex.append(r"\midrule")
        rows_latex.append(" & ".join(cells) + r" \\")

        df_row = {"Method": method}
        for (ds_key, ds_disp, attr, hdr), cell in zip(_DS_COLS, ds_vals):
            gm = ds_metrics.get(ds_key)
            v  = getattr(gm, attr, None) if gm else None
            df_row[f"{ds_disp} {hdr}"] = round(v * 100, 2) if v else None
        df_row["AUROC %"]    = round(dm.auroc * 100, 2) if dm and not math.isnan(dm.auroc) else None
        df_row["Spearman ρ"] = round(dm.spearman_rho, 3) if dm and not math.isnan(getattr(dm, "spearman_rho", float("nan"))) else None
        df_row["Avg QA F1 %"] = round(float(np.mean(qa_f1s)) * 100, 2) if qa_f1s else None
        rows_df.append(df_row)

    # ── column format ──
    # l  +  n_ds r's  +  r r  (detection)  +  r  (avg)
    col_fmt = "l " + " ".join(["r"] * n_data_col)

    # ── header rows ──
    # Row 1: dataset group headers + detection group header + avg
    ds_group_hdrs = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{\textbf{{{disp}}}}}"
        for _, disp, _, _ in _DS_COLS
    )
    hdr_row1 = (
        rf"\textbf{{Method}} & {ds_group_hdrs} & "
        rf"\multicolumn{{2}}{{c}}{{\textbf{{Detection}}}} & \textbf{{Avg}} \\"
    )
    # cmidrule under detection group
    det_start = n_ds + 2   # 1-indexed: col 1 = method, cols 2..n_ds+1 = datasets
    det_end   = det_start + 1
    avg_col   = det_end + 1
    cmidrule = rf"\cmidrule(lr){{{det_start}-{det_end}}}"

    # Row 2: per-column metric names
    ds_metric_hdrs = " & ".join(
        rf"\textit{{{hdr}}}" for _, _, _, hdr in _DS_COLS
    )
    hdr_row2 = rf" & {ds_metric_hdrs} & \textit{{AUROC}} & \textit{{$\rho$}} & \textit{{F1}} \\"

    hdr_rows = [hdr_row1, cmidrule, hdr_row2]

    caption = (
        r"Comparison of OSCAR against baseline methods across all four datasets. "
        r"Per-dataset primary metric: F1 for short-answer QA datasets; "
        r"ROUGE-L (R-L) for RAGTruth (long-form generation). "
        r"Detection metrics (AUROC, Spearman $\rho$) are on RAGTruth only; "
        r"`---' indicates the method does not produce a multi-path entropy signal. "
        r"Avg F1 = macro average over the three QA datasets."
    )
    latex = _booktabs_wrap(
        label="tab:baseline_comparison",
        caption=caption,
        col_fmt=col_fmt,
        header_rows=hdr_rows,
        body_rows=rows_latex,
    )
    df = pd.DataFrame(rows_df)
    return latex, df.to_markdown(index=False, floatfmt=".2f"), df


# ══════════════════════════════════════════════════════════════════════════════
# Table 5 — Demasking order comparison  (per-dataset sub-columns)
# ══════════════════════════════════════════════════════════════════════════════

def render_table5(
    order_gen_by_ds: dict[str, dict[str, DatasetGenMetrics]],
    order_det:       dict[str, DetectionMetrics],
    order_list:      list[str] | None = None,
) -> tuple[str, str, pd.DataFrame]:
    """
    Render Table 5: Demasking order comparison with per-dataset sub-columns.

    Args:
        order_gen_by_ds: {order_name → {dataset_key → DatasetGenMetrics}}
        order_det:       {order_name → DetectionMetrics (RAGTruth, subset='all')}
        order_list:      display order (defaults to canonical 5-order list)

    Column layout
    -------------
      Order | TriviaQA F1 | CommQA F1 | HotpotQA F1 | RAGTruth R-L | AUROC | ρ | Avg F1
    """
    order_list = order_list or [
        ORDER_ALL_LEARNED, ORDER_ALL_RANDOM,
        ORDER_HYBRID_50, ORDER_ENTROPY_ORDER, ORDER_OSCAR,
    ]

    n_ds       = len(_DS_COLS)
    n_data_col = n_ds + 3   # per-ds + AUROC + ρ + Avg

    rows_latex = []
    rows_df    = []

    for order in order_list:
        is_oscar = order == ORDER_OSCAR
        fmt = (lambda v: rf"\textbf{{{v}}}") if is_oscar else (lambda v: v)

        ds_metrics = order_gen_by_ds.get(order, {})
        dm         = order_det.get(order)

        # Per-dataset primary metric cells
        ds_vals = []
        for ds_key, _, attr, _ in _DS_COLS:
            gm = ds_metrics.get(ds_key)
            if gm:
                v = getattr(gm, attr, None)
                ds_vals.append(fmt(f"{v * 100:.2f}") if v is not None else "—")
            else:
                ds_vals.append("—")

        # Detection cells
        auroc_s = (fmt(f"{dm.auroc * 100:.2f}")
                   if dm and not math.isnan(dm.auroc) else "—")
        rho_s   = (fmt(f"{dm.spearman_rho:.3f}")
                   if dm and not math.isnan(getattr(dm, "spearman_rho", float("nan")))
                   else "—")

        # Macro-avg QA F1
        qa_f1s = [
            getattr(ds_metrics[k], "f1_after", None)
            for k, _, attr, _ in _DS_COLS
            if k != "ragtruth" and k in ds_metrics and attr == "f1_after"
        ]
        avg_f1_s = fmt(f"{float(np.mean(qa_f1s)) * 100:.2f}") if qa_f1s else "—"

        name = _ORDER_SHORT.get(order, order)
        cells = [name] + ds_vals + [auroc_s, rho_s, avg_f1_s]
        if is_oscar:
            rows_latex.append(r"\midrule")
        rows_latex.append(" & ".join(cells) + r" \\")

        df_row = {"Demasking order": order}
        for (ds_key, ds_disp, attr, hdr), cell in zip(_DS_COLS, ds_vals):
            gm = ds_metrics.get(ds_key)
            v  = getattr(gm, attr, None) if gm else None
            df_row[f"{ds_disp} {hdr}"] = round(v * 100, 2) if v else None
        df_row["AUROC %"]      = round(dm.auroc * 100, 2) if dm and not math.isnan(dm.auroc) else None
        df_row["Spearman ρ"]   = round(dm.spearman_rho, 3) if dm and not math.isnan(getattr(dm, "spearman_rho", float("nan"))) else None
        df_row["Avg QA F1 %"]  = round(float(np.mean(qa_f1s)) * 100, 2) if qa_f1s else None
        rows_df.append(df_row)

    col_fmt = "l " + " ".join(["r"] * n_data_col)

    ds_group_hdrs = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{\textbf{{{disp}}}}}"
        for _, disp, _, _ in _DS_COLS
    )
    det_start = n_ds + 2
    det_end   = det_start + 1
    cmidrule  = rf"\cmidrule(lr){{{det_start}-{det_end}}}"

    hdr_row1 = (
        rf"\textbf{{Order}} & {ds_group_hdrs} & "
        rf"\multicolumn{{2}}{{c}}{{\textbf{{Detection}}}} & \textbf{{Avg}} \\"
    )
    ds_metric_hdrs = " & ".join(
        rf"\textit{{{hdr}}}" for _, _, _, hdr in _DS_COLS
    )
    hdr_row2 = rf" & {ds_metric_hdrs} & \textit{{AUROC}} & \textit{{$\rho$}} & \textit{{F1}} \\"

    hdr_rows = [hdr_row1, cmidrule, hdr_row2]

    caption = (
        r"Effect of demasking order across all four datasets. "
        r"OSCAR uses 1 learned path $+$ $(N{-}1)$ random paths. "
        r"Per-dataset metric: F1 for QA; ROUGE-L for RAGTruth. "
        r"Detection columns (AUROC, $\rho$) use RAGTruth span labels. "
        r"Avg F1 = macro average over the three QA datasets."
    )
    latex = _booktabs_wrap(
        label="tab:demasking_order",
        caption=caption,
        col_fmt=col_fmt,
        header_rows=hdr_rows,
        body_rows=rows_latex,
    )
    df = pd.DataFrame(rows_df)
    return latex, df.to_markdown(index=False, floatfmt=".2f"), df
