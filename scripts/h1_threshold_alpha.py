#!/usr/bin/env python3
"""
H1 — Threshold α Sensitivity Ablation
======================================
Validates default α=0.20. Sweeps α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}
and re-runs correction from existing chain outputs (no new generation needed).

Usage:
    python scripts/h1_threshold_alpha.py \
        --results_dir results/parade_triviaqa \
        --output results/h1_alpha/
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def token_f1(pred, ref):
    p = normalize(pred).split()
    r = normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0
    prec = n / len(p)
    rec = n / len(r)
    return 2 * prec * rec / (prec + rec)


def simulate_alpha(records, alpha):
    """
    Simulate OSCAR correction at a different alpha threshold.
    """
    entropy_values = []
    for rec in records:
        ent_traj = rec.get("per_position_entropy", rec.get("entropy_trajectory", []))
        if isinstance(ent_traj, list) and ent_traj:
            entropy_values.extend([e for e in ent_traj if e > 0])

    if not entropy_values:
        return None

    threshold = np.percentile(entropy_values, 100 * (1 - alpha))
    pct_remasked = alpha * 100

    return {
        "alpha": alpha,
        "threshold": threshold,
        "pct_tokens_remasked": pct_remasked,
    }


def main():
    parser = argparse.ArgumentParser(description="H1: Alpha Sensitivity")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="results/h1_alpha/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl = Path(args.results_dir) / "raw_results.jsonl"
    records = []
    with open(jsonl) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(records)} records")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    results = []

    print(f"\n{'α':<8} {'% remasked':<12} {'threshold':<12}")
    print("-" * 32)

    for alpha in alphas:
        res = simulate_alpha(records, alpha)
        if res:
            results.append(res)
            print(f"{alpha:<8.2f} {res['pct_tokens_remasked']:<12.1f} "
                  f"{res['threshold']:<12.4f}")

    with open(out_dir / "h1_alpha_table.tex", "w") as f:
        f.write("% H1: Alpha sensitivity ablation\n")
        f.write("% NOTE: F1 and AUROC columns need to be filled from\n")
        f.write("% re-running correction at each alpha. This table shows\n")
        f.write("% the threshold values and remasking percentages.\n")
        f.write("% For proper F1/AUROC, re-run OSCAR pipeline with --alpha X\n\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Threshold sensitivity (LLaDA-8B, QA macro-avg).}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("$\\alpha$ & \\% remasked & F1 (\\%) & AUROC & Span Red.\\% \\\\\n\\midrule\n")
        for r in results:
            bold = "\\textbf" if r["alpha"] == 0.20 else ""
            default = " (default)" if r["alpha"] == 0.20 else ""
            f.write(
                f"{bold}{{{r['alpha']:.2f}}}{default} & "
                f"$\\sim${r['pct_tokens_remasked']:.0f}\\% & "
                f"\\todo{{???.?}} & \\todo{{???.?}} & \\todo{{???.?}} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(out_dir / "h1_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")
    print()
    print("NOTE: For proper F1/AUROC at each alpha, re-run the full OSCAR pipeline:")
    print("  for ALPHA in 0.05 0.10 0.15 0.20 0.25 0.30 0.40; do")
    print("    python run_experiment.py --top_k_percent $ALPHA \\")
    print("        --output_dir results/h1_alpha/alpha_${ALPHA}/ ...")
    print("  done")


if __name__ == "__main__":
    main()
