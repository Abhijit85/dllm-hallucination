#!/usr/bin/env python3
"""
C4 — Aggregate 3-seed results into mean ± std for Tables 1 & 2.

Usage:
    python scripts/c4_aggregate_seeds.py --seeds_dir results/c4_seeds/
"""

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path

import numpy as np


def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_metrics_from_jsonl(jsonl_path):
    """Compute AUROC, F1, EM from a raw_results.jsonl file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return None

    ems_before, ems_after = [], []
    f1s_before, f1s_after = [], []
    entropy_scores, labels = [], []

    for rec in records:
        gold = rec.get("gold_answers", rec.get("gold", []))
        if isinstance(gold, str):
            gold = [gold]

        gen_before = rec.get("generated_before", rec.get("generated", ""))
        gen_after = rec.get("generated_after", rec.get("refined", gen_before))

        em_b = int(any(normalize_answer(g) in normalize_answer(gen_before) for g in gold))
        em_a = int(any(normalize_answer(g) in normalize_answer(gen_after) for g in gold))
        ems_before.append(em_b)
        ems_after.append(em_a)

        def token_f1(pred, ref):
            pred_toks = normalize_answer(pred).split()
            ref_toks = normalize_answer(ref).split()
            if not pred_toks or not ref_toks:
                return float(pred_toks == ref_toks)
            from collections import Counter
            common = Counter(pred_toks) & Counter(ref_toks)
            n = sum(common.values())
            if n == 0:
                return 0.0
            p = n / len(pred_toks)
            r = n / len(ref_toks)
            return 2 * p * r / (p + r)

        f1_b = max(token_f1(gen_before, g) for g in gold) if gold else 0
        f1_a = max(token_f1(gen_after, g) for g in gold) if gold else 0
        f1s_before.append(f1_b)
        f1s_after.append(f1_a)

        ent = rec.get("mean_cross_entropy", rec.get("entropy_score", 0))
        entropy_scores.append(ent)
        labels.append(1 - em_b)

    auroc = float("nan")
    if len(set(labels)) >= 2 and entropy_scores:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(labels, entropy_scores)
        except Exception:
            pass

    return {
        "n": len(records),
        "em_before": np.mean(ems_before) * 100,
        "em_after": np.mean(ems_after) * 100,
        "f1_before": np.mean(f1s_before) * 100,
        "f1_after": np.mean(f1s_after) * 100,
        "auroc": auroc * 100 if not np.isnan(auroc) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_dir", default="results/c4_seeds/")
    parser.add_argument("--output", default="results/c4_seeds/")
    args = parser.parse_args()

    seeds_dir = Path(args.seeds_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for subdir in sorted(seeds_dir.iterdir()):
        if not subdir.is_dir():
            continue
        jsonl = subdir / "raw_results.jsonl"
        if not jsonl.exists():
            continue
        parts = subdir.name.rsplit("_seed", 1)
        if len(parts) == 2:
            groups[parts[0]].append(jsonl)

    print(f"{'='*80}")
    print("Table 1 & 2 with ± std across 3 seeds")
    print(f"{'='*80}")
    print()

    all_results = {}

    for group_name, jsonl_files in sorted(groups.items()):
        print(f"\n{group_name} ({len(jsonl_files)} seeds)")
        print("-" * 60)

        seed_metrics = []
        for jf in jsonl_files:
            metrics = compute_metrics_from_jsonl(jf)
            if metrics:
                seed_metrics.append(metrics)
                print(
                    f"  {jf.parent.name}: EM={metrics['em_after']:.1f} "
                    f"F1={metrics['f1_after']:.1f} AUROC={metrics['auroc']:.1f}"
                )

        if len(seed_metrics) < 2:
            print(f"  WARNING: Only {len(seed_metrics)} seeds — need at least 2 for std")
            continue

        result = {}
        for metric in ["em_before", "em_after", "f1_before", "f1_after", "auroc"]:
            vals = [m[metric] for m in seed_metrics if not np.isnan(m[metric])]
            if vals:
                result[f"{metric}_mean"] = np.mean(vals)
                result[f"{metric}_std"] = np.std(vals)

        all_results[group_name] = result

        print("\n  Summary:")
        for metric in ["em_after", "f1_after", "auroc"]:
            mean = result.get(f"{metric}_mean", float("nan"))
            std = result.get(f"{metric}_std", float("nan"))
            print(f"    {metric}: {mean:.1f} ± {std:.1f}")

    latex_path = out_dir / "tables_with_variance.tex"
    with open(latex_path, "w") as f:
        f.write("% Tables 1 & 2 with ± std from 3 seeds\n")
        f.write("% Generated by c4_aggregate_seeds.py\n\n")

        f.write("% Table 2: Generation quality with variance\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Generation quality (mean $\\pm$ std over 3 seeds).}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Dataset & EM Before & EM After & F1 Before & F1 After \\\\\n\\midrule\n")

        for name, res in sorted(all_results.items()):
            em_b = f"{res.get('em_before_mean', 0):.1f}"
            em_a = f"{res.get('em_after_mean', 0):.1f}$\\pm${res.get('em_after_std', 0):.1f}"
            f1_b = f"{res.get('f1_before_mean', 0):.1f}"
            f1_a = f"{res.get('f1_after_mean', 0):.1f}$\\pm${res.get('f1_after_std', 0):.1f}"
            f.write(f"{name} & {em_b} & {em_a} & {f1_b} & {f1_a} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(out_dir / "c4_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")
    print("  c4_summary.json         — raw mean/std values")
    print("  tables_with_variance.tex — LaTeX tables with ±")


if __name__ == "__main__":
    main()
