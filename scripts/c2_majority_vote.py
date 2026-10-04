#!/usr/bin/env python3
"""
C2 — Majority Vote Baseline
============================
PAPER-FATAL: Table 6 is missing the most obvious correction baseline.

This script:
1. Loads existing N=8 chain outputs from raw_results.jsonl
2. Computes token-level majority vote across chains
3. Evaluates F1/EM of majority-vote output vs gold
4. Adds MV row to Table 6 (complementarity ablation)

Zero GPU needed — pure post-processing on existing chain outputs.

Usage:
    python scripts/c2_majority_vote.py \
        --results_dirs results/parade_triviaqa results/parade_hotpotqa \
        --output results/c2_majority_vote/
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

# ── Text normalization (matches existing eval) ───────────────────────────────


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def compute_f1(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction, ground_truth):
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def best_score(prediction, gold_answers, metric_fn):
    """Best score across multiple gold answers."""
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    return max(metric_fn(prediction, g) for g in gold_answers) if gold_answers else 0.0


# ── Majority vote computation ────────────────────────────────────────────────


def token_majority_vote(chain_outputs):
    """
    Token-level majority vote across N chain outputs.

    For each position i, pick the most common token across all N chains.
    Handles variable-length outputs by padding shorter ones.
    """
    if not chain_outputs:
        return ""

    tokenized = [s.split() if isinstance(s, str) else s for s in chain_outputs]
    max_len = max(len(t) for t in tokenized)

    mv_tokens = []
    for i in range(max_len):
        tokens_at_i = []
        for chain in tokenized:
            if i < len(chain):
                tokens_at_i.append(chain[i])
        if tokens_at_i:
            mv_tokens.append(Counter(tokens_at_i).most_common(1)[0][0])

    return " ".join(mv_tokens)


def string_majority_vote(chain_outputs):
    """String-level majority vote: pick the most common full output."""
    if not chain_outputs:
        return ""
    normalized = [normalize_answer(s) for s in chain_outputs]
    counter = Counter(normalized)
    most_common_norm = counter.most_common(1)[0][0]
    for orig, norm in zip(chain_outputs, normalized):
        if norm == most_common_norm:
            return orig
    return chain_outputs[0]


# ── Main pipeline ────────────────────────────────────────────────────────────


def process_results_dir(results_dir, output_dir):
    """Process one results directory and compute MV metrics."""
    del output_dir
    jsonl_path = Path(results_dir) / "raw_results.jsonl"
    if not jsonl_path.exists():
        print(f"  WARNING: {jsonl_path} not found, skipping")
        return None

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"  Loaded {len(records)} records from {results_dir}")

    results = []
    n_has_chains = 0

    for rec in records:
        gold = rec.get("gold_answers", rec.get("gold", []))
        if isinstance(gold, str):
            gold = [gold]

        chain_outputs = rec.get(
            "chain_outputs", rec.get("path_outputs", rec.get("all_outputs", None))
        )

        if chain_outputs is None:
            chain_outputs = []
            for i in range(32):
                key = f"path_{i}_output"
                alt_key = f"chain_{i}_output"
                val = rec.get(key, rec.get(alt_key, None))
                if val is not None:
                    chain_outputs.append(val)
                else:
                    break

        if not chain_outputs:
            gen_before = rec.get("generated_before", rec.get("generated", ""))
            gen_after = rec.get("generated_after", rec.get("refined", ""))
            results.append(
                {
                    "sample_id": rec.get("sample_id", ""),
                    "gold": gold,
                    "baseline": gen_before,
                    "oscar": gen_after if gen_after else gen_before,
                    "mv_token": gen_before,
                    "mv_string": gen_before,
                    "has_chains": False,
                }
            )
            continue

        n_has_chains += 1
        baseline = rec.get("generated_before", chain_outputs[0])
        oscar = rec.get("generated_after", rec.get("refined", baseline))

        mv_tok = token_majority_vote(chain_outputs)
        mv_str = string_majority_vote(chain_outputs)

        results.append(
            {
                "sample_id": rec.get("sample_id", ""),
                "gold": gold,
                "baseline": baseline,
                "oscar": oscar,
                "mv_token": mv_tok,
                "mv_string": mv_str,
                "n_chains": len(chain_outputs),
                "has_chains": True,
            }
        )

    print(f"  Records with chain outputs: {n_has_chains}/{len(records)}")

    if not results:
        return None

    methods = {
        "Unguided decoding": "baseline",
        "Majority vote (token)": "mv_token",
        "Majority vote (string)": "mv_string",
        "OSCAR (full pipeline)": "oscar",
    }

    metrics = {}
    for method_name, field_name in methods.items():
        f1s, ems = [], []
        for r in results:
            pred = r[field_name]
            f1s.append(best_score(pred, r["gold"], compute_f1))
            ems.append(best_score(pred, r["gold"], compute_em))
        metrics[method_name] = {
            "F1": np.mean(f1s) * 100,
            "EM": np.mean(ems) * 100,
            "F1_std": np.std(f1s) * 100,
            "n": len(f1s),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="C2: Majority Vote Baseline")
    parser.add_argument(
        "--results_dirs",
        nargs="+",
        required=True,
        help="Directories containing raw_results.jsonl",
    )
    parser.add_argument("--output", default="results/c2_majority_vote/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for rdir in args.results_dirs:
        name = Path(rdir).name
        print(f"\nProcessing {name}...")
        metrics = process_results_dir(rdir, out_dir)
        if metrics:
            all_metrics[name] = metrics

    print(f"\n{'='*70}")
    print("Table 6 (revised) — Complementarity ablation with majority vote")
    print(f"{'='*70}")
    print(f"{'Configuration':<45} {'F1 (%)':<10} {'ΔF1':<10}")
    print("-" * 65)

    method_names = [
        "Unguided decoding",
        "Majority vote (token)",
        "Majority vote (string)",
        "OSCAR (full pipeline)",
    ]
    baseline_f1 = None

    for method in method_names:
        f1s = [
            all_metrics[d][method]["F1"]
            for d in all_metrics
            if method in all_metrics[d]
        ]
        if f1s:
            avg_f1 = np.mean(f1s)
            if baseline_f1 is None:
                baseline_f1 = avg_f1
            delta = avg_f1 - baseline_f1
            print(f"{method:<45} {avg_f1:>6.1f}     {delta:>+5.1f}")

    summary = {
        "per_dataset": all_metrics,
        "note": "If chain_outputs not in raw_results.jsonl, re-run OSCAR with --save_chains flag",
    }
    with open(out_dir / "c2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(out_dir / "table6_revised.tex", "w") as f:
        f.write("% Table 6 (revised) — with Majority Vote row\n")
        f.write("% Generated by c2_majority_vote.py\n\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(
            "\\caption{Complementarity ablation with majority vote baseline (LLaDA-8B, QA macro-avg).}\n"
        )
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n")
        f.write("Configuration & F1 (\\%) & $\\Delta$F1 \\\\\n\\midrule\n")

        for method in method_names:
            f1s = [
                all_metrics[d][method]["F1"]
                for d in all_metrics
                if method in all_metrics[d]
            ]
            if f1s:
                avg = np.mean(f1s)
                delta = avg - baseline_f1 if baseline_f1 is not None else 0
                bold = "\\textbf" if "OSCAR" in method else ""
                marker = "$\\star$ " if "Majority" in method else ""
                f.write(f"{marker}{method} & {bold}{{{avg:.1f}}} & {delta:+.1f} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nSaved to {out_dir}/")
    print("  c2_summary.json    — raw numbers")
    print("  table6_revised.tex — LaTeX table")
    print()
    print("NOTE: If your raw_results.jsonl doesn't store per-chain outputs,")
    print("you need to re-run OSCAR with a --save_chains flag to save them.")
    print("Alternatively, modify run_experiment.py to dump chain_outputs.")


if __name__ == "__main__":
    main()
