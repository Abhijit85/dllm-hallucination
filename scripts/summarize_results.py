#!/usr/bin/env python3
"""
scripts/summarize_results.py
-----------------------------
Reads all aggregate.json files under a results directory and prints a
markdown comparison table + saves results/summary.md.

Usage:
    python scripts/summarize_results.py --results_dir results/20260318_213311
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_aggregate(path: Path) -> dict | None:
    agg = path / "aggregate.json"
    if not agg.exists():
        return None
    with open(agg, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, decimals=4, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    fmt_str = f"{{:+.{decimals}f}}" if sign else f"{{:.{decimals}f}}"
    return fmt_str.format(v)


def infer_n_paths(label: str) -> str:
    if label == "Full (n=8)":
        return "8"
    if label.startswith("Ablation n="):
        return label.split("=")[-1]
    return "—"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()

    root = Path(args.results_dir)
    rows = []

    full_agg = load_aggregate(root / "full")
    if full_agg:
        rows.append({"label": "Full (n=8)", **full_agg})

    abl_root = root / "ablation"
    if abl_root.exists():
        for n in [1, 2, 4, 8, 16]:
            agg = load_aggregate(abl_root / f"n_paths_{n}")
            if agg:
                rows.append({"label": f"Ablation n={n}", **agg})

    if not rows:
        print("  No aggregate.json files found yet.")
        return

    header = (
        "| Run | N | Samples | Hall% | FactScore↑ | ΔFactScore | "
        "F1 | Spearman ρ | ρ samples | Time(min) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    lines = ["# Experiment Results", "", header, sep]
    for r in rows:
        n_val = r.get("n_paths", infer_n_paths(r["label"]))
        hall_pct = f"{r.get('hallucinated_sample_fraction', 0) * 100:.0f}%"
        line = (
            f"| {r['label']} "
            f"| {n_val} "
            f"| {r.get('n_samples', '—')} "
            f"| {hall_pct} "
            f"| {fmt(r.get('mean_fact_score_after'))} "
            f"| {fmt(r.get('mean_refinement_delta'), sign=True)} "
            f"| {fmt(r.get('mean_token_f1'))} "
            f"| {fmt(r.get('mean_spearman_rho'))} "
            f"| {r.get('n_rho_computed', '—')} "
            f"| {fmt(r.get('total_time_min'), decimals=1)} |"
        )
        lines.append(line)

    lines += ["", "## Key Findings", ""]

    if full_agg:
        delta = full_agg.get("mean_refinement_delta", float("nan"))
        rho = full_agg.get("mean_spearman_rho", float("nan"))
        n_rho = full_agg.get("n_rho_computed", 0)

        h2 = (
            "✅ SUPPORTED"
            if not math.isnan(delta) and delta > 0
            else "❌ NOT SUPPORTED"
        )
        h1 = (
            "✅ SUPPORTED"
            if not math.isnan(rho) and rho > 0
            else "❌ WEAK / NOT SUPPORTED"
        )

        lines += [
            f"- **H1** (entropy predicts hallucination): {h1}",
            f"  - Spearman ρ = {fmt(rho)} over {n_rho} hallucinated samples",
            f"- **H2** (remasking reduces hallucination): {h2}",
            f"  - Mean ΔFactScore = {fmt(delta, sign=True)}",
        ]

    abl_rows = [r for r in rows if r["label"].startswith("Ablation")]
    if len(abl_rows) >= 2:
        lines += ["", "### N-Paths Ablation"]
        best = max(
            abl_rows,
            key=lambda r: r.get("mean_refinement_delta", float("-inf")),
        )
        lines.append(
            f"- Best ΔFactScore at **n={best.get('n_paths', infer_n_paths(best['label']))}** "
            f"({fmt(best.get('mean_refinement_delta'), sign=True)})"
        )
        deltas = [
            (
                int(r.get("n_paths", infer_n_paths(r["label"]))),
                r.get("mean_refinement_delta", 0),
            )
            for r in abl_rows
            if str(r.get("n_paths", infer_n_paths(r["label"]))).isdigit()
        ]
        deltas.sort()
        if len(deltas) >= 3:
            gains = [deltas[i + 1][1] - deltas[i][1] for i in range(len(deltas) - 1)]
            if gains and gains[-1] < gains[0] * 0.1:
                lines.append(f"- Signal saturates around n={deltas[-2][0]} paths")

    table = "\n".join(lines)
    print(table)

    out = root / "summary.md"
    out.write_text(table, encoding="utf-8")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
