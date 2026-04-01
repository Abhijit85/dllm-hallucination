#!/usr/bin/env python3
"""
notebooks/analysis.py
----------------------
Analysis script for DLLM hallucination experiment results.
Run as a plain script or convert to Jupyter notebook with:
    jupytext --to notebook notebooks/analysis.py

Usage:
    python notebooks/analysis.py --results_dir results/20260318_213311
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 140,
})
BLUE = "#378ADD"
TEAL = "#1D9E75"
CORAL = "#D85A30"
AMBER = "#BA7517"
PURPLE = "#7F77DD"
GRAY = "#888780"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    out = {}

    full_path = results_dir / "full" / "raw_results.jsonl"
    if full_path.exists():
        out["Full (n=8)"] = load_jsonl(full_path)

    abl_dir = results_dir / "ablation"
    if abl_dir.exists():
        for n in [1, 2, 4, 8, 16]:
            p = abl_dir / f"n_paths_{n}" / "raw_results.jsonl"
            if p.exists():
                out[f"n={n}"] = load_jsonl(p)

    return out


def safe_mean(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def rho_not_skipped(record: dict) -> bool:
    return not record.get("rho_skipped", record.get("spearman_skipped", False))


def plot_factscore_scatter(records: list[dict], out_dir: Path):
    before = [r["fact_score_before"] for r in records if "fact_score_before" in r and "fact_score_after" in r]
    after = [r["fact_score_after"] for r in records if "fact_score_before" in r and "fact_score_after" in r]
    hall = [r.get("has_hallucination", False) for r in records if "fact_score_before" in r and "fact_score_after" in r]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [CORAL if h else BLUE for h in hall]
    ax.scatter(before, after, c=colors, alpha=0.55, s=25, linewidths=0)

    lim = max(max(before, default=1), max(after, default=1)) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=GRAY, lw=1, label="no change")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("FactScore before refinement")
    ax.set_ylabel("FactScore after refinement")
    ax.set_title("Refinement effect on source grounding")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORAL, markersize=8, label="Hallucinated"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Grounded"),
        Line2D([0], [0], linestyle="--", color=GRAY, label="No change"),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    improved = sum(1 for b, a in zip(before, after) if a > b)
    ax.text(0.04, 0.96, f"{improved}/{len(before)} samples improved", transform=ax.transAxes, va="top", fontsize=10, color=TEAL)

    fig.tight_layout()
    path = out_dir / "factscore_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_delta_distribution(records: list[dict], out_dir: Path):
    hall_deltas = [r["refinement_delta"] for r in records if r.get("has_hallucination") and "refinement_delta" in r]
    clean_deltas = [r["refinement_delta"] for r in records if not r.get("has_hallucination") and "refinement_delta" in r]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.5, 0.7, 30)

    ax.hist(clean_deltas, bins=bins, alpha=0.6, color=BLUE, label=f"Grounded (n={len(clean_deltas)})")
    ax.hist(hall_deltas, bins=bins, alpha=0.7, color=CORAL, label=f"Hallucinated (n={len(hall_deltas)})")
    ax.axvline(0, color=GRAY, lw=1, linestyle="--")

    if hall_deltas:
        ax.axvline(safe_mean(hall_deltas), color=CORAL, lw=2, linestyle="-", label=f"Hall mean={safe_mean(hall_deltas):+.3f}")
    if clean_deltas:
        ax.axvline(safe_mean(clean_deltas), color=BLUE, lw=2, linestyle="-", label=f"Clean mean={safe_mean(clean_deltas):+.3f}")

    ax.set_xlabel("Delta FactScore (after - before)")
    ax.set_ylabel("Count")
    ax.set_title("Refinement delta by hallucination label")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / "delta_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_entropy_trajectory(records: list[dict], out_dir: Path):
    hall_trajs = [r["entropy_trajectory"] for r in records if r.get("has_hallucination") and r.get("entropy_trajectory")]
    clean_trajs = [r["entropy_trajectory"] for r in records if not r.get("has_hallucination") and r.get("entropy_trajectory")]

    if not hall_trajs and not clean_trajs:
        print("  No trajectory data found (track_trajectory may be disabled)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    def plot_band(trajs, color, label):
        if not trajs:
            return
        max_len = max(len(t) for t in trajs)
        padded = [t + [float("nan")] * (max_len - len(t)) for t in trajs]
        arr = np.array(padded, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        steps = np.arange(max_len)
        ax.plot(steps, mean, color=color, lw=2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

    plot_band(hall_trajs, CORAL, f"Hallucinated (n={len(hall_trajs)})")
    plot_band(clean_trajs, BLUE, f"Grounded (n={len(clean_trajs)})")

    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Mean token entropy (nats)")
    ax.set_title("Entropy trajectory: do hallucinations crystallize early?")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "entropy_trajectory.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_npaths_ablation(all_records: dict[str, list[dict]], out_dir: Path):
    abl = {k: v for k, v in all_records.items() if k.startswith("n=")}
    if len(abl) < 2:
        print("  Skipping N-paths ablation plot (< 2 runs)")
        return

    ns, deltas, stds, rhos = [], [], [], []
    for label, records in sorted(abl.items(), key=lambda x: int(x[0].split("=")[1])):
        n = int(label.split("=")[1])
        d = [r["refinement_delta"] for r in records if "refinement_delta" in r]
        rho_vals = [
            r["spearman_rho"]
            for r in records
            if rho_not_skipped(r) and "spearman_rho" in r and not math.isnan(r["spearman_rho"])
        ]
        ns.append(n)
        deltas.append(safe_mean(d))
        stds.append(float(np.std(d)) if d else 0.0)
        rhos.append(safe_mean(rho_vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(ns, deltas, yerr=stds, fmt="o-", color=TEAL, capsize=4, lw=2, markersize=7)
    ax1.axhline(0, color=GRAY, lw=1, linestyle="--")
    ax1.set_xlabel("Number of paths (N)")
    ax1.set_ylabel("Mean Delta FactScore")
    ax1.set_title("Refinement benefit vs N paths")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ns)
    ax1.set_xticklabels(ns)

    ax2.plot(ns, rhos, "o-", color=PURPLE, lw=2, markersize=7)
    ax2.axhline(0, color=GRAY, lw=1, linestyle="--")
    ax2.set_xlabel("Number of paths (N)")
    ax2.set_ylabel("Mean Spearman rho")
    ax2.set_title("Entropy-hallucination correlation vs N paths")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ns)
    ax2.set_xticklabels(ns)

    fig.suptitle("N-Paths Ablation", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "npaths_ablation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_entropy_vs_label(records: list[dict], out_dir: Path):
    hall_flag_rate = [r["n_flagged"] / max(1, r.get("n_flagged", 1) + 1) for r in records if r.get("has_hallucination") and "n_flagged" in r]
    clean_flag_rate = [r["n_flagged"] / max(1, r.get("n_flagged", 1) + 1) for r in records if not r.get("has_hallucination") and "n_flagged" in r]

    rho_vals = [
        r["spearman_rho"]
        for r in records
        if rho_not_skipped(r) and "spearman_rho" in r and not math.isnan(r["spearman_rho"])
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(clean_flag_rate, bins=15, alpha=0.6, color=BLUE, label=f"Grounded (n={len(clean_flag_rate)})")
    ax1.hist(hall_flag_rate, bins=15, alpha=0.7, color=CORAL, label=f"Hallucinated (n={len(hall_flag_rate)})")
    ax1.set_xlabel("Fraction of tokens flagged as high-entropy")
    ax1.set_ylabel("Samples")
    ax1.set_title("High-entropy token rate by label")
    ax1.legend(frameon=False, fontsize=9)

    if rho_vals:
        ax2.hist(rho_vals, bins=15, color=PURPLE, alpha=0.75)
        ax2.axvline(safe_mean(rho_vals), color=CORAL, lw=2, linestyle="--", label=f"Mean rho = {safe_mean(rho_vals):.3f}")
        ax2.axvline(0, color=GRAY, lw=1, linestyle=":")
        ax2.set_xlabel("Spearman rho (entropy vs hallucination label)")
        ax2.set_ylabel("Samples")
        ax2.set_title(f"H1 validation\n(n={len(rho_vals)} samples with valid rho)")
        ax2.legend(frameon=False, fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No rho data yet\n(need hallucinated samples)", ha="center", va="center", transform=ax2.transAxes, color=GRAY)
        ax2.set_title("H1 validation")

    fig.tight_layout()
    path = out_dir / "h1_validation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dashboard(all_records: dict[str, list[dict]], out_dir: Path):
    full = all_records.get("Full (n=8)", [])
    if not full:
        full = next(iter(all_records.values()), [])
    if not full:
        return

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("DLLM Hallucination Reduction - Results Dashboard", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    before = [r.get("fact_score_before", 0) for r in full]
    after = [r.get("fact_score_after", 0) for r in full]
    hall = [r.get("has_hallucination", False) for r in full]
    colors = [CORAL if h else BLUE for h in hall]
    ax1.scatter(before, after, c=colors, alpha=0.5, s=15, linewidths=0)
    lim = max(max(before, default=1), max(after, default=1)) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color=GRAY, lw=1)
    ax1.set_xlabel("Before")
    ax1.set_ylabel("After")
    ax1.set_title("FactScore before vs after")

    ax2 = fig.add_subplot(gs[0, 1])
    deltas = [r.get("refinement_delta", 0) for r in full]
    ax2.hist(deltas, bins=20, color=TEAL, alpha=0.75)
    ax2.axvline(safe_mean(deltas), color=CORAL, lw=2, linestyle="--", label=f"Mean {safe_mean(deltas):+.3f}")
    ax2.axvline(0, color=GRAY, lw=1)
    ax2.set_xlabel("Delta FactScore")
    ax2.set_ylabel("Count")
    ax2.set_title("Refinement delta distribution")
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    rho_vals = [
        r["spearman_rho"]
        for r in full
        if rho_not_skipped(r) and "spearman_rho" in r and not math.isnan(r["spearman_rho"])
    ]
    if rho_vals:
        ax3.hist(rho_vals, bins=15, color=PURPLE, alpha=0.75)
        ax3.axvline(safe_mean(rho_vals), color=CORAL, lw=2, linestyle="--", label=f"Mean {safe_mean(rho_vals):+.3f}")
        ax3.axvline(0, color=GRAY, lw=1)
        ax3.legend(frameon=False, fontsize=8)
    ax3.set_xlabel("Spearman rho")
    ax3.set_ylabel("Count")
    ax3.set_title(f"H1: entropy->hallucination rho\n({len(rho_vals)} valid-rho samples)")

    ax4 = fig.add_subplot(gs[1, 0:2])
    hall_trajs = [r["entropy_trajectory"] for r in full if r.get("has_hallucination") and r.get("entropy_trajectory")]
    clean_trajs = [r["entropy_trajectory"] for r in full if not r.get("has_hallucination") and r.get("entropy_trajectory")]

    def _plot_band(ax, trajs, color, label):
        if not trajs:
            return
        n = max(len(t) for t in trajs)
        arr = np.array([t + [float("nan")] * (n - len(t)) for t in trajs], float)
        m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
        steps = np.arange(n)
        ax.plot(steps, m, color=color, lw=2, label=label)
        ax.fill_between(steps, m - s, m + s, color=color, alpha=0.12)

    _plot_band(ax4, hall_trajs, CORAL, f"Hallucinated (n={len(hall_trajs)})")
    _plot_band(ax4, clean_trajs, BLUE, f"Grounded (n={len(clean_trajs)})")
    ax4.set_xlabel("Denoising step")
    ax4.set_ylabel("Mean entropy (nats)")
    ax4.set_title("H3: when do hallucinations crystallize?")
    ax4.legend(frameon=False, fontsize=9)

    ax5 = fig.add_subplot(gs[1, 2])
    abl = {k: v for k, v in all_records.items() if k.startswith("n=")}
    if len(abl) >= 2:
        ns = sorted(int(k.split("=")[1]) for k in abl)
        d_means = [safe_mean([r.get("refinement_delta", 0) for r in abl[f"n={n}"]]) for n in ns]
        ax5.plot(ns, d_means, "o-", color=AMBER, lw=2, markersize=7)
        ax5.axhline(0, color=GRAY, lw=1, linestyle="--")
        ax5.set_xscale("log", base=2)
        ax5.set_xticks(ns)
        ax5.set_xticklabels(ns)
        ax5.set_xlabel("N paths")
        ax5.set_ylabel("Mean Delta FactScore")
        ax5.set_title("N-paths ablation")
    else:
        ax5.text(0.5, 0.5, "Run ablation\nto populate", ha="center", va="center", transform=ax5.transAxes, color=GRAY)
        ax5.set_title("N-paths ablation")

    path = out_dir / "dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_stats(records: list[dict], label: str = "Full experiment"):
    hall = [r for r in records if r.get("has_hallucination")]
    clean = [r for r in records if not r.get("has_hallucination")]

    print(f"\n{'=' * 54}")
    print(f"  {label}")
    print(f"{'=' * 54}")
    print(f"  Samples total     : {len(records)}")
    print(f"  Hallucinated      : {len(hall)} ({100 * len(hall) / max(1, len(records)):.1f}%)")
    print(f"  Grounded          : {len(clean)}")
    print()

    d_all = safe_mean([r.get("refinement_delta", float('nan')) for r in records])
    d_hall = safe_mean([r.get("refinement_delta", float('nan')) for r in hall])
    d_clean = safe_mean([r.get("refinement_delta", float('nan')) for r in clean])

    print(f"  Delta FactScore (all)  : {d_all:+.4f}")
    print(f"  Delta FactScore (hall) : {d_hall:+.4f}")
    print(f"  Delta FactScore (clean): {d_clean:+.4f}")

    rho_vals = [
        r["spearman_rho"]
        for r in records
        if rho_not_skipped(r) and "spearman_rho" in r and not math.isnan(r.get("spearman_rho", float("nan")))
    ]
    print(f"\n  Spearman rho        : {safe_mean(rho_vals):+.4f}  (n={len(rho_vals)} valid-rho samples)")

    fs_b = safe_mean([r.get("fact_score_before", float("nan")) for r in records])
    fs_a = safe_mean([r.get("fact_score_after", float("nan")) for r in records])
    print(f"  FactScore before    : {fs_b:.4f}")
    print(f"  FactScore after     : {fs_a:.4f}")

    n_improved = sum(1 for r in records if r.get("refinement_delta", 0) > 0)
    print(f"\n  Samples improved    : {n_improved}/{len(records)} ({100 * n_improved / max(1, len(records)):.1f}%)")

    h2 = "SUPPORTED" if d_all > 0 else "NOT SUPPORTED"
    h1 = "SUPPORTED" if len(rho_vals) > 0 and safe_mean(rho_vals) > 0 else "WEAK"
    print(f"\n  H1 (entropy -> hallucination) : {h1}")
    print(f"  H2 (remasking reduces hall.) : {h2}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="Path to timestamped results directory")
    p.add_argument("--no_plots", action="store_true", help="Print stats only, skip plot generation")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_dir}")
    all_records = load_results(results_dir)

    if not all_records:
        print("  No raw_results.jsonl files found. Run the experiment first.")
        return

    for label, records in all_records.items():
        print_stats(records, label)

    if args.no_plots:
        return

    print(f"\nGenerating plots -> {plots_dir}")

    full = all_records.get("Full (n=8)", next(iter(all_records.values()), []))
    if full:
        plot_factscore_scatter(full, plots_dir)
        plot_delta_distribution(full, plots_dir)
        plot_entropy_trajectory(full, plots_dir)
        plot_entropy_vs_label(full, plots_dir)

    plot_npaths_ablation(all_records, plots_dir)
    plot_dashboard(all_records, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
