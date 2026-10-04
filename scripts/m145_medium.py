#!/usr/bin/env python3
"""
M1 — Refinement Steps T_r Sensitivity
M4 — GPU Memory Profiling
M5 — Bootstrap Confidence Intervals

Usage:
    # M1: T_r sensitivity (needs GPU)
    CUDA_VISIBLE_DEVICES=1 python scripts/m145_medium.py m1 \
        --model_path /path/to/LLaDA-8B-Instruct \
        --output results/m1_tr/

    # M4: Memory profiling (needs GPU)
    CUDA_VISIBLE_DEVICES=1 python scripts/m145_medium.py m4 \
        --model_path /path/to/LLaDA-8B-Instruct \
        --output results/m4_memory/

    # M5: Bootstrap CIs (no GPU)
    python scripts/m145_medium.py m5 \
        --results_base results/ \
        --output results/m5_bootstrap/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# M1 — T_r Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def run_m1(args):
    """Run OSCAR with different T_r values."""
    import subprocess

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_values = [2, 4, 8, 16]

    for tr in tr_values:
        run_dir = out_dir / f"tr_{tr}"
        if (run_dir / "raw_results.jsonl").exists():
            print(f"  SKIP: T_r={tr} already done")
            continue

        print(f"\n  Running T_r={tr}...")
        cmd = [
            sys.executable, "run_experiment.py",
            "--model_id", args.model_path,
            "--data_path", args.data_path,
            "--n_paths", "8",
            "--num_steps", "128",
            "--gen_len", "128",
            "--steps_per_token", str(tr),
            "--max_samples", str(args.n_samples),
            "--output_dir", str(run_dir),
        ]
        subprocess.run(cmd, check=True)

    # ── Aggregate ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"M1: T_r Sensitivity Results")
    print(f"{'='*60}")
    print(f"{'T_r':<8} {'F1 (%)':<10} {'ΔFS':<10}")
    print("-" * 28)

    for tr in tr_values:
        jsonl = out_dir / f"tr_{tr}" / "raw_results.jsonl"
        if jsonl.exists():
            records = [json.loads(l) for l in open(jsonl) if l.strip()]
            deltas = [r.get("refinement_delta", 0) for r in records]
            f1s = [r.get("token_f1", 0) for r in records]
            print(f"{tr:<8} {np.mean(f1s)*100:<10.1f} {np.mean(deltas):+<10.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# M4 — GPU Memory Profiling
# ═══════════════════════════════════════════════════════════════════════════════

def run_m4(args):
    """Profile peak VRAM for N ∈ {1, 4, 8, 16}."""
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import time

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model for memory profiling...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    mask_token_id = getattr(config, "mask_token_id", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    # Model size
    model_params = sum(p.numel() for p in model.parameters())
    model_mem_gb = model_params * 2 / (1024**3)  # bf16 = 2 bytes
    print(f"  Model parameters: {model_params/1e9:.1f}B")
    print(f"  Model memory: {model_mem_gb:.1f} GB")

    # Test prompt
    prompt = "Given the following context, answer the question.\n\nContext: The Eiffel Tower is in Paris.\n\nQuestion: Where is the Eiffel Tower?\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    prompt_len = input_ids.shape[1]
    gen_len = 64

    n_values = [1, 4, 8, 16]
    results = []

    print(f"\n{'N':<6} {'Peak VRAM (GB)':<16} {'Wall-clock (s)':<16} {'Overhead':<10}")
    print("-" * 48)

    baseline_time = None

    for N in n_values:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create batch of N copies
        batch_ids = torch.cat([
            torch.cat([input_ids,
                       torch.full((1, gen_len), mask_token_id,
                                  dtype=torch.long, device="cuda")], dim=1)
            for _ in range(N)
        ], dim=0)

        # Run a few denoising steps to measure peak memory
        t0 = time.time()
        num_test_steps = 10
        for step in range(num_test_steps):
            with torch.no_grad():
                logits = model(batch_ids).logits
            # Simulate unmasking
            for b in range(N):
                masked = (batch_ids[b] == mask_token_id).nonzero(as_tuple=True)[0]
                if len(masked) > 0:
                    probs = torch.softmax(logits[b, masked] / 0.7, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    n_unmask = max(1, len(masked) // max(1, num_test_steps - step))
                    perm = torch.randperm(len(masked))[:n_unmask]
                    for idx in perm:
                        batch_ids[b, masked[idx]] = sampled[idx]

        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        if baseline_time is None:
            baseline_time = elapsed

        overhead = elapsed / baseline_time

        results.append({
            "N": N,
            "peak_vram_gb": round(peak_mem, 1),
            "wall_clock_s": round(elapsed, 2),
            "overhead": round(overhead, 2),
        })

        print(f"{N:<6} {peak_mem:<16.1f} {elapsed:<16.2f} {overhead:<10.2f}×")

    # ── Save ─────────────────────────────────────────────────────────────

    with open(out_dir / "m4_memory.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "m4_memory_table.tex", "w") as f:
        f.write("% M4: GPU Memory Profiling\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Memory and wall-clock overhead vs. number of chains $N$.}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n")
        f.write("$N$ & Peak VRAM (GB) & Wall-clock & Overhead \\\\\n\\midrule\n")
        for r in results:
            bold = "\\textbf" if r["N"] == 8 else ""
            f.write(f"{bold}{{{r['N']}}} & {r['peak_vram_gb']} & "
                    f"{r['wall_clock_s']:.1f}s & {r['overhead']:.2f}$\\times$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n  Key argument: Model weights ({model_mem_gb:.0f} GB) are shared.")
    print(f"  The N× applies only to activation/sequence buffers.")
    print(f"  Actual scaling: {model_mem_gb:.0f} + ~{(results[-1]['peak_vram_gb']-model_mem_gb)/16:.1f}×N GB")


# ═══════════════════════════════════════════════════════════════════════════════
# M5 — Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════

def run_m5(args):
    """Bootstrap 95% CIs on AUROC for all methods."""
    from sklearn.metrics import roc_auc_score

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(args.results_base)
    n_bootstrap = args.n_bootstrap

    print(f"Bootstrap CIs ({n_bootstrap} resamples)")
    print(f"{'='*60}")

    results = {}

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        jsonl = subdir / "raw_results.jsonl"
        if not jsonl.exists():
            continue

        records = []
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            continue

        # Extract labels and scores
        labels = []
        scores = []
        for rec in records:
            em = rec.get("em_correct", rec.get("is_correct", rec.get("correct", None)))
            if em is None:
                continue
            ent = rec.get("mean_cross_entropy", rec.get("entropy_score", None))
            if ent is None:
                continue
            labels.append(1 - int(em))  # 1=hallucinated
            scores.append(float(ent))

        if len(set(labels)) < 2 or len(labels) < 10:
            continue

        labels = np.array(labels)
        scores = np.array(scores)

        # Point estimate
        auroc = roc_auc_score(labels, scores)

        # Bootstrap
        rng = np.random.RandomState(42)
        bootstrap_aurocs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(labels), size=len(labels), replace=True)
            bl = labels[idx]
            bs = scores[idx]
            if len(set(bl)) < 2:
                continue
            try:
                bootstrap_aurocs.append(roc_auc_score(bl, bs))
            except ValueError:
                continue

        ci_low = np.percentile(bootstrap_aurocs, 2.5)
        ci_high = np.percentile(bootstrap_aurocs, 97.5)

        results[subdir.name] = {
            "auroc": auroc * 100,
            "ci_low": ci_low * 100,
            "ci_high": ci_high * 100,
            "n_samples": len(labels),
            "n_bootstrap": len(bootstrap_aurocs),
        }

        print(f"  {subdir.name:<35} {auroc*100:.1f} [{ci_low*100:.1f}, {ci_high*100:.1f}]")

    with open(out_dir / "m5_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/m5_bootstrap.json")
    print("  If OSCAR's CI doesn't overlap with DynHD's, the improvement is significant.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M1/M4/M5 Medium Priority Scripts")
    parser.add_argument("mode", choices=["m1", "m4", "m5"],
                        help="Which experiment to run")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--results_base", default="results/")
    parser.add_argument("--output", default="results/medium/")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    if args.mode == "m1":
        assert args.model_path, "--model_path required for M1"
        run_m1(args)
    elif args.mode == "m4":
        assert args.model_path, "--model_path required for M4"
        run_m4(args)
    elif args.mode == "m5":
        run_m5(args)


if __name__ == "__main__":
    main()
