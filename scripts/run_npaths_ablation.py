#!/usr/bin/env python3
"""
Run the N-paths ablation in a single Python process.

This avoids reloading the LLaDA checkpoint for every ablation sub-run.
Outputs are written in the same layout expected by summarize_results.py:

  <output_root>/n_paths_1/raw_results.jsonl
  <output_root>/n_paths_1/aggregate.json
  ...
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.ragtruth_loader import load_ragtruth
from eval.metrics import aggregate
from models import create_harness
from run_experiment import run_single_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--n_paths_list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--top_k_percent", type=float, default=0.20)
    parser.add_argument("--entropy_threshold", type=float, default=0.8)
    parser.add_argument("--steps_per_token", type=float, default=0.5)
    parser.add_argument("--min_refine_steps", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--task_type", type=str, default=None, choices=["QA", "Summary", "Data2txt"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_aggregate(
    run_dir: Path,
    args: argparse.Namespace,
    records: list[dict],
    n_paths: int,
    attempted_samples: int,
) -> None:
    agg = aggregate(records)
    agg_dict = dataclasses.asdict(agg)
    agg_dict["n_paths"] = n_paths
    agg_dict["num_steps"] = args.num_steps
    agg_dict["gen_len"] = args.gen_len
    agg_dict["top_k_percent"] = args.top_k_percent
    agg_dict["total_time_min"] = round(
        sum(record.get("elapsed_s", 0) for record in records if "elapsed_s" in record) / 60,
        2,
    )
    agg_dict["n_attempted_samples"] = attempted_samples
    agg_dict["n_failed_samples"] = max(0, attempted_samples - len(records))
    with open(run_dir / "aggregate.json", "w", encoding="utf-8") as handle:
        json.dump(agg_dict, handle, indent=2)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading RAGTruth ({args.split}, task={args.task_type}, max={args.max_samples})")
    samples = load_ragtruth(
        split=args.split,
        task_type=args.task_type,
        max_samples=args.max_samples,
        data_path=args.data_path,
    )
    print(f"  {len(samples)} samples loaded")

    print(f"Loading {args.model_id}")
    harness = create_harness(model_id=args.model_id)
    print(f"  resolved to {harness.model_path}")

    base_args = copy.copy(args)
    for n_paths in args.n_paths_list:
        run_dir = output_root / f"n_paths_{n_paths}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_args = copy.copy(base_args)
        run_args.n_paths = n_paths

        print()
        print(f"-- N-paths ablation: n_paths={n_paths} ({len(samples)} samples) --")

        records: list[dict] = []
        raw_results_path = run_dir / "raw_results.jsonl"
        if raw_results_path.exists():
            raw_results_path.unlink()

        for sample in tqdm(samples, desc=f"n_paths={n_paths}"):
            try:
                record = run_single_sample(sample, harness, run_args)
                records.append(record)
            except Exception as exc:
                print(f"  [skip] sample {sample.sample_id}: {exc}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            with open(raw_results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_aggregate(run_dir, args, records, n_paths, len(samples))
        print(f"  Saved ablation run to {run_dir}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
