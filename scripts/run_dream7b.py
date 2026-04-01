#!/usr/bin/env python3
"""
scripts/run_dream7b.py
----------------------
Runs PaRaDe on Dream-v0-Instruct-7B for the second DLLM generalization result.

WHY THIS SCRIPT EXISTS
----------------------
Dream-7B may use a model-specific mask token id and is stored in the local HF
cache using the standard snapshots layout. This script:

1. Resolves the installed local Dream snapshot path
2. Detects Dream's mask token id from config.json when available
3. Sets DLLM_MASK_TOKEN_ID for the harness if needed
4. Runs the existing PaRaDe pipeline via run_experiment.py

USAGE
-----
cd /mnt/data2/achakr40/dllm-hallucination

# Step 1: inspect config only
.venv/bin/python scripts/run_dream7b.py --check_config_only

# Step 2: smoke test
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python scripts/run_dream7b.py \
    --n_samples 20 \
    --task_type QA \
    --output_dir results/dream7b_smoke

# Step 3: full run
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python scripts/run_dream7b.py \
    --n_samples 500 \
    --task_type QA \
    --output_dir results/dream7b_qa_500
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL_REF = "Dream-org/Dream-v0-Instruct-7B"
DEFAULT_DATA_PATH = "/mnt/data2/achakr40/dllm-hallucination/external/RAGTruth/dataset"
LLADA_MASK_TOKEN_ID = 126336


def resolve_model_dir(model_ref: str) -> Path:
    """
    Resolve Dream model reference to a concrete local directory.

    Accepts:
      - absolute local paths
      - HF cache directories
      - Dream HF model ids that are already cached on this server
    """
    candidate = Path(model_ref).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        known = {
            "Dream-org/Dream-v0-Instruct-7B": Path(
                "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B"
            ),
            "dream-7b-instruct": Path(
                "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B"
            ),
        }
        if model_ref not in known:
            raise ValueError(
                f"Unknown Dream model reference: {model_ref}. "
                "Use Dream-org/Dream-v0-Instruct-7B or an absolute local path."
            )
        resolved = known[model_ref]

    if not resolved.exists():
        raise FileNotFoundError(f"Dream model directory does not exist: {resolved}")

    if (resolved / "config.json").exists():
        return resolved

    refs_main = resolved / "refs" / "main"
    if refs_main.exists():
        snapshot = refs_main.read_text().strip()
        snapshot_dir = resolved / "snapshots" / snapshot
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = resolved / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshot_dirs:
            return snapshot_dirs[-1]

    raise FileNotFoundError(f"Could not resolve Dream snapshot under: {resolved}")


def detect_mask_token_id(model_dir: Path) -> int:
    """
    Detect the mask token id from config.json.
    Falls back to the known LLaDA value if not present.
    """
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            mask_id = cfg.get("mask_token_id")
            architectures = cfg.get("architectures", [])
            print(f"  [Dream config] path = {cfg_path}")
            print(f"  [Dream config] architectures = {architectures}")
            if mask_id is not None:
                print(f"  [Dream config] mask_token_id = {mask_id}")
                return int(mask_id)
        except Exception as e:
            print(f"  [warn] Could not parse {cfg_path}: {e}")

    print(
        f"  [warn] mask_token_id not found in config; "
        f"assuming LLaDA default ({LLADA_MASK_TOKEN_ID})"
    )
    return LLADA_MASK_TOKEN_ID


def build_command(args: argparse.Namespace, model_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--model_id",
        str(model_dir),
        "--data_path",
        args.data_path,
        "--n_paths",
        str(args.n_paths),
        "--num_steps",
        str(args.num_steps),
        "--gen_len",
        str(args.gen_len),
        "--max_samples",
        str(args.n_samples),
        "--output_dir",
        args.output_dir,
        "--split",
        args.split,
    ]
    if args.task_type:
        cmd += ["--task_type", args.task_type]
    if args.ablate_thresholds:
        cmd.append("--ablate_thresholds")
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL_REF)
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--task_type", default="QA", choices=["QA", "Summary", "Data2txt"])
    p.add_argument("--n_paths", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default="results/dream7b_qa_500")
    p.add_argument("--ablate_thresholds", action="store_true")
    p.add_argument("--check_config_only", action="store_true")
    args = p.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    mask_token_id = detect_mask_token_id(model_dir)

    print("=" * 60)
    print("  Dream-7B PaRaDe Runner")
    print("=" * 60)
    print(f"  Model ref  : {args.model_path}")
    print(f"  Model dir  : {model_dir}")
    print(f"  Data       : {args.data_path}")
    print(f"  Split      : {args.split}")
    print(f"  Task       : {args.task_type}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  N paths    : {args.n_paths}")
    print(f"  Num steps  : {args.num_steps}")
    print(f"  Gen len    : {args.gen_len}")
    print(f"  Output     : {args.output_dir}")
    print()

    if args.check_config_only:
        print(f"  mask_token_id = {mask_token_id}")
        if mask_token_id == LLADA_MASK_TOKEN_ID:
            print("  Same as LLaDA; no extra harness patch is needed at runtime.")
        else:
            print(f"  Different from LLaDA ({LLADA_MASK_TOKEN_ID}); DLLM_MASK_TOKEN_ID will be set.")
        return

    env = os.environ.copy()
    env["DLLM_MASK_TOKEN_ID"] = str(mask_token_id)

    cmd = build_command(args, model_dir)
    print("Running:")
    print(f"  {' '.join(cmd)}")
    print()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc != 0:
        print(f"ERROR: experiment exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
