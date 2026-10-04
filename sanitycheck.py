#!/usr/bin/env python3
"""
sanitycheck.py
--------------
End-to-end validation on 20 QA samples with 4 paths.

Usage (local model + local data - recommended):
    python sanitycheck.py \
        --model_path /path/to/LLaDA-8B-Instruct \
        --data_path  /path/to/ragtruth

    # If your data is a single JSONL file with source_info included:
    python sanitycheck.py \
        --model_path /path/to/LLaDA-8B-Instruct \
        --data_path  /path/to/ragtruth/test.jsonl

    # Fall back to HuggingFace hub:
    python sanitycheck.py --model_path /path/to/LLaDA-8B-Instruct

Why LLaDA-8B-Instruct, not Base?
    RAGTruth tasks are RAG-style (context + question -> answer).
    Base model has no instruction-following capability, so it produces
    incoherent outputs for structured prompts and metrics become noise.
    Instruct is where RAG hallucination actually manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

RUN_DIR = Path("sanity_results")
CHECKPOINT_FILE = RUN_DIR / "checkpoint.jsonl"
SUMMARY_FILE = RUN_DIR / "summary.json"
VALIDATION_FILE = RUN_DIR / "validation.json"
RUN_CONFIG_FILE = RUN_DIR / "run_config.json"
CONSOLE_LOG_FILE = RUN_DIR / "console.log"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or local alias for the LLaDA-8B-Instruct weights directory",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Local path to a RAGTruth dataset directory or merged JSONL file. "
            "Omit to use the HuggingFace hub."
        ),
    )
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--n_paths", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--gen_len", type=int, default=64)
    p.add_argument("--entropy_threshold", type=float, default=0.5)
    p.add_argument("--refine_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default="sanity_results")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def hdr(msg):
    print("\n" + "=" * 58)
    print(f"  {msg}")
    print("=" * 58)


def ok(msg):
    print(f"  [OK] {msg}")


def info(msg):
    print(f"  [..] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def preview_text(value, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
    return repr(text[:limit])


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def init_run_dir(args) -> Path:
    global RUN_DIR, CHECKPOINT_FILE, SUMMARY_FILE, VALIDATION_FILE, RUN_CONFIG_FILE, CONSOLE_LOG_FILE

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sanity_{timestamp}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    RUN_DIR = run_dir
    CHECKPOINT_FILE = run_dir / "checkpoint.jsonl"
    SUMMARY_FILE = run_dir / "summary.json"
    VALIDATION_FILE = run_dir / "validation.json"
    RUN_CONFIG_FILE = run_dir / "run_config.json"
    CONSOLE_LOG_FILE = run_dir / "console.log"
    return run_dir


def save_run_config(args, resolved_model_path: str | None = None):
    payload = vars(args).copy()
    payload["resolved_model_path"] = resolved_model_path
    payload["created_at"] = datetime.now().isoformat()
    with open(RUN_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_validation(summary: dict) -> dict:
    validation = {
        "n_failed_is_zero": summary["n_failed"] == 0,
        "mean_spearman_rho_positive": (
            not math.isnan(summary["mean_spearman_rho"])
            and summary["mean_spearman_rho"] > 0
        ),
        "mean_refinement_delta_non_negative": (
            not math.isnan(summary["mean_refinement_delta"])
            and summary["mean_refinement_delta"] >= 0
        ),
    }
    validation["passed"] = all(validation.values())
    with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    return validation


def resolve_data_path(data_path: str | None) -> tuple[str, Path | None]:
    if data_path is None:
        return ("default", None)

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"data_path does not exist: {path}")

    if path.is_file():
        return ("jsonl", path)

    if (path / "response.jsonl").exists() and (path / "source_info.jsonl").exists():
        return ("dataset_dir", path)

    dataset_subdir = path / "dataset"
    if (dataset_subdir / "response.jsonl").exists() and (dataset_subdir / "source_info.jsonl").exists():
        return ("dataset_dir", dataset_subdir)

    raise FileNotFoundError(
        "Could not find RAGTruth dataset files under "
        f"{path}. Expected response.jsonl/source_info.jsonl, "
        "or a merged JSONL file."
    )


def filter_samples(samples, task_type: str, n_samples: int):
    filtered = [s for s in samples if s.task_type == task_type]
    return filtered[:n_samples]


def check_imports():
    hdr("Checkpoint 1 - imports")
    required = [
        ("torch", "torch", True),
        ("transformers", "transformers", True),
        ("scipy", "scipy", True),
        ("sklearn", "sklearn", True),
        ("numpy", "numpy", True),
        ("datasets", "datasets", False),
    ]
    for name, mod, mandatory in required:
        try:
            module = __import__(mod)
            ok(f"{name} {getattr(module, '__version__', '?')}")
        except ImportError as exc:
            if mandatory:
                fail(f"{name}: {exc}")
                sys.exit(1)
            warn(f"{name}: optional import missing ({exc})")

    try:
        from data.ragtruth_loader import load_ragtruth  # noqa: F401
        from eval.metrics import (  # noqa: F401
            disagreement_hallucination_correlation,
            fact_score,
            token_level_f1,
        )
        from models.llada_harness import DemaskingOrder, LLaDAHarness  # noqa: F401
        from strategies.parallel_remask import (  # noqa: F401
            compute_disagreement,
            random_remask_and_refine,
        )
        ok("all project modules import cleanly")
    except ImportError as exc:
        fail(f"project module import failed: {exc}")
        fail("Run from repo root so local imports resolve correctly.")
        sys.exit(1)


def check_gpu():
    hdr("Checkpoint 2 - GPU")
    if not torch.cuda.is_available():
        fail("CUDA not available")
        sys.exit(1)

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        vram = props.total_memory / 1024**3
        suffix = "OK" if vram >= 40 else "WARN (<40GB may OOM with 4 paths)"
        ok(f"GPU {idx}: {props.name}  {vram:.0f} GB  {suffix}")

    x = torch.ones(4, 4, dtype=torch.bfloat16, device="cuda")
    _ = x @ x
    ok("bfloat16 matmul on CUDA")


def check_model_path(model_path: str) -> str:
    hdr("Checkpoint 3 - model path")
    from models.llada_harness import resolve_local_llada_model

    try:
        resolved_path = Path(resolve_local_llada_model(model_path))
    except Exception as exc:
        fail(str(exc))
        sys.exit(1)

    config_file = resolved_path / "config.json"
    if not config_file.exists():
        fail(f"config.json not found in {resolved_path}")
        sys.exit(1)

    with open(config_file, encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg.get("_name_or_path", cfg.get("architectures", ["?"])[0])
    ok("config.json found")
    info(f"resolved path : {resolved_path}")
    info(f"model name    : {model_name}")
    info(f"architecture  : {cfg.get('architectures', ['?'])[0]}")
    info(f"hidden size   : {cfg.get('hidden_size', '?')}")
    info(f"num layers    : {cfg.get('num_hidden_layers', '?')}")

    path_str = str(resolved_path).lower()
    name_str = str(model_name).lower()
    if "instruct" in path_str or "instruct" in name_str:
        ok("Confirmed: Instruct variant")
    elif "base" in path_str or "base" in name_str:
        warn("This looks like LLaDA-8B-Base, not Instruct.")
        warn("Base does not follow RAG-style prompts well.")
        answer = input("  Continue anyway? [y/N]: ").strip().lower()
        if answer != "y":
            sys.exit(1)
    else:
        warn("Could not auto-detect Instruct vs Base from path/config.")

    for fname in ("tokenizer.json", "tokenizer_config.json"):
        if (resolved_path / fname).exists():
            ok(f"{fname} present")
        else:
            warn(f"{fname} not found - tokenizer may fail to load")

    return str(resolved_path)


def check_data(n_samples: int, data_path: str | None):
    hdr("Checkpoint 4 - RAGTruth data")
    from data.ragtruth_loader import (
        load_ragtruth,
        load_ragtruth_from_jsonl,
    )

    mode, resolved = resolve_data_path(data_path)
    if mode == "default":
        info("Using HuggingFace hub")
        t0 = time.time()
        samples = load_ragtruth(split="test", task_type="QA", max_samples=n_samples)
    elif mode == "dataset_dir":
        info(f"Loading local dataset directory: {resolved}")
        t0 = time.time()
        samples = load_ragtruth(
            split="test",
            task_type="QA",
            max_samples=n_samples,
            data_path=str(resolved),
        )
    else:
        info(f"Loading merged JSONL file: {resolved}")
        t0 = time.time()
        samples = filter_samples(load_ragtruth_from_jsonl(resolved), "QA", n_samples)

    elapsed = time.time() - t0
    if not samples:
        fail("No QA samples loaded")
        sys.exit(1)

    ok(f"{len(samples)} QA samples loaded ({elapsed:.1f}s)")
    hall = sum(1 for sample in samples if sample.has_hallucination)
    ok(f"hallucinated: {hall} / {len(samples)} ({100 * hall / len(samples):.0f}%)")

    sample = samples[0]
    info(f"sample[0] id={sample.sample_id} llm={sample.llm_name}")
    info(f"source   : {preview_text(sample.source_info)}")
    info(f"response : {preview_text(sample.response)}")
    if sample.spans:
        span = sample.spans[0]
        info(f"span[0]  : {span.text!r} ({span.kind} intensity={span.intensity})")

    return samples


def check_model_load(model_path: str):
    hdr("Checkpoint 5 - model load")
    from models.llada_harness import LLaDAHarness

    info(f"Loading LLaDA from: {model_path}")
    info("(bfloat16, CUDA)")
    t0 = time.time()
    harness = LLaDAHarness(model_id=model_path)
    ok(f"Model loaded ({time.time() - t0:.1f}s)")

    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ok(f"VRAM: {vram_used:.1f} / {vram_total:.0f} GB ({vram_total - vram_used:.1f} GB free)")
    if vram_total - vram_used < 10:
        warn("Less than 10GB VRAM free - 4 paths may OOM. Try --n_paths 2")

    return harness


def check_single_sample(harness, sample, args):
    hdr("Checkpoint 6 - single sample end-to-end")
    from eval.metrics import disagreement_hallucination_correlation, fact_score
    from models.llada_harness import DemaskingOrder
    from strategies.parallel_remask import (
        compute_disagreement,
        random_remask_and_refine,
    )

    info(f"Running {args.n_paths} paths ({args.num_steps} steps x {args.gen_len} tokens)")
    info("path 0: LEARNED order")
    info(f"paths 1-{args.n_paths - 1}: RANDOM order")

    t0 = time.time()
    result = harness.run_parallel_paths(
        prompt=sample.response[:50],
        source_info=sample.source_info,
        n_paths=args.n_paths,
        gen_len=args.gen_len,
        num_steps=args.num_steps,
        learned_paths=1,
        base_seed=args.seed,
    )
    elapsed = time.time() - t0
    ok(f"{args.n_paths} paths done ({elapsed:.1f}s, {elapsed / args.n_paths:.1f}s/path)")

    ent = result.token_entropy()
    gen_ent = ent[len(result.prompt_tokens):]
    info(
        "token entropy "
        f"mean={gen_ent.mean():.4f} max={gen_ent.max():.4f} "
        f"top-{int(100 * 0.2)}% flagged: pending"
    )

    report = compute_disagreement(result, entropy_threshold=args.entropy_threshold)
    info(
        f"flagged {len(report.high_entropy_positions)} generated tokens "
        f"(top {int(report.selection_fraction * 100)}%, cutoff={report.entropy_threshold:.4f})"
    )

    prompt_len = len(result.prompt_tokens)
    majority_txt = harness.decode(result.majority_vote()[prompt_len:])
    ok(f"majority vote: {majority_txt[:100]!r}")

    t0 = time.time()
    refinement = random_remask_and_refine(
        harness=harness,
        result=result,
        report=report,
        source_info=sample.source_info,
        refine_steps=args.refine_steps,
        refine_order=DemaskingOrder.LEARNED,
    )
    refined_txt = harness.decode(refinement.refined_tokens[prompt_len:])
    adaptive_steps = max(args.refine_steps, 16, int(math.ceil(refinement.n_remasked * 0.5)))
    ok(
        f"refinement done ({time.time() - t0:.1f}s, remasked {refinement.n_remasked} tokens, "
        f"steps={adaptive_steps})"
    )
    ok(f"refined text: {refined_txt[:100]!r}")

    fs_before = fact_score(majority_txt, sample.source_info)
    fs_after = fact_score(refined_txt, sample.source_info)
    ok(f"FactScore {fs_before:.4f} -> {fs_after:.4f} (delta {fs_after - fs_before:+.4f})")

    corr = disagreement_hallucination_correlation(
        report=report,
        sample=sample,
        tokenizer=harness.tokenizer,
        prompt_len=prompt_len,
    )
    if corr.skipped:
        warn("Spearman rho skipped for this sample (constant entropy or all-zero/all-one labels)")
    else:
        ok(f"Spearman rho = {corr.spearman_rho:.4f} (p={corr.p_value:.3f})")


def check_full_loop(harness, samples, args):
    hdr(f"Checkpoint 7 - full loop ({len(samples)} samples)")
    from eval.metrics import (
        disagreement_hallucination_correlation,
        fact_score,
        token_level_f1,
    )
    from models.llada_harness import DemaskingOrder
    from strategies.parallel_remask import (
        compute_disagreement,
        random_remask_and_refine,
    )

    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()

    records = []
    t_loop = time.time()

    for idx, sample in enumerate(samples):
        t0 = time.time()
        print(
            f"\n  [{idx + 1:02d}/{len(samples)}] id={sample.sample_id} "
            f"hall={sample.has_hallucination}",
            flush=True,
        )
        try:
            result = harness.run_parallel_paths(
                prompt=sample.response[:50],
                source_info=sample.source_info,
                n_paths=args.n_paths,
                gen_len=args.gen_len,
                num_steps=args.num_steps,
                learned_paths=1,
                base_seed=args.seed,
            )
            report = compute_disagreement(result, entropy_threshold=args.entropy_threshold)
            refinement = random_remask_and_refine(
                harness=harness,
                result=result,
                report=report,
                source_info=sample.source_info,
                refine_steps=args.refine_steps,
                refine_order=DemaskingOrder.LEARNED,
            )

            prompt_len = len(result.prompt_tokens)
            original_text = harness.decode(refinement.original_tokens[prompt_len:])
            refined_text = harness.decode(refinement.refined_tokens[prompt_len:])
            tf1 = token_level_f1(refinement.refined_tokens, sample, harness.tokenizer, prompt_len)
            corr = disagreement_hallucination_correlation(
                report=report,
                sample=sample,
                tokenizer=harness.tokenizer,
                prompt_len=prompt_len,
            )
            fs_before = fact_score(original_text, sample.source_info)
            fs_after = fact_score(refined_text, sample.source_info)

            rec = {
                "sample_id": sample.sample_id,
                "has_hallucination": sample.has_hallucination,
                "n_flagged": len(report.high_entropy_positions),
                "n_remasked": refinement.n_remasked,
                "token_f1": tf1.f1,
                "fact_score_before": fs_before,
                "fact_score_after": fs_after,
                "refinement_delta": fs_after - fs_before,
                "spearman_rho": corr.spearman_rho,
                "rho_skipped": corr.skipped,
                "elapsed_s": round(time.time() - t0, 2),
            }
            rho_text = "rho=SKIP" if corr.skipped else f"rho={corr.spearman_rho:.3f}"
            print(
                f"       F1={tf1.f1:.3f} "
                f"FS:{fs_before:.3f}->{fs_after:.3f}({fs_after - fs_before:+.3f}) "
                f"{rho_text} "
                f"remasked={refinement.n_remasked} "
                f"t={rec['elapsed_s']}s"
            )
        except Exception as exc:
            warn(f"sample {sample.sample_id} failed: {exc}")
            traceback.print_exc()
            rec = {"sample_id": sample.sample_id, "error": str(exc)}

        records.append(rec)
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    total = time.time() - t_loop
    clean = [rec for rec in records if "error" not in rec]

    def smean(key: str):
        vals = [
            rec[key]
            for rec in clean
            if rec.get(key) is not None and not math.isnan(rec[key])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "n_samples": len(samples),
        "n_succeeded": len(clean),
        "n_failed": len(records) - len(clean),
        "mean_token_f1": smean("token_f1"),
        "mean_fact_score_before": smean("fact_score_before"),
        "mean_fact_score_after": smean("fact_score_after"),
        "mean_refinement_delta": smean("refinement_delta"),
        "mean_spearman_rho": smean("spearman_rho"),
        "n_rho_computed": sum(1 for rec in clean if not rec.get("rho_skipped", False)),
        "total_time_min": round(total / 60, 2),
        "avg_time_per_sample_s": round(total / len(samples), 1),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    validation = write_validation(summary)

    hdr("Results")
    print(f"  Samples          : {summary['n_succeeded']}/{summary['n_samples']} OK")
    print(f"  Total time       : {summary['total_time_min']} min")
    print(f"  Avg per sample   : {summary['avg_time_per_sample_s']}s")
    print(f"  Mean token F1    : {summary['mean_token_f1']:.4f}")
    print(f"  FactScore before : {summary['mean_fact_score_before']:.4f}")
    print(f"  FactScore after  : {summary['mean_fact_score_after']:.4f}")
    print(f"  Refinement delta : {summary['mean_refinement_delta']:+.4f}")
    print(f"  Mean Spearman rho: {summary['mean_spearman_rho']:.4f}")
    print(f"  Samples with rho : {summary['n_rho_computed']}")
    print(f"\n  Run dir        : {RUN_DIR}")
    print(f"  Console log    : {CONSOLE_LOG_FILE}")
    print(f"  Raw checkpoint : {CHECKPOINT_FILE}")
    print(f"  Summary        : {SUMMARY_FILE}")
    print(f"  Validation     : {VALIDATION_FILE}")

    print()
    all_ok = True
    if summary["n_failed"]:
        warn(f"{summary['n_failed']} samples failed")
        all_ok = False
    else:
        ok("all samples completed without errors")

    rho = summary["mean_spearman_rho"]
    if summary["n_rho_computed"] > 0 and not math.isnan(rho) and rho > 0:
        ok(f"H1 looks healthy - entropy correlates with hallucination (rho={rho:.4f})")
    elif summary["n_rho_computed"] == 0:
        warn("H1 skipped - no samples had a valid non-constant hallucination signal")
    else:
        warn(f"H1 weak - rho={rho:.4f} (expected > 0; try more samples)")

    delta = summary["mean_refinement_delta"]
    if not math.isnan(delta) and delta >= 0:
        ok(f"H2 looks healthy - remasking non-negative FactScore delta ({delta:+.4f})")
    else:
        warn(f"H2 negative - delta={delta:+.4f} (try lower --entropy_threshold)")

    if all_ok:
        print("\n  [OK] Sanity check passed - ready for a larger run")
        print(
            "       python run_experiment.py "
            f"--model_id {args.model_path} "
            "--n_paths 8 --num_steps 64 --max_samples 500"
        )
    else:
        print("\n  [FAIL] Issues found - resolve before a larger run")

    return summary, validation


def main():
    args = parse_args()
    run_dir = init_run_dir(args)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fp = open(CONSOLE_LOG_FILE, "w", encoding="utf-8")
    sys.stdout = TeeStream(original_stdout, log_fp)
    sys.stderr = TeeStream(original_stderr, log_fp)

    try:
        print("\n" + "=" * 58)
        print("  DLLM Hallucination - Sanity Check")
        print(f"  run   : {run_dir}")
        print(f"  model : {args.model_path}")
        print(f"  data  : {args.data_path or 'HuggingFace hub'}")
        print(f"  {args.n_samples} samples x {args.n_paths} paths x {args.num_steps} steps")
        print("=" * 58)

        check_imports()
        check_gpu()
        resolved_model_path = check_model_path(args.model_path)
        save_run_config(args, resolved_model_path=resolved_model_path)
        samples = check_data(args.n_samples, args.data_path)
        harness = check_model_load(resolved_model_path)
        check_single_sample(harness, samples[0], args)
        check_full_loop(harness, samples, args)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()


if __name__ == "__main__":
    main()
