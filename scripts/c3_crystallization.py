#!/usr/bin/env python3
"""
C3 — Hallucination Crystallization Analysis
============================================
HIGH IMPACT: Transforms paper from methods → methods + discovery.
"When do DLM hallucinations crystallize during denoising?"

This script:
1. Runs N=8 chains with step-level logging (save H× at every denoising step)
2. Separates hallucinated vs grounded positions
3. Computes ΔH(t) = E[H× | hallucinated, step t] - E[H× | grounded, step t]
4. Generates Figure 3 data showing when the entropy gap emerges

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
    python scripts/c3_crystallization.py \
        --model_path /mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct \
        --dataset triviaqa \
        --n_samples 200 \
        --output results/c3_crystallization/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")


def load_dataset_samples(dataset_name, n_samples, cache_dir=None):
    """Load dataset samples."""
    try:
        from scripts.run_detection_auroc import load_triviaqa, load_hotpotqa
        if dataset_name == "triviaqa":
            return load_triviaqa(n_samples, cache_dir)
        if dataset_name == "hotpotqa":
            return load_hotpotqa(n_samples, cache_dir)
    except ImportError:
        pass
    from datasets import load_dataset
    if dataset_name == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                          cache_dir=cache_dir)
        return [{
            "id": str(i),
            "question": row["question"],
            "context": (row.get("search_results", {}).get("search_context", [""])[0][:2000]),
            "gold_answers": row["answer"]["aliases"],
        } for i, row in enumerate(ds) if i < n_samples]
    if dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation",
                          cache_dir=cache_dir)
        samples = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            ctx = " ".join(
                f"{t}: {''.join(s)}"
                for t, s in zip(row["context"]["title"], row["context"]["sentences"])
            )[:2000]
            samples.append({
                "id": str(i),
                "question": row["question"],
                "context": ctx,
                "gold_answers": [row["answer"]],
            })
        return samples
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_prompt(question, context=""):
    if context:
        return (
            "Given the following context, answer the question.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
    return f"Question: {question}\n\nAnswer:"


def run_chains_with_step_logging(model, tokenizer, prompt, gen_len=64,
                                 num_steps=128, n_paths=8, mask_token_id=None):
    """
    Run N chains and log the token distribution at EVERY denoising step.
    Returns:
        step_tokens: np.array of shape (n_paths, num_steps, gen_len) — token at each position
        final_outputs: list of N decoded strings
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    prompt_len = input_ids.shape[1]

    step_tokens = np.full((n_paths, num_steps + 1, gen_len), -1, dtype=np.int64)
    final_outputs = []

    for path_idx in range(n_paths):
        seed = 42 + path_idx * 1000
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        full_ids = torch.cat([
            input_ids,
            torch.full((1, gen_len), mask_token_id, dtype=torch.long, device="cuda"),
        ], dim=1)

        for step in range(num_steps):
            with torch.no_grad():
                logits = model(full_ids).logits

            gen_region = full_ids[0, prompt_len:]
            masked_pos = (gen_region == mask_token_id).nonzero(as_tuple=True)[0]

            if len(masked_pos) == 0:
                for s in range(step, num_steps + 1):
                    step_tokens[path_idx, s, :] = gen_region.cpu().numpy()
                break

            logits_masked = logits[0, prompt_len + masked_pos]
            probs = torch.softmax(logits_masked / 0.7, dim=-1)
            confidences = probs.max(dim=-1).values
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            n_unmask = max(1, len(masked_pos) // max(1, num_steps - step))

            if path_idx == 0:
                _, order = confidences.sort(descending=True)
            else:
                order = torch.randperm(len(masked_pos), device="cuda")

            unmask_indices = order[:n_unmask]
            for idx in unmask_indices:
                pos = masked_pos[idx]
                full_ids[0, prompt_len + pos] = sampled[idx]

            current = full_ids[0, prompt_len:].cpu().numpy()
            current_clean = np.where(current == mask_token_id, -1, current)
            step_tokens[path_idx, step + 1, :] = current_clean

        gen_ids = full_ids[0, prompt_len:]
        eos_id = tokenizer.eos_token_id
        clean = [
            t.item() for t in gen_ids
            if t.item() != mask_token_id and t.item() != eos_id
        ]
        text = tokenizer.decode(clean, skip_special_tokens=True).strip()
        final_outputs.append(text)

    return step_tokens, final_outputs


def compute_step_entropy(step_tokens, n_paths, num_steps, gen_len):
    """
    Compute cross-chain entropy at each step for each position.
    Returns: np.array of shape (num_steps+1, gen_len)
    """
    del n_paths
    entropy_by_step = np.zeros((num_steps + 1, gen_len))

    for step in range(num_steps + 1):
        for pos in range(gen_len):
            tokens_at_pos = step_tokens[:, step, pos]
            valid = tokens_at_pos[tokens_at_pos >= 0]
            if len(valid) < 2:
                entropy_by_step[step, pos] = 0.0
                continue
            from collections import Counter
            counts = Counter(valid.tolist())
            n = len(valid)
            ent = -sum((c / n) * np.log(c / n + 1e-10) for c in counts.values())
            entropy_by_step[step, pos] = ent

    return entropy_by_step


def is_position_hallucinated(pos_idx, final_outputs, gold_answers):
    """
    Simple heuristic: if the token at this position varies across chains
    AND the most common token doesn't appear in any gold answer, mark as hallucinated.
    """
    tokens_at_pos = set()
    for output in final_outputs:
        words = output.split()
        if pos_idx < len(words):
            tokens_at_pos.add(words[pos_idx].lower())

    if len(tokens_at_pos) <= 1:
        return False

    gold_tokens = set()
    for g in gold_answers:
        gold_tokens.update(g.lower().split())

    return not any(t in gold_tokens for t in tokens_at_pos)


def run(args):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    mask_token_id = getattr(config, "mask_token_id", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    print(f"Loading {args.dataset}...")
    samples = load_dataset_samples(args.dataset, args.n_samples)

    entropy_hall_by_step = defaultdict(list)
    entropy_ground_by_step = defaultdict(list)

    results = []

    for i, sample in enumerate(tqdm(samples, desc="Crystallization")):
        prompt = build_prompt(sample["question"], sample.get("context", ""))

        step_tokens, final_outputs = run_chains_with_step_logging(
            model, tokenizer, prompt,
            gen_len=args.gen_len,
            num_steps=args.num_steps,
            n_paths=args.n_paths,
            mask_token_id=mask_token_id,
        )

        entropy_by_step = compute_step_entropy(
            step_tokens, args.n_paths, args.num_steps, args.gen_len
        )

        for pos in range(min(args.gen_len, 20)):
            is_hall = is_position_hallucinated(pos, final_outputs, sample["gold_answers"])
            for step in range(0, args.num_steps + 1, max(1, args.num_steps // 20)):
                ent = entropy_by_step[step, pos]
                if is_hall:
                    entropy_hall_by_step[step].append(ent)
                else:
                    entropy_ground_by_step[step].append(ent)

        results.append({
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold": sample["gold_answers"],
            "outputs": final_outputs,
            "final_entropy": entropy_by_step[-1].tolist(),
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] Processed")

    steps_sorted = sorted(set(entropy_hall_by_step.keys()) | set(entropy_ground_by_step.keys()))

    curve_data = []
    for step in steps_sorted:
        hall_vals = entropy_hall_by_step.get(step, [])
        ground_vals = entropy_ground_by_step.get(step, [])
        h_hall = np.mean(hall_vals) if hall_vals else 0
        h_ground = np.mean(ground_vals) if ground_vals else 0
        delta_h = h_hall - h_ground
        curve_data.append({
            "step": step,
            "h_hallucinated": h_hall,
            "h_grounded": h_ground,
            "delta_h": delta_h,
            "n_hall": len(hall_vals),
            "n_ground": len(ground_vals),
        })

    print(f"\n{'='*60}")
    print("Crystallization Analysis — ΔH(t) Curve")
    print(f"{'='*60}")
    print(f"{'Step':<8} {'H(hall)':<10} {'H(ground)':<10} {'ΔH':<10}")
    print("-" * 40)
    for cd in curve_data:
        print(f"{cd['step']:<8} {cd['h_hallucinated']:<10.4f} "
              f"{cd['h_grounded']:<10.4f} {cd['delta_h']:<10.4f}")

    with open(out_dir / "crystallization_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)

    with open(out_dir / "raw_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    plot_script = out_dir / "plot_crystallization.py"
    with open(plot_script, "w") as f:
        f.write("""#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('crystallization_curve.json') as f:
    data = json.load(f)

steps = [d['step'] for d in data]
h_hall = [d['h_hallucinated'] for d in data]
h_ground = [d['h_grounded'] for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(steps, h_hall, 'r-o', label='Hallucinated', markersize=4)
ax1.plot(steps, h_ground, 'b-o', label='Grounded', markersize=4)
ax1.fill_between(steps, h_ground, h_hall, alpha=0.15, color='orange')
ax1.set_xlabel('Denoising step')
ax1.set_ylabel('Mean cross-chain entropy H×')
ax1.legend()
ax1.set_title('Entropy by position type')

delta = [d['delta_h'] for d in data]
ax2.plot(steps, delta, 'k-o', markersize=4)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Denoising step')
ax2.set_ylabel('ΔH(t) = H(hall) − H(ground)')
ax2.set_title('Entropy gap over denoising')

plt.tight_layout()
plt.savefig('figure3_crystallization.pdf', bbox_inches='tight')
plt.savefig('figure3_crystallization.png', dpi=300, bbox_inches='tight')
print('Saved figure3_crystallization.pdf/png')
""")

    print(f"\nSaved to {out_dir}/")
    print("  crystallization_curve.json — ΔH(t) data")
    print("  plot_crystallization.py    — matplotlib script for Figure 3")
    print("  raw_results.jsonl          — per-sample data")


def main():
    parser = argparse.ArgumentParser(description="C3: Hallucination Crystallization")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_paths", type=int, default=8)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--output", default="results/c3_crystallization/")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
