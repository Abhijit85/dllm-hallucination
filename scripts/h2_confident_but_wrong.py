#!/usr/bin/env python3
"""
H2 — Confident-but-Wrong (CBW) Rate
====================================
Quantifies Limitation 2: "What fraction of hallucinations are
undetectable because all N chains agree on the wrong answer?"

If CBW < 20%, OSCAR covers the vast majority of hallucinations.
If CBW > 30%, honest about ceiling, position retrieval as complementary.

Zero GPU — post-processing on existing chain outputs.

Usage:
    python scripts/h2_confident_but_wrong.py \
        --results_dirs results/parade_triviaqa results/parade_hotpotqa \
        --output results/h2_cbw/
"""

import argparse
import json
import re
import string
from pathlib import Path


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def is_correct_any(text, gold_answers):
    """Check if text matches any gold answer."""
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    text_n = normalize(text)
    return any(normalize(g) in text_n for g in gold_answers)


def process_dir(results_dir):
    """Analyze CBW from a results directory."""
    jsonl_path = Path(results_dir) / "raw_results.jsonl"
    if not jsonl_path.exists():
        print(f"  SKIP: {jsonl_path} not found")
        return None

    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"  Loaded {len(records)} records")

    total_hallucinated = 0
    cbw_count = 0
    detectable_count = 0

    for rec in records:
        gold = rec.get("gold_answers", rec.get("gold", []))
        if isinstance(gold, str):
            gold = [gold]

        chain_outputs = rec.get("chain_outputs", rec.get("path_outputs",
                        rec.get("all_outputs", None)))

        if chain_outputs is None:
            chain_outputs = []
            for i in range(32):
                val = rec.get(f"path_{i}_output", rec.get(f"chain_{i}_output"))
                if val is not None:
                    chain_outputs.append(val)
                else:
                    break

        if not chain_outputs:
            gen = rec.get("generated_before", rec.get("generated", ""))
            if not is_correct_any(gen, gold):
                total_hallucinated += 1
                ent = rec.get("mean_cross_entropy", rec.get("entropy_score", None))
                if ent is not None and float(ent) < 0.01:
                    cbw_count += 1
                elif ent is not None:
                    detectable_count += 1
            continue

        primary = chain_outputs[0]
        if is_correct_any(primary, gold):
            continue

        total_hallucinated += 1

        normalized_outputs = [normalize(o)[:100] for o in chain_outputs]
        unique_outputs = set(normalized_outputs)

        if len(unique_outputs) == 1:
            cbw_count += 1
        else:
            detectable_count += 1

    if total_hallucinated == 0:
        return {
            "total_samples": len(records),
            "total_hallucinated": 0,
            "cbw_count": 0,
            "detectable_count": 0,
            "cbw_rate": 0,
            "detectable_rate": 0,
        }

    return {
        "total_samples": len(records),
        "total_hallucinated": total_hallucinated,
        "cbw_count": cbw_count,
        "detectable_count": detectable_count,
        "cbw_rate": cbw_count / total_hallucinated,
        "detectable_rate": detectable_count / total_hallucinated,
    }


def main():
    parser = argparse.ArgumentParser(description="H2: CBW Rate")
    parser.add_argument("--results_dirs", nargs="+", required=True)
    parser.add_argument("--output", default="results/h2_cbw/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_hall = 0
    total_cbw = 0

    for rdir in args.results_dirs:
        name = Path(rdir).name
        print(f"\n{name}:")
        result = process_dir(rdir)
        if result:
            all_results[name] = result
            total_hall += result["total_hallucinated"]
            total_cbw += result["cbw_count"]

    print(f"\n{'='*60}")
    print("H2: Confident-but-Wrong Analysis")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'Hall.':<8} {'CBW':<8} {'Detect.':<8} {'CBW%':<8}")
    print("-" * 57)

    for name, res in all_results.items():
        print(f"{name:<25} {res['total_hallucinated']:<8} "
              f"{res['cbw_count']:<8} {res['detectable_count']:<8} "
              f"{res['cbw_rate']*100:<7.1f}%")

    if total_hall > 0:
        overall = total_cbw / total_hall * 100
        print(f"\n  Overall CBW rate: {total_cbw}/{total_hall} = {overall:.1f}%")
        print()
        if overall < 20:
            print("  ✓ CBW < 20% — OSCAR covers the vast majority of hallucinations.")
            print("    Paper narrative: 'Knowledge gaps account for only X% of errors.'")
        elif overall < 30:
            print("  ~ CBW 20-30% — moderate. Frame as: 'OSCAR detects 70-80% of cases;")
            print("    retrieval augmentation addresses the remainder.'")
        else:
            print("  ⚠ CBW > 30% — significant. Be honest: 'OSCAR's entropy signal")
            print("    cannot detect errors where the model lacks knowledge entirely.")
            print("    This motivates integration with retrieval (Table 4).'")

    with open(out_dir / "h2_cbw_table.tex", "w") as f:
        f.write("% H2: Confident-but-wrong analysis\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Confident-but-wrong (CBW) analysis: fraction of hallucinated ")
        f.write("positions where all $N=8$ chains agree on the wrong answer.}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n")
        f.write("Dataset & Hallucinated & CBW (undetectable) & CBW Rate \\\\\n\\midrule\n")
        for name, res in all_results.items():
            f.write(f"{name} & {res['total_hallucinated']} & {res['cbw_count']} "
                    f"& {res['cbw_rate']*100:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(out_dir / "h2_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
