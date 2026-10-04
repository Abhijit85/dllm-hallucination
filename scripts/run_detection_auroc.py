#!/usr/bin/env python3
"""
scripts/run_detection_auroc.py
-------------------------------
Runs PaRaDe on TriviaQA, HotpotQA, and CommonsenseQA in two modes:

  Default (--full_pipeline):  detection + remasking  -> AUROC + DeltaAccuracy
  Detection only:             parallel paths + entropy -> AUROC only

HALLUCINATION LABEL
  label=1  LLaDA answer does NOT contain any gold answer  (hallucinated)
  label=0  LLaDA answer contains a gold answer            (grounded)
  Same definition used by TraceDet, TDGNet, DynHD.

METRICS
  AUROC           sample-level entropy vs hallucination label
  DeltaAccuracy   (full_pipeline) fraction wrong->correct minus fraction correct->wrong

WHY BETTER AUROC THAN RAGTRUTH
  RAGTruth:            hallucination = 2-3 spans in 128-token response
                       sample-level entropy washed out by ~125 correct tokens
                       -> AUROC ~0.54
  TriviaQA/HotpotQA:  hallucination = entire answer is wrong
                       all answer tokens uncertain -> cleaner signal
                       -> expected AUROC 0.60-0.68

USAGE
-----
# Step 1 — download (needs internet, CPU only, ~2 min):
python scripts/run_detection_auroc.py --download_only

# Step 2 — full pipeline on TriviaQA (~3.5 hr):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7 \\
python scripts/run_detection_auroc.py \\
    --model_path /mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct \\
    --dataset triviaqa --n_samples 500 --n_paths 8 \\
    --num_steps 32 --gen_len 64 --full_pipeline \\
    --output results/parade_triviaqa/

# Step 3 — full pipeline on HotpotQA (~3.5 hr, separate GPU):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6 \\
python scripts/run_detection_auroc.py \\
    --model_path /mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct \\
    --dataset hotpotqa --n_samples 500 --n_paths 8 \\
    --num_steps 32 --gen_len 64 --full_pipeline \\
    --output results/parade_hotpotqa/

# Step 4 — full pipeline on CommonsenseQA (~2.5 hr, separate GPU):
#   gen_len=32 sufficient — MC answers are short ("A: ...", ~5-10 tokens)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7 \\
python scripts/run_detection_auroc.py \\
    --model_path /mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct \\
    --dataset commonsenseqa --n_samples 500 --n_paths 8 \\
    --num_steps 32 --gen_len 32 --full_pipeline \\
    --output results/parade_commonsenseqa/
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class DetectionSample:
    sample_id:    str
    question:     str
    context:      str
    gold_answers: list
    meta:         dict = None


# ── dataset loading ───────────────────────────────────────────────────────────

def load_triviaqa(n_samples=500, cache_dir=None):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia",
                      split="validation", cache_dir=cache_dir)
    out = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        sr      = item.get("search_results", {})
        ctxs    = sr.get("search_context", [])
        context = " ".join(str(c) for c in ctxs[:3])[:1500]
        answers = list(item["answer"].get("normalized_aliases", []))
        main    = item["answer"].get("normalized_value", "")
        if main and main not in answers:
            answers = [main] + answers
        out.append(DetectionSample(
            sample_id    = item["question_id"],
            question     = item["question"],
            context      = context,
            gold_answers = [a.lower().strip() for a in answers if a],
        ))
    print(f"  TriviaQA: {len(out)} samples")
    return out


def load_hotpotqa(n_samples=500, cache_dir=None):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor",
                      split="validation", cache_dir=cache_dir)
    out = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        ctx    = item.get("context", {})
        titles = ctx.get("title", [])
        sents  = ctx.get("sentences", [])
        # per-document text — needed for abductive bridge score
        docs = {}
        for title, grp in zip(titles, sents):
            docs[title] = " ".join(grp if isinstance(grp, list) else [grp])
        flat   = " ".join(docs.values())[:1500]
        answer = item["answer"].lower().strip()
        gold   = ["yes", "no"] if answer in ("yes", "no") else [answer]
        # supporting_facts: titles of the two key documents for multi-hop
        sf_titles = list({f[0] for f in
                          item.get("supporting_facts", {}).get("title", [])})
        out.append(DetectionSample(
            sample_id    = item["id"],
            question     = item["question"],
            context      = flat,
            gold_answers = gold,
            meta         = {"docs": docs, "sf_titles": sf_titles[:2]},
        ))
    print(f"  HotpotQA: {len(out)} samples")
    return out


def load_commonsenseqa(n_samples=500, cache_dir=None):
    """
    Load CommonsenseQA validation split.

    CommonsenseQA is multiple-choice (5 options A-E).
    No retrieval context — model must answer from parametric knowledge.
    Gold answers: correct choice letter + choice text (both checked).

    This is the third benchmark used by DynHD alongside TriviaQA and HotpotQA.
    MC format means hallucination label is unambiguous: right letter or wrong.
    """
    from datasets import load_dataset
    ds = load_dataset("commonsense_qa", split="validation", cache_dir=cache_dir)
    out = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        # Build choices string for the prompt: "A) cat  B) dog  ..."
        labels = item["choices"]["label"]   # ["A","B","C","D","E"]
        texts  = item["choices"]["text"]    # ["cat","dog",...]
        choices_str = "  ".join(f"{lbl}) {t}" for lbl, t in zip(labels, texts))

        # Gold: correct letter AND correct text (both accepted in is_correct)
        correct_label = item.get("answerKey", "").upper().strip()
        correct_text  = ""
        for lbl, txt in zip(labels, texts):
            if lbl == correct_label:
                correct_text = txt.lower().strip()
                break

        gold = [correct_label.lower()]           # "a", "b", etc.
        if correct_text:
            gold.append(correct_text)            # also accept the answer text

        out.append(DetectionSample(
            sample_id    = item["id"],
            question     = item["question"],
            context      = choices_str,
            gold_answers = gold,
            meta         = {
                "choices":       dict(zip(labels, texts)),
                "correct_label": correct_label,
            },
        ))
    print(f"  CommonsenseQA: {len(out)} samples")
    return out


# ── answer correctness ────────────────────────────────────────────────────────

def _norm(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(generated, gold_answers):
    gn = _norm(generated)
    return any(_norm(a) and _norm(a) in gn for a in gold_answers)


def hall_label(generated, gold_answers):
    return 0 if is_correct(generated, gold_answers) else 1


# ── prompt ────────────────────────────────────────────────────────────────────

def build_prompt(sample, dataset=""):
    """
    Build prompt. CommonsenseQA uses chain-of-thought format.

    CoT is critical for CSQA: the model generates reasoning tokens before the
    answer letter. Entropy over reasoning tokens reflects genuine uncertainty
    (model doesn't know why), unlike entropy over a single letter token which
    just reflects MC option ambiguity. This is the key fix for CSQA AUROC.
    """
    if dataset == "commonsenseqa":
        return (f"Question: {sample.question}\n"
                f"Choices: {sample.context}\n\n"
                f"Let's think step by step to find the correct answer.\n"
                f"Reasoning:")
    if sample.context:
        return (f"Context: {sample.context[:800]}\n\n"
                f"Question: {sample.question}\n\nAnswer:")
    return f"Question: {sample.question}\n\nAnswer:"


# ── entropy scoring ───────────────────────────────────────────────────────────

def compute_scores(ent_raw, report=None, result=None, plen=0,
                   source_context="", dataset="", harness=None):
    """
    Compute sample-level detection scores from per-token cross-path entropy.

    Base scores (all datasets):
      mean_entropy, top20_entropy, max_entropy, std_entropy, above_median_frac

    HotpotQA-specific additions:
      source_weighted_entropy  — entropy weighted by inverse source coverage.
          Tokens NOT in the source passage get weight 1.0; tokens that appear
          in the source passage get weight 0.3. This boosts the signal for
          tokens that are uncertain AND ungrounded — the true hallucination
          pattern in multi-hop QA where the model may ground on one document
          but hallucinate facts from the other.
      src_top20_entropy        — top-20% of source-weighted entropy values.
      traj_early_entropy       — trajectory entropy weighted toward early steps.
          H3 showed hallucinations crystallize early; this gives more weight to
          high-entropy tokens at early denoising steps (exp decay schedule).

    CommonsenseQA-specific additions:
      neg_confidence_gap       — negative vote gap across N paths for the final
          answer token. When paths strongly agree on one letter, gap is high
          (confident = likely correct). When paths scatter across A/B/C/D/E,
          gap is low (uncertain = likely hallucinated). Negated for AUROC.
      vote_entropy             — Shannon entropy of the vote distribution across
          paths. Uniform votes = high entropy = likely wrong.
    """
    ent   = ent_raw.cpu().float()
    n     = max(1, len(ent))
    top_k = max(1, int(0.20 * n))

    scores = {
        "mean_entropy":      float(ent.mean()),
        "top20_entropy":     float(torch.topk(ent, k=top_k).values.mean()),
        "max_entropy":       float(ent.max()),
        "std_entropy":       float(ent.std()),
        "above_median_frac": float((ent > ent.median()).float().mean()),
    }

    # ── HotpotQA: source-weighted entropy ────────────────────────────────────
    if source_context and harness is not None and result is not None:
        try:
            src_ids = set(harness.tokenizer(
                source_context[:1200], add_special_tokens=False
            )["input_ids"])
            gen_ids = result.majority_vote()[plen:].tolist()
            if gen_ids:
                weights = torch.tensor([
                    0.3 if tok in src_ids else 1.0
                    for tok in gen_ids[:len(ent)]
                ], dtype=torch.float)
                sw = ent * weights
                scores["source_weighted_entropy"] = float(
                    sw.sum() / weights.sum())
                top_k2 = max(1, int(0.20 * len(sw)))
                scores["src_top20_entropy"] = float(
                    torch.topk(sw, k=top_k2).values.mean())
        except Exception:
            pass

    # ── HotpotQA: trajectory early-step weighting ────────────────────────────
    if report is not None and getattr(report, "path_entropy_trajectory", None) is not None:
        try:
            traj = report.path_entropy_trajectory.cpu().float()
            if traj.dim() == 2:
                # shape: (n_steps, gen_len)
                n_steps = traj.shape[0]
                sw = torch.exp(-0.05 * torch.arange(n_steps).float())
                sw /= sw.sum()
                weighted = (traj * sw[:, None]).sum(dim=0)  # (gen_len,)
                scores["traj_early_entropy"] = float(weighted.mean())
                top_k3 = max(1, int(0.20 * len(weighted)))
                scores["traj_early_top20"] = float(
                    torch.topk(weighted, k=top_k3).values.mean())
            elif traj.dim() == 1:
                # shape: (n_steps,) — already per-step aggregated
                n_steps = traj.shape[0]
                sw = torch.exp(-0.05 * torch.arange(n_steps).float())
                sw /= sw.sum()
                scores["traj_early_entropy"] = float((traj * sw).sum())
        except Exception:
            pass

    # ── CommonsenseQA: vote confidence gap ───────────────────────────────────
    if dataset == "commonsenseqa" and result is not None:
        try:
            pt = getattr(result, "path_tokens", None)
            if pt is not None:
                # last generated token per path = the answer letter
                final_toks = pt[:, -1].tolist()
                n_paths    = len(final_toks)
                votes      = Counter(final_toks)
                top2       = votes.most_common(2)
                if len(top2) >= 2:
                    gap = (top2[0][1] - top2[1][1]) / n_paths
                else:
                    gap = 1.0  # unanimous
                # negated: high gap = confident = correct = label 0
                # low gap = uncertain = hallucinated = label 1 → higher score
                scores["neg_confidence_gap"] = -gap
                # vote entropy: high = uncertain across many options = hallucinated
                scores["vote_entropy"] = float(
                    -sum((c / n_paths) * math.log(c / n_paths + 1e-10)
                         for c in votes.values()))
        except Exception:
            pass

    return scores



# ── abductive scoring (inference to the best explanation) ────────────────────

def build_abductive_prompt(question, answer, context, dataset,
                            choices_str="", letter=None, choice_text=None):
    """
    Abductive prompt: ask the model to justify its answer.

    Holmes principle: accept the hypothesis that best explains all evidence.
    A hallucinated answer cannot sustain a coherent explanation — parallel
    chains diverge when asked to justify a false claim. A correct answer
    produces consistent low-entropy explanations because the model has a
    stable internal representation of the true fact.

    For CommonsenseQA (contrastive): generate explanations for EVERY choice.
    The best-explained choice (lowest explanation entropy) is the correct one.
    If the model did not pick it, it hallucinated.
    """
    if dataset == "commonsenseqa":
        return (f"Question: {question}\n"
                f"Choices: {choices_str}\n\n"
                f"The answer is {letter} ({choice_text}) because:")
    ctx_prefix = f"Context: {context[:600]}\n\n" if context else ""
    return (f"{ctx_prefix}Question: {question}\n"
            f"Answer: {answer}\n\nThis answer is correct because:")


def _explanation_entropy(prompt, harness, n_paths, gen_len, num_steps):
    """
    Run N parallel chains on an explanation prompt.
    Returns (mean_entropy, top20_entropy) of explanation tokens.
    Lower = more coherent explanation = better-supported answer.
    """
    try:
        from strategies.parallel_remask import compute_disagreement
        result  = harness.run_parallel_paths(
            prompt        = prompt,
            source_info   = "",
            n_paths       = n_paths,
            gen_len       = gen_len,
            num_steps     = num_steps,
            learned_paths = 1,
            base_seed     = 99,
        )
        report  = compute_disagreement(result, top_k_percent=0.20)
        plen    = len(result.prompt_tokens)
        ent     = report.token_entropy[plen:].cpu().float()
        n       = max(1, len(ent))
        top_k   = max(1, int(0.20 * n))
        return (float(ent.mean()),
                float(torch.topk(ent, k=top_k).values.mean()),
                result)
    except Exception:
        return (float("nan"), float("nan"), None)



def _self_verify(question, answer, context, harness, n_paths, num_steps,
                 choices_str="", letter=None, choice_text=None, dataset=""):
    """
    DLLM-native forced-choice self-verification via direct logit scoring.

    WHY THE PREVIOUS VERSION FAILED
    --------------------------------
    The original implementation called run_parallel_paths(gen_len=4) which
    appends 4 MASK tokens and fills them freely. LLaDA does not reliably
    emit "Yes" or "No" as the first generated token — it generates whatever
    maximizes likelihood given the context, which is usually punctuation or
    a continuation of the question. All paths hit "n_other" → constant score
    → AUROC = 0.5 exactly.

    THE FIX: FORCED-CHOICE VIA DIRECT LOGIT SCORING
    -------------------------------------------------
    Instead of generating freely, we score two choices directly:

        "[context] Q: ... A: ... Is this answer correct? [MASK]"

    We run ONE forward pass with a single MASK at the verification position.
    The DLLM predicts the full vocabulary distribution at that MASK token.
    We extract P("Yes" token) and P("No" token) directly from the logits.

    This is mechanistically native to DLLMs — filling a single masked token
    given full bidirectional context is exactly what they are trained to do.
    It is also faster than parallel chain generation (one forward pass vs
    n_paths × num_steps forward passes).

    MECHANISTIC DISTINCTION FROM AR SELF-VERIFICATION
    --------------------------------------------------
    AR methods (Prism, SelfCheckGPT) compute P(Yes) from next-token logits
    after generating the entire answer — a single causal forward pass using
    left-to-right context only. PaRaDe uses a DLLM's bidirectional attention
    over the FULL sequence including the answer: the model sees both the
    question AND the answer simultaneously when predicting Yes/No, allowing
    it to directly compare the answer against the question context. This
    bidirectional verification signal is structurally unavailable to AR models.

    Returns:
        fc_p_no       raw P(No token) at the MASK position
        fc_p_yes      raw P(Yes token)
        fc_no_frac    P(No)/(P(Yes)+P(No)) — the primary AUROC signal
                      Higher = model assigns more probability to "No"
                      = model doubts its own answer = hallucinated
        fc_log_ratio  log(P(No)/P(Yes)) — unbounded version for ranking
    """
    try:
        # ── 1. build verification stem ────────────────────────────────────────
        if dataset == "commonsenseqa" and letter is not None:
            # Full choice text in verification — significantly stronger signal
            # than plain "Is this answer correct?" for MC tasks.
            # Probe result: 0.842 AUROC on 20 samples vs 0.621 baseline.
            # "Is this the best answer?" phrasing outperforms "Is this correct?"
            # because it activates comparative reasoning across all choices.
            stem = (f"Question: {question}\n"
                    f"Choices: {choices_str}\n"
                    f"Selected: {letter}) {choice_text}\n"
                    f"Is this the best answer?")
        else:
            ctx_prefix = f"Context: {context[:400]}\n\n" if context else ""
            stem = (f"{ctx_prefix}Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Is this answer correct?")

        # ── 2. tokenize and append MASK ───────────────────────────────────────
        device  = next(harness.model.parameters()).device
        stem_ids = harness.tokenizer.encode(
            stem, add_special_tokens=True, return_tensors="pt"
        ).to(device)

        mask_id  = harness.mask_token_id  # 126336 LLaDA, 151666 Dream
        mask_tok = torch.tensor([[mask_id]], device=device)
        full_ids = torch.cat([stem_ids, mask_tok], dim=1)  # (..., stem+1)

        # ── 3. single forward pass — get logits at MASK position ──────────────
        # Use Dream-compatible forward pass if available (tok_idx + 4D attn mask)
        # Falls back to standard forward pass for LLaDA
        if hasattr(harness, 'forward_single'):
            logits = harness.forward_single(full_ids).float()  # (1, seq_len, vocab)
        else:
            with torch.no_grad():
                outputs = harness.model(full_ids)
                logits  = outputs.logits.float()        # (1, seq_len, vocab)
        mask_pos    = full_ids.shape[1] - 1             # last position = MASK
        mask_logits = logits[0, mask_pos, :]            # (vocab,)
        probs       = torch.softmax(mask_logits, dim=-1)

        # ── 4. get P(Yes) and P(No) ────────────────────────────────────────
        # Try both space-prefixed and bare variants — tokenizers differ
        yes_candidates = [" Yes", "Yes", " yes", "yes", " True", "True"]
        no_candidates  = [" No",  "No",  " no",  "no",  " False","False"]

        def best_prob(candidates):
            best = 0.0
            for w in candidates:
                ids = harness.tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    best = max(best, probs[ids[0]].item())
            return best

        p_yes = best_prob(yes_candidates)
        p_no  = best_prob(no_candidates)
        total = p_yes + p_no + 1e-10

        return {
            "fc_p_no":      p_no,
            "fc_p_yes":     p_yes,
            "fc_no_frac":   p_no  / total,                           # primary signal
            "fc_log_ratio": math.log((p_no + 1e-10) / (p_yes + 1e-10)),  # unbounded
        }

    except Exception:
        return {}

def compute_abductive_scores(sample, gen_before, harness, dataset,
                              n_paths=4, gen_len=32, num_steps=16):
    """
    Abductive PaRaDe: inference to the best explanation.

    Open-ended tasks (HotpotQA, TriviaQA):
      Explanation entropy of model's own answer.
      High entropy = cannot justify = hallucinated.

      HotpotQA bridge score:
      explanation must reference BOTH supporting docs (multi-hop necessity).
      bridge = min(coverage_doc1, coverage_doc2) — bottlenecked by the weaker.
      Low bridge = explanation only uses one doc = likely hallucinated.

    CommonsenseQA (contrastive abduction):
      Run explanation pass for each of 5 choices independently.
      best_choice = argmin(explanation_entropy) = most coherent option.
      abduct_chose_best = 1 if model did NOT pick best_choice (hallucinated).
      abduct_entropy_gap = entropy(model's choice) - entropy(best choice).
      Larger gap = model's explanation less coherent = more hallucinated.
    """
    scores = {}

    if dataset == "commonsenseqa":
        meta    = sample.meta or {}
        choices = meta.get("choices", {})
        if not choices:
            return scores
        choices_str = "  ".join(f"{lbl}) {t}" for lbl, t in choices.items())
        exp_ents = {}
        for letter, text in choices.items():
            prompt = build_abductive_prompt(
                question=sample.question, answer=gen_before,
                context="", dataset=dataset,
                choices_str=choices_str, letter=letter, choice_text=text)
            mean_e, top20_e, _ = _explanation_entropy(
                prompt, harness, n_paths, gen_len, num_steps)
            exp_ents[letter] = mean_e

        valid = {k: v for k, v in exp_ents.items() if not math.isnan(v)}
        if not valid:
            return scores

        best_letter = min(valid, key=valid.get)
        # Extract model's answer letter
        gen_letter = ""
        for ch in gen_before.upper():
            if ch in choices:
                gen_letter = ch
                break
        if not gen_letter:
            gen_letter = gen_before.strip().upper()[:1]

        # Hallucination signals
        scores["abduct_chose_best"]   = float(gen_letter != best_letter)
        scores["abduct_exp_entropy"]  = valid.get(gen_letter, float("nan"))
        scores["abduct_best_entropy"] = valid[best_letter]
        scores["abduct_entropy_gap"]  = (
            valid.get(gen_letter, valid[best_letter]) - valid[best_letter])

        # Self-verification: "Is [model's chosen answer] correct? Yes/No"
        # Run on model's actual choice — no_frac high = model doubts itself
        choices_str_sv = "  ".join(f"{lbl}) {t}" for lbl, t in choices.items())
        chosen_text    = choices.get(gen_letter, "")
        sv = _self_verify(
            question=sample.question, answer=gen_before, context="",
            harness=harness, n_paths=n_paths, num_steps=num_steps,
            choices_str=choices_str_sv, letter=gen_letter,
            choice_text=chosen_text, dataset=dataset)
        scores.update(sv)

    else:
        # Open-ended: TriviaQA / HotpotQA
        prompt = build_abductive_prompt(
            question=sample.question, answer=gen_before,
            context=sample.context, dataset=dataset)
        mean_e, top20_e, exp_result = _explanation_entropy(
            prompt, harness, n_paths, gen_len, num_steps)
        scores["abduct_exp_mean"]  = mean_e
        scores["abduct_exp_top20"] = top20_e

        # Self-verification: "Is this answer correct? Yes/No"
        # Mechanistically distinct from AR self-verification: N parallel
        # DLLM chains vote on a masked Yes/No completion simultaneously.
        sv = _self_verify(
            question=sample.question, answer=gen_before,
            context=sample.context, harness=harness,
            n_paths=n_paths, num_steps=num_steps, dataset=dataset)
        scores.update(sv)

        # HotpotQA bridge score
        if dataset == "hotpotqa" and exp_result is not None:
            meta      = sample.meta or {}
            docs      = meta.get("docs", {})
            sf_titles = meta.get("sf_titles", [])
            if len(sf_titles) >= 2:
                doc1_text = docs.get(sf_titles[0], "")
                doc2_text = docs.get(sf_titles[1], "")
                if doc1_text and doc2_text:
                    try:
                        exp_plen = len(exp_result.prompt_tokens)
                        exp_ids  = set(exp_result.majority_vote()[exp_plen:].tolist())
                        doc1_ids = set(harness.tokenizer(
                            doc1_text[:600], add_special_tokens=False)["input_ids"])
                        doc2_ids = set(harness.tokenizer(
                            doc2_text[:600], add_special_tokens=False)["input_ids"])
                        cov1 = len(exp_ids & doc1_ids) / max(1, len(doc1_ids))
                        cov2 = len(exp_ids & doc2_ids) / max(1, len(doc2_ids))
                        bridge = min(cov1, cov2)
                        # Negate: low bridge = hallucinated → higher score for AUROC
                        scores["neg_bridge_score"] = -bridge
                        scores["abduct_bridge"]    = bridge
                        scores["abduct_cov1"]      = cov1
                        scores["abduct_cov2"]      = cov2
                    except Exception:
                        pass
    return scores


# ── main loop ─────────────────────────────────────────────────────────────────

def run(args):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.llada_harness import DemaskingOrder, LLaDAHarness
    from strategies.parallel_remask import (
        compute_disagreement,
        random_remask_and_refine,
    )

    print(f"\nLoading {args.dataset} ({args.n_samples} samples)...")
    loaders = {
        "triviaqa":      load_triviaqa,
        "hotpotqa":      load_hotpotqa,
        "commonsenseqa": load_commonsenseqa,
    }
    samples = loaders[args.dataset](args.n_samples, args.cache_dir)

    print(f"Loading model: {args.model_path}")
    _is_dream = any(k in args.model_path.lower()
                    for k in ("dream",))
    if _is_dream:
        # Dream-7B requires transformers==4.46.2 — must run under .venv_dream
        # See models/dream_harness_native.py for compatibility notes
        try:
            import transformers as _tf
            _tv = _tf.__version__
            if not _tv.startswith("4.46"):
                print(
                    f"\n  WARNING: Dream-7B requires transformers==4.46.2 "
                    f"but found {_tv}.\n"
                    f"  Generation will be incoherent. Use .venv_dream:\n"
                    f"    PYTHONPATH=. .venv_dream/bin/python {__file__} ...\n"
                )
        except Exception:
            pass
        from models.dream_harness_native import DreamHarnessNative
        harness = DreamHarnessNative(model_id=args.model_path)
    else:
        harness = LLaDAHarness(model_id=args.model_path)

    mode_parts = []
    if args.full_pipeline:
        mode_parts.append("detection + remasking")
    if args.abductive:
        mode_parts.append("abductive + self-verify")
    mode = " | ".join(mode_parts) if mode_parts else "detection only"
    print(f"Mode: {mode}\n")

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "raw_results.jsonl"

    per_sample = []

    for sample in tqdm(samples, desc=f"PaRaDe [{args.dataset}]"):
        try:
            result = harness.run_parallel_paths(
                prompt        = build_prompt(sample, args.dataset),
                source_info   = sample.context,
                n_paths       = args.n_paths,
                gen_len       = args.gen_len,
                num_steps     = args.num_steps,
                learned_paths = 1,
                base_seed     = 42,
            )
            report = compute_disagreement(result, top_k_percent=0.20)
            plen   = len(result.prompt_tokens)
            ent    = report.token_entropy[plen:]

            gen_before = harness.tokenizer.decode(
                result.majority_vote()[plen:].tolist(),
                skip_special_tokens=True).strip()
            label_before = hall_label(gen_before, sample.gold_answers)
            scores       = compute_scores(
                ent, report=report, result=result, plen=plen,
                source_context=sample.context,
                dataset=args.dataset, harness=harness)

            # ── abductive second pass ────────────────────────────────────
            if args.abductive:
                abd = compute_abductive_scores(
                    sample     = sample,
                    gen_before = gen_before,
                    harness    = harness,
                    dataset    = args.dataset,
                    n_paths    = args.abduct_paths,
                    gen_len    = args.abduct_gen_len,
                    num_steps  = args.abduct_steps,
                )
                scores.update(abd)

            # ── first-step entropy (Dream only) ──────────────────────────
            # Compute entropy at the first denoising step before any token
            # is committed. Equivalent to LLaDA's traj_early_entropy but
            # implemented via a single forward_single() on a fully masked
            # sequence. Captures pre-commitment uncertainty — strongest
            # signal when the model is uncertain from the start (multi-hop).
            if hasattr(harness, 'compute_first_step_entropy'):
                try:
                    fse = harness.compute_first_step_entropy(
                        prompt      = build_prompt(sample, args.dataset),
                        source_info = sample.context,
                        gen_len     = args.gen_len,
                    )
                    k20 = max(1, int(len(fse) * 0.20))
                    scores['first_step_entropy_mean']  = fse.mean().item()
                    scores['first_step_entropy_top20'] = fse.topk(k20).values.mean().item()
                    scores['first_step_entropy_max']   = fse.max().item()
                except Exception as _fse_err:
                    scores['first_step_entropy_mean']  = 0.5
                    scores['first_step_entropy_top20'] = 0.5
                    scores['first_step_entropy_max']   = 0.5

            rec = {
                "sample_id":        sample.sample_id,
                "question":         sample.question[:120],
                "generated_before": gen_before[:200],
                "gold_answers":     sample.gold_answers[:3],
                "label":            label_before,
                "correct_before":   label_before == 0,
                **scores,
            }

            # CommonsenseQA: skip remasking — choice-token entropy reflects
            # option ambiguity, not factual error. Remasking breaks correct answers.
            do_remask = args.full_pipeline and args.dataset != "commonsenseqa"
            if do_remask:
                refinement = random_remask_and_refine(
                    harness          = harness,
                    result           = result,
                    report           = report,
                    source_info      = sample.context,
                    refine_steps     = None,
                    steps_per_token  = 0.5,
                    min_refine_steps = 16,
                    refine_order     = DemaskingOrder.LEARNED,
                )
                gen_after    = harness.decode(
                    refinement.refined_tokens[plen:]).strip()
                label_after  = hall_label(gen_after, sample.gold_answers)
                rec.update({
                    "generated_after": gen_after[:200],
                    "label_after":     label_after,
                    "correct_after":   label_after == 0,
                    "corrected":       (label_before == 1 and label_after == 0),
                    "broken":          (label_before == 0 and label_after == 1),
                    "n_remasked":      refinement.n_remasked,
                })

            per_sample.append(rec)
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        except Exception as e:
            print(f"  [skip] {sample.sample_id}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── aggregate ─────────────────────────────────────────────────────────────
    n_total     = len(per_sample)
    n_hall      = sum(r["label"] for r in per_sample)
    n_grnd      = n_total - n_hall
    acc_before  = sum(r["correct_before"] for r in per_sample) / max(1, n_total)
    all_labels  = [r["label"] for r in per_sample]

    print(f"\n{'='*62}")
    print(f"  PaRaDe [{args.dataset.upper()}]  —  {mode}")
    print(f"{'='*62}")
    print(f"  N           : {n_total}")
    print(f"  Hallucinated: {n_hall} ({100*n_hall/max(1,n_total):.1f}%)")
    print(f"  Grounded    : {n_grnd} ({100*n_grnd/max(1,n_total):.1f}%)")
    print(f"  LLaDA acc   : {100*acc_before:.1f}%")

    # AUROC
    auroc_results = {}
    if n_hall > 0 and n_grnd > 0:
        print()
        print(f"  {'Score':<22}  {'AUROC':>7}  {'AP':>7}")
        print(f"  {'-'*22}  {'-'*7}  {'-'*7}")
        # Include dataset-specific improved scores
        all_score_keys = [
            "mean_entropy", "top20_entropy", "max_entropy",
            "std_entropy", "above_median_frac",
            # HotpotQA improvements
            "source_weighted_entropy", "src_top20_entropy",
            "traj_early_entropy", "traj_early_top20",
            # CommonsenseQA improvements
            "neg_confidence_gap", "vote_entropy",
            # Abductive PaRaDe — open-ended
            "abduct_exp_mean", "abduct_exp_top20",
            # Abductive PaRaDe — HotpotQA bridge
            "neg_bridge_score",
            # Abductive PaRaDe — CommonsenseQA contrastive
            "abduct_chose_best", "abduct_exp_entropy",
            "abduct_entropy_gap",
            # First-step entropy (Dream-native, pre-commitment uncertainty)
            "first_step_entropy_mean", "first_step_entropy_top20", "first_step_entropy_max",
            # Forced-choice verification (DLLM-native, single forward pass)
            "fc_no_frac", "fc_log_ratio", "fc_p_no",
        ]
        for key in all_score_keys:
            sc = [r[key] for r in per_sample
                  if key in r and not math.isnan(r[key])]
            lb = all_labels[:len(sc)]
            if len(sc) < 5 or len(set(lb)) < 2:
                continue
            auroc = roc_auc_score(lb, sc)
            ap    = average_precision_score(lb, sc)
            auroc_results[key] = {"AUROC": round(auroc,4), "AP": round(ap,4)}
            print(f"  {key:<22}  {auroc:>7.4f}  {ap:>7.4f}")

    best_name  = max(auroc_results,
                     key=lambda k: auroc_results[k]["AUROC"],
                     default="—")
    best_auroc = (auroc_results[best_name]["AUROC"]
                  if auroc_results else float("nan"))

    # Delta accuracy
    delta_acc   = float("nan")
    n_corrected = n_broken = 0
    if args.full_pipeline:
        acc_after   = sum(r["correct_after"] for r in per_sample) / max(1, n_total)
        n_corrected = sum(r.get("corrected", False) for r in per_sample)
        n_broken    = sum(r.get("broken",    False) for r in per_sample)
        delta_acc   = acc_after - acc_before
        print()
        print("  --- Reduction ---")
        print(f"  Acc before : {100*acc_before:.1f}%")
        print(f"  Acc after  : {100*acc_after:.1f}%")
        print(f"  DeltaAcc   : {delta_acc:+.4f}  ({100*delta_acc:+.1f} pp)")
        print(f"  Wrong->OK  : {n_corrected}  "
              f"({100*n_corrected/max(1,n_hall):.1f}% of hallucinated)")
        print(f"  OK->Wrong  : {n_broken}  "
              f"({100*n_broken/max(1,n_grnd):.1f}% of grounded)")

    # Comparison table
    competitors = {
        "triviaqa": [
            ("TraceDet (Chang+25)", "~0.71"),
            ("DynHD (2603.16459)",  "~0.73"),
        ],
        "hotpotqa": [
            ("TraceDet (Chang+25)", "~0.69"),
            ("TDGNet (2602.08048)", "~0.72"),
            ("DynHD (2603.16459)",  "~0.71"),
        ],
        "commonsenseqa": [
            ("DynHD (2603.16459)",  "~0.72"),
        ],
    }
    da_header = f"  {'DeltaAcc':>10}" if args.full_pipeline else ""
    print()
    print(f"  {'Method':<28} {'Train-free':>10} {'Reduces':>8} {'AUROC':>7}{da_header}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*7}", end="")
    if args.full_pipeline:
        print(f"  {'-'*10}", end="")
    print()
    for name, auroc_str in competitors.get(args.dataset, []):
        print(f"  {name:<28} {'No':>10} {'No':>8} {auroc_str:>7}", end="")
        if args.full_pipeline:
            print(f"  {'N/A':>10}", end="")
        print()
    pa_str  = f"{best_auroc:.4f}" if not math.isnan(best_auroc) else "N/A"
    da_str  = f"{delta_acc:+.4f}" if not math.isnan(delta_acc) else "N/A"
    print(f"  {'PaRaDe (ours)':<28} {'Yes':>10} {'Yes':>8} {pa_str:>7}", end="")
    if args.full_pipeline:
        print(f"  {da_str:>10}", end="")
    print()

    # Save
    summary = {
        "dataset":         args.dataset,
        "mode":            "full_pipeline" if args.full_pipeline else "detection_only",
        "n_total":         n_total,
        "n_hallucinated":  n_hall,
        "n_grounded":      n_grnd,
        "hall_rate":       round(n_hall/max(1,n_total), 4),
        "accuracy_before": round(acc_before, 4),
        "auroc":           auroc_results,
        "best_auroc":      round(best_auroc, 4) if not math.isnan(best_auroc) else None,
        "best_score_name": best_name,
        "delta_accuracy":  round(delta_acc, 4) if not math.isnan(delta_acc) else None,
        "n_corrected":     n_corrected,
        "n_broken":        n_broken,
        "config":          vars(args),
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}/summary.json  |  {jsonl_path}")
    return summary


# ── download helper ───────────────────────────────────────────────────────────

def download_datasets(cache_dir=None):
    from datasets import load_dataset
    datasets_to_load = [
        ("trivia_qa",      "rc.wikipedia"),
        ("hotpot_qa",      "distractor"),
        ("commonsense_qa", None),           # no config needed
    ]
    for name, cfg in datasets_to_load:
        if cfg:
            print(f"Downloading {name} ({cfg}) validation...")
            ds = load_dataset(name, cfg, split="validation", cache_dir=cache_dir)
        else:
            print(f"Downloading {name} validation...")
            ds = load_dataset(name, split="validation", cache_dir=cache_dir)
        print(f"  {len(ds)} samples cached.")
    print("\nAll three datasets cached. Ready to run.")


# ── parse args ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default="/mnt/shared/shared_hf_home/hub/"
                           "GSAI-ML--LLaDA-8B-Instruct")
    p.add_argument("--dataset",       choices=["triviaqa","hotpotqa","commonsenseqa"],
                   default="triviaqa")
    p.add_argument("--n_samples",     type=int, default=500)
    p.add_argument("--n_paths",       type=int, default=8)
    p.add_argument("--num_steps",     type=int, default=32)
    p.add_argument("--gen_len",       type=int, default=64)
    p.add_argument("--full_pipeline", action="store_true",
                   help="Run detection + remasking (RECOMMENDED)")
    p.add_argument("--output",        default="results/parade_triviaqa/")
    p.add_argument("--cache_dir",     default=None)
    p.add_argument("--abductive",      action="store_true",
                   help="Enable abductive second-pass explanation scoring")
    p.add_argument("--abduct_paths",   type=int, default=4,
                   help="Parallel chains for explanation pass (default 4)")
    p.add_argument("--abduct_gen_len", type=int, default=32,
                   help="Token budget per explanation (default 32)")
    p.add_argument("--abduct_steps",   type=int, default=16,
                   help="Denoising steps for explanation pass (default 16)")
    p.add_argument("--download_only",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.download_only:
        download_datasets(args.cache_dir)
    else:
        run(args)
