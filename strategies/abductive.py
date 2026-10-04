from __future__ import annotations

import re
import string
from dataclasses import dataclass

import torch

from strategies.parallel_remask import compute_disagreement


@dataclass
class AbductiveScore:
    abductive_entropy: float
    abductive_top20_entropy: float
    abductive_hall_score: float
    bridge_score: float | None = None
    choice_entropies: dict[str, float] | None = None
    best_explained_choice: str | None = None


def _norm_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [tok for tok in text.split() if tok]


def _overlap_fraction(explanation: str, doc: str) -> float:
    exp_toks = set(_norm_text(explanation))
    doc_toks = set(_norm_text(doc))
    if not exp_toks or not doc_toks:
        return 0.0
    return len(exp_toks & doc_toks) / max(1, len(exp_toks))


def _explanation_prompt(
    dataset: str,
    question: str,
    predicted_answer: str,
    context: str,
    choice_letter: str | None = None,
    choice_text: str | None = None,
) -> str:
    if dataset == "commonsenseqa" and choice_letter and choice_text:
        return (
            f"Question: {question}\n"
            f"Choices: {context}\n\n"
            f"Explain briefly why choice {choice_letter}) {choice_text} is correct.\n"
            f"Explanation:"
        )
    if context:
        return (
            f"Context: {context[:800]}\n\n"
            f"Question: {question}\n"
            f"Answer: {predicted_answer}\n\n"
            "Explain briefly why this answer is correct.\n"
            "Explanation:"
        )
    return (
        f"Question: {question}\n"
        f"Answer: {predicted_answer}\n\n"
        "Explain briefly why this answer is correct.\n"
        "Explanation:"
    )


def _run_explanation_entropy(
    harness,
    prompt: str,
    source_info: str,
    n_paths: int,
    num_steps: int,
    gen_len: int,
    base_seed: int,
) -> tuple[float, float, str]:
    result = harness.run_parallel_paths(
        prompt=prompt,
        source_info=source_info,
        n_paths=n_paths,
        gen_len=gen_len,
        num_steps=num_steps,
        learned_paths=1,
        base_seed=base_seed,
    )
    report = compute_disagreement(result, top_k_percent=0.20)
    plen = len(result.prompt_tokens)
    ent = report.token_entropy[plen:].cpu().float()
    if len(ent) == 0:
        mean_ent = 0.0
        top20_ent = 0.0
    else:
        top_k = max(1, int(0.20 * len(ent)))
        mean_ent = float(ent.mean())
        top20_ent = float(torch.topk(ent, k=top_k).values.mean())
    text = harness.decode(result.majority_vote()[plen:]).strip()
    return mean_ent, top20_ent, text


def compute_abductive_score(
    harness,
    dataset: str,
    question: str,
    predicted_answer: str,
    context: str,
    supporting_docs: list[str] | None = None,
    choices: dict[str, str] | None = None,
    n_abductive_paths: int = 4,
    num_steps: int = 20,
    base_seed: int = 99,
    gen_len: int = 64,
) -> AbductiveScore:
    supporting_docs = supporting_docs or []
    choices = choices or {}

    if dataset == "commonsenseqa" and choices:
        choice_scores: dict[str, float] = {}
        for idx, (letter, text) in enumerate(choices.items()):
            prompt = _explanation_prompt(
                dataset=dataset,
                question=question,
                predicted_answer=predicted_answer,
                context=context,
                choice_letter=letter,
                choice_text=text,
            )
            mean_ent, _, _ = _run_explanation_entropy(
                harness=harness,
                prompt=prompt,
                source_info=context,
                n_paths=n_abductive_paths,
                num_steps=num_steps,
                gen_len=gen_len,
                base_seed=base_seed + idx,
            )
            choice_scores[letter] = mean_ent

        best_choice = min(choice_scores, key=choice_scores.get) if choice_scores else None
        pred_letter_match = re.search(r"[A-E]", str(predicted_answer).upper())
        pred_letter = pred_letter_match.group(0) if pred_letter_match else str(predicted_answer).strip()[:1].upper()
        pred_score = choice_scores.get(pred_letter, max(choice_scores.values(), default=0.0))
        best_score = min(choice_scores.values(), default=0.0)
        return AbductiveScore(
            abductive_entropy=pred_score,
            abductive_top20_entropy=pred_score,
            abductive_hall_score=pred_score - best_score,
            choice_entropies=choice_scores,
            best_explained_choice=best_choice,
        )

    prompt = _explanation_prompt(
        dataset=dataset,
        question=question,
        predicted_answer=predicted_answer,
        context=context,
    )
    mean_ent, top20_ent, explanation = _run_explanation_entropy(
        harness=harness,
        prompt=prompt,
        source_info=context,
        n_paths=n_abductive_paths,
        num_steps=num_steps,
        gen_len=gen_len,
        base_seed=base_seed,
    )

    bridge_score = None
    if dataset == "hotpotqa" and len(supporting_docs) >= 2:
        coverages = [_overlap_fraction(explanation, doc) for doc in supporting_docs[:2]]
        bridge_score = min(coverages) if coverages else 0.0

    hall_score = top20_ent
    if bridge_score is not None:
        hall_score = top20_ent + (1.0 - bridge_score)

    return AbductiveScore(
        abductive_entropy=mean_ent,
        abductive_top20_entropy=top20_ent,
        abductive_hall_score=hall_score,
        bridge_score=bridge_score,
    )
