"""
Strengthened H1 analysis using NLI-based source grounding instead of bigram overlap.

WHY NLI OVER BIGRAM
--------------------
Bigram overlap is too strict: "located in the French capital" vs "Paris, France"
scores zero overlap even though it's semantically grounded. This artificially
inflates the "ungrounded" fraction and dilutes the Spearman rho signal.

NLI-based grounding asks: does the source passage ENTAIL this generated sentence?
This correctly handles paraphrases, synonyms, and rephrasings.

THREE GROUNDING BACKENDS (in order of preference)
--------------------------------------------------
1. AlignScore  -- specifically designed for factual consistency in generated text
   pip install git+https://github.com/yuh-zha/AlignScore.git
   Model: AlignScore-base (~125M, fast) or AlignScore-large (~355M, better)

2. cross-encoder/nli-deberta-v3-large  -- strong general NLI, sentence-level
   Sentence-level: entailment probability per sentence in generation
   pip install sentence-transformers

3. facebook/bart-large-mnli  -- widely used zero-shot NLI baseline
   pip install transformers

The code automatically picks the best available backend.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import PreTrainedTokenizer

GroundingMode = Literal["alignscore", "nli_deberta", "nli_bart", "bigram"]
GroundingModeArg = GroundingMode | Literal["auto"]


def _detect_best_backend() -> GroundingMode:
    """Detect the best available grounding backend."""
    try:
        import alignscore  # noqa: F401

        return "alignscore"
    except ImportError:
        pass

    try:
        from sentence_transformers import CrossEncoder  # noqa: F401

        return "nli_deberta"
    except ImportError:
        pass

    try:
        from transformers import pipeline  # noqa: F401

        return "nli_bart"
    except ImportError:
        pass

    return "bigram"


class GroundingScorer:
    """
    Compute sentence-level grounding scores against a source passage.

    Sentence scores are later mapped back onto generated token positions.
    """

    def __init__(self, mode: GroundingModeArg = "auto", device: str = "cuda"):
        if mode == "auto":
            mode = _detect_best_backend()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.mode: GroundingMode = mode
        self.device = device
        self._model = None
        print(f"[H1] Grounding backend: {mode}")
        self._load(mode, device)

    def _load(self, mode: GroundingMode, device: str) -> None:
        if mode == "alignscore":
            from alignscore import AlignScore

            self._model = AlignScore(
                model="AlignScore-base",
                batch_size=8,
                device=device,
                evaluation_mode="bin_sum",
            )
            return

        if mode == "nli_deberta":
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                "cross-encoder/nli-deberta-v3-large",
                device=device,
                max_length=512,
            )
            return

        if mode == "nli_bart":
            from transformers import pipeline as hf_pipeline

            self._model = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if device == "cuda" else -1,
            )
            return

        if mode == "bigram":
            self._model = None
            return

        raise ValueError(f"Unsupported grounding mode: {mode}")

    def score_sentences(self, source: str, sentences: list[str]) -> list[float]:
        """
        Return a grounding score per sentence in [0.0, 1.0].

        1.0 means strongly grounded by the source, 0.0 means unsupported.
        """
        if not sentences:
            return []

        if self.mode == "alignscore":
            scores = self._model.score(
                contexts=[source] * len(sentences),
                claims=sentences,
            )
            return [float(score) for score in scores]

        if self.mode == "nli_deberta":
            pairs = [[source, sentence] for sentence in sentences]
            logits = self._model.predict(pairs, apply_softmax=True)
            return [float(row[1]) for row in logits]

        if self.mode == "nli_bart":
            scores: list[float] = []
            truncated_source = source[:300]
            for sentence in sentences:
                result = self._model(
                    sentence,
                    candidate_labels=["supported", "not supported"],
                    hypothesis_template=(
                        "This claim is {} by the document: " + truncated_source
                    ),
                )
                label_score = dict(zip(result["labels"], result["scores"]))
                scores.append(float(label_score.get("supported", 0.0)))
            return scores

        if self.mode == "bigram":
            return _bigram_scores(source, sentences)

        return [0.5] * len(sentences)


def _bigram_scores(source: str, sentences: list[str]) -> list[float]:
    """Bigram overlap fallback kept for comparison and no-extra-deps mode."""
    src_words = re.findall(r"\w+", source.lower())
    src_bigrams = {(src_words[i], src_words[i + 1]) for i in range(len(src_words) - 1)}

    scores: list[float] = []
    for sentence in sentences:
        words = re.findall(r"\w+", sentence.lower())
        if len(words) < 2:
            scores.append(0.0)
            continue
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        overlap = sum(1 for bigram in bigrams if bigram in src_bigrams)
        scores.append(overlap / len(bigrams) if bigrams else 0.0)
    return scores


def split_sentences(text: str) -> list[str]:
    """Split generated text into sentences for NLI scoring."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in raw if len(sentence.strip()) > 10]


def sentences_to_token_labels(
    sentences: list[str],
    sent_scores: list[float],
    generated_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    entailment_threshold: float = 0.5,
) -> np.ndarray:
    """
    Map sentence-level grounding scores back to token positions.

    Returns a binary array of length ``len(generated_ids)``:
    0 = grounded
    1 = ungrounded
    """
    gen_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        idx = gen_text.find(sentence, cursor)
        if idx == -1:
            first_word = sentence.split()[0] if sentence.split() else ""
            idx = gen_text.find(first_word, cursor) if first_word else -1
        if idx != -1:
            spans.append((idx, idx + len(sentence)))
            cursor = idx + len(sentence)
        else:
            spans.append((-1, -1))

    enc = tokenizer(
        gen_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]
    labels = np.ones(len(offsets), dtype=int)

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for j, (span_start, span_end) in enumerate(spans):
            if span_start == -1:
                continue
            if tok_start >= span_start and tok_end <= span_end + 5:
                score = sent_scores[j] if j < len(sent_scores) else 0.0
                labels[i] = 0 if score >= entailment_threshold else 1
                break

    return labels[: len(generated_ids)]


@dataclass
class H1ResultNLI:
    spearman_rho: float
    p_value: float
    n_tokens: int
    n_ungrounded: int
    mean_entropy_ungrounded: float
    mean_entropy_grounded: float
    entropy_gap: float
    sent_scores: list[float]
    mode: str
    skipped: bool = False
    skip_reason: str = ""


def compute_h1_nli(
    entropy: torch.Tensor,
    generated_ids: torch.Tensor,
    source_info: str,
    tokenizer: PreTrainedTokenizer,
    scorer: GroundingScorer,
    entailment_threshold: float = 0.5,
    min_ungrounded_tokens: int = 3,
) -> H1ResultNLI:
    """
    Compute H1 using NLI-style sentence grounding.

    Steps:
    1. Decode generated ids to text
    2. Split into sentences
    3. Score sentence grounding against source
    4. Map sentence scores to token labels
    5. Measure Spearman rho between entropy and ungrounded labels
    """
    ent = entropy.detach().cpu().numpy()
    min_len = min(len(ent), int(generated_ids.shape[0]))
    ent = ent[:min_len]
    gen_ids = generated_ids[:min_len]

    gen_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
    sentences = split_sentences(gen_text)
    if not sentences:
        return H1ResultNLI(
            spearman_rho=float("nan"),
            p_value=1.0,
            n_tokens=min_len,
            n_ungrounded=0,
            mean_entropy_ungrounded=float("nan"),
            mean_entropy_grounded=float("nan"),
            entropy_gap=float("nan"),
            sent_scores=[],
            mode=scorer.mode,
            skipped=True,
            skip_reason="no_sentences",
        )

    sent_scores = scorer.score_sentences(source_info, sentences)
    labels = sentences_to_token_labels(
        sentences=sentences,
        sent_scores=sent_scores,
        generated_ids=gen_ids,
        tokenizer=tokenizer,
        entailment_threshold=entailment_threshold,
    )

    n_ungrounded = int(labels.sum())
    if n_ungrounded < min_ungrounded_tokens:
        return H1ResultNLI(
            spearman_rho=float("nan"),
            p_value=1.0,
            n_tokens=min_len,
            n_ungrounded=n_ungrounded,
            mean_entropy_ungrounded=float("nan"),
            mean_entropy_grounded=float("nan"),
            entropy_gap=float("nan"),
            sent_scores=sent_scores,
            mode=scorer.mode,
            skipped=True,
            skip_reason="too_few_ungrounded",
        )

    if n_ungrounded == len(labels) or n_ungrounded == 0:
        return H1ResultNLI(
            spearman_rho=float("nan"),
            p_value=1.0,
            n_tokens=min_len,
            n_ungrounded=n_ungrounded,
            mean_entropy_ungrounded=float("nan"),
            mean_entropy_grounded=float("nan"),
            entropy_gap=float("nan"),
            sent_scores=sent_scores,
            mode=scorer.mode,
            skipped=True,
            skip_reason="constant_labels",
        )

    rho, p_value = spearmanr(ent, labels)

    mask_ungrounded = labels == 1
    mask_grounded = labels == 0
    mean_ungrounded = (
        float(ent[mask_ungrounded].mean()) if mask_ungrounded.any() else float("nan")
    )
    mean_grounded = (
        float(ent[mask_grounded].mean()) if mask_grounded.any() else float("nan")
    )
    gap = (
        mean_ungrounded - mean_grounded
        if not math.isnan(mean_ungrounded) and not math.isnan(mean_grounded)
        else float("nan")
    )

    return H1ResultNLI(
        spearman_rho=float(rho),
        p_value=float(p_value),
        n_tokens=min_len,
        n_ungrounded=n_ungrounded,
        mean_entropy_ungrounded=mean_ungrounded,
        mean_entropy_grounded=mean_grounded,
        entropy_gap=gap,
        sent_scores=sent_scores,
        mode=scorer.mode,
    )


def aggregate_h1_nli(results: list[H1ResultNLI]) -> dict:
    """Aggregate H1 NLI results across samples."""
    valid = [
        result
        for result in results
        if not result.skipped and not math.isnan(result.spearman_rho)
    ]

    def smean(values: list[float]) -> float:
        filtered = [value for value in values if not math.isnan(value)]
        return float(np.mean(filtered)) if filtered else float("nan")

    rhos = [result.spearman_rho for result in valid]

    return {
        "mode": results[0].mode if results else "unknown",
        "n_total": len(results),
        "n_valid": len(valid),
        "n_skipped": len(results) - len(valid),
        "mean_rho": smean(rhos),
        "median_rho": float(np.median(rhos)) if rhos else float("nan"),
        "pct_positive_rho": (
            sum(1 for rho in rhos if rho > 0) / len(rhos) if rhos else float("nan")
        ),
        "pct_rho_gt_02": (
            sum(1 for rho in rhos if rho > 0.2) / len(rhos) if rhos else float("nan")
        ),
        "mean_entropy_gap": smean([result.entropy_gap for result in valid]),
        "pct_gap_positive": (
            sum(
                1
                for result in valid
                if not math.isnan(result.entropy_gap) and result.entropy_gap > 0
            )
            / len(valid)
            if valid
            else float("nan")
        ),
        "mean_ungrounded_frac": smean(
            [result.n_ungrounded / max(1, result.n_tokens) for result in valid]
        ),
    }
