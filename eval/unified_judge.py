"""Compatibility wrapper for unified-judge utilities."""

from oscar_eval import (
    FLUENCY_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
    auroc,
    binarize,
    bootstrap_auroc_ci,
    call_judge,
    cohen,
    fleiss,
    paired_bootstrap_diff,
    parse_label,
)

__all__ = [
    "FLUENCY_PROMPT_TEMPLATE",
    "JUDGE_PROMPT_TEMPLATE",
    "auroc",
    "binarize",
    "bootstrap_auroc_ci",
    "call_judge",
    "cohen",
    "fleiss",
    "paired_bootstrap_diff",
    "parse_label",
]
