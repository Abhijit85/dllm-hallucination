import numpy as np
import pytest

import oscar_eval
from eval.unified_judge import (
    auroc,
    binarize,
    bootstrap_auroc_ci,
    cohen,
    fleiss,
    paired_bootstrap_diff,
    parse_label,
)


def test_parse_label_handles_substrings_and_case():
    assert parse_label(" incorrect.\n") == "INCORRECT"
    assert parse_label("Partial credit") == "PARTIAL"
    assert parse_label("correct") == "CORRECT"
    assert parse_label("maybe") == "UNKNOWN"


def test_oscar_eval_exports_same_parser():
    assert oscar_eval.parse_label("incorrect") == "INCORRECT"


def test_binarize_primary_and_drop_partial_modes():
    labels = ["CORRECT", "INCORRECT", "PARTIAL", "UNKNOWN"]
    y_primary, keep_primary = binarize(labels, partial_positive=True)
    assert y_primary.tolist() == [0, 1, 1, 0]
    assert keep_primary.tolist() == [True, True, True, False]

    y_drop, keep_drop = binarize(labels, partial_positive=False)
    assert y_drop.tolist() == [0, 1, 0, 0]
    assert keep_drop.tolist() == [True, True, False, False]


def test_auroc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    y = [0, 0, 1, 1]
    assert auroc(scores, y) == pytest.approx(1.0)


def test_bootstrap_auroc_ci_contains_observed():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    observed, lo, hi = bootstrap_auroc_ci(scores, y, B=400, seed=7)
    assert lo <= observed <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_paired_bootstrap_diff_detects_better_method():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score_a = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])
    score_b = np.array([0.10, 0.70, 0.20, 0.80, 0.30, 0.60, 0.40, 0.90])
    result = paired_bootstrap_diff(score_a, score_b, y, B=600, seed=11)
    assert result["diff"] > 0
    assert result["lo"] <= result["diff"] <= result["hi"]
    assert 0.0 <= result["p"] <= 1.0
    assert result["significant"] == (result["lo"] > 0 or result["hi"] < 0)


def test_cohen_matches_perfect_agreement():
    labels = ["CORRECT", "PARTIAL", "INCORRECT", "CORRECT"]
    assert cohen(labels, labels) == pytest.approx(1.0)


def test_fleiss_matches_perfect_agreement():
    mat = [
        ["CORRECT", "CORRECT", "CORRECT"],
        ["PARTIAL", "PARTIAL", "PARTIAL"],
        ["INCORRECT", "INCORRECT", "INCORRECT"],
    ]
    assert fleiss(mat) == pytest.approx(1.0)


def test_fleiss_rejects_unknown_label():
    with pytest.raises(ValueError):
        fleiss([["CORRECT", "BAD_LABEL", "CORRECT"]])
