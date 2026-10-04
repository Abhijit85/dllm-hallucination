"""
CPU-safe unit tests — no GPU or model weights required.
These run in CI. GPU-dependent tests are marked @pytest.mark.gpu.
"""

import json
import math

import pytest
import torch

from data.ragtruth_loader import (
    HallucinationSpan,
    RAGTruthSample,
    load_ragtruth,
)
from eval.metrics import aggregate, disagreement_hallucination_correlation, fact_score
from models import llada_harness
from strategies.parallel_remask import ablate_threshold, compute_disagreement

# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def grounded_sample():
    return RAGTruthSample(
        sample_id="test-001",
        source_id="source-001",
        task_type="QA",
        llm_name="test-model",
        temperature=0.0,
        prompt="Where is the Eiffel Tower located?",
        source_info="The Eiffel Tower is located in Paris, France.",
        response="The Eiffel Tower is located in Paris, France.",
        split="test",
        quality="good",
        spans=[],
    )


@pytest.fixture
def hallucinated_sample():
    return RAGTruthSample(
        sample_id="test-002",
        source_id="source-002",
        task_type="QA",
        llm_name="test-model",
        temperature=0.0,
        prompt="Where is the Eiffel Tower located?",
        source_info="The Eiffel Tower is located in Paris, France.",
        response="The Eiffel Tower is located in London, England.",
        split="test",
        quality="good",
        spans=[
            HallucinationSpan(start=38, end=53, text="London, England",
                              kind="evident_conflict", intensity=1.0)
        ],
    )


# ── RAGTruthSample ─────────────────────────────────────────────────────────────

def test_grounded_sample_no_hallucination(grounded_sample):
    assert not grounded_sample.has_hallucination
    assert len(grounded_sample.spans) == 0


def test_hallucinated_sample_detected(hallucinated_sample):
    assert hallucinated_sample.has_hallucination
    assert len(hallucinated_sample.spans) == 1
    assert hallucinated_sample.spans[0].kind == "evident_conflict"


# ── fact_score ─────────────────────────────────────────────────────────────────

def test_fact_score_perfect():
    src = "The cat sat on the mat"
    gen = "The cat sat on the mat"
    assert fact_score(gen, src, n=2) == pytest.approx(1.0)


def test_fact_score_zero():
    src = "The cat sat on the mat"
    gen = "Quantum computers use superconducting qubits"
    assert fact_score(gen, src, n=2) == pytest.approx(0.0)


def test_fact_score_partial():
    src = "The cat sat on the mat near the window"
    gen = "The cat sat on the floor"
    score = fact_score(gen, src, n=2)
    assert 0.0 < score < 1.0


def test_fact_score_empty_gen():
    assert fact_score("", "some source text", n=2) == 0.0


# ── token entropy (mock ParallelPathResult) ────────────────────────────────────

class MockPath:
    def __init__(self, tokens):
        self.final_tokens = torch.tensor(tokens)
        self.steps = []


class MockParallelResult:
    def __init__(self, paths_tokens):
        self.paths = [MockPath(t) for t in paths_tokens]
        self.prompt_tokens = torch.tensor([1, 2, 3])

    def token_entropy(self):
        seq_len = len(self.paths[0].final_tokens)
        n = len(self.paths)
        stacked = torch.stack([p.final_tokens for p in self.paths], dim=0)
        entropy = torch.zeros(seq_len)
        for pos in range(seq_len):
            counts = torch.bincount(stacked[:, pos], minlength=1).float()
            probs = counts[counts > 0] / n
            entropy[pos] = -(probs * probs.log()).sum()
        return entropy

    def majority_vote(self):
        stacked = torch.stack([p.final_tokens for p in self.paths], dim=0)
        return stacked.mode(dim=0).values

    def disagreement_mask(self, entropy_threshold=0.5):
        return self.token_entropy() > entropy_threshold


class MockTokenizer:
    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        if return_offsets_mapping:
            offsets = []
            start = 0
            for token in text.split():
                offsets.append((start, start + len(token)))
                start += len(token) + 1
            return {"offset_mapping": offsets}
        return {"input_ids": list(range(len(text.split())))}


def test_entropy_zero_when_paths_agree():
    # All 4 paths produce identical tokens
    tokens = [10, 20, 30, 40]
    result = MockParallelResult([tokens] * 4)
    ent = result.token_entropy()
    assert (ent == 0.0).all(), "Entropy should be 0 when all paths agree"


def test_entropy_nonzero_when_paths_disagree():
    # Each path produces a different token at position 2
    result = MockParallelResult([
        [10, 20, 11, 40],
        [10, 20, 22, 40],
        [10, 20, 33, 40],
        [10, 20, 44, 40],
    ])
    ent = result.token_entropy()
    assert ent[2] > 0, "Entropy should be >0 when paths disagree"
    assert ent[0] == 0.0, "Entropy should be 0 for agreed positions"


def test_majority_vote_selects_most_common():
    result = MockParallelResult([
        [10, 20, 99],
        [10, 20, 99],
        [10, 20, 42],
        [10, 20, 99],
    ])
    vote = result.majority_vote()
    assert vote[2].item() == 99  # 99 appears 3/4 times


def test_disagreement_mask_threshold():
    result = MockParallelResult([
        [10, 20, 11, 40],
        [10, 20, 22, 40],
        [10, 20, 33, 40],
        [10, 20, 44, 40],
    ])
    mask = result.disagreement_mask(entropy_threshold=0.0)
    # Position 2 always disagrees → should be flagged at threshold=0
    assert mask[2].item() is True
    # Positions 0, 1, 3 all agree → should not be flagged
    assert mask[0].item() is False


def test_compute_disagreement_uses_top_percent_selection():
    result = MockParallelResult([
        [10, 20, 11, 40, 50],
        [10, 20, 22, 40, 50],
        [10, 20, 33, 40, 50],
        [10, 20, 44, 40, 50],
    ])
    report = compute_disagreement(result, selection_fraction=0.5, track_trajectory=False)
    assert len(report.high_entropy_positions) == 1
    assert report.high_entropy_positions[0] == 3
    assert report.selection_fraction == 0.5


# ── ablate_threshold ───────────────────────────────────────────────────────────

def test_ablate_threshold_all_correct():
    # When flagged positions perfectly match ground truth
    result = MockParallelResult([
        [10, 99, 30],
        [10, 88, 30],
        [10, 77, 30],
        [10, 66, 30],
    ])
    ground_truth_hall = [1]  # position 1 is hallucinated
    ablations = ablate_threshold(result, ground_truth_hall, thresholds=[0.0])
    # At threshold 0.0, position 1 (high entropy) should be flagged
    a = ablations[0]
    assert a.n_hallucinated_flagged == 1
    assert a.recall == pytest.approx(1.0)


# ── aggregate ─────────────────────────────────────────────────────────────────

def test_aggregate_basic():
    records = [
        {"token_f1": 0.8, "fact_score_before": 0.6, "fact_score_after": 0.7,
         "refinement_delta": 0.1, "spearman_rho": 0.4, "change_rate": 0.1,
         "has_hallucination": True, "rho_skipped": False},
        {"token_f1": 0.9, "fact_score_before": 0.7, "fact_score_after": 0.75,
         "refinement_delta": 0.05, "spearman_rho": 0.5, "change_rate": 0.05,
         "has_hallucination": False, "rho_skipped": True},
    ]
    agg = aggregate(records)
    assert agg.n_samples == 2
    assert agg.mean_token_f1 == pytest.approx(0.85)
    assert agg.hallucinated_sample_fraction == pytest.approx(0.5)
    assert agg.mean_refinement_delta > 0
    assert agg.n_rho_computed == 1


def test_resolve_local_llada_model_alias(tmp_path, monkeypatch):
    model_dir = tmp_path / "llada-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        llada_harness.LOCAL_LLADA_MODEL_DIRS,
        "llada-8b-instruct",
        str(model_dir),
    )
    path = llada_harness.resolve_local_llada_model("llada-8b-instruct")
    assert path == str(model_dir)


def test_resolve_local_llada_model_rejects_unknown_remote_id():
    with pytest.raises(ValueError):
        llada_harness.resolve_local_llada_model("some-org/some-remote-model")


def test_load_ragtruth_reads_local_split_file(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    row = {
        "id": "resp-1",
        "task_type": "QA",
        "llm_name": "llama-test",
        "source_info": "Paris is the capital of France.",
        "response": "Lyon is the capital of France.",
        "labels": [
            {
                "start": 0,
                "end": 5,
                "text": "Lyon",
                "hallucination_type": "evident_conflict",
            }
        ],
    }

    (dataset_dir / "test.jsonl").write_text(
        json_line(row),
        encoding="utf-8",
    )

    samples = load_ragtruth(split="test", data_path=str(dataset_dir))
    assert len(samples) == 1
    assert samples[0].sample_id == "resp-1"
    assert samples[0].task_type == "QA"
    assert samples[0].llm_name == "llama-test"
    assert samples[0].source_info == "Paris is the capital of France."
    assert samples[0].spans[0].kind == "evident_conflict"


def test_load_ragtruth_reads_upstream_clone_layout(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    source_row = {
        "source_id": "src-1",
        "task_type": "QA",
        "source_info": "Paris is the capital of France.",
    }
    response_row = {
        "id": "resp-1",
        "source_id": "src-1",
        "model": "llama-test",
        "response": "Lyon is the capital of France.",
        "labels": [
            {
                "start": 0,
                "end": 4,
                "text": "Lyon",
                "label_type": "Evident Conflict",
            }
        ],
        "split": "test",
    }

    (dataset_dir / "source_info.jsonl").write_text(
        json_line(source_row),
        encoding="utf-8",
    )
    (dataset_dir / "response.jsonl").write_text(
        json_line(response_row),
        encoding="utf-8",
    )

    samples = load_ragtruth(split="test", data_path=str(dataset_dir))
    assert len(samples) == 1
    assert samples[0].sample_id == "resp-1"
    assert samples[0].task_type == "QA"
    assert samples[0].llm_name == "llama-test"
    assert samples[0].source_info == "Paris is the capital of France."
    assert samples[0].spans[0].kind == "evident_conflict"


def test_disagreement_hallucination_correlation_skips_constant_labels(grounded_sample):
    report = type(
        "MockReport",
        (),
        {"token_entropy": torch.tensor([0.1, 0.2, 0.3, 0.4])},
    )()
    corr = disagreement_hallucination_correlation(
        report=report,
        sample=grounded_sample,
        tokenizer=MockTokenizer(),
        prompt_len=0,
    )
    assert corr.skipped is True
    assert math.isnan(corr.spearman_rho)


def json_line(row: dict) -> str:
    return json.dumps(row) + "\n"
