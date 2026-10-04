from pathlib import Path

from oscar_judge_runner import MockJudge, judge_all_methods, judge_records


def test_judge_records_uses_cache_and_resume(tmp_path):
    records = [
        {
            "id": "q1",
            "question": "Capital of France?",
            "reference": "Paris",
            "candidate": "Paris",
        },
        {
            "id": "q2",
            "question": "Capital of France?",
            "reference": "Paris",
            "candidate": "Paris, France",
        },
        {
            "id": "q3",
            "question": "Capital of France?",
            "reference": "Paris",
            "candidate": "Lyon",
        },
    ]
    cache_path = tmp_path / "judge_cache.jsonl"
    judge = MockJudge()

    first = judge_records(records, judge, cache_path=cache_path, workers=2)
    second = judge_records(records, judge, cache_path=cache_path, workers=2)

    assert first == {"q1": "CORRECT", "q2": "PARTIAL", "q3": "INCORRECT"}
    assert second == first
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 3


def test_judge_all_methods_judges_only_base_records(tmp_path):
    base_records = [
        {
            "id": "q1",
            "question": "Capital of France?",
            "reference": "Paris",
            "candidate": "Paris",
        },
    ]
    labels = judge_all_methods(
        base_records,
        method_outputs={"oscar": {"q1": 0.9}, "dynhd": {"q1": 0.6}},
        client=MockJudge(),
        cache_dir=tmp_path,
    )
    assert labels == {"q1": "CORRECT"}
    assert Path(tmp_path, "base_mock.jsonl").exists()
