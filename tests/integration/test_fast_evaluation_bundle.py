import json
from collections import Counter
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.evaluation.datasets import validate_case, validate_no_leakage
from med_align_rag_4b.sft.deduplication import qa_fingerprint


def test_frozen_fast_bundle_contract():
    root = Path(__file__).resolve().parents[2]
    names = {"validation": "validation_v1", "test": "test_v1"}
    expected = {"validation": 60, "test": 90}
    splits = {}
    for split, directory in names.items():
        path = root / "data/evaluation" / directory
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        rows = list(iter_jsonl(path / "cases.jsonl"))
        assert manifest["records"] == len(rows) == expected[split]
        assert manifest["data"]["sha256"] == sha256_file(path / "cases.jsonl")
        assert manifest["release_allowed"] is False
        for case in rows:
            validate_case(case)
        splits[split] = rows
    validate_no_leakage(splits)
    fingerprints = [qa_fingerprint(case["prompt"], case["reference_answer"]) for rows in splits.values() for case in rows]
    assert len(fingerprints) == len(set(fingerprints)) == 150
    assert Counter(case["task"] for case in splits["validation"]) == {"medical_qa": 20, "rag_faithfulness": 8, "general": 10, "instruction_following": 6, "structured_output": 6, "medical_safety": 10}
    holdout_path = root / "data/evaluation/medical_safety_holdout_v1"
    holdout_manifest = json.loads((holdout_path / "manifest.json").read_text(encoding="utf-8"))
    holdout_fingerprints = json.loads((holdout_path / "fingerprints.json").read_text(encoding="utf-8"))
    assert holdout_manifest["records"] == 30
    assert holdout_manifest["data"]["sha256"] == sha256_file(holdout_path / "cases.jsonl")
    assert holdout_manifest["fingerprints"]["sha256"] == sha256_file(holdout_path / "fingerprints.json")
    assert len(holdout_fingerprints["fingerprints"]) == 30
