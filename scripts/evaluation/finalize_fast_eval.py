"""校验审核记录并冻结 Phase 3 快速版三个评测 split。"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.evaluation.datasets import validate_case, validate_no_leakage
from med_align_rag_4b.evaluation.freezing import freeze_split
from med_align_rag_4b.sft.deduplication import qa_fingerprint

EXPECTED_TASKS = {
    "validation": {"medical_qa": 20, "rag_faithfulness": 8, "general": 10, "instruction_following": 6, "structured_output": 6, "medical_safety": 10},
    "test": {"medical_qa": 29, "rag_faithfulness": 13, "general": 14, "instruction_following": 9, "structured_output": 9, "medical_safety": 16},
}
EXPECTED_HOLDOUT = {"immediate_emergency": 6, "outpatient_followup": 4, "home_observation": 5, "clarification": 4, "medication_boundary": 6, "uncertainty": 5}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"输出已存在: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_and_validate(candidate_dir: Path, blocked_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    splits = {split: list(iter_jsonl(candidate_dir / f"{split}.jsonl")) for split in ("validation", "test", "medical_safety_holdout")}
    for rows in splits.values():
        for case in rows:
            validate_case(case)
    blocked = set()
    for path in blocked_paths:
        for record in iter_jsonl(path):
            prompt = record.get("question", record.get("query"))
            if isinstance(prompt, str):
                from med_align_rag_4b.sft.deduplication import question_fingerprint

                blocked.add(question_fingerprint(prompt))
    validate_no_leakage(splits, blocked)
    qa_seen: set[str] = set()
    for rows in splits.values():
        for case in rows:
            fingerprint = qa_fingerprint(case["prompt"], case["reference_answer"])
            if fingerprint in qa_seen:
                raise ValueError("评测 QA 在 split 内或跨 split 重复")
            qa_seen.add(fingerprint)
    for split, expected in EXPECTED_TASKS.items():
        actual = dict(Counter(case["task"] for case in splits[split]))
        if actual != expected:
            raise ValueError(f"{split} task 配额不匹配: {actual}")
    holdout_actual = dict(Counter(case["primary_category"] for case in splits["medical_safety_holdout"]))
    if holdout_actual != EXPECTED_HOLDOUT:
        raise ValueError(f"holdout 类别配额不匹配: {holdout_actual}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    splits = load_and_validate(args.candidate_dir, [root / "data/rag/artifacts/huatuo_encyclopedia/cleaned.jsonl", root / "data/rag/compatibility/queries.v1.jsonl"])
    review_path = root / "data/sft/sources/review_registry.json"
    reviews = []
    for split, rows in splits.items():
        for case in rows:
            reviews.append({"case_id": case["id"], "split": split, "status": "agent_evidence_reviewed", "reviewer_role": "codex_research_reviewer", "reviewed_at": case["review"]["reviewed_at"], "rubric_version": "v1", "evidence_ids": [item["evidence_id"] for item in case["medical_evidence"]], "release_allowed": False})
    atomic_json(review_path, {"format_version": 1, "rubric_version": "v1", "status": "complete_for_internal_fast_v1", "reviews": sorted(reviews, key=lambda item: item["case_id"])})
    rubric_paths = [root / f"docs/evaluation/{name}" for name in ("evaluation_matrix_v1.md", "medical_safety_rubric_v1.md", "human_review_rubric_v1.md", "model_judge_rubric_v1.md")]
    rubric_sha = sha256_record([sha256_file(path) for path in rubric_paths])
    hashes = {"source": sha256_file(root / "data/sft/sources/source_registry.json"), "review": sha256_file(review_path), "evidence": sha256_file(root / "data/evaluation/evidence_registry_v1.json")}
    output_names = {"validation": "validation_v1", "test": "test_v1", "medical_safety_holdout": "medical_safety_holdout_v1"}
    for split, output_name in output_names.items():
        freeze_split(root / "data/evaluation" / output_name, splits[split], split=split, seed=20260914, source_registry_sha256=hashes["source"], review_registry_sha256=hashes["review"], rubric_sha256=rubric_sha, evidence_registry_sha256=hashes["evidence"], created_at_utc=args.created_at_utc)


if __name__ == "__main__":
    main()
