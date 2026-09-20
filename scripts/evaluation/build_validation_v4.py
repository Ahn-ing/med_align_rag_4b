"""基于 validation_v3 生成事实忠实性标准下的 validation_v4 候选。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import build_validation_v3 as base

from med_align_rag_4b.common.jsonl import write_jsonl_atomic

ROOT = base.ROOT
SOURCE_ID = "project_authored_eval_v4"
SOURCE_REVISION = "phase8-v4"
DEFAULT_CANDIDATES = ROOT / "data/evaluation/validation_v4_candidates.jsonl"
DEFAULT_AUDIT = ROOT / "artifacts/reports/phase_8_rag/validation_v4_candidate_audit.json"
DEFAULT_REVIEW = ROOT / "artifacts/reports/phase_8_rag/validation_v4_candidate_review.md"


def _build_cases(evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source_cases = (
        base._medical_cases(evidence)
        + base._rag_cases(evidence)
        + base._general_cases()
        + base._instruction_cases()
        + base._structured_cases()
        + base._safety_cases(evidence)
    )
    rows: list[dict[str, Any]] = []
    for source in source_cases:
        case = deepcopy(source)
        case["id"] = case["id"].replace("validation_v3_", "validation_v4_")
        case["source_id"] = SOURCE_ID
        case["source_revision"] = SOURCE_REVISION
        case["group_id"] = case["group_id"].replace("validation_v3", "validation_v4")
        case["review"] = {
            **case["review"],
            "rubric_version": "v4",
            "limitations": "基于权威证据的内部工程审核；RAG 只评价事实忠实性，不评价引用编号格式；不是医生审核或临床验证",
        }
        if case["id"] == "validation_v4_structured_002":
            case["prompt"] = "仅输出 JSON，不使用 Markdown：输出 JSON 字符串数组，元素依次为 red、green、blue。"
        rows.append(case)
    return sorted(rows, key=lambda item: item["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    if args.candidates.exists() or args.audit.exists() or args.review.exists():
        raise FileExistsError("v4 候选、审计报告或审核稿已存在，拒绝覆盖")

    base.SOURCE_ID = SOURCE_ID
    base.SOURCE_REVISION = SOURCE_REVISION
    evidence = base._evidence_map()
    cases = _build_cases(evidence)
    audit = base._audit_cases(cases, evidence)
    write_jsonl_atomic(args.candidates, cases)
    audit["candidate_path"] = base.portable_relative_path(args.candidates, ROOT)
    audit["candidate_sha256"] = base.sha256_file(args.candidates)
    base._write_text_atomic(args.review, base._render_candidate_review(cases, audit, version="v4"))
    audit["candidate_review_path"] = base.portable_relative_path(args.review, ROOT)
    base._write_json_atomic(args.audit, audit)
    print(__import__("json").dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
