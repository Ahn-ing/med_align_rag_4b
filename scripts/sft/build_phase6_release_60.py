"""构建 rule_pool/API/general 约为 60/20/20 的内部 SFT v2 release。"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer

SEED = "20260915-phase-s1r-release-60"
API_TARGET_TOKENS = 217_152
GENERAL_TARGET_TOKENS = 217_152


def _message(record: dict[str, Any], role: str) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == role)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_prompt(row: dict[str, Any]) -> str | None:
    for key in ("prompt", "question", "query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(item.get("content", ""))
            for item in messages
            if item.get("role") == "user"
        )
    return None


def _blocked_fingerprints(
    validation_v1: Path,
    validation_v2: Path,
    test_fingerprints: Path,
    medical_safety_holdout: Path,
    rag_queries: Path,
) -> set[str]:
    blocked: set[str] = set()
    for path in (validation_v1, validation_v2):
        for row in iter_jsonl(path):
            prompt = _case_prompt(row)
            if prompt:
                blocked.add(question_fingerprint(prompt))
    for row in iter_jsonl(rag_queries):
        prompt = _case_prompt(row)
        if prompt:
            blocked.add(question_fingerprint(prompt))
    for path in (test_fingerprints, medical_safety_holdout):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("fingerprints", payload)
        if not isinstance(values, list):
            raise ValueError(f"指纹文件格式无效: {path}")
        blocked.update(str(value) for value in values)
    return blocked


def _normalize_rule_pool(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.update(
        {
            "domain": "medical",
            "ability": record["primary_category"],
            "language": "zh",
            "secondary_categories": [],
            "data_component": "rule_pool_salvaged",
            "release_allowed": False,
            "truncated": False,
            "unresolved_safety_flags": [],
        }
    )
    if "assistant_supervised_tokens" not in normalized:
        raise ValueError(f"rule_pool 缺少 token 统计: {record.get('id')}")
    return normalized


def _normalize_api(record: dict[str, Any], tokenizer: QwenAssistantTokenizer) -> dict[str, Any]:
    messages = record["messages"]
    metrics = tokenizer.analyze(messages)
    selection = record.get("selection") or {}
    review = dict(record.get("review") or {})
    review.update(
        {
            "status": "agent_evidence_reviewed",
            "reviewer_role": "codex_research_reviewer",
            "evidence_checked": True,
        }
    )
    return {
        "id": record["id"],
        "source_id": "api_evidence_constrained_v5",
        "source_record_id": str(selection.get("task_id") or record["id"]),
        "source_revision": "evidence-registry-v5-20260915",
        "messages": messages,
        "domain": "medical",
        "ability": record["primary_category"],
        "language": "zh",
        "risk_level": record["risk_level"],
        "primary_category": record["primary_category"],
        "secondary_categories": [],
        "license_status": "approved",
        "medical_evidence": list(record.get("medical_evidence") or []),
        "review": review,
        "quality_flags": [],
        "unresolved_safety_flags": [],
        "fingerprints": {
            "question": question_fingerprint(_message(record, "user")),
            "qa": qa_fingerprint(_message(record, "user"), _message(record, "assistant")),
            "question_skeleton": sha256_text(_message(record, "user")),
            "answer_skeleton": sha256_text(_message(record, "assistant")),
        },
        "provenance": {
            **(record.get("provenance") or {}),
            "source_url": selection.get("source_url"),
            "focus_claim": selection.get("focus_claim"),
        },
        "total_tokens": metrics["total_tokens"],
        "assistant_supervised_tokens": metrics["assistant_supervised_tokens"],
        "truncated": False,
        "data_component": "api_evidence",
        "release_allowed": False,
    }


def _normalize_general(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    review = dict(record.get("review") or {})
    review.update(
        {
            "status": "agent_quality_reviewed",
            "reviewer_role": "codex_research_reviewer",
            "reviewed_at": "2026-09-15",
            "rubric_version": "phase-a-general-retention-v1",
            "evidence_checked": False,
        }
    )
    normalized.update(
        {
            "review": review,
            "domain": "general",
            "ability": record.get("ability", "instruction_following"),
            "language": "en",
            "secondary_categories": [],
            "medical_evidence": [],
            "data_component": "general",
            "release_allowed": False,
            "truncated": False,
            "unresolved_safety_flags": [],
        }
    )
    return normalized


def _select_general(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("ability", "instruction_following")].append(row)
    ordered_groups = []
    for group in groups.values():
        group.sort(key=lambda row: sha256_text(f"{SEED}:general:{row['id']}"))
        ordered_groups.append(group)
    ordered_groups.sort(key=lambda group: group[0]["id"])
    selected: list[dict[str, Any]] = []
    tokens = 0
    while ordered_groups:
        next_groups = []
        for group in ordered_groups:
            row = group[0]
            row_tokens = int(row["assistant_supervised_tokens"])
            if tokens + row_tokens <= target:
                selected.append(row)
                tokens += row_tokens
                group.pop(0)
            else:
                group.pop(0)
            if group:
                next_groups.append(group)
        ordered_groups = next_groups
    if abs(tokens - target) > target * 0.01:
        raise ValueError(f"general token 超出 ±1%: {tokens}")
    return selected


def _deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    priority = {"api_evidence": 0, "rule_pool_salvaged": 1, "general": 2}
    ordered = sorted(records, key=lambda row: (priority[row["data_component"]], row["id"]))
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        qfp = question_fingerprint(_message(row, "user"))
        by_question[qfp].append(row)
    conflicts = {
        qfp for qfp, rows in by_question.items()
        if len({qa_fingerprint(_message(row, "user"), _message(row, "assistant")) for row in rows}) > 1
    }
    clean = []
    rejected = []
    seen_qa: set[str] = set()
    seen_question: set[str] = set()
    for row in ordered:
        question = _message(row, "user")
        answer = _message(row, "assistant")
        qfp = question_fingerprint(question)
        qa = qa_fingerprint(question, answer)
        if qfp in conflicts:
            rejected.append({"id": row["id"], "reason": "conflicting_answers"})
            continue
        if qa in seen_qa:
            rejected.append({"id": row["id"], "reason": "duplicate_qa"})
            continue
        if qfp in seen_question:
            rejected.append({"id": row["id"], "reason": "duplicate_question"})
            continue
        seen_qa.add(qa)
        seen_question.add(qfp)
        clean.append(row)
    return clean, rejected


def _smoke_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_tokens = sum(int(row["assistant_supervised_tokens"]) for row in records)
    target = round(total_tokens * 0.02)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["data_component"], row["primary_category"])].append(row)
    ordered_groups = []
    for group in groups.values():
        group.sort(key=lambda row: sha256_text(f"{SEED}:smoke:{row['id']}"))
        ordered_groups.append(group)
    selected: list[dict[str, Any]] = []
    tokens = 0
    while ordered_groups:
        next_groups = []
        for group in ordered_groups:
            row = group.pop(0)
            row_tokens = int(row["assistant_supervised_tokens"])
            if tokens + row_tokens <= target:
                selected.append(row)
                tokens += row_tokens
            if group:
                next_groups.append(group)
        ordered_groups = next_groups
    return sorted(selected, key=lambda row: row["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-pool-selected", type=Path, required=True)
    parser.add_argument("--api-selected", type=Path, required=True)
    parser.add_argument("--general-pool", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--validation-v1", type=Path, required=True)
    parser.add_argument("--validation-v2", type=Path, required=True)
    parser.add_argument("--test-fingerprints", type=Path, required=True)
    parser.add_argument("--medical-safety-holdout", type=Path, required=True)
    parser.add_argument("--rag-queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    outputs = {
        "train": args.output_dir / "train.jsonl",
        "smoke": args.output_dir / "smoke_train.jsonl",
        "manifest": args.output_dir / "manifest.json",
        "rejections": args.output_dir / "merge_rejections.jsonl",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    tokenizer = QwenAssistantTokenizer(args.tokenizer_path, max_length=2048)
    blocked = _blocked_fingerprints(
        args.validation_v1,
        args.validation_v2,
        args.test_fingerprints,
        args.medical_safety_holdout,
        args.rag_queries,
    )
    rule_rows = [_normalize_rule_pool(row) for row in iter_jsonl(args.rule_pool_selected)]
    api_rows = [_normalize_api(row, tokenizer) for row in iter_jsonl(args.api_selected)]
    general_source = [
        row for row in iter_jsonl(args.general_pool) if row["domain"] == "general"
    ]
    general_rows = [
        _normalize_general(row)
        for row in _select_general(general_source, GENERAL_TARGET_TOKENS)
    ]

    merged = rule_rows + api_rows + general_rows
    leakage = []
    not_leaked = []
    for row in merged:
        question = _message(row, "user")
        if question_fingerprint(question) in blocked:
            leakage.append({"id": row["id"], "reason": "frozen_eval_or_rag_leakage"})
        else:
            not_leaked.append(row)
    clean, duplicates = _deduplicate(not_leaked)
    smoke = _smoke_subset(clean)

    write_jsonl_atomic(outputs["train"], clean)
    write_jsonl_atomic(outputs["smoke"], smoke)
    write_jsonl_atomic(outputs["rejections"], leakage + duplicates)

    composition = Counter()
    tokens = Counter()
    records = Counter()
    for row in clean:
        component = row["data_component"]
        composition[component] += int(row["assistant_supervised_tokens"])
        tokens[component] += int(row["assistant_supervised_tokens"])
        records[component] += 1
    total_tokens = sum(tokens.values())
    fractions = {key: composition[key] / total_tokens for key in composition}
    if abs(composition["rule_pool_salvaged"] - 651_456) > 651_456 * 0.01:
        raise ValueError("rule_pool token 超出 60% 目标 ±1%")
    if abs(composition["api_evidence"] - API_TARGET_TOKENS) > API_TARGET_TOKENS * 0.001:
        raise ValueError("API token 偏离固定值")
    if abs(composition["general"] - GENERAL_TARGET_TOKENS) > GENERAL_TARGET_TOKENS * 0.01:
        raise ValueError("general token 超出 20% 目标 ±1%")

    manifest = {
        "format_version": 1,
        "dataset_version": "v2",
        "status": "completed",
        "purpose": "internal_phase_6_b2_60_engineering_training",
        "data_mode": "rule_pool_60_api_20_general_20",
        "training_allowed": True,
        "release_allowed": False,
        "doctor_reviewed": False,
        "clinical_validation": False,
        "tokenizer": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "max_length": 2048,
            "enable_thinking": False,
        },
        "composition": {
            component: {
                "records": records[component],
                "tokens": tokens[component],
                "fraction": fractions.get(component, 0.0),
            }
            for component in ("rule_pool_salvaged", "api_evidence", "general")
        },
        "counts": {
            "train": len(clean),
            "smoke_train": len(smoke),
            "leakage_rejected": len(leakage),
            "duplicate_or_conflict_rejected": len(duplicates),
        },
        "inputs": {
            "rule_pool_selected": {"path": args.rule_pool_selected.as_posix(), "sha256": sha256_file(args.rule_pool_selected)},
            "api_selected": {"path": args.api_selected.as_posix(), "sha256": sha256_file(args.api_selected)},
            "general_pool": {"path": args.general_pool.as_posix(), "sha256": sha256_file(args.general_pool)},
        },
        "artifacts": {
            "train": {"path": outputs["train"].as_posix(), "sha256": sha256_file(outputs["train"]), "records": len(clean)},
            "smoke_train": {"path": outputs["smoke"].as_posix(), "sha256": sha256_file(outputs["smoke"]), "records": len(smoke)},
            "merge_rejections": {"path": outputs["rejections"].as_posix(), "sha256": sha256_file(outputs["rejections"]), "records": len(leakage) + len(duplicates)},
        },
        "limitations": [
            "rule_pool 部分是放宽后的内部工程数据，缺少逐条权威证据核验",
            "API 部分是代理证据审核，不是医生审核",
            "不得公开发布、不得用于临床诊断或声称临床有效性",
        ],
        "prohibited_actions_confirmed": {
            "training_run": False,
            "test_cases_read": False,
            "medical_safety_holdout_cases_read": False,
        },
    }
    _atomic_json(outputs["manifest"], manifest)
    print(json.dumps({"status": "completed", "records": len(clean), "tokens": dict(tokens), "fractions": fractions, "smoke": len(smoke)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
