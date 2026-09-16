"""构建 v2_plus_repair_r2：统一 system prompt，并追加医疗行为覆盖补丁。"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import portable_relative_path
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer

SYSTEM_PROMPT = "你是一个谨慎、清晰且遵循证据边界的助手。"
PATCH_SOURCE_ID = "project_authored_training_v2"
PATCH_REVISION = "medical-behavior-patch-v1"


def _message(record: dict[str, Any], role: str) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == role)


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


def _blocked(
    validation_v1: Path,
    validation_v2: Path,
    test_fingerprints: Path,
    medical_safety_holdout: Path,
    rag_queries: Path,
) -> set[str]:
    blocked: set[str] = set()
    for path in (validation_v1, validation_v2, rag_queries):
        for row in iter_jsonl(path):
            prompt = _case_prompt(row)
            if prompt:
                blocked.add(question_fingerprint(prompt))
    for path in (test_fingerprints, medical_safety_holdout):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("fingerprints", payload)
        blocked.update(str(value) for value in values)
    return blocked


def _with_system_prompt(record: dict[str, Any], tokenizer: QwenAssistantTokenizer) -> dict[str, Any]:
    updated = dict(record)
    messages = [dict(message) for message in record["messages"]]
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    elif messages[0].get("content") != SYSTEM_PROMPT:
        raise ValueError(f"记录已有不同 system prompt: {record.get('id')}")
    metrics = tokenizer.analyze(messages)
    if metrics["overlength"]:
        raise ValueError(f"加入 system prompt 后超过 max_length: {record.get('id')}")
    updated["messages"] = messages
    updated["total_tokens"] = metrics["total_tokens"]
    updated["assistant_supervised_tokens"] = metrics["assistant_supervised_tokens"]
    updated["truncated"] = False
    return updated

def _patch_record(
    raw: dict[str, Any],
    index: int,
    tokenizer: QwenAssistantTokenizer,
    evidence_ids: set[str],
) -> dict[str, Any]:
    missing = sorted(set(raw["evidence_ids"]) - evidence_ids)
    if missing:
        raise ValueError(f"补丁引用未知证据: {raw['slug']}: {missing}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw["question"]},
        {"role": "assistant", "content": raw["answer"]},
    ]
    metrics = tokenizer.analyze(messages)
    identifier = "sft_patch_" + sha256_text(
        f"{PATCH_REVISION}:{raw['slug']}:{raw['question']}:{raw['answer']}"
    )[:24]
    return {
        "id": identifier,
        "source_id": PATCH_SOURCE_ID,
        "source_record_id": f"{PATCH_REVISION}:{index:02d}",
        "source_revision": PATCH_REVISION,
        "messages": messages,
        "domain": "medical",
        "ability": raw["primary_category"],
        "language": "zh",
        "risk_level": raw["risk_level"],
        "primary_category": raw["primary_category"],
        "secondary_categories": [],
        "license_status": "approved",
        "medical_evidence": raw["evidence_ids"],
        "review": {
            "status": "agent_evidence_reviewed",
            "reviewer_role": "codex_research_reviewer",
            "reviewed_at": "2026-09-16",
            "rubric_version": "medical-behavior-patch-v1",
            "evidence_checked": True,
            "doctor_reviewed": False,
            "clinical_validation": False,
        },
        "quality_flags": [],
        "unresolved_safety_flags": [],
        "fingerprints": {
            "question": question_fingerprint(raw["question"]),
            "qa": qa_fingerprint(raw["question"], raw["answer"]),
            "question_skeleton": sha256_text(raw["question"]),
            "answer_skeleton": sha256_text(raw["answer"]),
        },
        "provenance": {
            "generation_method": "project_authored",
            "patch_purpose": "medical_behavior_coverage",
            "patch_slug": raw["slug"],
        },
        "total_tokens": metrics["total_tokens"],
        "assistant_supervised_tokens": metrics["assistant_supervised_tokens"],
        "truncated": False,
        "data_component": "medical_behavior_patch",
        "release_allowed": False,
    }


def _deduplicate_patch(
    base_records: list[dict[str, Any]], patch_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_questions = {question_fingerprint(_message(row, "user")) for row in base_records}
    seen_qa = {
        qa_fingerprint(_message(row, "user"), _message(row, "assistant"))
        for row in base_records
    }
    accepted = []
    rejected = []
    for row in patch_records:
        question = _message(row, "user")
        answer = _message(row, "assistant")
        qfp = question_fingerprint(question)
        qa = qa_fingerprint(question, answer)
        if qfp in seen_questions:
            rejected.append({"id": row["id"], "reason": "duplicate_question_with_base"})
            continue
        if qa in seen_qa:
            rejected.append({"id": row["id"], "reason": "duplicate_qa_with_base"})
            continue
        seen_questions.add(qfp)
        seen_qa.add(qa)
        accepted.append(row)
    return accepted, rejected


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-smoke", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--patch-source", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--validation-v1", type=Path, required=True)
    parser.add_argument("--validation-v2", type=Path, required=True)
    parser.add_argument("--test-fingerprints", type=Path, required=True)
    parser.add_argument("--medical-safety-holdout", type=Path, required=True)
    parser.add_argument("--rag-queries", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = {
        "train": args.output_dir / "train.jsonl",
        "smoke": args.output_dir / "smoke_train.jsonl",
        "manifest": args.output_dir / "manifest.json",
        "rejections": args.output_dir / "patch_rejections.jsonl",
    }
    for path in [*outputs.values(), args.patch_output]:
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    root = Path(__file__).resolve().parents[2]
    tokenizer = QwenAssistantTokenizer(args.tokenizer_path, max_length=2048)
    evidence_ids = {
        source["evidence_id"]
        for source in json.loads(args.evidence_registry.read_text(encoding="utf-8"))["sources"]
    }
    raw_patches = list(iter_jsonl(args.patch_source))
    patch_records = [
        _patch_record(raw, index, tokenizer, evidence_ids)
        for index, raw in enumerate(raw_patches, 1)
    ]
    base_records = [_with_system_prompt(row, tokenizer) for row in iter_jsonl(args.base_train)]
    accepted_patches, duplicate_rejections = _deduplicate_patch(base_records, patch_records)

    blocked = _blocked(
        args.validation_v1,
        args.validation_v2,
        args.test_fingerprints,
        args.medical_safety_holdout,
        args.rag_queries,
    )
    leakage = []
    clean_base = []
    for row in base_records:
        if question_fingerprint(_message(row, "user")) in blocked:
            leakage.append({"id": row["id"], "reason": "frozen_eval_or_rag_leakage", "source": "base"})
        else:
            clean_base.append(row)
    clean_patches = []
    for row in accepted_patches:
        if question_fingerprint(_message(row, "user")) in blocked:
            leakage.append({"id": row["id"], "reason": "frozen_eval_or_rag_leakage", "source": "patch"})
        else:
            clean_patches.append(row)

    train = clean_base + clean_patches
    base_smoke = [_with_system_prompt(row, tokenizer) for row in iter_jsonl(args.base_smoke)]
    smoke_ids = {row["id"] for row in base_smoke} | {row["id"] for row in clean_patches}
    smoke = [row for row in train if row["id"] in smoke_ids]
    write_jsonl_atomic(args.patch_output, clean_patches)
    write_jsonl_atomic(outputs["train"], train)
    write_jsonl_atomic(outputs["smoke"], smoke)
    write_jsonl_atomic(outputs["rejections"], duplicate_rejections + leakage)

    component_tokens = Counter()
    component_records = Counter()
    for row in train:
        component = row.get("data_component", "unknown")
        component_tokens[component] += int(row["assistant_supervised_tokens"])
        component_records[component] += 1
    total_tokens = sum(component_tokens.values())
    manifest = {
        "format_version": 1,
        "dataset_version": "v2_plus_repair_r2",
        "status": "completed",
        "purpose": "internal_engineering_training_with_medical_behavior_patch",
        "training_allowed": True,
        "release_allowed": False,
        "doctor_reviewed": False,
        "clinical_validation": False,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "base_release": {
            "path": portable_relative_path(args.base_manifest.parent, root),
            "manifest_sha256": sha256_file(args.base_manifest),
            "train_sha256": sha256_file(args.base_train),
            "records": len(base_records),
        },
        "patch": {
            "source_path": portable_relative_path(args.patch_source, root),
            "source_sha256": sha256_file(args.patch_source),
            "reviewed_path": portable_relative_path(args.patch_output, root),
            "reviewed_sha256": sha256_file(args.patch_output),
            "raw_records": len(patch_records),
            "accepted": len(clean_patches),
            "rejected": len(duplicate_rejections) + len(leakage),
        },
        "composition": {
            component: {
                "records": component_records[component],
                "tokens": component_tokens[component],
                "fraction": component_tokens[component] / total_tokens,
            }
            for component in sorted(component_tokens)
        },
        "counts": {"train": len(train), "smoke_train": len(smoke)},
        "artifacts": {
            "train": {"path": outputs["train"].as_posix(), "sha256": sha256_file(outputs["train"]), "records": len(train)},
            "smoke_train": {"path": outputs["smoke"].as_posix(), "sha256": sha256_file(outputs["smoke"]), "records": len(smoke)},
            "patch_rejections": {"path": outputs["rejections"].as_posix(), "sha256": sha256_file(outputs["rejections"]), "records": len(duplicate_rejections) + len(leakage)},
        },
        "limitations": [
            "补丁为项目自编内部工程数据，不是医生审核或临床验证",
            "基础 v2_plus_repair 保持只读",
            "不得公开发布或用于临床诊断",
        ],
        "prohibited_actions_confirmed": {"training_run": False, "release_created": False},
    }
    _write_json_atomic(outputs["manifest"], manifest)
    print(json.dumps({"status": "completed", "train": len(train), "smoke": len(smoke), "patches": len(clean_patches), "tokens": total_tokens}, ensure_ascii=False))


if __name__ == "__main__":
    main()
