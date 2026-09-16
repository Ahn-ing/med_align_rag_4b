"""将审核通过的修复候选追加到冻结 v2，生成 v2_plus_repair。"""

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
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint

SEED = "20260916-phase-s2-append-repair"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-smoke", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--repair-approved", type=Path, required=True)
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
        "rejections": args.output_dir / "append_rejections.jsonl",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    base_rows = list(iter_jsonl(args.base_train))
    base_smoke = list(iter_jsonl(args.base_smoke))
    repair_rows = list(iter_jsonl(args.repair_approved))
    blocked = _blocked(
        args.validation_v1,
        args.validation_v2,
        args.test_fingerprints,
        args.medical_safety_holdout,
        args.rag_queries,
    )

    seen_question = {question_fingerprint(_message(row, "user")) for row in base_rows}
    seen_qa = {
        qa_fingerprint(_message(row, "user"), _message(row, "assistant"))
        for row in base_rows
    }
    appended = []
    rejected = []
    for row in sorted(repair_rows, key=lambda item: sha256_text(f"{SEED}:{item['id']}")):
        question = _message(row, "user")
        answer = _message(row, "assistant")
        qfp = question_fingerprint(question)
        qa = qa_fingerprint(question, answer)
        if qfp in blocked:
            rejected.append({"id": row["id"], "reason": "frozen_eval_or_rag_leakage"})
            continue
        if qfp in seen_question:
            rejected.append({"id": row["id"], "reason": "duplicate_question_with_base"})
            continue
        if qa in seen_qa:
            rejected.append({"id": row["id"], "reason": "duplicate_qa_with_base"})
            continue
        seen_question.add(qfp)
        seen_qa.add(qa)
        appended.append(row)

    combined = base_rows + appended
    repair_smoke_target = round(
        sum(int(row["assistant_supervised_tokens"]) for row in appended) * 0.02
    )
    repair_smoke = []
    repair_tokens = 0
    for row in appended:
        row_tokens = int(row["assistant_supervised_tokens"])
        if repair_tokens + row_tokens <= repair_smoke_target:
            repair_smoke.append(row)
            repair_tokens += row_tokens
    smoke = base_smoke + repair_smoke

    write_jsonl_atomic(outputs["train"], combined)
    write_jsonl_atomic(outputs["smoke"], smoke)
    write_jsonl_atomic(outputs["rejections"], rejected)

    component_tokens = Counter()
    component_records = Counter()
    for row in combined:
        component = row.get("data_component", "unknown")
        component_tokens[component] += int(row["assistant_supervised_tokens"])
        component_records[component] += 1
    total_tokens = sum(component_tokens.values())
    manifest = {
        "format_version": 1,
        "dataset_version": "v2_plus_repair",
        "status": "completed",
        "purpose": "internal_engineering_training_append_only_repair",
        "training_allowed": True,
        "release_allowed": False,
        "doctor_reviewed": False,
        "clinical_validation": False,
        "base_release": {
            "path": args.base_manifest.parent.as_posix(),
            "manifest_sha256": sha256_file(args.base_manifest),
            "train_sha256": sha256_file(args.base_train),
            "records": len(base_rows),
        },
        "repair_append": {
            "approved_input": len(repair_rows),
            "appended": len(appended),
            "rejected": len(rejected),
            "tokens": sum(int(row["assistant_supervised_tokens"]) for row in appended),
        },
        "composition": {
            component: {
                "records": component_records[component],
                "tokens": component_tokens[component],
                "fraction": component_tokens[component] / total_tokens,
            }
            for component in sorted(component_tokens)
        },
        "ratio_is_descriptive_only": True,
        "counts": {"train": len(combined), "smoke_train": len(smoke)},
        "artifacts": {
            "train": {"path": outputs["train"].as_posix(), "sha256": sha256_file(outputs["train"]), "records": len(combined)},
            "smoke_train": {"path": outputs["smoke"].as_posix(), "sha256": sha256_file(outputs["smoke"]), "records": len(smoke)},
            "append_rejections": {"path": outputs["rejections"].as_posix(), "sha256": sha256_file(outputs["rejections"]), "records": len(rejected)},
        },
        "limitations": [
            "基础 v2 属放宽后的内部工程数据，不是医生审核或临床验证",
            "修复追加数据已完成代理证据审核，但仍不是医生审核",
            "不得公开发布或用于临床诊断",
        ],
        "prohibited_actions_confirmed": {
            "training_run": False,
            "base_release_modified": False,
            "test_cases_read": False,
            "medical_safety_holdout_cases_read": False,
        },
    }
    _atomic_json(outputs["manifest"], manifest)
    print(json.dumps({"status": manifest["status"], "base_records": len(base_rows), "appended": len(appended), "rejected": len(rejected), "total_records": len(combined), "repair_tokens": manifest["repair_append"]["tokens"], "smoke": len(smoke)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
