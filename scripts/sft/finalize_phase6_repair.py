"""审核证据约束 API 修复候选，并生成可直接追加的训练记录。"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer
from med_align_rag_4b.sft.v2_quality import automatic_quality_flags

SEED = "20260916-phase-s2-repair-audit"
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:%|mg|mmHg|mmol/L|天|周|个月|年|岁)?")
DANGEROUS = re.compile(r"一定能治好|肯定能治好|绝对能治好|肯定不需要就医|绝对不需要就医|不用就医|无需就医|不需要治疗|完全可以自愈")
LOWERCASE_LATIN = re.compile(r"\b[a-z]{4,}\b")
ALLOWED_LATIN = {"covid", "hiv", "who", "mmhg"}
TREATMENT_TERMS = (
    "抗病毒",
    "抗生素",
    "手术",
    "放疗",
    "化疗",
    "抗抑郁",
    "免疫抑制剂",
)


def _message(record: dict[str, Any], role: str) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == role)


def _bigrams(text: str) -> set[str]:
    value = re.sub(r"\s+", "", text)
    return {value[index : index + 2] for index in range(max(0, len(value) - 1))}


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


def _audit_reasons(record: dict[str, Any], task: dict[str, Any]) -> list[str]:
    question = _message(record, "user")
    answer = _message(record, "assistant")
    evidence_text = task["focus_claim"] + "\n" + "\n".join(task["supported_claims"])
    reasons = list(automatic_quality_flags(question, answer))
    if DANGEROUS.search(answer):
        reasons.append("dangerous_certainty")
    for token in LOWERCASE_LATIN.findall(answer):
        if token not in ALLOWED_LATIN:
            reasons.append(f"stray_lowercase_english:{token}")
    allowed_numbers = set(NUMBER.findall(evidence_text)) | set(NUMBER.findall(question))
    for number in NUMBER.findall(answer):
        if number not in allowed_numbers:
            reasons.append(f"unregistered_numeric:{number}")
    evidence_folded = evidence_text.casefold()
    for term in TREATMENT_TERMS:
        recommendation = re.search(rf"(?:建议|需要|可以|应当|必须|使用|服用)[^。；\n]{{0,12}}{re.escape(term)}", answer)
        if recommendation and term not in evidence_folded:
            reasons.append(f"unregistered_specific_treatment:{term}")
    if not (_bigrams(answer) & _bigrams(evidence_text)):
        reasons.append("no_evidence_term_overlap")
    return sorted(set(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    approved_path = args.output_dir / "approved_candidates.jsonl"
    quarantine_path = args.output_dir / "quarantine.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    for path in (approved_path, quarantine_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    tasks = list(iter_jsonl(args.tasks))
    task_by_id = {task["task_id"]: task for task in tasks}
    generated = []
    shards = sorted(args.generated_dir.glob("repair_shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError("没有生成分片")
    for shard in shards:
        generated.extend(iter_jsonl(shard))
    if len(generated) != len(tasks):
        raise ValueError(f"生成数量与任务数量不一致: {len(generated)} != {len(tasks)}")
    if len({row["id"] for row in generated}) != len(generated):
        raise ValueError("生成记录 ID 重复")

    tokenizer = QwenAssistantTokenizer(args.tokenizer_path, max_length=2048)
    approved = []
    quarantine = []
    for record in generated:
        task = task_by_id.get(record["id"])
        if task is None:
            quarantine.append({"id": record["id"], "reasons": ["task_not_found"]})
            continue
        reasons = _audit_reasons(record, task)
        if reasons:
            quarantine.append(
                {
                    "id": record["id"],
                    "reasons": reasons,
                    "question": _message(record, "user"),
                    "answer": _message(record, "assistant"),
                    "evidence_id": task["evidence_id"],
                }
            )
            continue
        metrics = tokenizer.analyze(record["messages"])
        approved.append(
            {
                "id": record["id"],
                "source_id": "api_evidence_constrained_v5",
                "source_record_id": task["task_id"],
                "source_revision": "evidence-registry-v5-20260915",
                "messages": record["messages"],
                "domain": "medical",
                "ability": record["primary_category"],
                "language": "zh",
                "risk_level": record["risk_level"],
                "primary_category": record["primary_category"],
                "secondary_categories": [],
                "license_status": "approved",
                "medical_evidence": [task["evidence_id"]],
                "review": {
                    "status": "agent_evidence_reviewed",
                    "reviewer_role": "codex_research_reviewer",
                    "reviewed_at": "2026-09-16",
                    "rubric_version": "s2-evidence-repair-v1",
                    "evidence_checked": True,
                    "doctor_reviewed": False,
                    "clinical_validation": False,
                },
                "quality_flags": [],
                "unresolved_safety_flags": [],
                "fingerprints": {
                    "question": question_fingerprint(_message(record, "user")),
                    "qa": qa_fingerprint(_message(record, "user"), _message(record, "assistant")),
                    "question_skeleton": sha256_text(_message(record, "user")),
                    "answer_skeleton": sha256_text(_message(record, "assistant")),
                },
                "provenance": {
                    "generation_method": "evidence_constrained_api",
                    "model": "deepseek-flash",
                    "repair_of_record_id": task["repair_of_record_id"],
                    "reference_question": task["reference_question"],
                    "source_url": task["source_url"],
                    "focus_claim": task["focus_claim"],
                },
                "total_tokens": metrics["total_tokens"],
                "assistant_supervised_tokens": metrics["assistant_supervised_tokens"],
                "truncated": False,
                "data_component": "rule_pool_repaired",
                "release_allowed": False,
            }
        )

    write_jsonl_atomic(approved_path, approved)
    write_jsonl_atomic(quarantine_path, quarantine)
    manifest = {
        "format_version": 1,
        "status": "repair_evidence_audit_completed",
        "training_allowed": False,
        "release_allowed": False,
        "records": len(generated),
        "approved": len(approved),
        "quarantined": len(quarantine),
        "approved_tokens": sum(row["assistant_supervised_tokens"] for row in approved),
        "quarantine_reasons": dict(Counter(reason for row in quarantine for reason in row["reasons"])),
        "artifacts": {
            "approved": {"path": approved_path.as_posix(), "sha256": sha256_file(approved_path), "records": len(approved)},
            "quarantine": {"path": quarantine_path.as_posix(), "sha256": sha256_file(quarantine_path), "records": len(quarantine)},
        },
        "inputs": {
            "tasks": {"path": args.tasks.as_posix(), "sha256": sha256_file(args.tasks)},
            "generated_dir": args.generated_dir.as_posix(),
        },
        "limitations": [
            "代理证据审核不是医生审核或临床验证",
            "只允许内部工程训练，release_allowed=false",
        ],
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "approved": len(approved), "quarantined": len(quarantine), "tokens": manifest["approved_tokens"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
