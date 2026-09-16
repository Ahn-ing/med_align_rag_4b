"""从硬剔除旧医疗问题构建证据约束的 API 重构任务。"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.evidence_routing import evidence_id_for_question

SEED = "20260916-phase-s2-evidence-repair"
CATEGORY_PRIORITY = {
    "medication_boundary": 0,
    "emergency_escalation": 0,
    "action_followup": 1,
    "symptom_risk_clarification": 2,
    "summary_synthesis": 3,
    "knowledge_explanation": 4,
}
CJK = re.compile(r"[\u3400-\u9fff]")


def _message(record: dict[str, Any], role: str) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == role)


def _bigrams(text: str) -> set[str]:
    normalized = "".join(text.split())
    return {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if CJK.search(normalized[index : index + 2])
    }


def _best_support(question: str, source: dict[str, Any]) -> tuple[str, int]:
    question_terms = _bigrams(question)
    ranked = []
    source_terms = [str(value) for value in source.get("topic_terms", [])]
    for index, support in enumerate(source.get("supports", [])):
        score = len(question_terms & _bigrams(str(support)))
        score += sum(len(term) for term in source_terms if term in question)
        ranked.append((score, sha256_text(f"{SEED}:{index}:{support}"), str(support)))
    if not ranked:
        raise ValueError("来源没有 supports")
    score, _, support = max(ranked)
    return support, score


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--hard-drop", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks-per-focus", type=int, default=1)
    args = parser.parse_args()
    if args.max_tasks_per_focus <= 0:
        raise ValueError("--max-tasks-per-focus 必须为正数")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有修复任务")

    registry = json.loads(args.evidence_registry.read_text(encoding="utf-8"))
    sources = registry["sources"]
    sources_by_id = {source["evidence_id"]: source for source in sources}
    records = {
        row["id"]: row
        for row in iter_jsonl(args.pool)
        if row.get("domain") == "medical"
    }
    drops = list(iter_jsonl(args.hard_drop))
    prepared = []
    route_counts = Counter()
    for drop in drops:
        record = records.get(drop["record_id"])
        if record is None:
            raise ValueError(f"原记录不存在: {drop['record_id']}")
        question = _message(record, "user")
        evidence_id = evidence_id_for_question(question, sources)
        if evidence_id is None:
            route_counts["no_route"] += 1
            continue
        source = sources_by_id[evidence_id]
        support, support_score = _best_support(question, source)
        if support_score <= 0:
            route_counts["route_without_support"] += 1
            continue
        route_counts["evidence_supported"] += 1
        prepared.append((record, drop, evidence_id, support, support_score))

    prepared.sort(
        key=lambda item: (
            CATEGORY_PRIORITY.get(item[0]["primary_category"], 9),
            -item[4],
            sha256_text(f"{SEED}:{item[0]['id']}"),
        )
    )
    tasks = []
    focus_counts: Counter[str] = Counter()
    for record, drop, evidence_id, support, support_score in prepared:
        focus_key = re.sub(r"\W+", "", support).casefold()
        if focus_counts[focus_key] >= args.max_tasks_per_focus:
            route_counts["duplicate_focus"] += 1
            continue
        source = sources_by_id[evidence_id]
        projected_tokens = min(768, max(96, int(record["assistant_supervised_tokens"])))
        task_id = "api_v2_repair_" + sha256_record(
            [record["id"], evidence_id, support]
        )[:20]
        tasks.append(
            {
                "task_id": task_id,
                "repair_of_record_id": record["id"],
                "evidence_id": evidence_id,
                "publisher": source["publisher"],
                "source_url": source["url"],
                "focus_claim": support,
                "supported_claims": source["supports"],
                "reference_question": _message(record, "user"),
                "primary_category": record["primary_category"],
                "risk_level": record.get("risk_level", "medium"),
                "hard_reasons": drop.get("hard_reasons", []),
                "hard_triage_codes": drop.get("hard_triage_codes", []),
                "support_overlap_score": support_score,
                "projected_assistant_tokens": projected_tokens,
                "style": "保留原问题的自然表达意图，但删除无证据前提并改写为安全、聚焦的问题",
                "requirements": [
                    "生成一个自然中文用户问题和一个自然中文助手回答",
                    "旧回答绝不作为医学依据，reference_question 只用于表达意图",
                    "如原问题包含未经证据支持的诊断、剂量、治疗或预后前提，应改写为证据范围内的问题",
                    "重点回答 focus_claim，可依据 supported_claims 作不改变原意的合理扩充",
                    "具体剂量、个体诊断、精确时间窗、治疗方案或因果机制必须有明确证据",
                    "不得使用固定字段模板、固定开场结尾或审核元话语",
                    "严格返回包含 question 和 answer 的 JSON 对象",
                ],
            }
        )
        focus_counts[focus_key] += 1

    write_jsonl_atomic(args.output, tasks)
    manifest = {
        "format_version": 1,
        "status": "repair_tasks_built_pending_api_and_audit",
        "training_allowed": False,
        "release_allowed": False,
        "hard_drop_input": len(drops),
        "route_counts": dict(route_counts),
        "tasks": len(tasks),
        "unique_focus_claims": len(focus_counts),
        "max_tasks_per_focus": args.max_tasks_per_focus,
        "projected_assistant_tokens": sum(
            int(task["projected_assistant_tokens"]) for task in tasks
        ),
        "tasks_sha256": sha256_file(args.output),
        "inputs": {
            "pool": {"path": args.pool.as_posix(), "sha256": sha256_file(args.pool)},
            "hard_drop": {
                "path": args.hard_drop.as_posix(),
                "sha256": sha256_file(args.hard_drop),
            },
            "evidence_registry": {
                "path": args.evidence_registry.as_posix(),
                "sha256": sha256_file(args.evidence_registry),
            },
        },
        "blockers": ["api_generation_not_run", "generated_rows_require_evidence_audit"],
        "limitations": [
            "证据支持分数只是路由门禁，不能替代生成后的逐条证据审核",
            "不是医生审核或临床验证",
        ],
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "tasks": len(tasks), "route_counts": dict(route_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
