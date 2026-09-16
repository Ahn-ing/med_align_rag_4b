"""按放宽门禁选择 60% 的 rule_pool 内部训练候选。"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint
from med_align_rag_4b.sft.preprocessing import clean_text

SEED = "20260915-phase-s1r-salvage-60"
TARGET_TOKENS = 651_456
CATEGORY_TARGETS = {
    "knowledge_explanation": 228_010,
    "symptom_risk_clarification": 130_291,
    "action_followup": 130_291,
    "summary_synthesis": 97_718,
    "medication_boundary": 65_146,
}
HARD_TRIAGE_CODES = {
    "control_or_replacement_char",
    "possible_personal_data",
    "unsupported_diagnosis",
    "unsupported_prognosis",
    "dangerous_certainty",
}
HARD_PATTERNS = {
    "personal_or_control": re.compile(
        r"1[3-9]\d{9}|身份证|微信号|QQ号|住院号|病历号|[\ufffd\x00-\x08]"
    ),
    "personal_diagnosis": re.compile(
        r"诊断为|确诊为|表明你是|说明你有|你的情况|你得的是|您是患了|你是患了"
    ),
    "specific_dose": re.compile(
        r"(?:一次|每天|每日|每次)[^。；\n]{0,8}(?:片|粒|毫克|mg|毫升)|"
        r"[0-9一二三四五六七八九十两]+片"
    ),
    "explicit_treatment": re.compile(
        r"建议[^。；\n]{0,16}(?:手术|化疗|放疗|服用|口服|用药|切除|碎石|"
        r"输液|住院|转院|灌肠|针灸|推拿)"
    ),
    "hard_prognosis": re.compile(r"活多久|能活多久|生存率|预后|治愈率|存活期|活不过"),
    "no_care_certainty": re.compile(
        r"不用治疗|不需要治疗|无须治疗|不用就医|无需就医|肯定会好|一定能治|完全可以自愈"
    ),
    "explicit_medication_or_authority_claim": re.compile(
        r"偏方|秘方|根治|彻底治愈|保证治愈|"
        r"(?:服用|口服|吃|用)[^。；\n]{0,8}(?:片|丸|胶囊|颗粒|口服液|注射液|药剂)|"
        r"抗抑郁剂|护肝片|降糖药|降压药|抗生素|激素"
    ),
}
QUALITY_WEIGHTS = {
    "list_dump": 5,
    "template_misassembly": 4,
    "specific_treatment_or_testing": 4,
    "unsupported_diagnosis": 6,
    "answer_topic_not_observed": 3,
    "no_registered_evidence_route": 3,
    "patient_specific_context": 2,
    "quantitative_clinical_claim": 2,
    "non_knowledge_requires_evidence_review": 1,
    "minor_formatting": 0,
}


def _message(record: dict[str, Any], role: str) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == role)


def _quality_score(triage: dict[str, Any]) -> int:
    return sum(QUALITY_WEIGHTS.get(code, 5) for code in triage["reason_codes"])


def _selection_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, str]:
    record, triage = item
    return _quality_score(triage), sha256_text(f"{SEED}:{record['id']}")


def _hard_reasons(question: str, answer: str) -> list[str]:
    text = question + "\n" + answer
    return sorted(name for name, pattern in HARD_PATTERNS.items() if pattern.search(text))


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


def _select_category(
    rows: list[tuple[dict[str, Any], dict[str, Any]]], target: int
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    tokens = 0
    seen_qa: set[str] = set()
    seen_question: set[str] = set()
    for item in sorted(rows, key=_selection_key):
        record = item[0]
        question = _message(record, "user")
        answer = _message(record, "assistant")
        qa = qa_fingerprint(question, answer)
        qfp = question_fingerprint(question)
        if qa in seen_qa or qfp in seen_question:
            continue
        record_tokens = int(record["assistant_supervised_tokens"])
        if tokens + record_tokens > target:
            continue
        selected.append(item)
        seen_qa.add(qa)
        seen_question.add(qfp)
        tokens += record_tokens
    return selected, tokens


def _select_audit_sample(
    rows: list[tuple[dict[str, Any], dict[str, Any]]], target: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    groups: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in rows:
        groups[(item[0]["primary_category"], min(_quality_score(item[1]), 5))].append(item)
    ordered_groups = []
    for group in groups.values():
        group.sort(key=lambda item: sha256_text(f"{SEED}:audit:{item[0]['id']}"))
        ordered_groups.append(group)
    ordered_groups.sort(key=lambda group: (group[0][0]["primary_category"], _quality_score(group[0][1])))
    selected = []
    while ordered_groups and len(selected) < target:
        next_groups = []
        for group in ordered_groups:
            selected.append(group.pop(0))
            if group:
                next_groups.append(group)
            if len(selected) == target:
                break
        ordered_groups = next_groups
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-size", type=int, default=200)
    args = parser.parse_args()
    if args.audit_size <= 0:
        raise ValueError("--audit-size 必须为正数")

    selected_path = args.output_dir / "rule_pool_selected.jsonl"
    audit_path = args.output_dir / "audit_sample.jsonl"
    hard_drop_path = args.output_dir / "hard_drop.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    for path in (selected_path, audit_path, hard_drop_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    triage_by_id = {row["record_id"]: row for row in iter_jsonl(args.triage)}
    medical_rows = [row for row in iter_jsonl(args.pool) if row["domain"] == "medical"]
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    hard_drops = []
    for record in medical_rows:
        triage = triage_by_id.get(record["id"])
        if triage is None:
            raise ValueError(f"triage 缺少记录: {record['id']}")
        question = _message(record, "user")
        answer = _message(record, "assistant")
        reasons = _hard_reasons(question, answer)
        severe_triage = sorted(set(triage["reason_codes"]) & HARD_TRIAGE_CODES)
        if reasons or severe_triage:
            hard_drops.append(
                {
                    "record_id": record["id"],
                    "primary_category": record["primary_category"],
                    "assistant_supervised_tokens": record["assistant_supervised_tokens"],
                    "hard_reasons": reasons,
                    "hard_triage_codes": severe_triage,
                }
            )
            continue
        candidates[record["primary_category"]].append((record, triage))

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    category_report = {}
    for category, target in CATEGORY_TARGETS.items():
        category_selected, tokens = _select_category(candidates[category], target)
        selected.extend(category_selected)
        category_report[category] = {
            "target_tokens": target,
            "selected_records": len(category_selected),
            "selected_tokens": tokens,
            "candidate_records": len(candidates[category]),
        }

    selected.sort(key=lambda item: item[0]["id"])
    total_tokens = sum(
        int(record["assistant_supervised_tokens"]) for record, _ in selected
    )
    if abs(total_tokens - TARGET_TOKENS) > TARGET_TOKENS * 0.01:
        raise ValueError(f"rule_pool 选择 token 超出 ±1%: {total_tokens}")

    output_rows = []
    for record, triage in selected:
        updated = dict(record)
        route = triage.get("evidence_id")
        updated["review"] = {
            "status": "agent_cleaned_legacy_reviewed",
            "reviewer_role": "codex_research_reviewer",
            "reviewed_at": "2026-09-15",
            "rubric_version": "s1r-relaxed-v1",
            "evidence_checked": False,
            "limitations": "内部工程训练审核；放宽了旧池语言、完整性和引用要求，不是医生审核或临床验证",
        }
        updated["quality_flags"] = sorted(set(record.get("quality_flags", []) + triage["reason_codes"]))
        updated["unresolved_safety_flags"] = []
        updated["release_allowed"] = False
        updated["provenance"] = {
            **(record.get("provenance") or {}),
            "selection_method": "s1r_relaxed_quality_gate",
            "routed_evidence_id": route,
            "evidence_route_checked": bool(route),
            "s0_decision": triage["decision"],
            "s0_reason_codes": triage["reason_codes"],
        }
        output_rows.append(updated)

    write_jsonl_atomic(selected_path, output_rows)
    write_jsonl_atomic(hard_drop_path, hard_drops)

    audit_rows = []
    for record, triage in _select_audit_sample(selected, args.audit_size):
        audit_rows.append(
            {
                "format_version": 1,
                "quality_score": _quality_score(triage),
                "s0_decision": triage["decision"],
                "s0_reason_codes": triage["reason_codes"],
                "record_id": record["id"],
                "primary_category": record["primary_category"],
                "question": _message(record, "user"),
                "answer": _message(record, "assistant"),
                "cleaned_question": clean_text(_message(record, "user")),
                "cleaned_answer": clean_text(_message(record, "assistant")),
            }
        )
    write_jsonl_atomic(audit_path, audit_rows)

    reason_counts = Counter()
    for _, triage in selected:
        reason_counts.update(triage["reason_codes"])
    manifest = {
        "format_version": 1,
        "status": "s1r_rule_pool_selected_pending_training_gate",
        "training_allowed": False,
        "release_allowed": False,
        "seed": SEED,
        "target_tokens": TARGET_TOKENS,
        "selected_tokens": total_tokens,
        "selected_records": len(output_rows),
        "hard_drop_records": len(hard_drops),
        "category_report": category_report,
        "selected_quality_reason_counts": dict(reason_counts),
        "inputs": {
            "pool": {"path": args.pool.as_posix(), "sha256": sha256_file(args.pool)},
            "triage": {"path": args.triage.as_posix(), "sha256": sha256_file(args.triage)},
        },
        "artifacts": {
            "selected": {"path": selected_path.as_posix(), "sha256": sha256_file(selected_path), "records": len(output_rows)},
            "audit_sample": {"path": audit_path.as_posix(), "sha256": sha256_file(audit_path), "records": len(audit_rows)},
            "hard_drop": {"path": hard_drop_path.as_posix(), "sha256": sha256_file(hard_drop_path), "records": len(hard_drops)},
        },
        "limitations": [
            "该池允许缺少逐条权威证据的低/中风险旧知识，只用于内部工程训练",
            "证据路由存在误匹配，routed_evidence_id 不是已核查证据",
            "不是医生审核或临床验证，禁止公开发布",
        ],
        "prohibited_actions_confirmed": {
            "training_run": False,
            "release_created": False,
            "test_or_medical_safety_holdout_read": False,
        },
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "records": len(output_rows), "tokens": total_tokens, "hard_drop": len(hard_drops)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
