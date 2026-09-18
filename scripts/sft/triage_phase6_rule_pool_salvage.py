"""执行旧医疗 rule pool 的 S0 清洗可行性分桶。"""

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
from med_align_rag_4b.sft.evidence_routing import evidence_id_for_question
from med_align_rag_4b.sft.preprocessing import clean_text

SEED = "20260915-phase-s0-rule-pool-salvage"
DEFAULT_AUDIT_SIZE = 200

_CONTROL = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
_LIST_DUMP = re.compile(r"(?:[^;；]{1,18}[;；]){5,}")
_TEMPLATE = re.compile(
    r"指导意见[:：]?|需要多咨询专家|正规医院|保持沟通|患者还需要|"
    r"得到良好的治疗|专家表示|患者朋友理应|以上就是对"
)
_DIAGNOSIS = re.compile(r"确诊|诊断为|患有|表明你是|说明你有|就是[^，。；]{0,10}(?:炎|病|癌)")
_TREATMENT = re.compile(
    r"口服|服用|用药|药物|服药|注射液|阿司匹林|抗生素|激素|化疗|放疗|"
    r"手术|碎石|灌肠|理疗|康复训练|针灸|推拿|按摩|补钙|维生素|"
    r"输液|住院|转院|检查|筛查|接种|疫苗|治疗|剂量|毫克|[0-9]+mg"
)
_PROGNOSIS = re.compile(
    r"预后|生存率|晚期|治愈率|活不过|还能活|延长寿命|活多久|能活多久|"
    r"彻底治愈|治好|痊愈|希望康复"
)
_CERTAINTY = re.compile(r"一定能|肯定不会|绝对|唯一|无须|不用治疗|不需要治疗|完全可以自愈")
_PATIENT_CONTEXT = re.compile(r"我的|我家|孩子|宝宝|小孩|患者|本人|朋友|妈妈|爸爸|家人|怀孕|孕周")
_QUANT_CLINICAL = re.compile(r"\d+(?:\.\d+)?\s*(?:mmol/L|mmHg|mg|毫升|毫克|片|天|周|个月|岁|年|支)")
_PERSONAL = re.compile(r"1[3-9]\d{9}|身份证|微信号|QQ号|详细地址|住院号|病历号")
_MINOR_FORMAT = re.compile(r"\*\*|(?:[!?。！？]\s*){2,}|\n{3,}")


def _message(record: dict[str, Any], role: str) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == role)


def _source_terms(source: dict[str, Any]) -> list[str]:
    terms = []
    for value in [source.get("title", ""), *source.get("topic_terms", [])]:
        term = str(value).strip().casefold()
        if 2 <= len(term) <= 16:
            terms.append(term)
    return terms


def classify_record(
    record: dict[str, Any],
    sources: list[dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """把一条旧医疗记录分成保留候选、证据重构或丢弃。"""
    question = _message(record, "user")
    answer = _message(record, "assistant")
    text = f"{question}\n{answer}"
    evidence_id = evidence_id_for_question(question, sources)
    repair_codes: list[str] = []
    drop_codes: list[str] = []
    minor_codes: list[str] = []

    if _CONTROL.search(text):
        drop_codes.append("control_or_replacement_char")
    if _PERSONAL.search(text):
        drop_codes.append("possible_personal_data")
    if _LIST_DUMP.search(answer):
        repair_codes.append("list_dump")
    if _TEMPLATE.search(answer):
        repair_codes.append("template_misassembly")
    if _DIAGNOSIS.search(answer):
        repair_codes.append("unsupported_diagnosis")
    if _TREATMENT.search(answer):
        repair_codes.append("specific_treatment_or_testing")
    if _PROGNOSIS.search(answer):
        repair_codes.append("unsupported_prognosis")
    if _CERTAINTY.search(answer):
        repair_codes.append("dangerous_certainty")
    if (_PATIENT_CONTEXT.search(question)):
        repair_codes.append("patient_specific_context")
    if (_QUANT_CLINICAL.search(answer)):
        repair_codes.append("quantitative_clinical_claim")
    if record["primary_category"] != "knowledge_explanation":
        repair_codes.append("non_knowledge_requires_evidence_review")
    if _MINOR_FORMAT.search(text):
        minor_codes.append("minor_formatting")

    if evidence_id is None:
        drop_codes.append("no_registered_evidence_route")
    else:
        source = sources_by_id[evidence_id]
        answer_folded = answer.casefold()
        if not any(term in answer_folded for term in _source_terms(source)):
            repair_codes.append("answer_topic_not_observed")

    if drop_codes:
        decision = "drop"
        reason_codes = sorted(set(drop_codes + repair_codes))
    elif repair_codes:
        decision = "repair_from_evidence"
        reason_codes = sorted(set(repair_codes + minor_codes))
    else:
        decision = "keep_after_minor_clean"
        reason_codes = sorted(set(minor_codes))

    return {
        "format_version": 1,
        "record_id": record["id"],
        "source_id": record["source_id"],
        "source_record_id": record["source_record_id"],
        "primary_category": record["primary_category"],
        "risk_level": record["risk_level"],
        "assistant_supervised_tokens": int(record["assistant_supervised_tokens"]),
        "decision": decision,
        "reason_codes": reason_codes,
        "evidence_id": evidence_id,
        "question": question,
        "answer": answer,
        "cleaned_question": clean_text(question),
        "cleaned_answer": clean_text(answer),
    }


def _triage_key(item: dict[str, Any]) -> str:
    return sha256_text(f"{SEED}:{item['record_id']}")
def _stratified_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["primary_category"], row["evidence_id"] or "none")].append(row)
    ordered_groups = sorted(groups.values(), key=lambda group: min(map(_triage_key, group)))
    for group in ordered_groups:
        group.sort(key=_triage_key)
    result: list[dict[str, Any]] = []
    while any(ordered_groups):
        next_groups = []
        for group in ordered_groups:
            result.append(group.pop(0))
            if group:
                next_groups.append(group)
        ordered_groups = next_groups
    return result


def _select_audit_sample(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    keep = [row for row in rows if row["decision"] == "keep_after_minor_clean"]
    repair = [row for row in rows if row["decision"] == "repair_from_evidence"]
    keep_target = min(len(keep), target // 2)
    repair_target = min(len(repair), target - keep_target)
    selected = _stratified_rows(keep)[:keep_target] + _stratified_rows(repair)[:repair_target]
    if len(selected) < target:
        selected_ids = {row["record_id"] for row in selected}
        remaining = [row for row in keep + repair if row["record_id"] not in selected_ids]
        selected.extend(_stratified_rows(remaining)[: target - len(selected)])
    return selected


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in rows)
    tokens = Counter()
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    reasons = Counter()
    for row in rows:
        tokens[row["decision"]] += row["assistant_supervised_tokens"]
        categories[row["primary_category"]][row["decision"]] += 1
        reasons.update(row["reason_codes"])
    return {
        "records": len(rows),
        "decisions": dict(decisions),
        "tokens": dict(tokens),
        "category_decisions": {key: dict(value) for key, value in sorted(categories.items())},
        "reason_codes": dict(reasons),
    }


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
    parser.add_argument("--phase-a-sample", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-size", type=int, default=DEFAULT_AUDIT_SIZE)
    args = parser.parse_args()
    if args.audit_size <= 0:
        raise ValueError("--audit-size 必须为正数")

    outputs = {
        "triage": args.output_dir / "triage.jsonl",
        "pilot_triage": args.output_dir / "pilot_triage.jsonl",
        "audit_sample": args.output_dir / "pilot_audit_sample.jsonl",
    }
    for path in [*outputs.values(), args.report]:
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    pool_manifest = json.loads((args.pool.parent / "manifest.json").read_text(encoding="utf-8"))
    pool_expected = pool_manifest["rule_screened_pool"]
    sample_manifest = json.loads(
        args.phase_a_sample.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    registry = json.loads(args.evidence_registry.read_text(encoding="utf-8"))
    sources = registry["sources"]
    sources_by_id = {source["evidence_id"]: source for source in sources}

    medical_rows = [record for record in iter_jsonl(args.pool) if record["domain"] == "medical"]
    if len(medical_rows) != pool_manifest["rule_screened_pool"]["medical_records"]:
        raise ValueError("medical 记录数与 manifest 不一致")
    triage_rows = [classify_record(record, sources, sources_by_id) for record in medical_rows]

    phase_a_rows = list(iter_jsonl(args.phase_a_sample))
    pilot_triage = [
        classify_record(item["record"], sources, sources_by_id)
        for item in phase_a_rows
        if item["record"]["domain"] == "medical"
    ]
    audit_sample = _select_audit_sample(triage_rows, args.audit_size)

    write_jsonl_atomic(outputs["triage"], triage_rows)
    write_jsonl_atomic(outputs["pilot_triage"], pilot_triage)
    write_jsonl_atomic(outputs["audit_sample"], audit_sample)

    full = _counts(triage_rows)
    phase_a = _counts(pilot_triage)
    keep_tokens = full["tokens"].get("keep_after_minor_clean", 0)
    repair_tokens = full["tokens"].get("repair_from_evidence", 0)
    report = {
        "format_version": 1,
        "status": "s0_triage_complete_pending_agent_audit",
        "training_allowed": False,
        "release_allowed": False,
        "seed": SEED,
        "inputs": {
            "pool": {"path": args.pool.as_posix(), "sha256": sha256_file(args.pool)},
            "phase_a_sample": {
                "path": args.phase_a_sample.as_posix(),
                "sha256": sha256_file(args.phase_a_sample),
            },
            "evidence_registry": {
                "path": args.evidence_registry.as_posix(),
                "sha256": sha256_file(args.evidence_registry),
                "sources": len(sources),
            },
        },
        "full_medical_triage": full,
        "phase_a_pilot_triage": phase_a,
        "estimated_tokens": {
            "keep_after_minor_clean": keep_tokens,
            "repair_from_evidence": repair_tokens,
            "candidate_after_successful_repair": keep_tokens + repair_tokens,
            "drop": full["tokens"].get("drop", 0),
        },
        "pilot_gate": {
            "minimum_candidate_medical_tokens": 50_000,
            "maximum_repair_tasks": 300,
            "observed_candidate_medical_tokens": keep_tokens + repair_tokens,
            "observed_repair_records": full["decisions"].get("repair_from_evidence", 0),
            "status": "pending_agent_audit",
        },
        "audit_sample": {
            "path": outputs["audit_sample"].as_posix(),
            "records": len(audit_sample),
            "sha256": sha256_file(outputs["audit_sample"]),
        },
        "artifacts": {
            name: {
                "path": path.as_posix(),
                "records": len(triage_rows)
                if name == "triage"
                else len(pilot_triage)
                if name == "pilot_triage"
                else len(audit_sample),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
        "limitations": [
            "S0 只估计清洗产出率，不代表任何旧记录已通过最终审核",
            "正则匹配会误报或漏报，审计样本必须人工复核",
            "未训练、未发布、未读取 test 或 medical safety holdout 正文",
        ],
        "prohibited_actions_confirmed": {
            "training_run": False,
            "release_created": False,
            "test_or_medical_safety_holdout_read": False,
        },
    }
    _write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "medical_records": full["records"],
                "decisions": full["decisions"],
                "tokens": full["tokens"],
                "audit_sample": len(audit_sample),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
