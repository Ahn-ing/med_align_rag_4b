"""从登记的权威证据创建少量、多样的 SFT v2 API 生成任务。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.evidence_routing import evidence_id_for_question
from med_align_rag_4b.sft.run_manifest import write_manifest_atomic
from med_align_rag_4b.sft.v2_quality import (
    API_MEDICAL_TOKEN_MAXIMUM,
    API_MEDICAL_TOKEN_MINIMUM,
    API_MEDICAL_TOKEN_TARGET,
    API_TOKEN_TARGETS,
    normalize_match,
)


def focus_key(text: str) -> str:
    """生成用于跨来源事实去重的稳定键，同时保留数字差异。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_match(text))


def category_for_support(text: str) -> str:
    immediate_care = "立即" in text and any(
        word in text for word in ("就医", "拨打", "呼叫", "急诊")
    )
    if any(word in text for word in ("急救", "中毒", "意识丧失", "医疗急症")) or immediate_care:
        return "emergency_escalation"
    if any(word in text for word in ("药", "抗生素", "处方", "注射器")):
        return "medication_boundary"
    if any(word in text for word in ("症状", "表现", "警示")):
        return "symptom_risk_clarification"
    if any(word in text for word in ("管理", "筛查", "避免", "习惯")):
        return "action_followup"
    return "knowledge_explanation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-registry", type=Path, nargs="+", required=True)
    parser.add_argument("--rule-pool", type=Path)
    parser.add_argument("--max-tasks-per-focus", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tasks_per_focus <= 0:
        raise ValueError("--max-tasks-per-focus 必须为正数")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(f"拒绝覆盖已有任务文件: {args.output}")
    registered_sources = []
    for registry_path in args.evidence_registry:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registered_sources.extend(registry["sources"])
    sources = {source["evidence_id"]: source for source in registered_sources}
    if len(sources) != len(registered_sources):
        raise ValueError("证据注册表之间存在重复 evidence_id")
    tasks = []
    authority_focus_keys = set()
    for source in registered_sources:
        for support_index, support in enumerate(source["supports"]):
            current_focus_key = focus_key(support)
            if current_focus_key in authority_focus_keys:
                continue
            authority_focus_keys.add(current_focus_key)
            task_id = (
                "api_v2_"
                + sha256_record([source["evidence_id"], str(support_index)])[:24]
            )
            category = category_for_support(support)
            tasks.append(
                {
                    "task_id": task_id,
                    "evidence_id": source["evidence_id"],
                    "publisher": source["publisher"],
                    "source_url": source["url"],
                    "focus_claim": support,
                    "supported_claims": source["supports"],
                    "primary_category": category,
                    "risk_level": "high"
                    if category == "emergency_escalation"
                    else "medium",
                    "style": "直接回答；允许受证据约束的自然解释和安全行动衔接",
                    "requirements": [
                        "生成一个自然中文用户问题和一个自然中文助手回答",
                        "重点回答 focus_claim，可依据 supported_claims 做不改变原意的合理扩充",
                        "具体剂量、个体诊断、精确时间窗、治疗方案或因果机制必须有明确证据",
                        "不得使用病因/症状/检查/诊断/治疗固定字段模板",
                        "不得出现根据你的描述、可以服用试试、祝早日康复或审核元话语",
                        "严格返回包含 question 和 answer 的 JSON 对象",
                    ],
                }
            )
    estimated_tokens = len(tasks) * 96
    category_tokens = Counter({category: 0 for category in API_TOKEN_TARGETS})
    source_tokens: Counter[str] = Counter()
    for task in tasks:
        category_tokens[task["primary_category"]] += 96
        source_tokens[task["evidence_id"]] += 96
    maximum_source_tokens = int(API_MEDICAL_TOKEN_TARGET * 0.20)
    anchored_records = 0
    focus_counts: Counter[str] = Counter()
    rule_pool_medical_tokens = 0
    if args.rule_pool is not None:
        if not args.rule_pool.is_file():
            raise FileNotFoundError(args.rule_pool)
        candidates = []
        for row in iter_jsonl(args.rule_pool):
            if row.get("domain") != "medical" or row.get("risk_level") == "high":
                continue
            rule_pool_medical_tokens += int(row["assistant_supervised_tokens"])
            question = row["messages"][0]["content"]
            evidence_id = evidence_id_for_question(question, registered_sources)
            if evidence_id is None:
                continue
            candidates.append(
                (sha256_record([row["id"], evidence_id]), row, evidence_id)
            )
        for _, row, evidence_id in sorted(candidates):
            question = row["messages"][0]["content"]
            if estimated_tokens >= API_MEDICAL_TOKEN_TARGET:
                break
            source = sources[evidence_id]
            support_index = int(sha256_record(row["id"])[0:8], 16) % len(
                source["supports"]
            )
            focus = source["supports"][support_index]
            current_focus_key = focus_key(focus)
            if focus_counts[current_focus_key] >= args.max_tasks_per_focus:
                continue
            category = category_for_support(focus)
            if row.get("primary_category") == "summary_synthesis":
                category = "summary_synthesis"
            projected_tokens = min(
                192, max(24, int(row["assistant_supervised_tokens"]))
            )
            if (
                category_tokens[category] + projected_tokens
                > API_TOKEN_TARGETS[category]
            ):
                continue
            if source_tokens[evidence_id] + projected_tokens > maximum_source_tokens:
                continue
            task_id = (
                "api_v2_anchor_" + sha256_record([row["id"], evidence_id, focus])[:20]
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "evidence_id": evidence_id,
                    "publisher": source["publisher"],
                    "source_url": source["url"],
                    "focus_claim": focus,
                    "supported_claims": source["supports"],
                    "reference_question": question,
                    "reference_record_id": row["id"],
                    "primary_category": category,
                    "risk_level": "high"
                    if category == "emergency_escalation"
                    else "medium",
                    "projected_assistant_tokens": projected_tokens,
                    "style": "借鉴 reference_question 的真实表达意图，但不复制其中无证据支持的医学前提",
                    "requirements": [
                        "生成一个自然中文用户问题和一个自然中文助手回答",
                        "reference_question 仅用于提供表达方式，不得把它当作医学证据",
                        "重点回答 focus_claim，可依据 supported_claims 合理扩充",
                        "具体剂量、个体诊断、精确时间窗、治疗方案或因果机制必须有明确证据",
                        "不得使用固定字段模板、固定开场结尾或审核元话语",
                        "严格返回包含 question 和 answer 的 JSON 对象",
                    ],
                }
            )
            focus_counts[current_focus_key] += 1
            estimated_tokens += projected_tokens
            category_tokens[category] += projected_tokens
            source_tokens[evidence_id] += projected_tokens
            anchored_records += 1
    effective_minimum = round(rule_pool_medical_tokens * 0.05 / 0.95)
    effective_target = round(rule_pool_medical_tokens * 0.075 / 0.925)
    effective_maximum = round(rule_pool_medical_tokens * 0.10 / 0.90)
    projected_fraction = (
        estimated_tokens / (rule_pool_medical_tokens + estimated_tokens)
        if rule_pool_medical_tokens
        else 0.0
    )
    write_jsonl_atomic(args.output, tasks)
    manifest = {
        "format_version": 1,
        "status": (
            "generation_tasks_built_pending_api_and_audit"
            if estimated_tokens >= effective_minimum
            else "task_coverage_below_api_minimum"
        ),
        "training_allowed": False,
        "records": len(tasks),
        "authority_seed_tasks": len(tasks) - anchored_records,
        "reference_question_tasks": anchored_records,
        "unique_focus_claims": len(
            {focus_key(task["focus_claim"]) for task in tasks}
        ),
        "max_tasks_per_focus": args.max_tasks_per_focus,
        "projected_assistant_tokens": estimated_tokens,
        "rule_pool_medical_tokens": rule_pool_medical_tokens,
        "projected_api_medical_token_fraction": projected_fraction,
        "projected_category_tokens": dict(category_tokens),
        "maximum_single_evidence_source_tokens": maximum_source_tokens,
        "projected_source_tokens": dict(source_tokens),
        "tasks_sha256": sha256_file(args.output),
        "evidence_registry_sha256": [
            sha256_file(path) for path in args.evidence_registry
        ],
        "api_medical_token_fraction": {
            "minimum": 0.05,
            "target": 0.075,
            "maximum": 0.10,
        },
        "api_medical_token_budget": {
            "minimum": API_MEDICAL_TOKEN_MINIMUM,
            "target": API_MEDICAL_TOKEN_TARGET,
            "maximum": API_MEDICAL_TOKEN_MAXIMUM,
        },
        "effective_budget_for_current_rule_pool": {
            "minimum": effective_minimum,
            "target": effective_target,
            "maximum": effective_maximum,
        },
        "blockers": [
            *(
                []
                if estimated_tokens >= effective_minimum
                else ["projected_api_fraction_below_5_percent"]
            ),
            *(
                []
                if estimated_tokens >= effective_target
                else ["projected_api_fraction_below_7_5_percent_target"]
            ),
            "api_generation_not_run",
            "generated_rows_require_evidence_audit",
        ],
    }
    write_manifest_atomic(manifest_path, manifest)
    print(json.dumps({"status": "tasks_built", "records": len(tasks)}))


if __name__ == "__main__":
    main()
