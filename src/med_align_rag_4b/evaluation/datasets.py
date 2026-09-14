"""评测案例门禁和泄漏检查。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from med_align_rag_4b.sft.deduplication import question_fingerprint

SPLITS = {"validation", "test", "medical_safety_holdout"}


def validate_case(case: dict[str, Any]) -> None:
    required = {
        "id", "split", "task", "primary_category", "secondary_categories", "risk_level",
        "prompt", "reference_answer", "required_facts", "forbidden_claims", "required_actions",
        "forbidden_actions", "clarification_expectations", "output_schema", "rag_requirement",
        "source_id", "source_revision", "medical_evidence", "license_status", "review",
    }
    missing = sorted(required - case.keys())
    if missing:
        raise ValueError(f"评测案例缺少字段: {', '.join(missing)}")
    if case["split"] not in SPLITS:
        raise ValueError("split 无效")
    review = case["review"]
    if case["license_status"] != "approved" or review.get("status") not in {"approved", "agent_evidence_reviewed"}:
        raise ValueError("未获许可证和审核批准的案例不能冻结")
    if case.get("contains_personal_data") is not False:
        raise ValueError("个人数据状态必须明确为 false")
    if review.get("status") == "agent_evidence_reviewed":
        if review.get("reviewer_role") != "codex_research_reviewer" or case.get("release_allowed") is not False:
            raise ValueError("代理证据审核案例必须固定审核角色且禁止发布")
    if case["task"] in {"medical_qa", "medical_safety", "rag_faithfulness"} and not case["medical_evidence"]:
        raise ValueError("医疗案例缺少权威来源")
    for evidence in case["medical_evidence"]:
        required_evidence = {"evidence_id", "publisher", "title", "url", "revision", "accessed_at", "supports"}
        if not isinstance(evidence, dict) or required_evidence - evidence.keys():
            raise ValueError("医疗证据字段不完整")
    if case["risk_level"] == "high" and (not case["required_actions"] or not case["forbidden_actions"]):
        raise ValueError("高风险案例必须定义 required_actions 和 forbidden_actions")
    category = case["primary_category"]
    if category == "home_observation" and not case["required_actions"]:
        raise ValueError("家庭观察案例必须定义升级条件")
    if category == "medication_boundary" and (not case["required_actions"] or not case["forbidden_actions"]):
        raise ValueError("药物案例必须定义边界和禁止行为")
    if category == "clarification" and not case["clarification_expectations"]:
        raise ValueError("澄清案例必须定义关键信息")
    if case["task"] == "structured_output" and case["output_schema"] is None:
        raise ValueError("结构化输出案例必须提供 schema")


def validate_no_leakage(splits: dict[str, Iterable[dict[str, Any]]], blocked_fingerprints: set[str] | None = None) -> None:
    """确保 ID、规范化问题与 group 不在 split 内或跨 split 泄漏。"""
    seen_ids: dict[str, str] = {}
    seen_questions: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    blocked = blocked_fingerprints or set()
    for split, records in splits.items():
        for case in records:
            if case["split"] != split:
                raise ValueError("记录 split 与容器不一致")
            qfp = question_fingerprint(case["prompt"])
            for value, seen, label in ((case["id"], seen_ids, "ID"), (qfp, seen_questions, "问题")):
                if value in seen:
                    raise ValueError(f"{label} 在 split 内或跨 split 重复")
                seen[value] = split
            group = case.get("group_id")
            if group and group in seen_groups and seen_groups[group] != split:
                raise ValueError("group 跨 split")
            if group:
                seen_groups[group] = split
            if qfp in blocked:
                raise ValueError("评测问题与 RAG 或排除指纹重叠")
