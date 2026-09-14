"""SFT 来源和候选记录的治理契约。"""

from __future__ import annotations

from typing import Any

SOURCE_STATUSES = {"approved", "pending", "rejected"}
REVIEW_STATUSES = {"unreviewed", "ready_for_review", "approved", "agent_evidence_reviewed", "needs_revision", "rejected"}
INTERNAL_REVIEWER_ROLE = "codex_research_reviewer"


def validate_source(source: dict[str, Any]) -> None:
    """校验数据源登记项；未知信息必须显式写为 unknown。"""
    required = {
        "source_id", "name", "purpose", "origin", "source_url", "revision",
        "license", "license_status", "contains_medical_claims",
        "contains_personal_data", "generation_method", "allowed_uses",
        "review_status", "notes",
    }
    missing = sorted(required - source.keys())
    if missing:
        raise ValueError(f"数据源缺少字段: {', '.join(missing)}")
    for field in ("source_id", "name", "origin", "revision", "license", "generation_method"):
        if not isinstance(source[field], str) or not source[field].strip():
            raise ValueError(f"{field} 必须是非空字符串")
    if source["license_status"] not in SOURCE_STATUSES or source["review_status"] not in SOURCE_STATUSES:
        raise ValueError("许可证或来源审核状态无效")
    if not isinstance(source["allowed_uses"], list):
        raise ValueError("allowed_uses 必须是列表")


def validate_candidate(record: dict[str, Any]) -> None:
    """校验统一分析记录的最小结构。"""
    required = {
        "id", "source_id", "source_record_id", "messages", "domain", "ability",
        "language", "risk_level", "primary_category", "secondary_categories",
        "source_revision", "license_status", "medical_evidence", "review",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"候选记录缺少字段: {', '.join(missing)}")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空列表")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
            raise ValueError("messages 中的 role/content 无效")
    review = record["review"]
    if not isinstance(review, dict) or review.get("status") not in REVIEW_STATUSES:
        raise ValueError("review.status 无效")


def validate_training_candidate(record: dict[str, Any], source: dict[str, Any]) -> None:
    """执行正式内部训练记录的来源、证据和代理审核门禁。"""
    validate_source(source)
    validate_candidate(record)
    if source["license_status"] != "approved" or source["review_status"] != "approved":
        raise ValueError("来源许可证或审核未批准")
    if "training_candidate" not in source["allowed_uses"]:
        raise ValueError("来源未获准用于 training_candidate")
    if record["license_status"] != "approved":
        raise ValueError("记录许可证未批准")
    if source["contains_personal_data"] is not False:
        raise ValueError("个人数据状态必须明确为 false")
    review = record["review"]
    if review.get("status") != "agent_evidence_reviewed":
        raise ValueError("记录尚未完成代理证据审核")
    if review.get("reviewer_role") != INTERNAL_REVIEWER_ROLE:
        raise ValueError("审核角色无效")
    if not review.get("evidence_checked"):
        raise ValueError("审核未确认检查证据")
    if record["domain"] in {"medical", "medical_safety"} and not record["medical_evidence"]:
        raise ValueError("医疗记录缺少权威证据")
    if record.get("unresolved_safety_flags"):
        raise ValueError("记录仍有未解决安全标记")
