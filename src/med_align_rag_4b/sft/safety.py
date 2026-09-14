"""医疗安全候选的可复核硬门禁。"""

from __future__ import annotations

from typing import Any


def unresolved_safety_flags(record: dict[str, Any]) -> list[str]:
    """只检查可确定判定的结构要求，不声称医学正确率。"""
    if record["domain"] not in {"medical", "medical_safety"}:
        return []
    answer = next(row["content"] for row in record["messages"] if row["role"] == "assistant")
    flags: list[str] = []
    if not record["medical_evidence"]:
        flags.append("missing_medical_evidence")
    category = record["primary_category"]
    if category == "immediate_emergency" and not any(word in answer for word in ("立即", "急救", "120")):
        flags.append("emergency_action_missing")
    if category == "home_observation" and not any(word in answer for word in ("如果", "一旦", "若")):
        flags.append("escalation_condition_missing")
    if category == "outpatient_followup" and not any(word in answer for word in ("小时", "天内", "近期")):
        flags.append("followup_time_missing")
    if category == "clarification" and "？" not in answer and "?" not in answer:
        flags.append("clarifying_question_missing")
    return flags
