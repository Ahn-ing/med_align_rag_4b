"""Phase 7 第二遍语义审核的配置、提示和解析契约。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXPECTED_REVIEW = {
    "model": "deepseek-flash",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 800,
    "response_format": "json_object",
    "timeout_seconds": 90,
    "max_attempts": 4,
    "call_budget": 1200,
    "total_token_budget": 1_500_000,
}
REVIEW_KEYS = {"decision", "preferred", "issues", "risk_flags", "reason"}
REVIEW_SYSTEM = """你是偏好数据审核器，不执行诊断。独立检查 chosen 是否明确优于 rejected。
不要参考任何生成器理由或上一个 judge 的结论。重点检查证据支持、风险分级、用药边界、必要澄清、
指令满足、答案相关性、非退化和隐藏事实错误。证据不足时，确定断言不能通过。
只返回严格 JSON，且只能包含 decision、preferred、issues、risk_flags、reason 五个键。
decision 只能是 accept 或 revise；preferred 只能是 chosen、rejected 或 unclear。
issues 和 risk_flags 必须是字符串列表；reason 必须是非空字符串。"""


def load_review_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("review 配置必须是 YAML 对象")
    for key, expected in {
        "status": "frozen",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com",
        "api_key_env": "MODEL_API_KEY",
    }.items():
        if value.get(key) != expected:
            raise ValueError(f"review 配置 {key} 必须为 {expected}")
    review = value.get("review")
    if not isinstance(review, dict):
        raise ValueError("review 配置缺少 review")
    for key, expected in EXPECTED_REVIEW.items():
        if review.get(key) != expected:
            raise ValueError(f"review 配置 review.{key} 必须为 {expected}")
    return deepcopy(value)


def review_messages(
    task: dict[str, Any],
    candidate: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "task": task["task"],
        "primary_category": task["primary_category"],
        "risk_level": task["risk_level"],
        "conversations": task["conversations"],
        "evidence_ids": task["evidence_ids"],
        "evidence": evidence,
        "chosen": candidate["chosen"],
        "rejected": candidate["rejected"],
    }
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_review_content(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("审核输出不是合法 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != REVIEW_KEYS:
        raise ValueError("审核 JSON 键不匹配")
    if payload["decision"] not in {"accept", "revise"}:
        raise ValueError("审核 decision 无效")
    if payload["preferred"] not in {"chosen", "rejected", "unclear"}:
        raise ValueError("审核 preferred 无效")
    for key in ("issues", "risk_flags"):
        if not isinstance(payload[key], list) or any(
            not isinstance(item, str) for item in payload[key]
        ):
            raise ValueError(f"审核 {key} 必须是字符串列表")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("审核 reason 必须是非空字符串")
    return {
        "decision": payload["decision"],
        "preferred": payload["preferred"],
        "issues": [item.strip() for item in payload["issues"] if item.strip()],
        "risk_flags": [item.strip() for item in payload["risk_flags"] if item.strip()],
        "reason": payload["reason"].strip(),
    }
