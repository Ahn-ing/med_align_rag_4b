"""Phase 5 loss-only validation 子集与工程 fixture 规则。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_text

SYSTEM_PROMPT = "你是一个谨慎、清晰且遵循证据边界的助手。"
SELECTION_SEED = 20260914
TASK_QUOTAS = {
    "general": 2,
    "instruction_following": 2,
    "medical_qa": 4,
    "rag_faithfulness": 2,
    "structured_output": 2,
}
SAFETY_CATEGORIES = {
    "clarification", "home_observation", "immediate_emergency",
    "medication_boundary", "outpatient_followup", "uncertainty",
}


def _selection_key(record: dict[str, Any]) -> tuple[str, str]:
    return sha256_text(f"{SELECTION_SEED}:{record['id']}"), record["id"]


def select_phase5_loss_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按冻结配额和稳定哈希选出 18 条 loss-only cases。"""
    selected: list[dict[str, Any]] = []
    for task, quota in TASK_QUOTAS.items():
        candidates = sorted((row for row in records if row["task"] == task), key=_selection_key)
        if len(candidates) < quota:
            raise ValueError(f"validation 中 {task} 不足 {quota} 条")
        selected.extend(candidates[:quota])
    safety = [row for row in records if row["task"] == "medical_safety"]
    for category in sorted(SAFETY_CATEGORIES):
        candidates = sorted((row for row in safety if row["primary_category"] == category), key=_selection_key)
        if not candidates:
            raise ValueError(f"validation 缺少安全子类: {category}")
        selected.append(candidates[0])
    if len(selected) != 18 or len({row["id"] for row in selected}) != 18:
        raise ValueError("Phase 5 loss 子集必须是 18 条唯一记录")
    return sorted(selected, key=lambda row: row["id"])


def build_loss_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将冻结 eval case 转为与 SFT 一致的 messages，不改写正文。"""
    output = []
    for row in select_phase5_loss_cases(records):
        output.append({
            "id": row["id"],
            "source_split": "validation",
            "task": row["task"],
            "primary_category": row["primary_category"],
            "risk_level": row["risk_level"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["reference_answer"]},
            ],
        })
    return output


def distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in records).items()))


def load_fixtures(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("工程 fixtures 必须恰好包含 3 条")
    if len({row.get("id") for row in value}) != 3:
        raise ValueError("工程 fixture ID 必须唯一")
    return value


def validate_fixture_output(fixture: dict[str, Any], output: str) -> dict[str, Any]:
    """只执行生成链的机械验收，不判断医疗质量。"""
    text = output.strip()
    if not text:
        raise ValueError(f"{fixture['id']} 输出为空")
    lowered = text.lower()
    if "<think>" in lowered or "</think>" in lowered:
        raise ValueError(f"{fixture['id']} 出现 think block")
    if any(token in text for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError(f"{fixture['id']} 出现异常控制 token")
    user = next(row["content"] for row in fixture["messages"] if row["role"] == "user")
    if user in text:
        raise ValueError(f"{fixture['id']} 完整复读 user prompt")
    parsed_json = None
    if fixture["kind"] == "structured_json":
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{fixture['id']} 输出不是合法 JSON") from exc
        if set(parsed_json) != set(fixture["required_keys"]):
            raise ValueError(f"{fixture['id']} JSON keys 不匹配")
    return {"status": "passed", "characters": len(text), "json_valid": parsed_json is not None}
