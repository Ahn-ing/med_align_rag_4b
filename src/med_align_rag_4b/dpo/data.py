"""Phase 7 偏好数据 schema、配额、泄漏和 TRL 转换契约。"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.sft.deduplication import normalize_text, question_fingerprint

SPLITS = ("train", "preference_validation")
TASKS = (
    "medical_behavior",
    "medical_safety",
    "general",
    "instruction_structured",
)
SAFETY_CATEGORIES = (
    "clarification",
    "home_observation",
    "immediate_emergency",
    "medication_boundary",
    "outpatient_followup",
    "uncertainty",
)
REVIEW_STATUSES = {"pending_agent_review", "agent_evidence_reviewed"}
INTERNAL_REVIEWER_ROLE = "codex_research_reviewer"
RISK_LEVELS = {"low", "medium", "high"}
REQUIRED_FIELDS = {
    "id",
    "prompt_group_id",
    "split",
    "source_id",
    "source_sha256",
    "conversations",
    "chosen",
    "rejected",
    "task",
    "primary_category",
    "risk_level",
    "evidence_ids",
    "review_status",
    "reviewer_role",
    "reviewed_at_utc",
    "evidence_checked",
    "prompt_tokens",
    "chosen_tokens",
    "rejected_tokens",
    "pair_reason",
    "chosen_source",
    "rejected_source",
    "contains_personal_data",
    "release_allowed",
}


def validate_preference_record(
    record: dict[str, Any],
    *,
    system_prompt: str,
    max_length: int,
    require_reviewed: bool = False,
) -> None:
    """校验一条内部工程偏好记录，不判断医学内容质量。"""
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError(f"偏好记录缺少字段: {', '.join(missing)}")
    for field in ("id", "prompt_group_id", "source_id"):
        _require_text(record[field], field)
    _require_sha256(record["source_sha256"], "source_sha256")
    if record["split"] not in SPLITS:
        raise ValueError("偏好记录 split 无效")
    if record["task"] not in TASKS:
        raise ValueError("偏好记录 task 无效")
    if record["risk_level"] not in RISK_LEVELS:
        raise ValueError("偏好记录 risk_level 无效")
    if record["contains_personal_data"] is not False:
        raise ValueError("偏好记录 contains_personal_data 必须为 false")
    if record["release_allowed"] is not False:
        raise ValueError("偏好记录 release_allowed 必须为 false")

    _validate_conversations(record["conversations"], system_prompt)
    chosen = _require_text(record["chosen"], "chosen")
    rejected = _require_text(record["rejected"], "rejected")
    if normalize_text(chosen) == normalize_text(rejected):
        raise ValueError("chosen 与 rejected 必须在归一化后不同")

    evidence_ids = record["evidence_ids"]
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in evidence_ids
    ):
        raise ValueError("evidence_ids 必须是非空字符串列表")
    if record["task"] in {"medical_behavior", "medical_safety"} and not evidence_ids:
        raise ValueError("医疗偏好记录必须包含 evidence_ids")
    if (
        record["task"] == "medical_safety"
        and record["primary_category"] not in SAFETY_CATEGORIES
    ):
        raise ValueError("医疗安全记录的 primary_category 无效")

    review_status = record["review_status"]
    if review_status not in REVIEW_STATUSES:
        raise ValueError("review_status 无效")
    if review_status == "agent_evidence_reviewed":
        if record["reviewer_role"] != INTERNAL_REVIEWER_ROLE:
            raise ValueError("代理证据审核角色必须为 codex_research_reviewer")
        _require_text(record["reviewed_at_utc"], "reviewed_at_utc")
        if record["evidence_checked"] is not True:
            raise ValueError("代理证据审核必须确认 evidence_checked=true")
    elif (
        any(
            value is not None
            for value in (record["reviewer_role"], record["reviewed_at_utc"])
        )
        or record["evidence_checked"] is not False
    ):
        raise ValueError("待审核记录的审核字段必须为空且 evidence_checked=false")
    if require_reviewed and review_status != "agent_evidence_reviewed":
        raise ValueError("偏好记录尚未完成代理证据审核")

    for field in ("prompt_tokens", "chosen_tokens", "rejected_tokens"):
        value = record[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} 必须为正整数")
    if record["prompt_tokens"] + record["chosen_tokens"] > max_length:
        raise ValueError("prompt + chosen 超过 max_length")
    if record["prompt_tokens"] + record["rejected_tokens"] > max_length:
        raise ValueError("prompt + rejected 超过 max_length")

    for field in ("pair_reason", "chosen_source", "rejected_source"):
        _require_text(record[field], field)


def validate_preference_dataset(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    config: dict[str, Any],
    *,
    blocked_fingerprints: set[str] | None = None,
    enforce_quotas: bool = True,
    require_reviewed: bool = False,
) -> dict[str, Any]:
    """校验整个偏好数据集，并返回不包含文本的分布摘要。"""
    rows = list(records)
    if not rows:
        raise ValueError("偏好数据集不能为空")
    data_config = config["data"]
    blocked = blocked_fingerprints or set()
    if any(not _is_sha256(item) for item in blocked):
        raise ValueError("blocked_fingerprints 必须是小写 SHA256")

    split_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    safety_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    prompt_group_splits: dict[str, str] = {}

    for record in rows:
        validate_preference_record(
            record,
            system_prompt=data_config["system_prompt"],
            max_length=data_config["max_length"],
            require_reviewed=require_reviewed,
        )
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"偏好记录 ID 重复: {record_id}")
        seen_ids.add(record_id)

        prompt = latest_user_content(record["conversations"])
        fingerprint = question_fingerprint(prompt)
        if fingerprint in seen_prompts:
            raise ValueError(f"偏好 prompt 重复: {record_id}")
        seen_prompts.add(fingerprint)
        if fingerprint in blocked:
            raise ValueError(f"偏好 prompt 与冻结评测集重叠: {record_id}")

        split = record["split"]
        group = record["prompt_group_id"]
        previous_split = prompt_group_splits.setdefault(group, split)
        if previous_split != split:
            raise ValueError(f"prompt_group_id 跨 split: {group}")
        split_counts[split] += 1
        task_counts[split][record["task"]] += 1
        if record["task"] == "medical_safety":
            safety_counts[split][record["primary_category"]] += 1

    if enforce_quotas:
        expected_counts = data_config["counts"]
        for split in SPLITS:
            if split_counts[split] != expected_counts[split]:
                raise ValueError(f"{split} 记录数必须为 {expected_counts[split]}")
            for task, expected in data_config["task_quotas"][split].items():
                if task_counts[split][task] != expected:
                    raise ValueError(f"{split} 的 {task} 必须为 {expected} 条")
            for category, expected in data_config["safety_quotas"][split].items():
                if safety_counts[split][category] != expected:
                    raise ValueError(f"{split} 的 {category} 必须为 {expected} 条")

    return {
        "records": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "tasks": {split: dict(sorted(task_counts[split].items())) for split in SPLITS},
        "safety": {
            split: dict(sorted(safety_counts[split].items())) for split in SPLITS
        },
        "fingerprint_algorithm": "sha256-normalized-question-v1",
        "blocked_fingerprints": len(blocked),
        "review_required": require_reviewed,
    }


def load_preference_dataset(
    path: str | Path,
    config: dict[str, Any],
    *,
    blocked_fingerprints: set[str] | None = None,
    enforce_quotas: bool = True,
    require_reviewed: bool = False,
) -> dict[str, Any]:
    """从 JSONL 加载并校验偏好数据。"""
    rows = list(iter_jsonl(path))
    return validate_preference_dataset(
        rows,
        config,
        blocked_fingerprints=blocked_fingerprints,
        enforce_quotas=enforce_quotas,
        require_reviewed=require_reviewed,
    )


def to_trl_record(record: dict[str, Any]) -> dict[str, Any]:
    """转换为 TRL DPO 使用的 conversational 字段，不修改原始记录。"""
    return {
        "prompt": deepcopy(record["conversations"]),
        "chosen": [{"role": "assistant", "content": record["chosen"]}],
        "rejected": [{"role": "assistant", "content": record["rejected"]}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def latest_user_content(conversations: list[dict[str, str]]) -> str:
    """返回最后一个 user turn，作为与冻结评测集比较的问题。"""
    user_turns = [
        message["content"] for message in conversations if message["role"] == "user"
    ]
    if not user_turns:
        raise ValueError("conversations 缺少 user turn")
    return user_turns[-1]


def load_fingerprint_file(path: str | Path) -> set[str]:
    """读取不含原文的冻结问题指纹文件。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("指纹文件必须是 JSON 对象")
    if value.get("algorithm") != "sha256-normalized-question-v1":
        raise ValueError("指纹文件 algorithm 不匹配")
    fingerprints = value.get("fingerprints")
    if not isinstance(fingerprints, list) or any(
        not _is_sha256(item) for item in fingerprints
    ):
        raise ValueError("指纹文件 fingerprints 必须是小写 SHA256 列表")
    return set(fingerprints)


def load_blocked_fingerprints(paths: list[str | Path]) -> set[str]:
    """合并多个冻结指纹文件。"""
    fingerprints: set[str] = set()
    for path in paths:
        fingerprints.update(load_fingerprint_file(path))
    return fingerprints


def _validate_conversations(conversations: Any, system_prompt: str) -> None:
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("conversations 至少需要 system 和 user")
    for message in conversations:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("conversations 中每条消息只能包含 role/content")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("conversations role 无效")
        _require_text(message["content"], "conversations.content")
    if (
        conversations[0]["role"] != "system"
        or conversations[0]["content"] != system_prompt
    ):
        raise ValueError("conversations 首条 system prompt 不匹配")
    if sum(message["role"] == "system" for message in conversations) != 1:
        raise ValueError("conversations 只能包含一个 system message")
    if conversations[-1]["role"] != "user":
        raise ValueError("conversations 最后一条必须是 user")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_sha256(value: Any, field: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field} 必须是小写 SHA256")
