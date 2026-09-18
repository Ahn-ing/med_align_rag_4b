"""Phase 7 prompt pool 的确定性选择、指纹和冻结契约。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from med_align_rag_4b.common.hashing import sha256_record, sha256_text
from med_align_rag_4b.sft.deduplication import question_fingerprint
from med_align_rag_4b.sft.evidence_routing import evidence_id_for_question

MEDICAL_SOURCE_COMPONENT = "api_evidence"
GENERAL_SOURCE_ID = "databricks_dolly_15k_bdd27f4"
STRUCTURED_ABILITIES = {
    "classification",
    "information_extraction",
    "summarization",
}
SAFETY_CATEGORIES = (
    "clarification",
    "home_observation",
    "immediate_emergency",
    "medication_boundary",
    "outpatient_followup",
    "uncertainty",
)
REQUIRED_PROMPT_FIELDS = {
    "format_version",
    "id",
    "prompt_group_id",
    "split",
    "task",
    "primary_category",
    "risk_level",
    "conversations",
    "source_id",
    "source_record_id",
    "source_revision",
    "source_sha256",
    "license_status",
    "evidence_ids",
    "origin",
    "data_component",
    "selection_reason",
    "contains_personal_data",
    "release_allowed",
    "source_answer_used",
    "prompt_fingerprint",
    "status",
}


def build_prompt_pool(
    train_rows: list[dict[str, Any]],
    evidence_sources: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    blocked_fingerprints: set[str],
    source_sha256: str,
) -> list[dict[str, Any]]:
    """按冻结配额选择互不重复的 prompt，不读取 assistant 回答。"""
    data_config = config["data"]
    system_prompt = data_config["system_prompt"]
    seen: set[str] = set()
    medical_candidates: list[dict[str, Any]] = []
    general_candidates: list[dict[str, Any]] = []

    for row in train_rows:
        candidate = _candidate(
            row,
            evidence_sources,
            blocked_fingerprints=blocked_fingerprints,
            system_prompt=system_prompt,
            source_sha256=source_sha256,
        )
        if candidate is None or candidate["prompt_fingerprint"] in seen:
            continue
        seen.add(candidate["prompt_fingerprint"])
        if row["domain"] == "medical":
            medical_candidates.append(candidate)
        elif row["domain"] == "general":
            general_candidates.append(candidate)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    safety = data_config["safety_quotas"]
    for category in SAFETY_CATEGORIES:
        candidates = [
            item
            for item in medical_candidates
            if item["safety_category"] == category
            and item["candidate_key"] not in selected_keys
        ]
        train_rows, validation_rows = select_stratum(
            candidates,
            train_count=safety["train"][category],
            validation_count=safety["preference_validation"][category],
            seed=data_config["split_seed"],
        )
        selected.extend(
            _records_for_stratum(
                train_rows,
                validation_rows,
                task="medical_safety",
                primary_category=category,
                selection_reason=f"medical_safety_{category}",
            )
        )
        selected_keys.update(
            item["candidate_key"] for item in train_rows + validation_rows
        )

    behavior = [
        item
        for item in medical_candidates
        if item["candidate_key"] not in selected_keys
    ]
    train_rows, validation_rows = select_stratum(
        behavior,
        train_count=data_config["task_quotas"]["train"]["medical_behavior"],
        validation_count=data_config["task_quotas"]["preference_validation"][
            "medical_behavior"
        ],
        seed=data_config["split_seed"],
    )
    selected.extend(
        _records_for_stratum(
            train_rows,
            validation_rows,
            task="medical_behavior",
            primary_category="medical_behavior",
            selection_reason="medical_behavior_remaining",
        )
    )
    selected_keys.update(item["candidate_key"] for item in train_rows + validation_rows)

    instruction = [
        item
        for item in general_candidates
        if item["ability"] in STRUCTURED_ABILITIES
        and item["candidate_key"] not in selected_keys
    ]
    train_rows, validation_rows = select_stratum(
        instruction,
        train_count=data_config["task_quotas"]["train"]["instruction_structured"],
        validation_count=data_config["task_quotas"]["preference_validation"][
            "instruction_structured"
        ],
        seed=data_config["split_seed"],
    )
    selected.extend(
        _records_for_stratum(
            train_rows,
            validation_rows,
            task="instruction_structured",
            primary_category="instruction_structured",
            selection_reason="structured_instruction_ability",
        )
    )
    selected_keys.update(item["candidate_key"] for item in train_rows + validation_rows)

    general = [
        item
        for item in general_candidates
        if item["candidate_key"] not in selected_keys
    ]
    train_rows, validation_rows = select_stratum(
        general,
        train_count=data_config["task_quotas"]["train"]["general"],
        validation_count=data_config["task_quotas"]["preference_validation"]["general"],
        seed=data_config["split_seed"],
    )
    selected.extend(
        _records_for_stratum(
            train_rows,
            validation_rows,
            task="general",
            primary_category="general",
            selection_reason="general_dolly_training",
        )
    )

    selected.sort(key=lambda item: item["id"])
    validate_prompt_pool(selected, config, blocked_fingerprints=blocked_fingerprints)
    return selected


def classify_safety_category(row: dict[str, Any]) -> str | None:
    """用固定关键词和原类别路由医疗 prompt，不判断医学正确性。"""
    text = _last_user_content(row["messages"])
    if row.get("primary_category") == "emergency_escalation" or _contains_any(
        text,
        (
            "急救",
            "立即就医",
            "马上就医",
            "立刻就医",
            "胸痛",
            "呼吸困难",
            "卒中",
            "中风",
            "昏迷",
            "抽搐",
            "大出血",
            "休克",
            "中毒",
            "误服",
            "过量",
            "自杀",
            "严重过敏",
            "过敏性休克",
        ),
    ):
        return "immediate_emergency"
    if row.get("primary_category") == "medication_boundary" or _contains_any(
        text,
        (
            "用药",
            "吃药",
            "服药",
            "药物",
            "剂量",
            "停药",
            "抗生素",
            "消炎药",
            "非处方药",
            "OTC",
            "止痛药",
            "疫苗",
        ),
    ):
        return "medication_boundary"
    if _contains_any(
        text,
        (
            "在家",
            "家里",
            "居家",
            "家庭护理",
            "家庭观察",
            "观察多久",
            "先观察",
            "自行观察",
            "在家怎么处理",
            "平时",
            "日常",
            "生活方式",
            "自我管理",
            "护理",
            "饮食",
            "运动",
            "休息",
            "睡眠",
            "监测",
        ),
    ):
        return "home_observation"
    if _contains_any(
        text,
        (
            "复诊",
            "随访",
            "复查",
            "门诊",
            "挂什么科",
            "哪个科",
            "后续检查",
            "检查报告",
            "结果异常",
            "多久去医院",
            "什么时候去医院",
            "后续",
            "下一步",
            "治疗",
            "管理",
            "康复",
            "就诊",
            "医生",
        ),
    ):
        return "outpatient_followup"
    if row.get("primary_category") == "symptom_risk_clarification" or _contains_any(
        text,
        (
            "哪些信息",
            "需要了解",
            "需要提供",
            "怎么判断",
            "如何判断",
            "是否严重",
            "严不严重",
            "是什么原因",
            "症状",
            "征兆",
            "表现",
            "危险信号",
        ),
    ):
        return "clarification"
    if _contains_any(
        text,
        (
            "不确定",
            "无法确定",
            "无法判断",
            "不一定",
            "是否一定",
            "一定会",
            "会不会",
            "可能是",
            "能否确诊",
            "确诊",
            "排除",
            "怀疑",
            "可能",
        ),
    ):
        return "uncertainty"
    return None


def select_stratum(
    candidates: list[dict[str, Any]],
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按稳定哈希选择并切分一个任务/安全子类。"""
    total = train_count + validation_count
    if total <= 0:
        raise ValueError("train_count + validation_count 必须为正数")
    if len(candidates) < total:
        raise ValueError(f"prompt 候选不足: 需要 {total}，实际 {len(candidates)}")
    ordered = sorted(
        candidates,
        key=lambda item: (
            sha256_text(f"{seed}:{item['prompt_fingerprint']}"),
            item["prompt_fingerprint"],
        ),
    )[:total]
    return ordered[:train_count], ordered[train_count:]


def validate_prompt_pool(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    blocked_fingerprints: set[str] | None = None,
) -> dict[str, Any]:
    """校验 prompt pool 的结构、配额、唯一性和泄漏边界。"""
    rows = list(records)
    if not rows:
        raise ValueError("prompt pool 不能为空")
    data_config = config["data"]
    blocked = blocked_fingerprints or set()
    split_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "preference_validation": Counter(),
    }
    safety_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "preference_validation": Counter(),
    }
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    group_splits: dict[str, str] = {}

    for record in rows:
        missing = sorted(REQUIRED_PROMPT_FIELDS - record.keys())
        if missing:
            raise ValueError(f"prompt pool 记录缺少字段: {', '.join(missing)}")
        if record["format_version"] != 1:
            raise ValueError("prompt pool format_version 必须为 1")
        if record["status"] != "prompt_frozen":
            raise ValueError("prompt pool status 必须为 prompt_frozen")
        if record["split"] not in {"train", "preference_validation"}:
            raise ValueError("prompt pool split 无效")
        if record["task"] not in {
            "medical_behavior",
            "medical_safety",
            "general",
            "instruction_structured",
        }:
            raise ValueError("prompt pool task 无效")
        if record["source_answer_used"] is not False:
            raise ValueError("prompt pool 不得复用 assistant 答案")
        if record["contains_personal_data"] is not False:
            raise ValueError("prompt pool contains_personal_data 必须为 false")
        if record["release_allowed"] is not False:
            raise ValueError("prompt pool release_allowed 必须为 false")
        if record["id"] in seen_ids:
            raise ValueError(f"prompt pool ID 重复: {record['id']}")
        seen_ids.add(record["id"])
        fingerprint = record["prompt_fingerprint"]
        if fingerprint != question_fingerprint(
            _last_user_content(record["conversations"])
        ):
            raise ValueError(f"prompt fingerprint 不匹配: {record['id']}")
        if fingerprint in seen_prompts:
            raise ValueError(f"prompt pool prompt 重复: {record['id']}")
        seen_prompts.add(fingerprint)
        if fingerprint in blocked:
            raise ValueError(f"prompt 与冻结评测集重叠: {record['id']}")
        group = record["prompt_group_id"]
        split = record["split"]
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(f"prompt_group_id 跨 split: {group}")
        split_counts[split] += 1
        task_counts[split][record["task"]] += 1
        if record["task"] == "medical_safety":
            safety_counts[split][record["primary_category"]] += 1

    for split in ("train", "preference_validation"):
        if split_counts[split] != data_config["counts"][split]:
            raise ValueError(f"{split} prompt 数不匹配")
        for task, expected in data_config["task_quotas"][split].items():
            if task_counts[split][task] != expected:
                raise ValueError(f"{split} 的 {task} 配额不匹配")
        for category, expected in data_config["safety_quotas"][split].items():
            if safety_counts[split][category] != expected:
                raise ValueError(f"{split} 的 {category} 配额不匹配")

    return {
        "records": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "tasks": {
            split: dict(sorted(task_counts[split].items())) for split in task_counts
        },
        "safety": {
            split: dict(sorted(safety_counts[split].items())) for split in safety_counts
        },
        "blocked_fingerprints": len(blocked),
    }


def _candidate(
    row: dict[str, Any],
    evidence_sources: list[dict[str, Any]],
    *,
    blocked_fingerprints: set[str],
    system_prompt: str,
    source_sha256: str,
) -> dict[str, Any] | None:
    if row.get("license_status") != "approved":
        return None

    domain = row.get("domain")
    review_status = (row.get("review") or {}).get("status")
    allowed_review_statuses = {
        "medical": {"approved", "agent_evidence_reviewed"},
        "general": {"approved", "agent_quality_reviewed"},
    }
    if review_status not in allowed_review_statuses.get(str(domain), set()):
        return None
    if domain == "medical":
        if row.get("data_component") != MEDICAL_SOURCE_COMPONENT:
            return None
        evidence_ids = list(row.get("medical_evidence") or [])
        if not evidence_ids:
            routed = evidence_id_for_question(
                _last_user_content(row["messages"]), evidence_sources
            )
            if routed is None:
                return None
            evidence_ids = [routed]
        safety_category = classify_safety_category(row)
    elif domain == "general":
        if row.get("source_id") != GENERAL_SOURCE_ID:
            return None
        evidence_ids = []
        safety_category = None
    else:
        return None

    messages = _prompt_messages(row["messages"])
    if messages[0]["content"] != system_prompt:
        return None
    prompt = messages[-1]["content"]
    fingerprint = question_fingerprint(prompt)
    if fingerprint in blocked_fingerprints:
        return None
    candidate_key = sha256_record(
        [
            row["source_id"],
            row.get("source_record_id"),
            fingerprint,
        ]
    )
    return {
        "candidate_key": candidate_key,
        "prompt_fingerprint": fingerprint,
        "conversations": messages,
        "source_id": row["source_id"],
        "source_record_id": row.get("source_record_id"),
        "source_revision": row.get("source_revision"),
        "source_sha256": source_sha256,
        "license_status": row["license_status"],
        "evidence_ids": evidence_ids,
        "risk_level": row.get("risk_level") if domain == "medical" else "low",
        "ability": row.get("ability"),
        "data_component": row["data_component"],
        "safety_category": safety_category,
    }


def _records_for_stratum(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    task: str,
    primary_category: str,
    selection_reason: str,
) -> list[dict[str, Any]]:
    records = []
    for split, rows in (
        ("train", train_rows),
        ("preference_validation", validation_rows),
    ):
        for candidate in rows:
            records.append(
                _prompt_record(
                    candidate,
                    split=split,
                    task=task,
                    primary_category=primary_category,
                    selection_reason=selection_reason,
                )
            )
    return records


def _prompt_record(
    candidate: dict[str, Any],
    *,
    split: str,
    task: str,
    primary_category: str,
    selection_reason: str,
) -> dict[str, Any]:
    group_id = (
        "dpo_pg_"
        + sha256_record(
            [
                candidate["source_id"],
                candidate["source_record_id"],
                candidate["prompt_fingerprint"],
            ]
        )[:24]
    )
    record_id = (
        "dpo_prompt_" + sha256_record([split, task, primary_category, group_id])[:24]
    )
    return {
        "format_version": 1,
        "id": record_id,
        "prompt_group_id": group_id,
        "split": split,
        "task": task,
        "primary_category": primary_category,
        "risk_level": candidate["risk_level"],
        "conversations": deepcopy(candidate["conversations"]),
        "source_id": candidate["source_id"],
        "source_record_id": candidate["source_record_id"],
        "source_revision": candidate["source_revision"],
        "source_sha256": candidate["source_sha256"],
        "license_status": candidate["license_status"],
        "evidence_ids": list(candidate["evidence_ids"]),
        "origin": "phase6_released_train_prompt_reuse",
        "data_component": candidate["data_component"],
        "selection_reason": selection_reason,
        "contains_personal_data": False,
        "release_allowed": False,
        "source_answer_used": False,
        "prompt_fingerprint": candidate["prompt_fingerprint"],
        "status": "prompt_frozen",
    }


def _prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    last_user = max(
        index for index, message in enumerate(messages) if message["role"] == "user"
    )
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages[: last_user + 1]
        if message["role"] in {"system", "user", "assistant"}
    ]


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    users = [message["content"] for message in messages if message["role"] == "user"]
    if not users:
        raise ValueError("messages 缺少 user")
    return users[-1]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(keyword.casefold() in folded for keyword in keywords)
