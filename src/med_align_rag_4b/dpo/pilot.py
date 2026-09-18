"""Phase 7 API pilot 的确定性分层选择。"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any

from med_align_rag_4b.common.hashing import sha256_text
from med_align_rag_4b.dpo.data import latest_user_content
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


def select_pilot_prompts(
    prompt_records: list[dict[str, Any]], api_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """按 API 配置中的冻结配额选择 pilot prompt。"""
    pilot = api_config["pilot"]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for split in SPLITS:
        for task in TASKS:
            candidates = [
                row
                for row in prompt_records
                if row["split"] == split
                and row["task"] == task
                and row["id"] not in selected_ids
            ]
            expected = pilot["task_quotas"][split][task]
            if task == "medical_safety":
                for category in SAFETY_CATEGORIES:
                    count = pilot["safety_quotas"][split][category]
                    if count == 0:
                        continue
                    category_candidates = [
                        row for row in candidates if row["primary_category"] == category
                    ]
                    chosen = _select(category_candidates, count, seed=pilot["seed"])
                    selected.extend(
                        _as_pilot_record(row, selected_rank=len(selected))
                        for row in chosen
                    )
                    selected_ids.update(row["id"] for row in chosen)
            else:
                chosen = _select(candidates, expected, seed=pilot["seed"])
                selected.extend(
                    _as_pilot_record(row, selected_rank=len(selected)) for row in chosen
                )
                selected_ids.update(row["id"] for row in chosen)
    summary = validate_pilot_prompts(selected, api_config)
    if summary["records"] != pilot["records"]:
        raise ValueError("pilot 记录数不匹配")
    return sorted(selected, key=lambda row: row["pilot_rank"])


def validate_pilot_prompts(
    records: list[dict[str, Any]], api_config: dict[str, Any]
) -> dict[str, Any]:
    """校验 pilot 数量、配额和唯一性。"""
    rows = list(records)
    pilot = api_config["pilot"]
    split_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    safety_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    ids: set[str] = set()
    groups: set[str] = set()
    for expected_rank, row in enumerate(
        sorted(rows, key=lambda item: item["pilot_rank"])
    ):
        if row["pilot_rank"] != expected_rank:
            raise ValueError("pilot_rank 必须从 0 连续编号")
        if row["id"] in ids or row["prompt_group_id"] in groups:
            raise ValueError("pilot prompt ID 或 group 重复")
        ids.add(row["id"])
        groups.add(row["prompt_group_id"])
        split = row["split"]
        split_counts[split] += 1
        task_counts[split][row["task"]] += 1
        if row["task"] == "medical_safety":
            safety_counts[split][row["primary_category"]] += 1
    for split in SPLITS:
        expected_split = sum(pilot["task_quotas"][split].values())
        if split_counts[split] != expected_split:
            raise ValueError(
                f"pilot {split} 记录数必须为 {expected_split}，实际为 {split_counts[split]}"
            )
    for split in SPLITS:
        for task, expected in pilot["task_quotas"][split].items():
            if task_counts[split][task] != expected:
                raise ValueError(f"pilot {split}/{task} 配额不匹配")
        for category, expected in pilot["safety_quotas"][split].items():
            if safety_counts[split][category] != expected:
                raise ValueError(f"pilot {split}/{category} 配额不匹配")
    return {
        "records": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "tasks": {split: dict(sorted(task_counts[split].items())) for split in SPLITS},
        "safety": {
            split: dict(sorted(safety_counts[split].items())) for split in SPLITS
        },
    }


def _select(
    candidates: list[dict[str, Any]], count: int, *, seed: int
) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if len(candidates) < count:
        raise ValueError(f"pilot 候选不足: 需要 {count}，实际 {len(candidates)}")
    return sorted(
        candidates,
        key=lambda row: (sha256_text(f"pilot:{seed}:{row['id']}"), row["id"]),
    )[:count]


def _as_pilot_record(row: dict[str, Any], *, selected_rank: int) -> dict[str, Any]:
    result = deepcopy(row)
    result["pilot_rank"] = selected_rank
    result["api_status"] = "pending"
    return result


CONTROL_MARKERS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)bearer\\s+[A-Za-z0-9._-]{12,}"),
)


def validate_generated_pilot(
    tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    api_config: dict[str, Any],
    *,
    blocked_fingerprints: set[str],
) -> dict[str, Any]:
    """校验 API pilot 的 task 对齐、pair 结构和冻结配额。"""
    errors: list[str] = []
    task_by_id = {row["id"]: row for row in tasks}
    seen_candidate_ids: set[str] = set()
    candidate_counts: Counter[tuple[str, str]] = Counter()
    safety_counts: Counter[tuple[str, str]] = Counter()
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []

    for candidate in candidates:
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or candidate_id in seen_candidate_ids:
            errors.append(f"candidate_id_invalid_or_duplicate:{candidate_id}")
            continue
        seen_candidate_ids.add(candidate_id)
        expected_task_id = candidate_id.replace("dpo_pair_", "dpo_prompt_", 1)
        task = task_by_id.get(expected_task_id)
        if task is None:
            errors.append(f"candidate_without_task:{candidate_id}")
            continue
        errors.extend(_candidate_errors(candidate, task))
        if (
            question_fingerprint(latest_user_content(candidate["conversations"]))
            in blocked_fingerprints
        ):
            errors.append(f"blocked_fingerprint:{candidate_id}")
        candidate_counts[(candidate["split"], candidate["task"])] += 1
        if candidate["task"] == "medical_safety":
            safety_counts[(candidate["split"], candidate["primary_category"])] += 1
        chosen_lengths.append(len(candidate["chosen"]))
        rejected_lengths.append(len(candidate["rejected"]))

    for item in quarantine:
        if not isinstance(item.get("reasons"), list) or not item["reasons"]:
            errors.append(f"quarantine_without_reason:{item.get('task_id')}")

    expected_counts = Counter((task["split"], task["task"]) for task in tasks)
    expected_safety = Counter(
        (task["split"], task["primary_category"])
        for task in tasks
        if task["task"] == "medical_safety"
    )
    quota_complete = (
        candidate_counts == expected_counts and safety_counts == expected_safety
    )
    if not quota_complete:
        errors.append("candidate_quotas_incomplete")
    if len(candidates) + len(quarantine) != len(tasks):
        errors.append("candidate_and_quarantine_count_do_not_match_tasks")
    if any(
        pattern.search(str(candidate))
        for candidate in candidates
        for pattern in SECRET_PATTERNS
    ):
        errors.append("secret_like_text_in_candidates")

    return {
        "passed": not errors,
        "errors": errors,
        "tasks": len(tasks),
        "candidates": len(candidates),
        "quarantined": len(quarantine),
        "quota_complete": quota_complete,
        "candidate_counts": {
            f"{split}/{task}": candidate_counts[(split, task)]
            for split in SPLITS
            for task in TASKS
        },
        "safety_counts": {
            f"{split}/{category}": safety_counts[(split, category)]
            for split in SPLITS
            for category in SAFETY_CATEGORIES
        },
        "chosen_char_length": _length_summary(chosen_lengths),
        "rejected_char_length": _length_summary(rejected_lengths),
        "blocked_fingerprints": len(blocked_fingerprints),
    }


def _candidate_errors(candidate: dict[str, Any], task: dict[str, Any]) -> list[str]:
    errors = []
    for key in (
        "prompt_group_id",
        "split",
        "source_id",
        "source_record_id",
        "source_revision",
        "source_sha256",
        "conversations",
        "task",
        "primary_category",
        "risk_level",
        "evidence_ids",
    ):
        if candidate.get(key) != task.get(key):
            errors.append(f"task_mismatch:{candidate['id']}:{key}")
    if candidate.get("review_status") != "pending_agent_review":
        errors.append(f"review_status_invalid:{candidate['id']}")
    if candidate.get("release_allowed") is not False:
        errors.append(f"release_allowed_invalid:{candidate['id']}")
    if candidate.get("contains_personal_data") is not False:
        errors.append(f"personal_data_flag_invalid:{candidate['id']}")
    for key in ("chosen", "rejected"):
        value = candidate.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"answer_empty:{candidate['id']}:{key}")
        elif any(marker in value for marker in CONTROL_MARKERS):
            errors.append(f"control_marker:{candidate['id']}:{key}")
    if normalize_text(candidate["chosen"]) == normalize_text(candidate["rejected"]):
        errors.append(f"answers_not_distinct:{candidate['id']}")
    used = set(candidate.get("evidence_ids_used", []))
    if not used.issubset(set(task["evidence_ids"])):
        errors.append(f"evidence_subset_invalid:{candidate['id']}")
    if task["task"].startswith("medical") and not used:
        errors.append(f"medical_evidence_missing:{candidate['id']}")
    if candidate.get("judge", {}).get("accepted") is not True:
        errors.append(f"judge_not_accepted:{candidate['id']}")
    if candidate.get("judge", {}).get("preferred") != "chosen":
        errors.append(f"judge_preference_invalid:{candidate['id']}")
    return errors


def _length_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }
