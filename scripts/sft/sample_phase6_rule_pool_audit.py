"""为 Phase 6 v2 rule pool 生成确定性的分层质量审计样本。"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl

SEED = "20260915-phase-a-rule-pool-audit"
CATEGORY_QUOTAS = {
    "knowledge_explanation": 50,
    "symptom_risk_clarification": 50,
    "medication_boundary": 50,
    "action_followup": 50,
    "summary_synthesis": 50,
    "general": 50,
}
TOKEN_BIN_ORDER = ("24-63", "64-127", "128-255", "256+")


def token_bin(tokens: int) -> str:
    """把 assistant token 数映射为审计用固定区间。"""
    if tokens < 64:
        return "24-63"
    if tokens < 128:
        return "64-127"
    if tokens < 256:
        return "128-255"
    return "256+"


def source_half_bounds(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """按来源记录编号范围定义前/后半段，而不是假设编号完全连续。"""
    indexes: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        indexes[row["source_id"]].append(int(row["source_record_id"]))
    bounds = {}
    for source_id, values in indexes.items():
        low, high = min(values), max(values)
        bounds[source_id] = (low, low + (high - low) // 2)
    return bounds


def stratify_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """按类别、来源半段和 token 区间建立主分层。"""
    bounds = source_half_bounds(rows)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        midpoint = bounds[row["source_id"]][1]
        half = "first" if int(row["source_record_id"]) <= midpoint else "second"
        key = (half, token_bin(int(row["assistant_supervised_tokens"])))
        strata[(row["primary_category"],) + key].append(row)
    return strata


def allocate_quotas(
    populations: dict[tuple[str, str], int], target: int
) -> dict[tuple[str, str], int]:
    """按层规模分配配额，同时给每个非空层至少 3 条。"""
    nonempty = {key: count for key, count in populations.items() if count}
    if not nonempty:
        raise ValueError("没有可用于抽样的分层")
    minimum = min(3, target // len(nonempty)) if target >= len(nonempty) else 0
    quotas = {key: min(minimum, count) for key, count in nonempty.items()}
    remaining = target - sum(quotas.values())
    if remaining < 0:
        raise ValueError("最低配额超过目标数量")
    total_population = sum(nonempty.values())
    exact = {key: remaining * count / total_population for key, count in nonempty.items()}
    for key, value in exact.items():
        quota = min(math.floor(value), nonempty[key] - quotas[key])
        quotas[key] += quota
    remaining = target - sum(quotas.values())
    remainders = sorted(
        nonempty,
        key=lambda key: (
            exact[key] - math.floor(exact[key]),
            sha256_text(f"{SEED}:quota:{key}"),
        ),
        reverse=True,
    )
    for key in remainders:
        if remaining == 0:
            break
        if quotas[key] < nonempty[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("分层容量不足以满足配额")
    return quotas


def select_from_stratum(rows: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    """优先覆盖不同问答骨架，再按固定哈希顺序补足配额。"""
    ordered = sorted(rows, key=lambda row: sha256_text(f"{SEED}:{row['id']}"))
    selected: list[dict[str, Any]] = []
    used_question_buckets: set[str] = set()
    used_answer_buckets: set[str] = set()
    for row in ordered:
        q_bucket = row["fingerprints"]["question_skeleton"][:2]
        a_bucket = row["fingerprints"]["answer_skeleton"][:2]
        if q_bucket in used_question_buckets and a_bucket in used_answer_buckets:
            continue
        selected.append(row)
        used_question_buckets.add(q_bucket)
        used_answer_buckets.add(a_bucket)
        if len(selected) == quota:
            return selected
    selected_ids = {row["id"] for row in selected}
    for row in ordered:
        if row["id"] in selected_ids:
            continue
        selected.append(row)
        if len(selected) == quota:
            break
    return selected


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """原子写入审计 manifest，拒绝覆盖已有文件。"""
    if path.exists():
        raise FileExistsError(f"输出已存在: {path}")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.pool))
    expected_count = sum(CATEGORY_QUOTAS.values())
    pool_manifest_path = args.pool.with_name("manifest.json")
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    expected_pool = pool_manifest["rule_screened_pool"]
    pool_sha256 = sha256_file(args.pool)
    if len(rows) != expected_pool["records"]:
        raise ValueError(
            f"rule pool 记录数应为 {expected_pool['records']}，实际为 {len(rows)}"
        )
    if pool_sha256 != expected_pool["sha256"]:
        raise ValueError("rule pool SHA256 与 manifest 不一致")
    observed_categories = Counter(row["primary_category"] for row in rows)
    if set(observed_categories) != set(CATEGORY_QUOTAS):
        raise ValueError(f"类别集合不符合预期: {sorted(observed_categories)}")
    for row in rows:
        if not all(key in row for key in ("id", "messages", "fingerprints")):
            raise ValueError(f"记录缺少必要字段: {row.get('id')}")
        messages = {message["role"]: message["content"] for message in row["messages"]}
        if set(messages) != {"user", "assistant"}:
            raise ValueError(f"记录消息角色异常: {row['id']}")

    strata = stratify_rows(rows)
    sample: list[dict[str, Any]] = []
    allocation_report: dict[str, Any] = {}
    for category, target in CATEGORY_QUOTAS.items():
        category_strata = {
            key[1:]: value for key, value in strata.items() if key[0] == category
        }
        quotas = allocate_quotas(
            {key: len(value) for key, value in category_strata.items()}, target
        )
        category_selected: list[dict[str, Any]] = []
        cell_report = []
        for key in sorted(category_strata, key=lambda item: (item[0], TOKEN_BIN_ORDER.index(item[1]))):
            selected = select_from_stratum(category_strata[key], quotas[key])
            for row in selected:
                category_selected.append(
                    {
                        "format_version": 1,
                        "audit_stratum": {
                            "primary_category": category,
                            "risk_level": row["risk_level"],
                            "source_half": key[0],
                            "token_bin": key[1],
                            "question_skeleton_bucket": row["fingerprints"]["question_skeleton"][:2],
                            "answer_skeleton_bucket": row["fingerprints"]["answer_skeleton"][:2],
                        },
                        "record": row,
                    }
                )
            cell_report.append(
                {
                    "source_half": key[0],
                    "token_bin": key[1],
                    "population": len(category_strata[key]),
                    "quota": quotas[key],
                }
            )
        if len(category_selected) != target:
            raise AssertionError(f"{category} 抽样数量错误: {len(category_selected)}")
        sample.extend(category_selected)
        allocation_report[category] = {
            "population": observed_categories[category],
            "sample": target,
            "cells": cell_report,
        }

    sample.sort(key=lambda item: (item["audit_stratum"]["primary_category"], item["record"]["id"]))
    if len(sample) != expected_count or len({item["record"]["id"] for item in sample}) != expected_count:
        raise AssertionError("样本数量或唯一性检查失败")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for item in sample:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "format_version": 1,
        "status": "sample_frozen_pending_manual_review",
        "training_allowed": False,
        "release_allowed": False,
        "seed": SEED,
        "method": "category_quota_then_source_half_and_token_bin_quota_then_skeleton_bucket_diversity_then_hash",
        "pool": {
            "path": args.pool.as_posix(),
            "sha256": pool_sha256,
            "records": len(rows),
            "manifest_path": pool_manifest_path.as_posix(),
            "manifest_sha256": sha256_file(pool_manifest_path),
        },
        "sample": {
            "path": args.output.as_posix(),
            "sha256": sha256_file(args.output),
            "records": len(sample),
        },
        "allocation": allocation_report,
        "review_taxonomy": [
            "garbled_or_concatenated",
            "question_answer_mismatch",
            "unsupported_diagnosis",
            "wrong_treatment_or_testing_advice",
            "unsupported_prognosis",
            "dangerous_certainty",
            "sensitive_personal_info",
            "normal",
        ],
        "prohibited_actions_confirmed": {
            "training_run": False,
            "release_created": False,
            "test_or_medical_safety_holdout_read": False,
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "records": len(sample), "path": args.output.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
