"""按有效 assistant token 的确定性分层选样。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from med_align_rag_4b.common.hashing import sha256_record


def deterministic_priority(record: dict[str, Any], seed: int) -> str:
    return sha256_record([seed, record["domain"], record["primary_category"], record["template_family"], record["id"]])


def select_nearest(records: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    """稳定排序后，在跨过配额的边界选择误差较小的一侧。"""
    ordered = sorted(records, key=lambda row: deterministic_priority(row, seed))
    selected: list[dict[str, Any]] = []
    total = 0
    for record in ordered:
        tokens = record["assistant_supervised_tokens"]
        if total + tokens > target:
            if abs(target - (total + tokens)) < abs(target - total):
                selected.append(record)
            break
        selected.append(record)
        total += tokens
    return selected


def token_distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[str(record[field])] += record["assistant_supervised_tokens"]
    return dict(sorted(counts.items()))
