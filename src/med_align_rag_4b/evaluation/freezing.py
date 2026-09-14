"""评测集的确定性、拒绝覆盖、原子冻结。"""

from __future__ import annotations

import json
import os
import random
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.common.jsonl import count_jsonl, write_jsonl_atomic
from med_align_rag_4b.evaluation.datasets import validate_case, validate_no_leakage
from med_align_rag_4b.sft.deduplication import question_fingerprint


def select_by_quota(cases: Iterable[dict[str, Any]], quotas: dict[str, int], seed: int) -> list[dict[str, Any]]:
    """按类别做稳定伪随机选择；输入顺序不影响结果。"""
    rng = random.Random(seed)
    keyed = [(rng.random(), case["id"], case) for case in sorted(cases, key=lambda item: item["id"])]
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for _, _, case in sorted(keyed):
        category = case["primary_category"]
        if counts[category] < quotas.get(category, 0):
            selected.append(case)
            counts[category] += 1
    return sorted(selected, key=lambda item: item["id"])


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_split(
    output_dir: str | Path,
    cases: Iterable[dict[str, Any]],
    *,
    split: str,
    seed: int,
    source_registry_sha256: str,
    review_registry_sha256: str,
    rubric_sha256: str,
    evidence_registry_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """审核后发布一个不可覆盖的冻结 split，并重新加载校验 count/hash。"""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"冻结目录已存在: {destination}")
    rows = sorted(list(cases), key=lambda item: item["id"])
    for case in rows:
        validate_case(case)
    validate_no_leakage({split: rows})
    destination.mkdir(parents=True)
    data_path = destination / "cases.jsonl"
    try:
        write_jsonl_atomic(data_path, rows)
        digest = sha256_file(data_path)
        if count_jsonl(data_path) != len(rows):
            raise RuntimeError("冻结集重新加载后的 count 不一致")
        fingerprint_descriptor = None
        if split == "medical_safety_holdout":
            fingerprint_path = destination / "fingerprints.json"
            write_holdout_fingerprints(fingerprint_path, rows)
            fingerprint_descriptor = {"path": f"data/evaluation/{destination.name}/fingerprints.json", "sha256": sha256_file(fingerprint_path), "records": len(rows)}
        manifest = {
            "format_version": 1, "schema_version": "evaluation_case_v1", "split": split,
            "status": "frozen", "purpose": "模型比较；禁止训练使用", "records": len(rows),
            "data": {"path": f"data/evaluation/{destination.name}/cases.jsonl", "sha256": digest},
            "source_registry_sha256": source_registry_sha256,
            "review_registry_sha256": review_registry_sha256,
            "rubric_sha256": rubric_sha256,
            "evidence_registry_sha256": evidence_registry_sha256,
            "category_distribution": dict(sorted(Counter(row["primary_category"] for row in rows).items())),
            "risk_distribution": dict(sorted(Counter(row["risk_level"] for row in rows).items())),
            "deduplication": "NFKC+whitespace+casefold question fingerprint",
            "overlap_method": "exact normalized question fingerprint", "seed": seed,
            "sources": sorted({(row["source_id"], row["source_revision"], row["license_status"]) for row in rows}),
            "created_at_utc": created_at_utc,
            "usage_gate": "不得进入训练；holdout 仅在模型和配置冻结后运行",
            "release_allowed": False,
        }
        if fingerprint_descriptor is not None:
            manifest["fingerprints"] = fingerprint_descriptor
        _write_json_atomic(destination / "manifest.json", manifest)
        return manifest
    except Exception:
        for child in destination.iterdir():
            child.unlink()
        destination.rmdir()
        raise


def holdout_fingerprints(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """仅公开不可逆问题指纹，不返回 prompt。"""
    values = sorted({question_fingerprint(case["prompt"]) for case in cases})
    return {"format_version": 1, "algorithm": "sha256-normalized-question-v1", "fingerprints": values, "set_sha256": sha256_record(values)}


def write_holdout_fingerprints(path: str | Path, cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """原子发布不包含 prompt 的 holdout 指纹文件。"""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"指纹文件已存在: {destination}")
    payload = holdout_fingerprints(cases)
    _write_json_atomic(destination, payload)
    return payload


def is_blocked_by_holdout(prompt: str, fingerprints: dict[str, Any]) -> bool:
    return question_fingerprint(prompt) in set(fingerprints["fingerprints"])
