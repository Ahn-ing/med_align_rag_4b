"""Phase 4 发布数据的 Phase 5 读取门禁。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import count_jsonl, iter_jsonl


def validate_phase4_release(project_root: str | Path, train_path: str | Path) -> dict[str, Any]:
    """校验冻结 manifest、文件哈希和 smoke 严格子集。"""
    root = Path(project_root).resolve()
    released = root / "data/sft/released/v1"
    manifest = json.loads((released / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "purpose": "internal_phase_5_engineering_training",
        "training_allowed": True,
        "release_allowed": False,
        "model_id": "Qwen/Qwen3-4B",
        "enable_thinking": False,
        "max_length": 2048,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Phase 4 manifest {key} 不符合冻结契约")
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    for relative, item in artifacts.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Phase 4 产物缺失或 SHA256 不匹配: {relative}")
        if item["records"] is not None and count_jsonl(path) != item["records"]:
            raise ValueError(f"Phase 4 产物记录数不匹配: {relative}")
    train_rows = list(iter_jsonl(released / "train.jsonl"))
    smoke_rows = list(iter_jsonl(released / "smoke_train.jsonl"))
    train_ids = {row["id"] for row in train_rows}
    smoke_ids = {row["id"] for row in smoke_rows}
    if len(train_ids) != len(train_rows) or len(smoke_ids) != len(smoke_rows):
        raise ValueError("train/smoke 存在重复稳定 ID")
    if not smoke_ids or smoke_ids == train_ids or not smoke_ids < train_ids:
        raise ValueError("smoke_train 必须是 train 的非空严格 ID 子集")
    selected = Path(train_path).resolve()
    allowed = {released / "train.jsonl", released / "smoke_train.jsonl"}
    if selected not in {path.resolve() for path in allowed}:
        raise ValueError("Phase 5 只允许使用冻结 train 或 smoke_train")
    return manifest


def load_and_encode(path: str | Path, encoder: Any) -> list[dict[str, Any]]:
    """编码并与 Phase 4 记录的 token 统计逐条对齐。"""
    encoded = []
    for row in iter_jsonl(path):
        item = encoder.encode(row["messages"])
        if len(item["input_ids"]) != row["total_tokens"]:
            raise ValueError(f"{row['id']} total_tokens 与 Phase 4 不一致")
        if item["active_label_tokens"] != row["assistant_supervised_tokens"]:
            raise ValueError(f"{row['id']} assistant token 统计与 Phase 4 不一致")
        encoded.append(item)
    return encoded


def validate_phase5_eval(project_root: str | Path, path: str | Path) -> dict[str, Any]:
    """验证 loss-only 子集授权、来源和内容哈希。"""
    root = Path(project_root).resolve()
    data_path = Path(path).resolve()
    expected_path = (root / "data/evaluation/phase5_loss_v1.jsonl").resolve()
    if data_path != expected_path:
        raise ValueError("Phase 5 只允许使用固定 loss-only 子集")
    manifest_path = data_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "status": "approved_for_phase5_loss",
        "purpose": "loss_only_engineering_validation",
        "generation_allowed": False,
        "gradient_update_allowed": False,
        "records": 18,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Phase 5 eval manifest {key} 不匹配")
    if manifest.get("data", {}).get("sha256") != sha256_file(data_path):
        raise ValueError("Phase 5 eval SHA256 不匹配")
    if count_jsonl(data_path) != manifest["records"]:
        raise ValueError("Phase 5 eval 记录数不匹配")
    validation = root / "data/evaluation/validation_v1/cases.jsonl"
    validation_manifest = root / "data/evaluation/validation_v1/manifest.json"
    source = manifest.get("source", {})
    if source.get("sha256") != sha256_file(validation):
        raise ValueError("Phase 5 eval 来源 validation SHA256 不匹配")
    if source.get("manifest_sha256") != sha256_file(validation_manifest):
        raise ValueError("Phase 5 eval 来源 manifest SHA256 不匹配")
    ids = [row["id"] for row in iter_jsonl(data_path)]
    if ids != manifest.get("selection", {}).get("selected_ids"):
        raise ValueError("Phase 5 eval 稳定 ID 与 manifest 不匹配")
    return manifest


def load_eval_and_encode(path: str | Path, encoder: Any) -> list[dict[str, Any]]:
    """使用训练同源 tokenizer 编码 eval，不将其加入梯度更新。"""
    encoded = []
    for row in iter_jsonl(path):
        item = encoder.encode(row["messages"])
        if item["active_label_tokens"] <= 0:
            raise ValueError(f"{row['id']} assistant supervision 为 0")
        encoded.append(item)
    return encoded
