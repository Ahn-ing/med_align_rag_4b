"""Phase 4 发布数据的 Phase 5/6 读取门禁。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import count_jsonl, iter_jsonl
from med_align_rag_4b.sft.evaluation import evaluation_case_messages


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


def validate_phase6_release(project_root: str | Path, train_path: str | Path) -> dict[str, Any]:
    """校验 Phase 6 冻结 release、基础版本哈希及 smoke 子集。"""
    root = Path(project_root).resolve()
    released = Path(train_path).resolve().parent
    manifest_path = released / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_version = manifest.get("dataset_version")
    allowed_versions = {"v2_plus_repair", "v2_plus_repair_r2"}
    if dataset_version not in allowed_versions:
        raise ValueError(f"不支持的数据 release: {dataset_version}")
    required = {
        "status": "completed",
        "training_allowed": True,
        "release_allowed": False,
        "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{dataset_version} manifest {key} 不符合冻结契约")
    expected_purpose = (
        "internal_engineering_training_append_only_repair"
        if dataset_version == "v2_plus_repair"
        else "internal_engineering_training_with_medical_behavior_patch"
    )
    if manifest.get("purpose") != expected_purpose:
        raise ValueError(f"{dataset_version} manifest purpose 不符合冻结契约")
    base = manifest.get("base_release", {})
    base_release = root / str(base.get("path", ""))
    if sha256_file(base_release / "manifest.json") != base.get("manifest_sha256"):
        raise ValueError("基础 release manifest SHA256 不匹配")
    if sha256_file(base_release / "train.jsonl") != base.get("train_sha256"):
        raise ValueError("基础 release train SHA256 不匹配")
    for relative, item in manifest.get("artifacts", {}).items():
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"{dataset_version} 产物缺失或 SHA256 不匹配: {relative}")
        if item.get("records") is not None and count_jsonl(path) != item["records"]:
            raise ValueError(f"{dataset_version} 产物记录数不匹配: {relative}")
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
        raise ValueError(f"Phase 6 {dataset_version} 只允许使用冻结 train 或 smoke_train")
    tokenizer_revision = "1cfa9a7208912126459214e8b04321603b3df60c"
    if dataset_version == "v2_plus_repair":
        tokenizer_files = manifest.get("tokenizer_files", {})
        if not tokenizer_files:
            raise ValueError("v2_plus_repair 缺少 tokenizer_files")
    else:
        base_manifest = json.loads((base_release / "manifest.json").read_text(encoding="utf-8"))
        tokenizer_files = base_manifest["tokenizer_files"]
    return {
        **manifest,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_files": tokenizer_files,
    }


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


def validate_phase6_validation(project_root: str | Path, path: str | Path) -> dict[str, Any]:
    """校验 Phase 6 完整 validation，仅允许无梯度 loss 评估。"""
    root = Path(project_root).resolve()
    data_path = Path(path).resolve()
    expected = (root / "data/evaluation/validation_v2/cases.jsonl").resolve()
    if data_path != expected:
        raise ValueError("Phase 6 只允许使用冻结的完整 validation")
    manifest_path = data_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "split": "validation",
        "status": "frozen",
        "records": 60,
        "release_allowed": False,
    }
    for key, expected_value in required.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"Phase 6 validation manifest {key} 不匹配")
    if manifest.get("data", {}).get("path") != data_path.relative_to(root).as_posix():
        raise ValueError("Phase 6 validation manifest 路径不匹配")
    if manifest["data"].get("sha256") != sha256_file(data_path):
        raise ValueError("Phase 6 validation SHA256 不匹配")
    rows = list(iter_jsonl(data_path))
    ids = [row.get("id") for row in rows]
    if len(rows) != manifest["records"] or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Phase 6 validation 记录数或稳定 ID 无效")
    if len(set(ids)) != len(ids):
        raise ValueError("Phase 6 validation 存在重复稳定 ID")
    return {
        **manifest,
        "generation_allowed": False,
        "gradient_update_allowed": False,
    }


def load_eval_and_encode(path: str | Path, encoder: Any) -> list[dict[str, Any]]:
    """使用训练同源 tokenizer 编码 loss eval，不将其加入梯度更新。"""
    encoded = []
    for row in iter_jsonl(path):
        messages = row.get("messages")
        if messages is None:
            messages = evaluation_case_messages(row)
        item = encoder.encode(messages)
        if item["active_label_tokens"] <= 0:
            raise ValueError(f"{row['id']} assistant supervision 为 0")
        encoded.append(item)
    return encoded
