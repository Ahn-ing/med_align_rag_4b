"""可迁移 manifest 的构建和校验。"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.paths import portable_relative_path
from med_align_rag_4b.common.schemas import ArtifactDescriptor
from med_align_rag_4b.common.utils import utc_now_iso


def describe_artifact(path: str | Path, project_root: str | Path, *, records: int | None = None) -> ArtifactDescriptor:
    """从项目内现有文件构建带哈希的产物描述。"""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"产物不存在: {artifact}")
    return ArtifactDescriptor(
        path=portable_relative_path(artifact, project_root),
        sha256=sha256_file(artifact),
        bytes=artifact.stat().st_size,
        records=records,
    )


def build_manifest(kind: str, artifacts: Iterable[ArtifactDescriptor], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建不包含机器绝对路径的版本化 manifest。"""
    return {
        "format_version": 1,
        "kind": kind,
        "created_at_utc": utc_now_iso(),
        "artifacts": [artifact.__dict__ for artifact in artifacts],
        "metadata": metadata or {},
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    """加载 Phase 0 manifest 并校验产物描述。"""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest JSON 无效: {source}") from exc
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("缺少或不支持 manifest format_version")
    if not isinstance(value.get("kind"), str) or not value["kind"].strip():
        raise ValueError("manifest kind 必须是非空字符串")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts 必须是列表")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("manifest 中每个 artifact 必须是对象")
        ArtifactDescriptor.from_dict(artifact)
    return value
