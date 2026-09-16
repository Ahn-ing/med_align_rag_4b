"""Phase 5 运行 manifest 的脱敏原子更新。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.utils import utc_now_iso


def assert_no_absolute_paths(value: Any) -> None:
    """递归拒绝 manifest 中的 Windows/POSIX 绝对路径。"""
    if isinstance(value, dict):
        for item in value.values():
            assert_no_absolute_paths(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_absolute_paths(item)
    elif isinstance(value, str):
        if Path(value).is_absolute() or (len(value) >= 3 and value[1:3] in {":\\", ":/"}):
            raise ValueError("run manifest 不得包含本机绝对路径")


def write_manifest_atomic(path: str | Path, value: dict[str, Any]) -> None:
    """校验后在同目录原子替换 JSON。"""
    assert_no_absolute_paths(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def initial_manifest(
    *, experiment_type: str, payload: dict[str, Any], phase: str = "phase_5"
) -> dict[str, Any]:
    """创建 running 状态的最小版本化 manifest。"""
    allowed = {"phase_5": {"smoke", "short_run"}, "phase_6": {"formal_train"}}
    if experiment_type not in allowed.get(phase, set()):
        raise ValueError("未知 experiment_type")
    result = {
        "format_version": 1, "phase": phase, "status": "running",
        "experiment_type": experiment_type, "started_at_utc": utc_now_iso(),
        **payload,
    }
    assert_no_absolute_paths(result)
    return result


def finish_manifest(manifest: dict[str, Any], status: str, **updates: Any) -> dict[str, Any]:
    """构建 completed/failed 终态，不改写输入对象。"""
    if status not in {"completed", "failed"}:
        raise ValueError("终态只能是 completed 或 failed")
    result = {**manifest, **updates, "status": status, "finished_at_utc": utc_now_iso()}
    assert_no_absolute_paths(result)
    return result
