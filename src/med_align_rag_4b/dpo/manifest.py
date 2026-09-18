"""Phase 7 run manifest 的最小装配入口。"""

from __future__ import annotations

from typing import Any

from med_align_rag_4b.common.utils import utc_now_iso
from med_align_rag_4b.sft.run_manifest import (
    assert_no_absolute_paths,
    finish_manifest,
    write_manifest_atomic,
)

ALLOWED_EXPERIMENT_TYPES = {"contract_check", "data_release", "smoke", "formal_dpo"}


def initial_manifest(
    *, experiment_type: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """创建不含绝对路径的 Phase 7 running manifest。"""
    if experiment_type not in ALLOWED_EXPERIMENT_TYPES:
        raise ValueError("未知 Phase 7 experiment_type")
    result = {
        "format_version": 1,
        "phase": "phase_7",
        "status": "running",
        "experiment_type": experiment_type,
        "started_at_utc": utc_now_iso(),
        **payload,
    }
    assert_no_absolute_paths(result)
    return result


__all__ = [
    "ALLOWED_EXPERIMENT_TYPES",
    "finish_manifest",
    "initial_manifest",
    "write_manifest_atomic",
]
