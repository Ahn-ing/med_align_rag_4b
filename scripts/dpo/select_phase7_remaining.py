"""选择尚未进入 pilot 的 900 条 Phase 7 prompt 任务，不调用 API。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-pool", default="data/dpo/phase7_v1/prompt_pool.jsonl")
    parser.add_argument("--pilot-tasks", default="data/dpo/pilot_v1/tasks.jsonl")
    parser.add_argument("--output-dir", default="data/dpo/full_v1/remaining")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prompt_pool_path = resolve_project_path(PROJECT_ROOT, args.prompt_pool)
    pilot_path = resolve_project_path(PROJECT_ROOT, args.pilot_tasks)
    output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在，不覆盖: {output_dir}")
        unexpected = [
            item
            for item in output_dir.iterdir()
            if item.name not in {"tasks.jsonl", "manifest.json"}
        ]
        if unexpected:
            raise FileExistsError(f"输出目录包含意外文件，拒绝覆盖: {unexpected}")

    prompt_rows = list(iter_jsonl(prompt_pool_path))
    pilot_ids = {row["id"] for row in iter_jsonl(pilot_path)}
    remaining = [row for row in prompt_rows if row["id"] not in pilot_ids]
    if len(prompt_rows) != 1000 or len(pilot_ids) != 100 or len(remaining) != 900:
        raise ValueError("remaining prompt 数量必须为 1000 - 100 = 900")
    _validate_remaining(remaining)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase7_remaining.", dir=output_dir.parent))
    try:
        tasks_path = staging / "tasks.jsonl"
        write_jsonl_atomic(tasks_path, remaining)
        manifest = initial_manifest(
            experiment_type="data_release",
            payload={
                "purpose": "phase_7_remaining_tasks",
                "inputs": {
                    "prompt_pool": _descriptor(prompt_pool_path, PROJECT_ROOT),
                    "pilot_tasks": _descriptor(pilot_path, PROJECT_ROOT),
                },
                "counts": {
                    "records": len(remaining),
                    "train": sum(row["split"] == "train" for row in remaining),
                    "preference_validation": sum(
                        row["split"] == "preference_validation" for row in remaining
                    ),
                    "tasks": _counts(remaining, "task"),
                    "safety": _counts(
                        [row for row in remaining if row["task"] == "medical_safety"],
                        "primary_category",
                    ),
                },
                "executed_actions": {
                    "commercial_api_called": False,
                    "preference_pairs_generated": False,
                    "model_loaded": False,
                    "dpo_training_started": False,
                },
                "release_allowed": False,
            },
        )
        completed = finish_manifest(
            manifest,
            "completed",
            output=_descriptor(tasks_path, staging, records=len(remaining)),
        )
        if output_dir.exists():
            write_jsonl_atomic(output_dir / "tasks.jsonl", remaining, overwrite=True)
            write_manifest_atomic(output_dir / "manifest.json", completed)
        else:
            write_manifest_atomic(staging / "manifest.json", completed)
            os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_remaining."):
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "records": len(remaining),
                "train": sum(row["split"] == "train" for row in remaining),
                "preference_validation": sum(
                    row["split"] == "preference_validation" for row in remaining
                ),
            },
            ensure_ascii=False,
        )
    )


def _validate_remaining(rows: list[dict[str, Any]]) -> None:
    if sum(row["split"] == "train" for row in rows) != 810:
        raise ValueError("remaining train 必须为 810")
    if sum(row["split"] == "preference_validation" for row in rows) != 90:
        raise ValueError("remaining preference_validation 必须为 90")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("remaining prompt ID 重复")


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def _descriptor(
    path: Path, relative_to: Path, *, records: int | None = None
) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "records": records,
    }


if __name__ == "__main__":
    main()
