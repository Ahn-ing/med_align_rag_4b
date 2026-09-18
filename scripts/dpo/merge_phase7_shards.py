"""合并 Phase 7 full_v1 分片为统一 generated 目录，不调用 API。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.manifests import build_manifest, describe_artifact
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.manifest import write_manifest_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", default="data/dpo/full_v1/shards")
    parser.add_argument("--output-dir", default="data/dpo/full_v1/generated")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    shards_dir = resolve_project_path(PROJECT_ROOT, args.shards_dir)
    output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖: {output_dir}")

    shards = sorted([p for p in shards_dir.iterdir() if p.is_dir()])
    if len(shards) != 9:
        raise ValueError(f"full_v1 分片必须为 9 个，实际 {len(shards)}")
    candidates: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    usage = Counter()
    for shard in shards:
        manifest_path = shard / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"分片未完成: {shard.name}")
        if manifest.get("counts", {}).get("tasks_selected") != 100:
            raise ValueError(f"分片数量不是 100: {shard.name}")
        shard_candidates = list(iter_jsonl(shard / "candidates.jsonl"))
        shard_quarantine = list(iter_jsonl(shard / "quarantine.jsonl"))
        if len(shard_candidates) + len(shard_quarantine) != 100:
            raise ValueError(f"分片 candidate+quarantine 不等于 100: {shard.name}")
        for row in shard_candidates:
            if row["id"] in seen_ids:
                raise ValueError(f"合并后 ID 重复: {row['id']}")
            seen_ids.add(row["id"])
        candidates.extend(shard_candidates)
        quarantine.extend(shard_quarantine)
        for key, value in (manifest.get("usage") or {}).items():
            if key in {"calls", "prompt_tokens", "completion_tokens", "total_tokens"}:
                usage[key] += int(value or 0)
    if len(candidates) + len(quarantine) != 900:
        raise ValueError(
            f"合并后总数必须为 900，实际 {len(candidates) + len(quarantine)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase7_merged.", dir=output_dir.parent))
    try:
        candidates_path = staging / "candidates.jsonl"
        quarantine_path = staging / "quarantine.jsonl"
        write_jsonl_atomic(candidates_path, candidates)
        write_jsonl_atomic(quarantine_path, quarantine)
        manifest = build_manifest(
            "phase_7_full_remaining_generated",
            [
                describe_artifact(candidates_path, staging, records=len(candidates)),
                describe_artifact(quarantine_path, staging, records=len(quarantine)),
            ],
            metadata={
                "phase": "phase_7",
                "shards": len(shards),
                "records": len(candidates) + len(quarantine),
                "candidates_pending_review": len(candidates),
                "quarantined": len(quarantine),
                "usage": dict(usage),
                "executed_actions": {
                    "commercial_api_called": False,
                    "preference_pairs_generated": False,
                    "model_loaded": False,
                    "dpo_training_started": False,
                },
                "release_allowed": False,
            },
        )
        write_manifest_atomic(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_merged."):
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "candidates": len(candidates),
                "quarantine": len(quarantine),
                "usage": dict(usage),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
