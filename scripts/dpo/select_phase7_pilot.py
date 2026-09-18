"""选择 Phase 7 API pilot 的分层 prompt，不调用 API。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.api_config import load_api_config
from med_align_rag_4b.dpo.manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)
from med_align_rag_4b.dpo.pilot import select_pilot_prompts

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/dpo/phase7_api.yaml")
    parser.add_argument("--prompt-pool", default="data/dpo/phase7_v1/prompt_pool.jsonl")
    parser.add_argument("--output-dir", default="data/dpo/pilot_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_config_path = resolve_project_path(PROJECT_ROOT, args.api_config)
    prompt_pool_path = resolve_project_path(PROJECT_ROOT, args.prompt_pool)
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

    api_config = load_api_config(api_config_path)
    prompt_rows = list(iter_jsonl(prompt_pool_path))
    tasks = select_pilot_prompts(prompt_rows, api_config)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase7_pilot.", dir=output_dir.parent))
    try:
        tasks_path = staging / "tasks.jsonl"
        write_jsonl_atomic(tasks_path, tasks)
        manifest = initial_manifest(
            experiment_type="data_release",
            payload={
                "purpose": "phase_7_api_pilot_tasks",
                "inputs": {
                    "api_config": _descriptor(api_config_path, PROJECT_ROOT),
                    "prompt_pool": _descriptor(prompt_pool_path, PROJECT_ROOT),
                },
                "api": {
                    "provider": api_config["provider"],
                    "endpoint": api_config["endpoint"],
                    "api_key_env": api_config["api_key_env"],
                    "generation_model": api_config["generation"]["model"],
                    "judge_model": api_config["judge"]["model"],
                    "call_budget": api_config["pilot"]["call_budget"],
                    "total_token_budget": api_config["pilot"]["total_token_budget"],
                },
                "counts": {
                    "records": len(tasks),
                    "train": sum(row["split"] == "train" for row in tasks),
                    "preference_validation": sum(
                        row["split"] == "preference_validation" for row in tasks
                    ),
                },
                "output": _descriptor(tasks_path, staging, records=len(tasks)),
                "executed_actions": {
                    "commercial_api_called": False,
                    "preference_pairs_generated": False,
                    "model_loaded": False,
                    "dpo_training_started": False,
                },
                "release_allowed": False,
            },
        )
        completed = finish_manifest(manifest, "completed")
        if output_dir.exists():
            write_jsonl_atomic(output_dir / "tasks.jsonl", tasks, overwrite=True)
            write_manifest_atomic(output_dir / "manifest.json", completed)
        else:
            write_manifest_atomic(staging / "manifest.json", completed)
            os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_pilot."):
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "records": len(tasks),
            },
            ensure_ascii=False,
        )
    )


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
