"""校验已生成的 Phase 7 API pilot，不调用 API。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.api_config import load_api_config
from med_align_rag_4b.dpo.data import load_fingerprint_file
from med_align_rag_4b.dpo.manifest import write_manifest_atomic
from med_align_rag_4b.dpo.pilot import validate_generated_pilot

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/dpo/phase7_api.yaml")
    parser.add_argument("--tasks", default="data/dpo/pilot_v1/tasks.jsonl")
    parser.add_argument("--generated-dir", default="data/dpo/pilot_v1/generated")
    parser.add_argument(
        "--blocked-fingerprints",
        default="data/dpo/phase7_v1/blocked_fingerprints.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_config_path = resolve_project_path(PROJECT_ROOT, args.api_config)
    tasks_path = resolve_project_path(PROJECT_ROOT, args.tasks)
    generated_dir = resolve_project_path(PROJECT_ROOT, args.generated_dir)
    blocked_path = resolve_project_path(PROJECT_ROOT, args.blocked_fingerprints)
    candidates_path = generated_dir / "candidates.jsonl"
    quarantine_path = generated_dir / "quarantine.jsonl"
    generation_manifest_path = generated_dir / "manifest.json"
    report_path = generated_dir / "validation.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"验证报告已存在，不覆盖: {report_path}")

    api_config = load_api_config(api_config_path)
    tasks = list(iter_jsonl(tasks_path))
    candidates = list(iter_jsonl(candidates_path))
    quarantine = list(iter_jsonl(quarantine_path))
    blocked = load_fingerprint_file(blocked_path)
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    result = validate_generated_pilot(
        tasks,
        candidates,
        quarantine,
        api_config,
        blocked_fingerprints=blocked,
    )
    result.update(
        {
            "format_version": 1,
            "phase": "phase_7",
            "purpose": "phase_7_api_pilot_validation",
            "inputs": {
                "api_config": _descriptor(api_config_path),
                "tasks": _descriptor(tasks_path),
                "candidates": _descriptor(candidates_path),
                "quarantine": _descriptor(quarantine_path),
                "generation_manifest": _descriptor(generation_manifest_path),
                "blocked_fingerprints": _descriptor(blocked_path),
            },
            "api_usage": generation_manifest.get("usage")
            or generation_manifest.get("metadata", {}).get("usage", {}),
            "generator_judge_independent": generation_manifest.get("api", {}).get(
                "judge_independent_from_generator", False
            ),
            "limitations": [
                "pilot 结果尚未完成代理语义审核",
                "生成器和 judge 使用同一模型，不能据此声称独立评测",
                "当前字符长度不是 Qwen token 长度，训练前仍需 tokenizer 门禁",
            ],
            "release_allowed": False,
        }
    )
    write_manifest_atomic(report_path, result)
    if not result["passed"]:
        raise SystemExit("Phase 7 API pilot 自动门禁失败，详见 validation.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    main()
