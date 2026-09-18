"""构建并原子发布 Phase 7 prompt pool，不调用 API 或模型。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.config import load_dpo_config
from med_align_rag_4b.dpo.data import load_fingerprint_file
from med_align_rag_4b.dpo.manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)
from med_align_rag_4b.dpo.prompt_pool import (
    build_prompt_pool,
    validate_prompt_pool,
)
from med_align_rag_4b.sft.deduplication import question_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo/qwen3_4b_lora_phase7.yaml")
    parser.add_argument(
        "--train-release", default="data/sft/released/v2_plus_repair_r2/train.jsonl"
    )
    parser.add_argument(
        "--source-registry", default="data/sft/sources/source_registry.json"
    )
    parser.add_argument(
        "--evidence-registry",
        default="data/sft/sources/evidence_registry_v5_scale.json",
    )
    parser.add_argument(
        "--validation-cases", default="data/evaluation/validation_v2/cases.jsonl"
    )
    parser.add_argument(
        "--test-fingerprints", default="data/evaluation/test_v1/fingerprints.json"
    )
    parser.add_argument(
        "--holdout-fingerprints",
        default="data/evaluation/medical_safety_holdout_v1/fingerprints.json",
    )
    parser.add_argument(
        "--rag-queries", default="data/rag/compatibility/queries.v1.jsonl"
    )
    parser.add_argument("--input-freeze", default="manifests/phase_7/input_freeze.json")
    parser.add_argument("--output-dir", default="data/dpo/phase7_v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    train_path = resolve_project_path(PROJECT_ROOT, args.train_release)
    source_registry_path = resolve_project_path(PROJECT_ROOT, args.source_registry)
    evidence_registry_path = resolve_project_path(PROJECT_ROOT, args.evidence_registry)
    validation_path = resolve_project_path(PROJECT_ROOT, args.validation_cases)
    test_fingerprints_path = resolve_project_path(PROJECT_ROOT, args.test_fingerprints)
    holdout_fingerprints_path = resolve_project_path(
        PROJECT_ROOT, args.holdout_fingerprints
    )
    rag_queries_path = resolve_project_path(PROJECT_ROOT, args.rag_queries)
    input_freeze_path = resolve_project_path(PROJECT_ROOT, args.input_freeze)
    output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖: {output_dir}")

    config = load_dpo_config(config_path)
    train_rows = list(iter_jsonl(train_path))
    evidence_registry = json.loads(evidence_registry_path.read_text(encoding="utf-8"))
    evidence_sources = evidence_registry["sources"]
    blocked, blocked_sources = _load_blocked_fingerprints(
        validation_path=validation_path,
        test_fingerprints_path=test_fingerprints_path,
        holdout_fingerprints_path=holdout_fingerprints_path,
        rag_queries_path=rag_queries_path,
    )
    records = build_prompt_pool(
        train_rows,
        evidence_sources,
        config=config,
        blocked_fingerprints=blocked,
        source_sha256=sha256_file(train_path),
    )
    summary = validate_prompt_pool(records, config, blocked_fingerprints=blocked)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase7_v1.", dir=output_dir.parent))
    try:
        prompt_path = staging / "prompt_pool.jsonl"
        prompt_count = write_jsonl_atomic(prompt_path, records)
        if prompt_count != len(records):
            raise RuntimeError("prompt pool 写入数量不一致")

        blocked_path = staging / "blocked_fingerprints.json"
        blocked_payload = {
            "format_version": 1,
            "algorithm": "sha256-normalized-question-v1",
            "records": len(blocked),
            "fingerprints": sorted(blocked),
            "set_sha256": sha256_record(sorted(blocked)),
            "sources": blocked_sources,
            "contains_case_text": False,
        }
        write_manifest_atomic(blocked_path, blocked_payload)

        manifest = initial_manifest(
            experiment_type="data_release",
            payload={
                "purpose": "phase_7_prompt_pool_freeze",
                "seed": config["data"]["split_seed"],
                "inputs": {
                    "config": _file_descriptor(config_path, PROJECT_ROOT),
                    "train_release": _file_descriptor(train_path, PROJECT_ROOT),
                    "source_registry": _file_descriptor(
                        source_registry_path, PROJECT_ROOT
                    ),
                    "evidence_registry": _file_descriptor(
                        evidence_registry_path, PROJECT_ROOT
                    ),
                    "validation_cases": _file_descriptor(validation_path, PROJECT_ROOT),
                    "test_fingerprints": _file_descriptor(
                        test_fingerprints_path, PROJECT_ROOT
                    ),
                    "holdout_fingerprints": _file_descriptor(
                        holdout_fingerprints_path, PROJECT_ROOT
                    ),
                    "rag_queries": _file_descriptor(rag_queries_path, PROJECT_ROOT),
                    "input_freeze": _file_descriptor(input_freeze_path, PROJECT_ROOT),
                },
                "counts": summary,
                "outputs": {
                    "prompt_pool": _file_descriptor(
                        prompt_path, staging, records=len(records)
                    ),
                    "blocked_fingerprints": _file_descriptor(
                        blocked_path, staging, records=len(blocked)
                    ),
                },
                "leakage_checks": {
                    "blocked_fingerprint_collisions": 0,
                    "duplicate_prompt_fingerprints": 0,
                    "cross_split_prompt_groups": 0,
                },
                "prompt_source": {
                    "medical": "api_evidence",
                    "general_and_instruction": "databricks_dolly_15k_bdd27f4",
                    "assistant_answers_reused": False,
                },
                "executed_actions": {
                    "commercial_api_called": False,
                    "model_loaded": False,
                    "preference_pairs_generated": False,
                    "dpo_training_started": False,
                },
                "release_allowed": False,
            },
        )
        completed = finish_manifest(
            manifest,
            "completed",
            outputs={
                "prompt_pool": _file_descriptor(
                    prompt_path, staging, records=len(records)
                ),
                "blocked_fingerprints": _file_descriptor(
                    blocked_path, staging, records=len(blocked)
                ),
            },
        )
        write_manifest_atomic(staging / "manifest.json", completed)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_v1."):
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "records": len(records),
                "blocked_fingerprints": len(blocked),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_blocked_fingerprints(
    *,
    validation_path: Path,
    test_fingerprints_path: Path,
    holdout_fingerprints_path: Path,
    rag_queries_path: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    validation_fingerprints = {
        question_fingerprint(row["prompt"]) for row in iter_jsonl(validation_path)
    }
    rag_fingerprints = {
        question_fingerprint(row["query"]) for row in iter_jsonl(rag_queries_path)
    }
    test_fingerprints = load_fingerprint_file(test_fingerprints_path)
    holdout_fingerprints = load_fingerprint_file(holdout_fingerprints_path)
    sources = [
        _fingerprint_source(
            "validation_v2",
            validation_path,
            validation_fingerprints,
            computed_from_field="prompt",
        ),
        _fingerprint_source(
            "test_v1",
            test_fingerprints_path,
            test_fingerprints,
            computed_from_field=None,
        ),
        _fingerprint_source(
            "medical_safety_holdout_v1",
            holdout_fingerprints_path,
            holdout_fingerprints,
            computed_from_field=None,
        ),
        _fingerprint_source(
            "rag_compatibility_queries",
            rag_queries_path,
            rag_fingerprints,
            computed_from_field="query",
        ),
    ]
    blocked = set().union(
        validation_fingerprints,
        test_fingerprints,
        holdout_fingerprints,
        rag_fingerprints,
    )
    return blocked, sources


def _fingerprint_source(
    name: str,
    path: Path,
    fingerprints: set[str],
    *,
    computed_from_field: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "records": len(fingerprints),
        "computed_from_field": computed_from_field,
    }


def _file_descriptor(
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
