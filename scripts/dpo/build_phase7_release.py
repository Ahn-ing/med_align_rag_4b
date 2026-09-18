"""从 Phase 7 reviewed candidates 构建冻结 release，不加载生成模型。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.manifests import build_manifest, describe_artifact
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.common.utils import utc_now_iso
from med_align_rag_4b.dpo.config import load_dpo_config
from med_align_rag_4b.dpo.data import (
    load_blocked_fingerprints,
    validate_preference_dataset,
)
from med_align_rag_4b.dpo.manifest import write_manifest_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", default="data/dpo/full_v2/reviewed/candidates.jsonl"
    )
    parser.add_argument("--config", default="configs/dpo/qwen3_4b_lora_phase7.yaml")
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument(
        "--blocked-fingerprints",
        type=Path,
        default=Path("data/dpo/phase7_v1/blocked_fingerprints.json"),
    )
    parser.add_argument("--output-dir", default="data/dpo/released/phase7_v2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidates_path = resolve_project_path(PROJECT_ROOT, args.candidates)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    blocked_path = resolve_project_path(PROJECT_ROOT, args.blocked_fingerprints)
    output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖: {output_dir}")

    config = load_dpo_config(config_path)
    blocked = load_blocked_fingerprints([blocked_path])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=False,
        local_files_only=True,
    )
    rows = list(iter_jsonl(candidates_path))
    release_rows = [_with_token_counts(row, tokenizer) for row in rows]
    summary = validate_preference_dataset(
        release_rows,
        config,
        blocked_fingerprints=blocked,
        enforce_quotas=True,
        require_reviewed=True,
    )
    train = [row for row in release_rows if row["split"] == "train"]
    validation = [
        row for row in release_rows if row["split"] == "preference_validation"
    ]
    if len(train) != config["data"]["counts"]["train"]:
        raise ValueError("train 数量与冻结配置不一致")
    if len(validation) != config["data"]["counts"]["preference_validation"]:
        raise ValueError("preference_validation 数量与冻结配置不一致")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".phase7_v2_release.", dir=output_dir.parent)
    )
    try:
        train_path = staging / "train.jsonl"
        validation_path = staging / "preference_validation.jsonl"
        write_jsonl_atomic(train_path, train)
        write_jsonl_atomic(validation_path, validation)
        token_summary = _token_summary(train, validation)
        manifest = build_manifest(
            "phase_7_preference_release",
            [
                describe_artifact(train_path, PROJECT_ROOT, records=len(train)),
                describe_artifact(validation_path, PROJECT_ROOT, records=len(validation)),
            ],
            metadata={
                "phase": "phase_7",
                "dataset_version": "phase7_v2",
                "status": "completed",
                "training_allowed": True,
                "training_authorized": False,
                "release_allowed": False,
                "doctor_reviewed": False,
                "clinical_validation": False,
                "counts": {
                    "train": len(train),
                    "preference_validation": len(validation),
                    "total": len(release_rows),
                },
                "tokens": token_summary,
                "summary": summary,
                "source_reviewed_candidates": candidates_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "source_reviewed_sha256": sha256_file(candidates_path),
                "reviewer_role": "codex_research_reviewer",
                "review_method": "codex_rule_audit_plus_priority_queue_plus_length_remediation_v1",
                "created_at_utc": utc_now_iso(),
            },
        )
        write_manifest_atomic(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_v2_release."):
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "train": len(train),
                "preference_validation": len(validation),
                "tokens": token_summary,
            },
            ensure_ascii=False,
        )
    )


def _with_token_counts(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    item = deepcopy(row)
    conversations = item["conversations"]
    prompt_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )["input_ids"]
    chosen_ids = tokenizer.apply_chat_template(
        conversations + [{"role": "assistant", "content": item["chosen"]}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )["input_ids"]
    rejected_ids = tokenizer.apply_chat_template(
        conversations + [{"role": "assistant", "content": item["rejected"]}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )["input_ids"]
    prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
    chosen_ids = chosen_ids[0] if isinstance(chosen_ids[0], list) else chosen_ids
    rejected_ids = (
        rejected_ids[0] if isinstance(rejected_ids[0], list) else rejected_ids
    )
    if chosen_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"chosen token 前缀不一致: {item['id']}")
    if rejected_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"rejected token 前缀不一致: {item['id']}")
    item["prompt_tokens"] = len(prompt_ids)
    item["chosen_tokens"] = len(chosen_ids) - len(prompt_ids)
    item["rejected_tokens"] = len(rejected_ids) - len(prompt_ids)
    item["review_issues"] = list(item.get("review_issues") or [])
    item["status"] = "agent_reviewed_pending_release"
    return item


def _token_summary(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> dict[str, Any]:
    def split_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "chosen_tokens": sum(row["chosen_tokens"] for row in rows),
            "rejected_tokens": sum(row["rejected_tokens"] for row in rows),
        }

    train_tokens = split_summary(train)
    validation_tokens = split_summary(validation)
    return {
        "train": train_tokens,
        "preference_validation": validation_tokens,
        "total": {
            key: train_tokens[key] + validation_tokens[key] for key in train_tokens
        },
    }


if __name__ == "__main__":
    main()
