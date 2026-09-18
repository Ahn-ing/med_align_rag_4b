"""调用 API 生成 Phase 7 100 对偏好数据 pilot。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.api_config import load_api_config
from med_align_rag_4b.dpo.api_generation import (
    evidence_context,
    generation_messages,
    judge_messages,
    parse_generation_content,
    parse_judge_content,
    request_json_completion,
)
from med_align_rag_4b.dpo.manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/dpo/phase7_api.yaml")
    parser.add_argument("--tasks", default="data/dpo/pilot_v1/tasks.jsonl")
    parser.add_argument(
        "--evidence-registry",
        default="data/sft/sources/evidence_registry_v5_scale.json",
    )
    parser.add_argument("--output-dir", default="data/dpo/pilot_v1/generated")
    parser.add_argument("--budget-profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--endpoint")
    parser.add_argument("--generation-model")
    parser.add_argument("--judge-model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_config_path = resolve_project_path(PROJECT_ROOT, args.api_config)
    tasks_path = resolve_project_path(PROJECT_ROOT, args.tasks)
    evidence_registry_path = resolve_project_path(PROJECT_ROOT, args.evidence_registry)
    output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖: {output_dir}")

    api_config = load_api_config(api_config_path)
    if args.endpoint:
        api_config["endpoint"] = args.endpoint.rstrip("/")
    if args.generation_model:
        api_config["generation"]["model"] = args.generation_model
    if args.judge_model:
        api_config["judge"]["model"] = args.judge_model
    budget_settings = api_config[args.budget_profile]
    api_key = os.environ.get(api_config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"环境变量 {api_config['api_key_env']} 未设置；请先在本机配置 API key"
        )
    all_tasks = list(iter_jsonl(tasks_path))
    if args.offset < 0:
        raise ValueError("--offset 不得为负数")
    if args.offset > len(all_tasks):
        raise ValueError("--offset 超出任务数量")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正数")
    stop = None if args.limit is None else args.offset + args.limit
    tasks = all_tasks[args.offset : stop]
    if not tasks:
        raise ValueError("offset/limit 未选择任何任务")
    evidence_registry = json.loads(evidence_registry_path.read_text(encoding="utf-8"))
    budget = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    with tqdm(total=len(tasks), desc="Phase 7 API pilot", unit="pair") as progress:
        for task in tasks:
            pair, rejection = _process_task(
                task,
                evidence_registry=evidence_registry,
                api_config=api_config,
                budget_profile=args.budget_profile,
                api_key=api_key,
                budget=budget,
            )
            if pair is not None:
                accepted.append(pair)
            if rejection is not None:
                quarantine.append(rejection)
            progress.update(1)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase7_generated.", dir=output_dir.parent))
    try:
        candidates_path = staging / "candidates.jsonl"
        quarantine_path = staging / "quarantine.jsonl"
        write_jsonl_atomic(candidates_path, accepted)
        write_jsonl_atomic(quarantine_path, quarantine)
        manifest = initial_manifest(
            experiment_type="data_release",
            payload={
                "purpose": f"phase_7_api_{args.budget_profile}_generation",
                "inputs": {
                    "api_config": _descriptor(api_config_path, PROJECT_ROOT),
                    "tasks": _descriptor(tasks_path, PROJECT_ROOT),
                    "evidence_registry": _descriptor(
                        evidence_registry_path, PROJECT_ROOT
                    ),
                },
                "api": {
                    "provider": api_config["provider"],
                    "endpoint": api_config["endpoint"],
                    "generation_model": api_config["generation"]["model"],
                    "judge_model": api_config["judge"]["model"],
                    "judge_independent_from_generator": (
                        api_config["generation"]["model"]
                        != api_config["judge"]["model"]
                    ),
                    "api_key_env": api_config["api_key_env"],
                    "api_key_persisted": False,
                    "budget_profile": args.budget_profile,
                    "call_budget": budget_settings["call_budget"],
                    "total_token_budget": budget_settings["total_token_budget"],
                },
                "counts": {
                    "tasks_total": len(all_tasks),
                    "tasks_selected": len(tasks),
                    "task_offset": args.offset,
                    "task_limit": args.limit,
                    "accepted_pending_review": len(accepted),
                    "quarantined": len(quarantine),
                },
                "usage": budget,
                "outputs": {
                    "candidates": _descriptor(
                        candidates_path, staging, records=len(accepted)
                    ),
                    "quarantine": _descriptor(
                        quarantine_path, staging, records=len(quarantine)
                    ),
                },
                "executed_actions": {
                    "commercial_api_called": True,
                    "preference_pairs_generated": True,
                    "model_loaded": False,
                    "dpo_training_started": False,
                },
                "review_status": "pending_agent_review",
                "release_allowed": False,
            },
        )
        completed = finish_manifest(manifest, "completed")
        write_manifest_atomic(staging / "manifest.json", completed)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_generated."):
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "accepted": len(accepted),
                "quarantined": len(quarantine),
                "usage": budget,
            },
            ensure_ascii=False,
        )
    )


def _process_task(
    task: dict[str, Any],
    *,
    evidence_registry: dict[str, Any],
    api_config: dict[str, Any],
    budget_profile: str,
    api_key: str,
    budget: dict[str, int],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    evidence = evidence_context(evidence_registry, task["evidence_ids"])
    generation_request = generation_messages(task, evidence)
    pair, generation_meta = _request(
        messages=generation_request,
        parser=parse_generation_content,
        api_config=api_config,
        budget_profile=budget_profile,
        api_key=api_key,
        budget=budget,
    )
    rejection_reasons = _pair_rejection_reasons(task, pair)
    if rejection_reasons:
        return None, _quarantine(task, rejection_reasons, pair=pair)

    judge_request = judge_messages(task, pair, evidence)
    judgment, judge_meta = _request(
        messages=judge_request,
        parser=parse_judge_content,
        api_config=api_config,
        budget_profile=budget_profile,
        api_key=api_key,
        budget=budget,
    )
    if not judgment["accepted"] or judgment["preferred"] != "chosen":
        return None, _quarantine(
            task,
            ["judge_rejected_pair"],
            pair=pair,
            judgment=judgment,
        )

    record = {
        "id": task["id"].replace("dpo_prompt_", "dpo_pair_", 1),
        "prompt_group_id": task["prompt_group_id"],
        "split": task["split"],
        "source_id": task["source_id"],
        "source_record_id": task["source_record_id"],
        "source_revision": task["source_revision"],
        "source_sha256": task["source_sha256"],
        "conversations": task["conversations"],
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
        "pair_reason": pair["pair_reason"],
        "task": task["task"],
        "primary_category": task["primary_category"],
        "risk_level": task["risk_level"],
        "evidence_ids": task["evidence_ids"],
        "evidence_ids_used": pair["evidence_ids_used"],
        "review_status": "pending_agent_review",
        "reviewer_role": None,
        "reviewed_at_utc": None,
        "evidence_checked": False,
        "chosen_source": (
            "api_evidence_constrained"
            if task["task"].startswith("medical")
            else "api_generated"
        ),
        "rejected_source": "api_controlled_negative",
        "contains_personal_data": False,
        "release_allowed": False,
        "status": "api_generated_pending_review",
        "generation": {
            "model": generation_meta["response_model"],
            "response_id": generation_meta["response_id"],
            "usage": generation_meta["usage"],
            "prompt_sha256": sha256_text(
                json.dumps(generation_request, ensure_ascii=False, sort_keys=True)
            ),
        },
        "judge": {
            "model": judge_meta["response_model"],
            "response_id": judge_meta["response_id"],
            "usage": judge_meta["usage"],
            "prompt_sha256": sha256_text(
                json.dumps(judge_request, ensure_ascii=False, sort_keys=True)
            ),
            "preferred": judgment["preferred"],
            "accepted": judgment["accepted"],
            "issues": judgment["issues"],
            "risk_flags": judgment["risk_flags"],
        },
    }
    return record, None


def _request(
    *,
    messages: list[dict[str, str]],
    parser: Any,
    api_config: dict[str, Any],
    budget_profile: str,
    api_key: str,
    budget: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    budget_settings = api_config[budget_profile]
    remaining_calls = budget_settings["call_budget"] - budget["calls"]
    if remaining_calls <= 0:
        raise RuntimeError("API 调用预算已耗尽")
    section = (
        api_config["generation"]
        if parser is parse_generation_content
        else api_config["judge"]
    )
    result, metadata = request_json_completion(
        endpoint=api_config["endpoint"],
        api_key=api_key,
        model=section["model"],
        messages=messages,
        temperature=section["temperature"],
        top_p=section["top_p"],
        max_tokens=section["max_tokens"],
        response_format=section["response_format"],
        timeout_seconds=section["timeout_seconds"],
        max_attempts=min(section["max_attempts"], remaining_calls),
        parser=parser,
    )
    budget["calls"] += metadata["attempts"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        budget[key] += metadata["usage"][key]
    if budget["total_tokens"] > budget_settings["total_token_budget"]:
        raise RuntimeError("API total token budget 已超出")
    return result, metadata


def _pair_rejection_reasons(task: dict[str, Any], pair: dict[str, Any]) -> list[str]:
    reasons = []
    allowed_evidence = set(task["evidence_ids"])
    used_evidence = set(pair["evidence_ids_used"])
    if not used_evidence.issubset(allowed_evidence):
        reasons.append("generated_evidence_ids_not_subset")
    if task["task"].startswith("medical") and not used_evidence:
        reasons.append("medical_pair_missing_evidence_use")
    if len(pair["chosen"]) > 12000 or len(pair["rejected"]) > 12000:
        reasons.append("answer_over_12000_characters")
    return reasons


def _quarantine(
    task: dict[str, Any],
    reasons: list[str],
    *,
    pair: dict[str, Any] | None = None,
    judgment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "task_id": task["id"],
        "prompt_group_id": task["prompt_group_id"],
        "reasons": reasons,
    }
    if pair is not None:
        result["pair"] = pair
    if judgment is not None:
        result["judgment"] = judgment
    return result


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
