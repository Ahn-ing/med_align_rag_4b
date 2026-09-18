"""调用 API 生成 Phase 7 v2 的 50 条 style pilot。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.api_config import load_api_config
from med_align_rag_4b.dpo.api_generation import (
    evidence_context,
    generation_messages_v2,
    judge_messages_v2,
    parse_generation_content,
    parse_judge_content,
    request_json_completion,
)
from med_align_rag_4b.dpo.manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)
from med_align_rag_4b.dpo.style import (
    pair_penalty,
    style_gate_failures,
    style_metrics,
    style_tag_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/dpo/phase7_api_v2.yaml")
    parser.add_argument("--tasks", default="data/dpo/pilot_v2/tasks.jsonl")
    parser.add_argument(
        "--evidence-registry",
        default="data/sft/sources/evidence_registry_v5_scale.json",
    )
    parser.add_argument("--output-dir", default="data/dpo/pilot_v2/generated")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-interval-seconds", type=float)
    parser.add_argument("--budget-profile", choices=("pilot", "full"), default="full")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--endpoint")
    parser.add_argument("--generation-model")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--wire-api",
        choices=("chat_completions", "responses"),
    )
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
    if args.wire_api:
        api_config["wire_api"] = args.wire_api
    if args.request_interval_seconds is not None:
        if args.request_interval_seconds < 0:
            raise ValueError("--request-interval-seconds 不得为负数")
        api_config["pilot"]["request_interval_seconds"] = args.request_interval_seconds
    if args.stream is not None:
        api_config["generation"]["stream"] = args.stream
        api_config["judge"]["stream"] = args.stream
    api_key = os.environ.get(api_config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"环境变量 {api_config['api_key_env']} 未设置；请先配置 API key"
        )

    all_tasks = list(iter_jsonl(tasks_path))
    if args.offset < 0 or args.offset > len(all_tasks):
        raise ValueError("--offset 超出任务范围")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正数")
    stop = None if args.limit is None else args.offset + args.limit
    tasks = all_tasks[args.offset : stop]
    if not tasks:
        raise ValueError("offset/limit 未选择任何任务")

    evidence_registry = json.loads(evidence_registry_path.read_text(encoding="utf-8"))
    budget = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    selected: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    selected_chosen: list[str] = []

    with tqdm(total=len(tasks), desc="Phase 7 v2 style pilot", unit="pair") as progress:
        for task in tasks:
            task_candidates = []
            profile = _profile_for(api_config, task["task"])
            evidence = evidence_context(evidence_registry, task["evidence_ids"])
            for candidate_index in range(profile["candidates"]):
                style_tag = style_tag_for(task["task"], candidate_index)
                generation_request = generation_messages_v2(
                    task,
                    evidence,
                    style_tag=style_tag,
                    candidate_index=candidate_index,
                )
                try:
                    pair, generation_meta = _request(
                        messages=generation_request,
                        parser=parse_generation_content,
                        section={**api_config["generation"], **profile},
                        budget_profile=args.budget_profile,
                        api_config=api_config,
                        api_key=api_key,
                        budget=budget,
                    )
                except Exception as exc:
                    quarantine.append(_quarantine(task, [f"generation_error:{exc}"]))
                    continue

                reasons = _pair_rejection_reasons(task, pair)
                candidate = _candidate_record(
                    task,
                    pair,
                    style_tag=style_tag,
                    candidate_index=candidate_index,
                    generation_request=generation_request,
                    generation_meta=generation_meta,
                )
                if reasons:
                    candidate["selected"] = False
                    candidate["rejection_reasons"] = reasons
                    all_candidates.append(candidate)
                    quarantine.append(_quarantine(task, reasons, pair=pair))
                    continue

                judge_request = judge_messages_v2(task, pair, evidence)
                judgment, judge_meta = _request(
                    messages=judge_request,
                    parser=parse_judge_content,
                    section=api_config["judge"],
                    budget_profile=args.budget_profile,
                    api_config=api_config,
                    api_key=api_key,
                    budget=budget,
                )
                candidate["judge"] = _judge_metadata(
                    judge_request, judge_meta, judgment
                )
                if not judgment["accepted"] or judgment["preferred"] != "chosen":
                    candidate["selected"] = False
                    candidate["rejection_reasons"] = ["judge_rejected_pair"]
                    all_candidates.append(candidate)
                    quarantine.append(
                        _quarantine(
                            task,
                            ["judge_rejected_pair"],
                            pair=pair,
                            judgment=judgment,
                        )
                    )
                    continue
                task_candidates.append(candidate)

            if task_candidates:
                for candidate in task_candidates:
                    candidate["style_penalty"] = pair_penalty(
                        {
                            "chosen": candidate["chosen"],
                            "rejected": candidate["rejected"],
                        },
                        selected_chosen,
                    )
                best = min(task_candidates, key=lambda item: item["style_penalty"])
                best["selected"] = True
                selected.append(
                    _selected_record(best, candidate_count=len(task_candidates))
                )
                selected_chosen.append(best["chosen"])
                all_candidates.extend(task_candidates)
            progress.update(1)

    style_report = _style_report(selected)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase7_v2_pilot.", dir=output_dir.parent))
    try:
        candidates_path = staging / "candidates.jsonl"
        all_candidates_path = staging / "all_candidates.jsonl"
        quarantine_path = staging / "quarantine.jsonl"
        style_path = staging / "style_report.json"
        write_jsonl_atomic(candidates_path, selected)
        write_jsonl_atomic(all_candidates_path, all_candidates)
        write_jsonl_atomic(quarantine_path, quarantine)
        write_manifest_atomic(style_path, style_report)
        manifest = initial_manifest(
            experiment_type="data_release",
            payload={
                "purpose": "phase_7_v2_style_pilot_api_generation",
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
                    "generation_profiles": api_config["generation"]["profiles"],
                    "api_key_env": api_config["api_key_env"],
                    "api_key_persisted": False,
                    "budget_profile": args.budget_profile,
                    "call_budget": api_config[args.budget_profile]["call_budget"],
                    "total_token_budget": api_config[args.budget_profile][
                        "total_token_budget"
                    ],
                },
                "counts": {
                    "tasks_total": len(all_tasks),
                    "tasks_selected": len(tasks),
                    "pairs_selected": len(selected),
                    "candidates_generated": len(all_candidates),
                    "quarantined": len(quarantine),
                },
                "usage": budget,
                "style_report": _descriptor(style_path, staging, records=len(selected)),
                "executed_actions": {
                    "commercial_api_called": budget["calls"] > 0,
                    "preference_pairs_generated": len(selected) > 0,
                    "model_loaded": False,
                    "dpo_training_started": False,
                },
                "release_allowed": False,
            },
        )
        expected_pairs = len(tasks)
        final_status = "completed" if len(selected) == expected_pairs else "failed"
        updates = (
            {}
            if final_status == "completed"
            else {"failure_reason": "incomplete_pair_count"}
        )
        completed = finish_manifest(manifest, final_status, **updates)
        write_manifest_atomic(staging / "manifest.json", completed)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.name.startswith(".phase7_v2_pilot."):
            shutil.rmtree(staging)
        raise

    if final_status != "completed":
        raise RuntimeError(
            f"v2 pilot 未完成: selected={len(selected)}, expected={len(tasks)}, "
            f"quarantined={len(quarantine)}"
        )

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "selected": len(selected),
                "quarantined": len(quarantine),
                "style_status": style_report["status"],
                "style_failures": style_report["failures"],
                "usage": budget,
            },
            ensure_ascii=False,
        )
    )


def _profile_for(api_config: dict[str, Any], task: str) -> dict[str, Any]:
    profile = api_config["generation"]["profiles"].get(task)
    if not isinstance(profile, dict):
        raise ValueError(f"缺少 generation profile: {task}")
    return profile


def _request(
    *,
    messages: list[dict[str, str]],
    parser: Any,
    section: dict[str, Any],
    budget_profile: str,
    api_config: dict[str, Any],
    api_key: str,
    budget: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    budget_settings = api_config[budget_profile]
    remaining_calls = budget_settings["call_budget"] - budget["calls"]
    if remaining_calls <= 0:
        raise RuntimeError("API 调用预算已耗尽")
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
        wire_api=api_config.get("wire_api", "chat_completions"),
        stream=bool(section.get("stream", False)),
        stream_include_usage=bool(
            (section.get("stream_options") or {}).get("include_usage", True)
        ),
    )
    interval = float(api_config["pilot"].get("request_interval_seconds", 0.0))
    if interval > 0:
        time.sleep(interval)
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


def _candidate_record(
    task: dict[str, Any],
    pair: dict[str, Any],
    *,
    style_tag: str,
    candidate_index: int,
    generation_request: list[dict[str, str]],
    generation_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": task["id"].replace("dpo_prompt_", "dpo_pair_", 1),
        "candidate_id": f"{task['id']}_c{candidate_index}",
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
        "style_tag": style_tag,
        "candidate_index": candidate_index,
        "review_status": "pending_agent_review",
        "reviewer_role": None,
        "reviewed_at_utc": None,
        "evidence_checked": False,
        "chosen_source": (
            "api_evidence_varied_v2"
            if task["task"].startswith("medical")
            else "api_generated_v2"
        ),
        "rejected_source": "api_hard_negative_v2",
        "contains_personal_data": False,
        "release_allowed": False,
        "status": "v2_api_generated_pending_review",
        "generation": {
            "model": generation_meta["response_model"],
            "response_id": generation_meta["response_id"],
            "usage": generation_meta["usage"],
            "prompt_sha256": sha256_text(
                json.dumps(generation_request, ensure_ascii=False, sort_keys=True)
            ),
        },
    }


def _judge_metadata(
    judge_request: list[dict[str, str]],
    judge_meta: dict[str, Any],
    judgment: dict[str, Any],
) -> dict[str, Any]:
    return {
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
    }


def _selected_record(
    candidate: dict[str, Any], *, candidate_count: int
) -> dict[str, Any]:
    result = dict(candidate)
    result.pop("style_penalty", None)
    result["candidate_count"] = candidate_count
    return result


def _style_report(selected: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_metrics = (
        style_metrics([row["chosen"] for row in selected]) if selected else None
    )
    rejected_metrics = (
        style_metrics([row["rejected"] for row in selected]) if selected else None
    )
    failures = (
        style_gate_failures(chosen_metrics) if chosen_metrics else ["no_selected_pairs"]
    )
    return {
        "format_version": 1,
        "phase": "phase_7",
        "kind": "phase_7_v2_style_pilot_report",
        "status": "passed" if not failures else "failed",
        "count": len(selected),
        "chosen": chosen_metrics,
        "rejected": rejected_metrics,
        "failures": failures,
        "thresholds": {
            "who_rate": 0.15,
            "citation_rate": 0.10,
            "citation_opening_rate": 0.05,
            "fixed_phrase_rate": 0.10,
            "max_first_12_share": 0.10,
        },
        "limitations": [
            "style gate 不判断医学事实或安全性",
            "pilot 通过后才允许进入全量 v2 生成",
        ],
    }


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
