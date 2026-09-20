"""完整 RAG 组件消融的 CPU 侧配置、消息和记录契约。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.evaluation.datasets import validate_case
from med_align_rag_4b.rag.prompting import SYSTEM_INSTRUCTION, build_prompt

ARM_ORDER = ("R0", "R1", "R2", "R3", "R4", "R5")
EXPECTED_ARMS = {
    "R0": ("base", None, False),
    "R1": ("base", None, True),
    "R2": ("base", "sft", False),
    "R3": ("base", "sft", True),
    "R4": ("merged_sft", "dpo_v2", False),
    "R5": ("merged_sft", "dpo_v2", True),
}


@dataclass(frozen=True)
class RAGAblationArm:
    arm_id: str
    model_source: str
    adapter: str | None
    rag: bool
    purpose: str


def load_rag_ablation_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("Phase 8 RAG 配置格式无效")
    if value.get("kind") != "phase8_rag_ablation_config":
        raise ValueError("Phase 8 RAG 配置 kind 无效")
    if value.get("base_system_prompt") != "你是一个谨慎、清晰且遵循证据边界的助手。":
        raise ValueError("base_system_prompt 已冻结")
    dataset = value.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("split") != "validation":
        raise ValueError("RAG 消融主评测只能使用 validation")
    for key in ("cases", "manifest"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError(f"dataset.{key} 无效")
    rag = value.get("rag")
    if not isinstance(rag, dict):
        raise ValueError("rag 配置必须是对象")
    if rag.get("controlled_context_for_required") is not True:
        raise ValueError("RAG 必需案例必须使用冻结 Context")
    for key in ("manifest",):
        if not isinstance(rag.get(key), str) or not rag[key]:
            raise ValueError(f"rag.{key} 无效")
    for key in ("top_k", "max_context_chars"):
        if not isinstance(rag.get(key), int) or rag[key] <= 0:
            raise ValueError(f"rag.{key} 必须为正整数")
    arms = value.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != ARM_ORDER:
        raise ValueError("arms 必须按 R0-R5 冻结")
    for arm_id in ARM_ORDER:
        arm = arms.get(arm_id)
        if not isinstance(arm, dict):
            raise ValueError(f"{arm_id} 必须是对象")
        actual = (arm.get("model_source"), arm.get("adapter"), arm.get("rag"))
        if actual != EXPECTED_ARMS[arm_id]:
            raise ValueError(f"{arm_id} 与冻结矩阵不匹配")
    return value


def get_arm(config: Mapping[str, Any], arm_id: str) -> RAGAblationArm:
    if arm_id not in ARM_ORDER:
        raise ValueError(f"未知 arm: {arm_id}")
    value = config["arms"][arm_id]
    return RAGAblationArm(
        arm_id=arm_id,
        model_source=value["model_source"],
        adapter=value["adapter"],
        rag=value["rag"],
        purpose=value["purpose"],
    )


def load_validation_cases(project_root: str | Path, dataset: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    root = Path(project_root).resolve()
    cases_path = resolve_project_path(root, dataset["cases"])
    manifest_path = resolve_project_path(root, dataset["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "validation" or manifest.get("status") != "frozen":
        raise ValueError("validation manifest 无效")
    if manifest.get("data", {}).get("path") != dataset["cases"]:
        raise ValueError("validation manifest 路径不匹配")
    digest = sha256_file(cases_path)
    if manifest.get("data", {}).get("sha256") != digest:
        raise ValueError("validation 内容哈希不匹配")
    rows = list(iter_jsonl(cases_path))
    if len(rows) != manifest.get("records"):
        raise ValueError("validation 记录数不匹配")
    for row in rows:
        validate_case(row)
        if row.get("split") != "validation":
            raise ValueError("RAG 消融拒绝非 validation 案例")
    return rows, digest


def build_no_rag_messages(case: Mapping[str, Any], system_prompt: str) -> list[dict[str, str]]:
    prompt = case.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("case prompt 无效")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.strip()},
    ]


def _controlled_result(case: Mapping[str, Any]) -> dict[str, Any]:
    context = case.get("context")
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"{case.get('id')} 缺少冻结 Context")
    return {
        "index": -1,
        "score": 1.0,
        "text": context.strip(),
        "metadata": {
            "chunk_id": "controlled_case_context",
            "question": case["id"],
        },
    }


def build_rag_messages(
    case: Mapping[str, Any],
    retrieve: Callable[[str, int], Mapping[str, Any]],
    *,
    top_k: int,
    max_context_chars: int,
    controlled_context_for_required: bool,
) -> tuple[list[dict[str, str]], str, str, list[dict[str, Any]]]:
    if case.get("rag_requirement") == "required" and controlled_context_for_required:
        result = _controlled_result(case)
        results = [result]
        source = "controlled_case_context"
    else:
        retrieval = retrieve(case["prompt"], top_k)
        results = retrieval.get("results")
        if not isinstance(results, list):
            raise ValueError("retriever.results 必须是列表")
        source = "frozen_retriever"
    payload = build_prompt(case["prompt"], results, max_context_chars=max_context_chars)
    if payload.messages[0]["content"] != SYSTEM_INSTRUCTION:
        raise ValueError("RAG ON 必须使用 rag.prompting.SYSTEM_INSTRUCTION")
    evidence = [
        {
            "rank": rank,
            "row_id": int(item.get("index", -1)),
            "score": float(item.get("score", 0.0)),
            "chunk_id": item.get("metadata", {}).get("chunk_id"),
            "text_sha256": sha256_text(str(item.get("text", ""))),
        }
        for rank, item in enumerate(results, start=1)
    ]
    return payload.messages, payload.context, source, evidence


def generate_case_record(
    case: Mapping[str, Any],
    *,
    arm: RAGAblationArm,
    base_system_prompt: str,
    generation_config: Any,
    generate: Callable[[list[dict[str, str]], Any], Mapping[str, Any]],
    retrieve: Callable[[str, int], Mapping[str, Any]] | None = None,
    rag_top_k: int = 5,
    max_context_chars: int = 12000,
    controlled_context_for_required: bool = True,
) -> dict[str, Any]:
    context: str | None = None
    source = "disabled"
    evidence: list[dict[str, Any]] = []
    if arm.rag:
        if retrieve is None:
            raise ValueError(f"{arm.arm_id} 需要 retriever")
        messages, context, source, evidence = build_rag_messages(
            case,
            retrieve,
            top_k=rag_top_k,
            max_context_chars=max_context_chars,
            controlled_context_for_required=controlled_context_for_required,
        )
    else:
        messages = build_no_rag_messages(case, base_system_prompt)
    started = time.perf_counter()
    generated = generate(messages, generation_config)
    elapsed = time.perf_counter() - started
    if not isinstance(generated, Mapping) or not isinstance(generated.get("answer"), str):
        raise ValueError("generator 返回无效")
    output = generated["answer"]
    return {
        "format_version": 1,
        "phase": 8,
        "arm": arm.arm_id,
        "case_id": case["id"],
        "task": case["task"],
        "primary_category": case["primary_category"],
        "risk_level": case["risk_level"],
        "prompt": case["prompt"],
        "messages": messages,
        "system_prompt": messages[0]["content"],
        "output": output,
        "output_sha256": sha256_text(output),
        "generation": generated.get("generation_config", {}),
        "generated_tokens": generated.get("generated_tokens"),
        "finish_reason": generated.get("finish_reason"),
        "truncated": bool(generated.get("truncated", False)),
        "runtime": {"latency_seconds": elapsed},
        "rag": {
            "enabled": arm.rag,
            "context_source": source,
            "top_k": rag_top_k if arm.rag else None,
            "max_context_chars": max_context_chars if arm.rag else None,
            "context": context,
            "evidence": evidence,
            "uses_rag_system_prompt": bool(arm.rag and messages[0]["content"] == SYSTEM_INSTRUCTION),
        },
    }


def limit_cases(cases: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    rows = list(cases)
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError("limit 必须为正数")
    return rows[:limit]