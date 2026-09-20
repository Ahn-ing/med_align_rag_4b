"""运行完整 RAG 消融 R0-R5 的固定模型 judge。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import portable_relative_path, resolve_project_path
from med_align_rag_4b.dpo.api_config import load_api_config
from med_align_rag_4b.dpo.api_generation import request_json_completion
from med_align_rag_4b.evaluation.rag_ablation import (
    ARM_ORDER,
    load_rag_ablation_config,
    load_validation_cases,
)
from med_align_rag_4b.evaluation.rag_judge import (
    JUDGE_PROMPT_VERSION,
    judge_input_sha256,
    judge_messages,
    parse_judge_score,
)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_judge_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "phase8_rag_judge_config" or value.get("status") != "frozen":
        raise ValueError("judge 配置无效")
    source_path = resolve_project_path(root, value["source_api_config"])
    source = load_api_config(source_path)
    judge = value.get("judge")
    if not isinstance(judge, dict):
        raise ValueError("judge 配置缺少 judge")
    for key, expected in source["judge"].items():
        if key == "model":
            continue
        if judge.get(key) != expected:
            raise ValueError(f"judge.{key} 与 Phase 7 v2 不一致")
    if not isinstance(judge.get("model"), str) or not judge["model"].strip():
        raise ValueError("judge.model 无效")
    return value


def _completed_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    values: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (str(row["arm"]), str(row["case_id"]))
        if key in values:
            raise ValueError(f"partial judge 存在重复: {key}")
        values[key] = row
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/phase8_rag.yaml"))
    parser.add_argument("--judge-config", type=Path, default=Path("configs/evaluation/phase8_rag_judge.yaml"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4/judge_scores_v4.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4/judge_scores_v4.manifest.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_rag_ablation_config(root / args.config, root)
    judge_config = _load_judge_config(root / args.judge_config, root)
    cases, cases_sha256 = load_validation_cases(root, config["dataset"])
    case_by_id = {case["id"]: case for case in cases}
    api_key = os.environ.get(judge_config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"环境变量 {judge_config['api_key_env']} 未设置")
    all_tasks: list[tuple[str, dict[str, Any]]] = []
    for arm in ARM_ORDER:
        rows = list(iter_jsonl(root / args.evaluation_root / f"{arm}.jsonl"))
        if [row["case_id"] for row in rows] != [case["id"] for case in cases]:
            raise ValueError(f"{arm} case 顺序不一致")
        all_tasks.extend((arm, row) for row in rows)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须为正数")
        all_tasks = all_tasks[: args.limit]
    output = root / args.output
    manifest_path = root / args.manifest
    partial_path = output.with_name(f".{output.name}.partial")
    if output.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(output) != existing.get("output_sha256"):
            raise ValueError("已有 judge 输出哈希不匹配")
        print(json.dumps({"status": "skipped_complete", "records": existing.get("records")}, ensure_ascii=False))
        return
    if output.exists() or manifest_path.exists():
        raise FileExistsError("已有 judge 输出不完整，拒绝覆盖")
    completed = _completed_records(partial_path)
    remaining: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for arm, row in all_tasks:
        case = case_by_id[row["case_id"]]
        digest = judge_input_sha256(case, row)
        previous = completed.get((arm, row["case_id"]))
        if previous is not None:
            if previous.get("input_sha256") != digest:
                raise ValueError(f"{arm}/{row['case_id']} partial input hash 不一致")
            continue
        remaining.append((arm, row, case, digest))
    judge = judge_config["judge"]
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8", newline="\n") as stream:
        progress = tqdm(
            remaining,
            desc="Phase 8 full RAG judge",
            unit="case",
            dynamic_ncols=True,
            initial=len(completed),
            total=len(completed) + len(remaining),
        )
        for arm, row, case, digest in progress:
            parsed, metadata = request_json_completion(
                endpoint=judge_config["endpoint"],
                api_key=api_key,
                model=judge["model"],
                messages=judge_messages(case, row),
                temperature=judge["temperature"],
                top_p=judge["top_p"],
                max_tokens=judge["max_tokens"],
                response_format=judge["response_format"],
                timeout_seconds=judge["timeout_seconds"],
                max_attempts=judge["max_attempts"],
                parser=parse_judge_score,
                wire_api=judge_config["wire_api"],
                stream=judge.get("stream", False),
                stream_include_usage=judge.get("stream_options", {}).get("include_usage", True),
            )
            record = {
                "format_version": 1,
                "phase": 8,
                "arm": arm,
                "case_id": case["id"],
                "task": case["task"],
                "primary_category": case["primary_category"],
                "risk_level": case["risk_level"],
                "prompt_version": JUDGE_PROMPT_VERSION,
                "input_sha256": digest,
                "judge_model": judge["model"],
                "result": parsed,
                "response": metadata,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            interval = float(judge_config.get("request_interval_seconds", 0.0))
            if interval > 0:
                time.sleep(interval)
    final_rows = _completed_records(partial_path)
    expected_keys = {(arm, row["case_id"]) for arm, row in all_tasks}
    if not expected_keys.issubset(final_rows):
        missing = sorted(expected_keys - set(final_rows))
        raise RuntimeError(f"partial judge 不完整: {missing[:5]}")
    partial_path.replace(output)
    manifest = {
        "format_version": 1,
        "phase": 8,
        "kind": "phase8_full_rag_judge",
        "status": "completed_pending_agent_review",
        "protocol": "full_rag_component",
        "provider": judge_config["provider"],
        "endpoint": judge_config["endpoint"],
        "wire_api": judge_config["wire_api"],
        "judge_model": judge["model"],
        "prompt_version": JUDGE_PROMPT_VERSION,
        "cases_sha256": cases_sha256,
        "records": len(final_rows),
        "full_matrix_records": 360,
        "limit": args.limit,
        "config_sha256": sha256_file(root / args.judge_config),
        "output": portable_relative_path(output, root),
        "output_sha256": sha256_file(output),
        "release_allowed": False,
        "limitations": ["模型 judge 是辅助评分，不是医生审核或临床验证", "test 与 medical safety holdout 未读取"],
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "records": len(final_rows), "judge_model": judge["model"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()