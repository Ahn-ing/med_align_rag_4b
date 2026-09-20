"""计算完整 RAG 消融 R0-R5 的规则评分与配对摘要。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import portable_relative_path
from med_align_rag_4b.evaluation.rag_ablation import (
    ARM_ORDER,
    load_rag_ablation_config,
    load_validation_cases,
)
from med_align_rag_4b.evaluation.rag_scoring import aggregate_scores, score_record

CONTRASTS = (
    ("R0", "R1"),
    ("R2", "R3"),
    ("R4", "R5"),
    ("R0", "R2"),
    ("R2", "R4"),
    ("R1", "R3"),
    ("R3", "R5"),
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


def _metric(score: dict[str, Any], name: str) -> float | None:
    if name == "hard_fail":
        return 1.0 if score["hard_fail"] else 0.0
    if name == "structured_schema_pass":
        return None if score["json_schema_ok"] is None else float(score["json_schema_ok"])
    value = score.get(name)
    return None if value is None else float(value)


def _contrast(scores: dict[str, list[dict[str, Any]]], source: str, target: str) -> dict[str, Any]:
    source_by_id = {row["case_id"]: row for row in scores[source]}
    target_by_id = {row["case_id"]: row for row in scores[target]}
    definitions = {
        "hard_fail": ("hard_fail", "lower"),
        "literal_fact_hit_rate": ("literal_fact_hit_rate", "higher"),
        "structured_schema_pass": ("structured_schema_pass", "higher"),
        "insufficient_evidence_statement": ("insufficient_evidence_statement", "higher"),
        "generated_tokens": ("generated_tokens", "lower"),
    }
    result: dict[str, Any] = {}
    for metric, (field, direction) in definitions.items():
        deltas = []
        for case_id in sorted(source_by_id):
            left = _metric(source_by_id[case_id], field)
            right = _metric(target_by_id[case_id], field)
            if left is not None and right is not None:
                deltas.append(right - left)
        if not deltas:
            result[metric] = {"n": 0, "mean_delta_target_minus_source": None}
            continue
        improved = sum(delta < 0 for delta in deltas) if direction == "lower" else sum(delta > 0 for delta in deltas)
        worse = sum(delta > 0 for delta in deltas) if direction == "lower" else sum(delta < 0 for delta in deltas)
        result[metric] = {
            "n": len(deltas),
            "direction": direction,
            "mean_delta_target_minus_source": statistics.fmean(deltas),
            "improved_cases": improved,
            "worse_cases": worse,
            "tied_cases": len(deltas) - improved - worse,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/phase8_rag.yaml"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4"))
    parser.add_argument("--scores", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4/rule_scores.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/phase_8_rag/formal_v4_rules.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_rag_ablation_config(root / args.config, root)
    cases, cases_sha256 = load_validation_cases(root, config["dataset"])
    case_by_id = {case["id"]: case for case in cases}
    scores: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, Any] = {}
    for arm in ARM_ORDER:
        output = root / args.evaluation_root / f"{arm}.jsonl"
        manifest_path = root / args.evaluation_root / f"{arm}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = list(iter_jsonl(output))
        if len(rows) != manifest.get("records") or sha256_file(output) != manifest.get("output_sha256"):
            raise ValueError(f"{arm} manifest/hash 不一致")
        scores[arm] = [score_record(case_by_id[row["case_id"]], row) for row in rows]
        sources[arm] = {"output": portable_relative_path(output, root), "manifest": portable_relative_path(manifest_path, root)}
    score_rows = [score for arm in ARM_ORDER for score in scores[arm]]
    write_jsonl_atomic(root / args.scores, score_rows)
    report = {
        "format_version": 1,
        "phase": 8,
        "kind": "phase8_full_rag_rules",
        "status": "completed_pending_judge",
        "protocol": "full_rag_component",
        "cases_sha256": cases_sha256,
        "summary": {arm: aggregate_scores(scores[arm]) for arm in ARM_ORDER},
        "paired_contrasts": {
            f"{source}->{target}": _contrast(scores, source, target)
            for source, target in CONTRASTS
        },
        "sources": sources,
        "outputs": {
            "rule_scores": portable_relative_path(root / args.scores, root),
            "rule_scores_sha256": sha256_file(root / args.scores),
        },
        "limitations": ["规则结果不评价医学事实正确性或临床安全", "test 与 medical safety holdout 未读取"],
    }
    _write_json_atomic(root / args.report, report)
    print(json.dumps({"status": report["status"], "records": len(score_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()