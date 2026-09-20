"""生成完整 RAG 消融的退化审核数据和详细 Markdown。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

CONTRASTS = (
    ("R0", "R1"),
    ("R2", "R3"),
    ("R4", "R5"),
    ("R0", "R2"),
    ("R2", "R4"),
    ("R1", "R3"),
    ("R3", "R5"),
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/phase8_rag.yaml"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4"))
    parser.add_argument("--judge-scores", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4/judge_scores_v4.jsonl"))
    parser.add_argument("--queue", type=Path, default=Path("artifacts/evaluations/phase_8_rag/formal_v4/degradation_review_queue_v4.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/phase_8_rag/formal_v4_degradation_review_v4.json"))
    parser.add_argument("--markdown", type=Path, default=Path("artifacts/reports/phase_8_rag/formal_v4_degradation_review_v4_detailed.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_rag_ablation_config(root / args.config, root)
    cases, cases_sha256 = load_validation_cases(root, config["dataset"])

    evaluation_root = root / args.evaluation_root
    outputs = {arm: {row["case_id"]: row for row in iter_jsonl(evaluation_root / f"{arm}.jsonl")} for arm in ARM_ORDER}
    judge = {(row["arm"], row["case_id"]): row for row in iter_jsonl(root / args.judge_scores)}
    rule = {(row["arm"], row["case_id"]): row for row in iter_jsonl(evaluation_root / "rule_scores.jsonl")}
    items = []
    for source, target in CONTRASTS:
        for case in cases:
            case_id = case["id"]
            sj, tj = judge[(source, case_id)]["result"], judge[(target, case_id)]["result"]
            sr, tr = rule[(source, case_id)], rule[(target, case_id)]
            reasons = []
            if tj["score_0_to_2"] < sj["score_0_to_2"]:
                reasons.append("judge_score_decrease")
            if tr["hard_fail"] and not sr["hard_fail"]:
                reasons.append("new_rule_hard_fail")
            if tr["truncated"] and not sr["truncated"]:
                reasons.append("new_truncation")
            if not reasons:
                continue
            items.append({
                "format_version": 1,
                "phase": 8,
                "review_status": "pending_manual_review",
                "contrast": f"{source}->{target}",
                "source_arm": source,
                "target_arm": target,
                "case_id": case_id,
                "task": case["task"],
                "primary_category": case["primary_category"],
                "risk_level": case["risk_level"],
                "degradation_reasons": reasons,
                "judge_score_source": sj["score_0_to_2"],
                "judge_score_target": tj["score_0_to_2"],
                "source_judge": sj,
                "target_judge": tj,
                "source_rule": sr,
                "target_rule": tr,
                "prompt": case["prompt"],
                "reference_answer": case["reference_answer"],
                "source_output": outputs[source][case_id]["output"],
                "target_output": outputs[target][case_id]["output"],
                "source_rag": outputs[source][case_id]["rag"],
                "target_rag": outputs[target][case_id]["rag"],
            })
    write_jsonl_atomic(root / args.queue, items)
    reason_counts = Counter(reason for row in items for reason in row["degradation_reasons"])
    report = {
        "format_version": 1,
        "phase": 8,
        "kind": "phase8_full_rag_degradation_review",
        "status": "pending_manual_review",
        "cases_sha256": cases_sha256,
        "records": len(items),
        "by_contrast": dict(sorted(Counter(row["contrast"] for row in items).items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "queue": {"path": portable_relative_path(root / args.queue, root), "sha256": sha256_file(root / args.queue)},
    }
    _write_json(root / args.report, report)
    lines = ["# Phase 8 完整 RAG 退化人工审核详情", "", f"共 {len(items)} 条退化记录。", ""]
    for index, row in enumerate(items, 1):
        lines.extend([
            f"## {index}. {row['contrast']} | {row['case_id']}",
            "",
            f"- Task: `{row['task']}`",
            f"- Risk: `{row['risk_level']}`",
            f"- 原因: {', '.join(row['degradation_reasons'])}",
            f"- Judge: `{row['judge_score_source']} -> {row['judge_score_target']}`",
            "",
            "### 原问题",
            "",
            row["prompt"],
            "",
            "### 参考回答",
            "",
            row["reference_answer"],
            "",
            f"### Source 回答：{row['source_arm']}",
            "",
            row["source_output"],
            "",
            f"### Target 回答：{row['target_arm']}",
            "",
            row["target_output"],
            "",
            "### Source judge",
            "",
            f"- score: `{row['source_judge']['score_0_to_2']}`",
            f"- violations: {json.dumps(row['source_judge']['violations'], ensure_ascii=False)}",
            f"- uncertainty: {row['source_judge']['uncertainty'] or '无'}",
            "",
            "### Target judge",
            "",
            f"- score: `{row['target_judge']['score_0_to_2']}`",
            f"- violations: {json.dumps(row['target_judge']['violations'], ensure_ascii=False)}",
            f"- uncertainty: {row['target_judge']['uncertainty'] or '无'}",
            "",
            "### 规则变化",
            "",
            f"- hard_fail: `{row['source_rule']['hard_fail']} -> {row['target_rule']['hard_fail']}`",
            f"- truncated: `{row['source_rule']['truncated']} -> {row['target_rule']['truncated']}`",
            "",
            "### 人工审核",
            "",
            "- [ ] 真实退化",
            "- [ ] judge 抖动",
            "- [ ] 规则误判",
            "- [ ] 需要额外证据",
            "",
            "审核意见：",
            "",
            "---",
            "",
        ])
    markdown_path = root / args.markdown
    if markdown_path.exists():
        raise FileExistsError(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": len(items), "by_contrast": report["by_contrast"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()