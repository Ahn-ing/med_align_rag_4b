"""从冻结 validation 确定性构建 Phase 5 loss-only 子集。"""

from __future__ import annotations

import json
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.evaluation import (
    SAFETY_CATEGORIES,
    SELECTION_SEED,
    TASK_QUOTAS,
    build_loss_records,
    distribution,
)
from med_align_rag_4b.sft.run_manifest import write_manifest_atomic


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "data/evaluation/validation_v1/cases.jsonl"
    source_manifest = root / "data/evaluation/validation_v1/manifest.json"
    output = root / "data/evaluation/phase5_loss_v1.jsonl"
    output_manifest = output.with_suffix(".manifest.json")
    frozen = json.loads(source_manifest.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or frozen.get("split") != "validation":
        raise ValueError("Phase 3 validation 未冻结")
    records = build_loss_records(list(iter_jsonl(source)))
    write_jsonl_atomic(output, records)
    manifest = {
        "format_version": 1,
        "dataset_version": "phase5_loss_v1",
        "status": "approved_for_phase5_loss",
        "purpose": "loss_only_engineering_validation",
        "generation_allowed": False,
        "gradient_update_allowed": False,
        "records": len(records),
        "source": {
            "split": "validation",
            "path": "data/evaluation/validation_v1/cases.jsonl",
            "sha256": sha256_file(source),
            "manifest_sha256": sha256_file(source_manifest),
        },
        "selection": {
            "seed": SELECTION_SEED,
            "method": "quota_then_sha256_seed_colon_stable_id",
            "task_quotas": TASK_QUOTAS,
            "safety_categories": sorted(SAFETY_CATEGORIES),
            "selected_ids": [row["id"] for row in records],
        },
        "task_distribution": distribution(records, "task"),
        "category_distribution": distribution(records, "primary_category"),
        "risk_distribution": distribution(records, "risk_level"),
        "data": {
            "path": "data/evaluation/phase5_loss_v1.jsonl",
            "sha256": sha256_file(output),
        },
        "limitations": [
            "仅用于 Phase 5 无梯度 validation loss 工程验收",
            "不运行生成式能力评测，不作为医疗质量结论",
            "不参与参数更新、调参或训练数据修改",
        ],
    }
    write_manifest_atomic(output_manifest, manifest)
    print(json.dumps({"status": "passed", "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
