"""从已审核候选冻结评测集；不会生成或补写案例。"""

from __future__ import annotations

import argparse
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.evaluation.freezing import freeze_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test", "medical_safety_holdout"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--review-registry", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    rows = list(iter_jsonl(args.candidates))
    freeze_split(args.output_dir, rows, split=args.split, seed=args.seed, source_registry_sha256=sha256_file(args.source_registry), review_registry_sha256=sha256_file(args.review_registry), rubric_sha256=sha256_file(args.rubric), evidence_registry_sha256=sha256_file(args.evidence_registry), created_at_utc=args.created_at_utc)


if __name__ == "__main__":
    main()
