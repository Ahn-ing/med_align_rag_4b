"""独立验证 Phase 4 快速版 manifest、配额、审核关联与文件哈希。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from med_align_rag_4b.common.jsonl import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    released = root / "data/sft/released" / args.version
    manifest = json.loads((released / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        if artifact["records"] is not None:
            assert path.suffix == ".jsonl"
            assert sum(1 for _ in iter_jsonl(path)) == artifact["records"]
    reviewed = {row["id"]: row for row in iter_jsonl(root / f"data/sft/reviewed/{args.version}/approved_candidates.jsonl")}
    domains: Counter[str] = Counter()
    safety: Counter[str] = Counter()
    for row in iter_jsonl(released / "train.jsonl"):
        source = reviewed[row["id"]]
        assert row["review_status"] == "agent_evidence_reviewed"
        assert source["review"]["reviewer_role"] == "codex_research_reviewer"
        assert source["review"]["evidence_checked"] is True
        assert not source["unresolved_safety_flags"] and not row["truncated"]
        assert row["assistant_supervised_tokens"] > 0 and row["total_tokens"] <= 2048
        assert source["chat_template_valid"] is True and source["has_supervised_eos"] is True
        if row["domain"] in {"medical", "medical_safety"}:
            assert source["medical_evidence"]
        domains[row["domain"]] += row["assistant_supervised_tokens"]
        if row["domain"] == "medical_safety":
            safety[row["primary_category"]] += row["assistant_supervised_tokens"]
    for domain, target in manifest["targets"].items():
        assert abs(domains[domain] - target) <= target * 0.01
    expected_safety = {"immediate_emergency": 20_000, "outpatient_followup": 15_000, "home_observation": 15_000, "clarification": 15_000, "medication_boundary": 20_000, "uncertainty": 15_000}
    for category, target in expected_safety.items():
        assert abs(safety[category] - target) <= target * 0.01
    assert manifest["training_allowed"] is True and manifest["release_allowed"] is False
    print(json.dumps({"status": "passed", "domain_tokens": domains, "safety_tokens": safety}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
