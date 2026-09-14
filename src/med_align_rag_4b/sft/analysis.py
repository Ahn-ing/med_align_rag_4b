"""候选记录的确定性只读统计。"""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from med_align_rag_4b.sft.deduplication import (
    answer_text,
    duplicate_groups,
    question_text,
)


def analyze_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总字段分布、长度和确定性重复；不调用 tokenizer 或模型。"""
    rows = list(records)
    duplicates = duplicate_groups(rows)
    lengths = [len(question_text(row)) + len(answer_text(row)) for row in rows]
    return {
        "records": len(rows),
        "empty_questions": sum(not question_text(row).strip() for row in rows),
        "empty_answers": sum(not answer_text(row).strip() for row in rows),
        "language_distribution": dict(sorted(Counter(row["language"] for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "risk_distribution": dict(sorted(Counter(row["risk_level"] for row in rows).items())),
        "review_distribution": dict(sorted(Counter(row["review"]["status"] for row in rows).items())),
        "license_distribution": dict(sorted(Counter(row["license_status"] for row in rows).items())),
        "missing_medical_evidence": sum(row["domain"].startswith("medical") and not row["medical_evidence"] for row in rows),
        "character_length": {"min": min(lengths, default=0), "max": max(lengths, default=0), "mean": sum(lengths) / len(lengths) if lengths else 0},
        **{name: len(groups) for name, groups in duplicates.items()},
    }
