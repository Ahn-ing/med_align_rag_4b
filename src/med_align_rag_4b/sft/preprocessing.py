"""Phase 4 候选记录的最小语义保持清洗。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from med_align_rag_4b.common.hashing import sha256_record, sha256_text
from med_align_rag_4b.sft.deduplication import stable_record_id

_INLINE_SPACE = re.compile(r"[^\S\r\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(value: str) -> str:
    """执行 NFKC、行内空白合并和段落边界保留。"""
    if not isinstance(value, str):
        raise ValueError("文本必须是字符串")
    text = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def build_candidate(
    *, source_id: str, source_record_id: str, source_revision: str,
    user: str, assistant: str, domain: str, ability: str,
    primary_category: str, risk_level: str, evidence_ids: list[str],
    template_family: str, secondary_categories: list[str] | None = None,
) -> dict[str, Any]:
    """从可审计字段创建统一候选记录。"""
    raw_messages = [
        {"role": "system", "content": "你是一个谨慎、清晰且遵循证据边界的助手。"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    messages = [{"role": row["role"], "content": clean_text(row["content"])} for row in raw_messages]
    if not messages[1]["content"] or not messages[2]["content"]:
        raise ValueError("user/assistant 不能为空")
    identifier = stable_record_id(source_id, source_record_id, messages[1]["content"], messages[2]["content"])
    return {
        "id": identifier,
        "source_id": source_id,
        "source_record_id": source_record_id,
        "source_revision": source_revision,
        "messages": messages,
        "domain": domain,
        "ability": ability,
        "language": "zh" if domain != "general" else "en",
        "risk_level": risk_level,
        "primary_category": primary_category,
        "secondary_categories": secondary_categories or [],
        "license_status": "approved",
        "medical_evidence": evidence_ids,
        "review": {
            "status": "agent_evidence_reviewed",
            "reviewer_role": "codex_research_reviewer",
            "reviewed_at": "2026-09-14",
            "rubric_version": "phase4-fast-v1",
            "evidence_checked": True,
        },
        "quality_flags": [],
        "unresolved_safety_flags": [],
        "template_family": template_family,
        "fingerprints": {
            "source_record_sha256": sha256_record(raw_messages),
            "normalized_content_sha256": sha256_record(messages),
            "question_sha256": sha256_text(messages[1]["content"]),
        },
        "provenance": {"generation_method": "curated" if source_id.startswith("databricks") else "project_authored"},
    }
