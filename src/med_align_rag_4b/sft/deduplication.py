"""不损伤医学含义的规范化、稳定 ID 和重复指纹。"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from med_align_rag_4b.common.hashing import sha256_record, sha256_text

_SPACE = re.compile(r"\s+")
_PUNCTUATION = str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")"})


def normalize_text(text: str) -> str:
    """仅做 NFKC、常见标点、空白和英文大小写规范化。"""
    normalized = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    return _SPACE.sub(" ", normalized).strip().casefold()


def question_text(record: dict[str, Any]) -> str:
    return "\n".join(m["content"] for m in record["messages"] if m["role"] == "user")


def answer_text(record: dict[str, Any]) -> str:
    return "\n".join(m["content"] for m in record["messages"] if m["role"] == "assistant")


def question_fingerprint(text: str) -> str:
    return sha256_text(normalize_text(text))


def qa_fingerprint(question: str, answer: str) -> str:
    return sha256_record([normalize_text(question), normalize_text(answer)])


def stable_record_id(source_id: str, source_record_id: str, question: str, answer: str) -> str:
    """生成不依赖输入顺序的稳定记录 ID。"""
    return "sft_" + sha256_record([source_id, source_record_id, normalize_text(question), normalize_text(answer)])[:24]


def duplicate_groups(records: Iterable[dict[str, Any]]) -> dict[str, list[list[str]]]:
    """返回规范化 QA 重复、问题重复及同问异答组。"""
    by_question: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_qa: dict[str, list[str]] = defaultdict(list)
    for record in records:
        question, answer = question_text(record), answer_text(record)
        qfp, qafp = question_fingerprint(question), qa_fingerprint(question, answer)
        by_question[qfp].append((record["id"], qafp))
        by_qa[qafp].append(record["id"])
    return {
        "normalized_qa_duplicates": [ids for ids in by_qa.values() if len(ids) > 1],
        "normalized_question_duplicates": [[item[0] for item in rows] for rows in by_question.values() if len(rows) > 1],
        "conflicting_answers": [[item[0] for item in rows] for rows in by_question.values() if len({item[1] for item in rows}) > 1],
    }


def overlap_ids(records: Iterable[dict[str, Any]], blocked_question_fingerprints: set[str]) -> list[str]:
    return [record["id"] for record in records if question_fingerprint(question_text(record)) in blocked_question_fingerprints]
