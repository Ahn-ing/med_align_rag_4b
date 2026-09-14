"""流式分析原始或清洗后的问答 JSONL。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = True, truncation: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class AnalysisReport:
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_questions: int
    duplicate_qa: int
    questions_with_multiple_answers: int
    question_chars: dict[str, float | int | None]
    answer_chars: dict[str, float | int | None]
    answer_tokens: dict[str, float | int | None] | None = None


def extract_qa(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """兼容平铺记录以及 Huatuo 原始嵌套问答格式。"""
    if "question" in record or "answer" in record:
        question, answer = record.get("question"), record.get("answer")
    else:
        questions, answers = record.get("questions"), record.get("answers")
        question = questions[0][0] if isinstance(questions, list) and questions and isinstance(questions[0], list) and questions[0] else None
        answer = answers[0] if isinstance(answers, list) and answers else None
    return (question if isinstance(question, str) else None, answer if isinstance(answer, str) else None)


def iter_records(path: str | Path) -> Iterator[dict[str, Any] | None]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            yield value if isinstance(value, dict) else None


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def analyze_records(records: Iterable[dict[str, Any] | None], *, tokenizer: Tokenizer | None = None) -> AnalysisReport:
    total = invalid = 0
    questions: list[str] = []
    answers: list[str] = []
    for record in records:
        total += 1
        if record is None:
            invalid += 1
            continue
        question, answer = extract_qa(record)
        if not question or not question.strip() or not answer or not answer.strip():
            invalid += 1
            continue
        questions.append(question)
        answers.append(answer)
    question_counts = Counter(questions)
    qa_counts = Counter(zip(questions, answers, strict=True))
    grouped: dict[str, set[str]] = defaultdict(set)
    for question, answer in zip(questions, answers, strict=True):
        grouped[question].add(answer)
    token_stats = None
    if tokenizer is not None:
        token_stats = _stats([len(tokenizer.encode(value, add_special_tokens=True, truncation=False)) for value in answers])
    return AnalysisReport(
        total, len(questions), invalid,
        sum(value - 1 for value in question_counts.values() if value > 1),
        sum(value - 1 for value in qa_counts.values() if value > 1),
        sum(len(values) > 1 for values in grouped.values()),
        _stats([len(value) for value in questions]), _stats([len(value) for value in answers]), token_stats,
    )


def analyze_dataset(path: str | Path, *, tokenizer: Tokenizer | None = None) -> AnalysisReport:
    return analyze_records(iter_records(path), tokenizer=tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(asdict(analyze_dataset(args.input)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
