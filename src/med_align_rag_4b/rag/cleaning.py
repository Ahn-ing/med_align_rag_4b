"""将 Huatuo 问答 JSONL 清洗为稳定的 RAG corpus。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.jsonl import write_jsonl_atomic
from med_align_rag_4b.rag.analysis import extract_qa, iter_records


@dataclass
class CleanStats:
    total_records: int = 0
    cleaned_records: int = 0
    invalid_json: int = 0
    malformed_questions: int = 0
    malformed_answers: int = 0
    empty_questions: int = 0
    empty_answers: int = 0
    duplicate_qa: int = 0

    @property
    def retention_rate(self) -> float:
        return self.cleaned_records / self.total_records * 100 if self.total_records else 0.0


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def iter_clean_records(path: str | Path, stats: CleanStats):
    seen: set[tuple[str, str]] = set()
    for record in iter_records(path):
        stats.total_records += 1
        if record is None:
            stats.invalid_json += 1
            continue
        raw_question, raw_answer = extract_qa(record)
        question, answer = clean_text(raw_question), clean_text(raw_answer)
        if question is None:
            if isinstance(raw_question, str):
                stats.empty_questions += 1
            else:
                stats.malformed_questions += 1
            continue
        if answer is None:
            if isinstance(raw_answer, str):
                stats.empty_answers += 1
            else:
                stats.malformed_answers += 1
            continue
        key = (question, answer)
        if key in seen:
            stats.duplicate_qa += 1
            continue
        seen.add(key)
        stats.cleaned_records += 1
        yield {"id": f"huatuo_{stats.cleaned_records:06d}", "question": question, "answer": answer}


def clean_dataset(input_path: str | Path, output_path: str | Path, *, overwrite: bool = False) -> CleanStats:
    stats = CleanStats()
    write_jsonl_atomic(output_path, iter_clean_records(input_path, stats), overwrite=overwrite)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(clean_dataset(args.input, args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
