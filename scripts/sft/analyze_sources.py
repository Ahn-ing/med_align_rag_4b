"""只读分析已登记的 COIG-CQIA 与 shibing624/medical 原始文件。"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.sft.deduplication import (
    qa_fingerprint,
    question_fingerprint,
)


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """兼容 JSONL 和 JSON 数组；全程只读。"""
    with path.open("r", encoding="utf-8") as stream:
        first = stream.read(1)
        stream.seek(0)
        if first == "[":
            value = json.load(stream)
            if not isinstance(value, list):
                raise ValueError(f"顶层 JSON 必须是数组: {path}")
            yield from value
        else:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"第 {line_number} 行不是对象: {path}")
                yield value


def prompt_answer(record: dict[str, Any]) -> tuple[str, str]:
    instruction = record.get("instruction")
    extra = record.get("input")
    answer = record.get("output")
    if not all(isinstance(value, str) for value in (instruction, extra, answer)):
        raise ValueError("instruction/input/output 必须为字符串")
    return instruction + ("\n" + extra if extra else ""), answer


def _rag_fingerprints(path: Path) -> set[str]:
    values: set[str] = set()
    for record in iter_json_records(path):
        question = record.get("question", record.get("query"))
        if isinstance(question, str):
            values.add(question_fingerprint(question))
    return values


def analyze(paths: dict[str, Path], rag_path: Path, compatibility_path: Path) -> dict[str, Any]:
    """用临时 SQLite 精确统计重复、冲突、跨 split 和 RAG overlap。"""
    rag = _rag_fingerprints(rag_path)
    compatibility = _rag_fingerprints(compatibility_path)
    with tempfile.TemporaryDirectory(prefix="phase3_analysis_") as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "fingerprints.sqlite3")
        connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE records(source TEXT, split TEXT, raw_q TEXT, raw_qa TEXT, qfp TEXT, qafp TEXT);
        """)
        report: dict[str, Any] = {"format_version": 1, "method": "NFKC+whitespace+casefold exact fingerprints", "sources": {}}
        for label, path in paths.items():
            counts: Counter[str] = Counter()
            lengths_q: list[int] = []
            lengths_a: list[int] = []
            batch: list[tuple[str, str, str, str, str, str]] = []
            for record in iter_json_records(path):
                counts["records"] += 1
                try:
                    question, answer = prompt_answer(record)
                except (ValueError, TypeError):
                    counts["unparseable"] += 1
                    continue
                counts["empty_question"] += not question.strip()
                counts["empty_answer"] += not answer.strip()
                qfp = question_fingerprint(question)
                qafp = qa_fingerprint(question, answer)
                counts["rag_question_overlap"] += qfp in rag
                counts["compatibility_query_overlap"] += qfp in compatibility
                lengths_q.append(len(question))
                lengths_a.append(len(answer))
                batch.append((label.split(":", 1)[0], label, sha256_record(question), sha256_record([question, answer]), qfp, qafp))
                if len(batch) >= 10_000:
                    connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", batch)
            connection.commit()
            print(f"analyzed {label}: {counts['records']} records", flush=True)
            report["sources"][label] = {
                "path_argument": label,
                "sha256": sha256_file(path),
                **dict(counts),
                "question_characters": {"min": min(lengths_q, default=0), "max": max(lengths_q, default=0), "mean": sum(lengths_q) / len(lengths_q) if lengths_q else 0},
                "answer_characters": {"min": min(lengths_a, default=0), "max": max(lengths_a, default=0), "mean": sum(lengths_a) / len(lengths_a) if lengths_a else 0},
            }
        connection.executescript("CREATE INDEX idx_qfp ON records(qfp); CREATE INDEX idx_qafp ON records(qafp); CREATE INDEX idx_raw_q ON records(raw_q); CREATE INDEX idx_raw_qa ON records(raw_qa);")
        query = connection.execute
        report["duplicates"] = {
            "exact_qa_extra_records": query("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY raw_qa HAVING n>1)").fetchone()[0],
            "exact_question_extra_records": query("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY raw_q HAVING n>1)").fetchone()[0],
            "normalized_qa_extra_records": query("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY qafp HAVING n>1)").fetchone()[0],
            "normalized_question_extra_records": query("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY qfp HAVING n>1)").fetchone()[0],
            "conflicting_question_groups": query("SELECT COUNT(*) FROM (SELECT qfp FROM records GROUP BY qfp HAVING COUNT(DISTINCT qafp)>1)").fetchone()[0],
            "cross_source_question_groups": query("SELECT COUNT(*) FROM (SELECT qfp FROM records GROUP BY qfp HAVING COUNT(DISTINCT source)>1)").fetchone()[0],
            "cross_original_split_question_groups": query("SELECT COUNT(*) FROM (SELECT qfp FROM records GROUP BY qfp HAVING COUNT(DISTINCT split)>1)").fetchone()[0],
        }
        connection.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coig", type=Path, required=True)
    parser.add_argument("--medical-train", type=Path, required=True)
    parser.add_argument("--medical-validation", type=Path, required=True)
    parser.add_argument("--medical-test", type=Path, required=True)
    parser.add_argument("--rag-cleaned", type=Path, required=True)
    parser.add_argument("--compatibility-queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出已存在: {args.output}")
    result = analyze({"coig:train": args.coig, "medical:train": args.medical_train, "medical:validation": args.medical_validation, "medical:test": args.medical_test}, args.rag_cleaned, args.compatibility_queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
