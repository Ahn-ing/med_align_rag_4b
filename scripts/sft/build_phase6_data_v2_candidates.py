"""从只读原始来源构建 SFT v2 审核队列；不发布可训练数据。"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer
from med_align_rag_4b.sft.v2_quality import (
    API_TOKEN_TARGETS,
    MEDICAL_CATEGORIES,
    MEDICAL_TOKEN_TARGETS,
    automatic_quality_flags,
    iter_json_array,
    raw_medical_record,
    template_skeleton,
    write_json_line,
)

SEED = 20260915
GENERAL_TARGET = 750_000
SOURCE_REVISION = "6e219f1a14856833ee436063d3b73c5f1ab9cfb9"
MEDICAL_WORDS = re.compile(
    r"(?i)\b(?:diagnos|disease|drug|medicine|patient|symptom|cancer|blood|"
    r"hospital|doctor)\w*\b|医疗|疾病|药物|症状|患者|医院|医生|血压|癌"
)


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                write_json_line(stream, record)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def load_blocked_fingerprints(root: Path, test_fingerprints: Path | None) -> tuple[set[str], list[str]]:
    blocked: set[str] = set()
    blockers: list[str] = []
    for row in iter_jsonl(root / "data/evaluation/validation_v1/cases.jsonl"):
        blocked.add(question_fingerprint(row["prompt"]))
    holdout = json.loads(
        (root / "data/evaluation/medical_safety_holdout_v1/fingerprints.json").read_text(
            encoding="utf-8"
        )
    )
    blocked.update(holdout["fingerprints"])
    for row in iter_jsonl(root / "data/rag/compatibility/queries.v1.jsonl"):
        blocked.add(question_fingerprint(row["query"]))
    if test_fingerprints is None:
        blockers.append("missing_test_v1_irreversible_fingerprints")
    else:
        payload = json.loads(test_fingerprints.read_text(encoding="utf-8"))
        blocked.update(payload["fingerprints"])
    return blocked, blockers


def init_dedup_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE questions (qfp TEXT PRIMARY KEY, first_qa TEXT NOT NULL, conflict INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE qa_pairs (qafp TEXT PRIMARY KEY, first_line INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE question_skeletons (skeleton TEXT PRIMARY KEY, first_line INTEGER NOT NULL)"
    )
    return connection


def scan_medical_source(
    source: Path, database: sqlite3.Connection, *, limit: int | None = None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line_number, raw in enumerate(iter_json_array(source), 1):
        if limit is not None and line_number > limit:
            break
        counts["input_records"] += 1
        try:
            record = raw_medical_record(raw, line_number)
        except ValueError as exc:
            counts[f"rejected_{exc}"] += 1
            continue
        if record["quality_flags"]:
            for flag in record["quality_flags"]:
                counts[f"flag_{flag}"] += 1
            continue
        qfp = record["fingerprints"]["question"]
        qafp = record["fingerprints"]["qa"]
        previous = database.execute(
            "SELECT first_qa FROM questions WHERE qfp = ?", (qfp,)
        ).fetchone()
        if previous is None:
            database.execute(
                "INSERT INTO questions(qfp, first_qa, conflict) VALUES (?, ?, 0)",
                (qfp, qafp),
            )
        elif previous[0] != qafp:
            database.execute("UPDATE questions SET conflict = 1 WHERE qfp = ?", (qfp,))
        database.execute(
            "INSERT OR IGNORE INTO qa_pairs(qafp, first_line) VALUES (?, ?)",
            (qafp, line_number),
        )
        database.execute(
            "INSERT OR IGNORE INTO question_skeletons(skeleton, first_line) VALUES (?, ?)",
            (record["fingerprints"]["question_skeleton"], line_number),
        )
        if line_number % 20_000 == 0:
            database.commit()
            print(json.dumps({"stage": "dedup_scan", "records": line_number}))
    database.commit()
    counts["conflicting_questions"] = database.execute(
        "SELECT COUNT(*) FROM questions WHERE conflict = 1"
    ).fetchone()[0]
    counts["unique_qa_pairs"] = database.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
    return counts


def push_reservoir(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]],
    category: str,
    record: dict[str, Any],
    max_records: int,
) -> None:
    priority = int(sha256_text(f"{SEED}:{record['id']}"), 16)
    item = (-priority, record["id"], record)
    heap = heaps[category]
    if len(heap) < max_records:
        heapq.heappush(heap, item)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, item)


def medical_review_pool(
    source: Path,
    database: sqlite3.Connection,
    tokenizer: QwenAssistantTokenizer,
    blocked: set[str],
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        category: [] for category in MEDICAL_CATEGORIES
    }
    counts: Counter[str] = Counter()
    max_records = {
        category: max(2_000, int(target / 24 * 1.25))
        for category, target in MEDICAL_TOKEN_TARGETS.items()
    }
    for line_number, raw in enumerate(iter_json_array(source), 1):
        if limit is not None and line_number > limit:
            break
        try:
            record = raw_medical_record(raw, line_number)
        except ValueError:
            continue
        if record["quality_flags"]:
            continue
        qfp = record["fingerprints"]["question"]
        qafp = record["fingerprints"]["qa"]
        state = database.execute(
            "SELECT conflict FROM questions WHERE qfp = ?", (qfp,)
        ).fetchone()
        first = database.execute(
            "SELECT first_line FROM qa_pairs WHERE qafp = ?", (qafp,)
        ).fetchone()
        skeleton_first = database.execute(
            "SELECT first_line FROM question_skeletons WHERE skeleton = ?",
            (record["fingerprints"]["question_skeleton"],),
        ).fetchone()
        if state is None or state[0]:
            counts["excluded_conflicting_question"] += 1
            continue
        if first is None or first[0] != line_number:
            counts["excluded_duplicate_qa"] += 1
            continue
        if skeleton_first is None or skeleton_first[0] != line_number:
            counts["excluded_question_skeleton_near_duplicate"] += 1
            continue
        if qfp in blocked:
            counts["excluded_blocked_fingerprint"] += 1
            continue
        if record["risk_level"] == "high":
            counts["excluded_high_risk_requires_evidence"] += 1
            continue
        push_reservoir(
            heaps, record["primary_category"], record, max_records[record["primary_category"]]
        )
        counts["eligible_before_tokenization"] += 1
        if line_number % 20_000 == 0:
            print(json.dumps({"stage": "quality_token_scan", "records": line_number}))

    selected: list[dict[str, Any]] = []
    category_report: dict[str, Any] = {}
    for category, heap in heaps.items():
        ordered = sorted(
            (item[2] for item in heap),
            key=lambda row: sha256_text(f"{SEED}:{row['id']}"),
        )
        target = MEDICAL_TOKEN_TARGETS[category] - API_TOKEN_TARGETS[category]
        skeleton_limit = max(1, round((target / 64) * 0.01))
        skeleton_counts: Counter[str] = Counter()
        category_selected: list[dict[str, Any]] = []
        tokens = 0
        for record in ordered:
            try:
                metrics = tokenizer.analyze(record["messages"])
            except ValueError:
                counts["excluded_tokenization"] += 1
                continue
            supervised = metrics["assistant_supervised_tokens"]
            if supervised < 24:
                counts["excluded_under_24_tokens"] += 1
                continue
            if supervised > 768 or metrics["overlength"]:
                counts["excluded_over_768_tokens"] += 1
                continue
            skeleton = record["fingerprints"]["answer_skeleton"]
            if skeleton_counts[skeleton] >= skeleton_limit:
                counts["excluded_answer_skeleton_cap"] += 1
                continue
            if tokens >= target:
                break
            record.update({
                "total_tokens": metrics["total_tokens"],
                "assistant_supervised_tokens": supervised,
                "truncated": False,
            })
            category_selected.append(record)
            skeleton_counts[skeleton] += 1
            tokens += supervised
        selected.extend(category_selected)
        category_report[category] = {
            "target_tokens": target,
            "candidate_records": len(heap),
            "selected_for_review": len(category_selected),
            "selected_tokens": tokens,
            "unique_answer_skeletons": len(skeleton_counts),
            "largest_answer_skeleton": max(skeleton_counts.values(), default=0),
            "skeleton_limit": skeleton_limit,
        }
    return sorted(selected, key=lambda row: row["id"]), counts, category_report


def general_review_pool(
    path: Path, tokenizer: QwenAssistantTokenizer, blocked: set[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, raw in enumerate(iter_jsonl(path), 1):
        instruction, response, context = (
            raw.get("instruction"), raw.get("response"), raw.get("context")
        )
        if not isinstance(instruction, str) or not isinstance(response, str):
            counts["excluded_schema"] += 1
            continue
        question = instruction.strip()
        if isinstance(context, str) and context.strip():
            question += f"\n\nContext:\n{context.strip()}"
        answer = response.strip()
        if not question or not answer or MEDICAL_WORDS.search(f"{question} {answer}"):
            counts["excluded_empty_or_medical"] += 1
            continue
        flags = automatic_quality_flags(question, answer)
        flags = [flag for flag in flags if flag != "answer_too_short_for_medical_review"]
        if flags:
            counts["excluded_quality"] += 1
            continue
        qfp = question_fingerprint(question)
        if qfp in blocked:
            counts["excluded_blocked_fingerprint"] += 1
            continue
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        try:
            metrics = tokenizer.analyze(messages)
        except ValueError:
            counts["excluded_tokenization"] += 1
            continue
        tokens = metrics["assistant_supervised_tokens"]
        if tokens > 768 or metrics["overlength"]:
            counts["excluded_over_768_tokens"] += 1
            continue
        identifier = "sftv2_" + sha256_record(
            ["databricks_dolly_15k_bdd27f4", str(index), question, answer]
        )[:24]
        rows.append({
            "id": identifier,
            "source_id": "databricks_dolly_15k_bdd27f4",
            "source_record_id": str(index),
            "source_revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
            "messages": messages,
            "domain": "general",
            "ability": str(raw.get("category", "instruction_following")),
            "primary_category": "general",
            "risk_level": "not_applicable",
            "license_status": "approved",
            "release_allowed": False,
            "quality_flags": [],
            "review": {"status": "source_approved_quality_screened"},
            "medical_evidence": [],
            "total_tokens": metrics["total_tokens"],
            "assistant_supervised_tokens": tokens,
            "truncated": False,
            "fingerprints": {
                "question": qfp,
                "qa": qa_fingerprint(question, answer),
                "question_skeleton": sha256_text(template_skeleton(question)),
                "answer_skeleton": sha256_text(template_skeleton(answer)),
            },
            "provenance": {"generation_method": "human_upstream"},
        })
    ordered = sorted(rows, key=lambda row: sha256_text(f"{SEED}:{row['id']}"))
    selected: list[dict[str, Any]] = []
    tokens = 0
    seen_qa: set[str] = set()
    for row in ordered:
        if row["fingerprints"]["qa"] in seen_qa:
            counts["excluded_duplicate_qa"] += 1
            continue
        if tokens >= GENERAL_TARGET:
            break
        selected.append(row)
        seen_qa.add(row["fingerprints"]["qa"])
        tokens += row["assistant_supervised_tokens"]
    counts["selected_records"] = len(selected)
    counts["selected_tokens"] = tokens
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-train", type=Path, required=True)
    parser.add_argument("--dolly", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--test-fingerprints", type=Path)
    parser.add_argument("--limit", type=int, help="仅供开发测试；正式构建必须省略")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    review_dir = root / "data/sft/reviewed/v2"
    report_dir = root / "artifacts/reports/phase_6_data_v2"
    if review_dir.exists() or report_dir.exists():
        raise FileExistsError("v2 审核队列或报告目录已存在，拒绝覆盖")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit 必须为正数")
    for source in (args.medical_train, args.dolly, args.tokenizer_path):
        if not source.exists():
            raise FileNotFoundError(source)

    blocked, blockers = load_blocked_fingerprints(root, args.test_fingerprints)
    database_path = root / ".tmp/sft_v2_dedup.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)
    database = init_dedup_database(database_path)
    try:
        scan_counts = scan_medical_source(args.medical_train, database, limit=args.limit)
        tokenizer = QwenAssistantTokenizer(args.tokenizer_path, max_length=2048)
        medical, medical_counts, category_report = medical_review_pool(
            args.medical_train, database, tokenizer, blocked, limit=args.limit
        )
        general, general_counts = general_review_pool(args.dolly, tokenizer, blocked)
    finally:
        database.close()

    for row in medical:
        row["review"] = {
            "status": "rule_screened_pending_stratified_audit",
            "reviewer_role": "deterministic_v2_rules",
            "evidence_checked": False,
        }
    review_queue = medical + general
    review_sha = atomic_jsonl(review_dir / "rule_screened_pool.jsonl", review_queue)
    manifest = {
        "format_version": 1,
        "dataset_version": "v2",
        "status": "built_pending_review",
        "training_allowed": False,
        "release_allowed": False,
        "source": {
            "medical": {
                "logical_path": "../med_align_rag/data/sft/raw/medical/finetune/train_zh_0.json",
                "sha256": sha256_file(args.medical_train),
                "revision": SOURCE_REVISION,
                "records_scanned": scan_counts["input_records"],
                "read_only": True,
            },
            "general": {
                "logical_path": "data/external/datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl",
                "sha256": sha256_file(args.dolly),
                "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
            },
        },
        "tokenizer": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "max_length": 2048,
            "enable_thinking": False,
            "system_prompt": None,
        },
        "targets": {"medical": MEDICAL_TOKEN_TARGETS, "general": GENERAL_TARGET},
        "rule_screened_pool": {
            "path": "data/sft/reviewed/v2/rule_screened_pool.jsonl",
            "sha256": review_sha,
            "records": len(review_queue),
            "medical_records": len(medical),
            "general_records": len(general),
        },
        "scan_counts": dict(scan_counts),
        "medical_candidate_counts": dict(medical_counts),
        "general_candidate_counts": dict(general_counts),
        "category_report": category_report,
        "blockers": blockers + [
            "rule_screened_pool_requires_stratified_audit",
            "api_evidence_constrained_samples_not_built",
            "validation_v2_not_frozen",
        ],
        "prohibited_actions_confirmed": {
            "legacy_project_written": False,
            "test_cases_read": False,
            "medical_safety_holdout_cases_read": False,
            "training_run": False,
            "dpo_run": False,
        },
    }
    atomic_json(review_dir / "manifest.json", manifest)
    atomic_json(report_dir / "candidate_report.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "review_records": len(review_queue),
        "blockers": manifest["blockers"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
