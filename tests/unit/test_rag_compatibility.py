import json

import pytest

from med_align_rag_4b.rag.compatibility import (
    compare_ranked_results,
    fingerprint_model,
    load_queries,
    write_json_atomic,
)


def _result(row, score, chunk_id="chunk"):
    metadata = {"chunk_id": chunk_id, "text": f"text-{row}"}
    return {"index": row, "score": score, "text": metadata["text"], "metadata": metadata}


def test_queries_require_stable_unique_records(tmp_path):
    path = tmp_path / "queries.jsonl"
    records = [
        {"query_id": f"q{i:03d}", "query": f"问题{i}", "source_row_id": i, "source_chunk_id": f"c{i}", "length_bucket": "short"}
        for i in range(2)
    ]
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    assert [item.query_id for item in load_queries(path, 2)] == ["q000", "q001"]
    records[1]["query"] = records[0]["query"]
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="文本必须唯一"):
        load_queries(path, 2)


def test_model_fingerprint_is_stable_and_rejects_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr("med_align_rag_4b.rag.compatibility.MODEL_FINGERPRINT_FILES", ("config.json", "weights.bin"))
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weights.bin"):
        fingerprint_model(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"weights")
    assert fingerprint_model(tmp_path) == fingerprint_model(tmp_path)


def test_ranked_comparison_accepts_tolerance_and_marks_ties():
    old = [_result(1, 0.9, "a"), _result(2, 0.9 - 5e-8, "b")]
    new = [_result(1, 0.9 + 1e-6, "a"), _result(2, 0.9 - 5e-8, "b")]
    result = compare_ranked_results(old, new)
    assert result["max_score_abs_error"] == pytest.approx(1e-6)
    assert result["tie_risk"] is True


@pytest.mark.parametrize(
    ("new", "message"),
    [
        ([_result(2, 0.9, "b")], "row id"),
        ([_result(1, 0.9, "other")], "chunk id"),
        ([_result(1, 0.89, "a")], "score"),
    ],
)
def test_ranked_comparison_rejects_incompatibility(new, message):
    with pytest.raises(AssertionError, match=message):
        compare_ranked_results([_result(1, 0.9, "a")], new)


def test_atomic_report_rejects_overwrite(tmp_path):
    path = tmp_path / "report.json"
    write_json_atomic(path, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_json_atomic(path, {"status": "changed"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "passed"
