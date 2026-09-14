import pytest

from med_align_rag_4b.common.jsonl import count_jsonl, iter_jsonl, write_jsonl_atomic


def test_jsonl_round_trip_and_overwrite_guard(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    records = [{"id": "一"}, {"id": "二"}]
    assert write_jsonl_atomic(path, records) == 2
    assert list(iter_jsonl(path)) == records
    assert count_jsonl(path) == 2
    with pytest.raises(FileExistsError):
        write_jsonl_atomic(path, records)


def test_iter_jsonl_reports_invalid_line(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        list(iter_jsonl(path))
