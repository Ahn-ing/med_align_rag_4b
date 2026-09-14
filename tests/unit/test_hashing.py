import pytest

from med_align_rag_4b.common.hashing import sha256_file, sha256_record, sha256_text


def test_hash_helpers_are_deterministic(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("医疗 RAG", encoding="utf-8")
    assert sha256_file(path) == sha256_text("医疗 RAG")
    assert sha256_record({"b": 2, "a": 1}) == sha256_record({"a": 1, "b": 2})


def test_sha256_file_rejects_invalid_chunk_size(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="正数"):
        sha256_file(path, chunk_size=0)
