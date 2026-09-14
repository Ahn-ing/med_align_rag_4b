from med_align_rag_4b.sft.deduplication import (
    duplicate_groups,
    normalize_text,
    overlap_ids,
    question_fingerprint,
    stable_record_id,
)


def _record(identifier, source, question, answer):
    return {"id": identifier, "source_id": source, "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def test_normalization_preserves_medical_meaning_and_stable_id():
    assert normalize_text(" ＮＯＴ  5ｍｇ，不过敏 ") == "not 5mg,不过敏"
    assert "不" in normalize_text("不使用 10 mg")
    first = stable_record_id("s", "42", "问题？", "回答")
    assert first == stable_record_id("s", "42", "问题?", "回答")


def test_duplicate_conflict_cross_source_and_overlap():
    rows = [_record("a", "s1", "问题？", "甲"), _record("b", "s2", "问题?", "甲"), _record("c", "s2", "问题?", "乙")]
    groups = duplicate_groups(reversed(rows))
    assert len(groups["normalized_qa_duplicates"]) == 1
    assert len(groups["normalized_question_duplicates"]) == 1
    assert len(groups["conflicting_answers"]) == 1
    assert overlap_ids(rows, {question_fingerprint("问题?")}) == ["a", "b", "c"]
