import pytest

from med_align_rag_4b.evaluation.datasets import validate_no_leakage
from med_align_rag_4b.sft.deduplication import question_fingerprint


def _case(identifier, split, prompt, group=None):
    return {"id": identifier, "split": split, "prompt": prompt, "group_id": group}


def test_cross_split_question_and_group_are_rejected():
    with pytest.raises(ValueError, match="问题"):
        validate_no_leakage({"validation": [_case("a", "validation", "同题？")], "test": [_case("b", "test", "同题?")]})
    with pytest.raises(ValueError, match="group"):
        validate_no_leakage({"validation": [_case("a", "validation", "甲", "g")], "test": [_case("b", "test", "乙", "g")]})


def test_rag_and_compatibility_fingerprint_overlap_is_rejected():
    with pytest.raises(ValueError, match="RAG"):
        validate_no_leakage({"test": [_case("a", "test", "重叠")]}, {question_fingerprint("重叠")})
