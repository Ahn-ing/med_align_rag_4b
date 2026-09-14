import pytest

from med_align_rag_4b.sft.governance import validate_candidate, validate_source


def test_source_registry_requires_explicit_fields():
    with pytest.raises(ValueError, match="缺少字段"):
        validate_source({"source_id": "x"})


def test_candidate_requires_valid_messages_and_review():
    record = {
        "id": "x", "source_id": "s", "source_record_id": "1",
        "messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}],
        "domain": "medical", "ability": "qa", "language": "zh", "risk_level": "low",
        "primary_category": "qa", "secondary_categories": [], "source_revision": "r",
        "license_status": "pending", "medical_evidence": [], "review": {"status": "unreviewed"},
    }
    validate_candidate(record)
    record["messages"][0]["role"] = "bad"
    with pytest.raises(ValueError, match="role/content"):
        validate_candidate(record)
