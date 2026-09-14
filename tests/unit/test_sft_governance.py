import pytest

from med_align_rag_4b.sft.governance import (
    validate_candidate,
    validate_source,
    validate_training_candidate,
)


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


def test_training_gate_requires_source_evidence_and_agent_review():
    source = {
        "source_id": "s", "name": "source", "purpose": "medical",
        "origin": "official", "source_url": "https://example.test", "revision": "r",
        "license": "Apache-2.0", "license_status": "approved",
        "contains_medical_claims": True, "contains_personal_data": False,
        "generation_method": "human", "allowed_uses": ["training_candidate"],
        "review_status": "approved", "notes": "test",
    }
    record = {
        "id": "x", "source_id": "s", "source_record_id": "1", "source_revision": "r",
        "messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}],
        "domain": "medical", "ability": "qa", "language": "zh", "risk_level": "low",
        "primary_category": "medical_qa", "secondary_categories": [],
        "license_status": "approved", "medical_evidence": ["e1"],
        "review": {"status": "agent_evidence_reviewed", "reviewer_role": "codex_research_reviewer", "evidence_checked": True},
        "unresolved_safety_flags": [],
    }
    validate_training_candidate(record, source)
    record["medical_evidence"] = []
    with pytest.raises(ValueError, match="缺少权威证据"):
        validate_training_candidate(record, source)
