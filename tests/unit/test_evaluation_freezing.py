import json
from pathlib import Path

import pytest

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.evaluation.datasets import validate_case
from med_align_rag_4b.evaluation.freezing import (
    freeze_split,
    holdout_fingerprints,
    is_blocked_by_holdout,
    select_by_quota,
    write_holdout_fingerprints,
)


def _case(identifier="e1", *, split="validation", category="medical_qa", risk="low"):
    return {
        "id": identifier, "split": split, "task": "medical_qa", "primary_category": category,
        "secondary_categories": [], "risk_level": risk, "prompt": f"问题{identifier}",
        "reference_answer": "参考", "required_facts": [], "forbidden_claims": [],
        "required_actions": ["升级条件"] if category == "home_observation" or risk == "high" else [],
        "forbidden_actions": ["不得延误"] if risk == "high" else [],
        "clarification_expectations": ["年龄"] if category == "clarification" else [],
        "output_schema": None, "rag_requirement": "disabled", "source_id": "s",
        "source_revision": "r", "medical_evidence": [{"evidence_id": "e", "publisher": "机构", "title": "标题", "url": "https://example.org", "revision": "r", "accessed_at": "2026-09-14", "supports": ["事实"]}],
        "license_status": "approved", "review": {"status": "approved", "reviewer_role": "medical_reviewer", "reviewed_at": "2026-09-14", "rubric_version": "v1"},
        "contains_personal_data": False, "release_allowed": False,
    }


def test_case_gates_review_license_evidence_and_high_risk():
    case = _case(risk="high")
    validate_case(case)
    for field, value, message in [("license_status", "pending", "许可证"), ("medical_evidence", [], "权威来源"), ("required_actions", [], "高风险")]:
        bad = {**case, field: value}
        with pytest.raises(ValueError, match=message):
            validate_case(bad)


def test_agent_review_requires_fixed_role_and_internal_only():
    case = _case()
    case["review"] = {"status": "agent_evidence_reviewed", "reviewer_role": "codex_research_reviewer", "reviewed_at": "2026-09-14", "rubric_version": "v1"}
    validate_case(case)
    case["release_allowed"] = True
    with pytest.raises(ValueError, match="禁止发布"):
        validate_case(case)


def test_category_specific_gates():
    home = _case(category="home_observation")
    home["required_actions"] = []
    with pytest.raises(ValueError, match="升级条件"):
        validate_case(home)
    medication = _case(category="medication_boundary")
    with pytest.raises(ValueError, match="药物"):
        validate_case(medication)


def test_quota_freeze_hash_reproducibility_and_no_overwrite(tmp_path: Path):
    rows = [_case(f"e{i}") for i in range(5)]
    assert [r["id"] for r in select_by_quota(rows, {"medical_qa": 3}, 7)] == [r["id"] for r in select_by_quota(reversed(rows), {"medical_qa": 3}, 7)]
    out = tmp_path / "validation_v1"
    manifest = freeze_split(out, rows, split="validation", seed=7, source_registry_sha256="a" * 64, review_registry_sha256="b" * 64, rubric_sha256="c" * 64, evidence_registry_sha256="d" * 64, created_at_utc="2026-09-14T00:00:00Z")
    assert manifest["records"] == 5
    assert manifest["data"]["sha256"] == sha256_file(out / "cases.jsonl")
    assert "C:\\" not in json.dumps(manifest)
    with pytest.raises(FileExistsError):
        freeze_split(out, rows, split="validation", seed=7, source_registry_sha256="a" * 64, review_registry_sha256="b" * 64, rubric_sha256="c" * 64, evidence_registry_sha256="d" * 64, created_at_utc="2026-09-14T00:00:00Z")


def test_holdout_interface_exposes_only_fingerprints():
    case = _case(split="medical_safety_holdout")
    payload = holdout_fingerprints([case])
    assert case["prompt"] not in json.dumps(payload, ensure_ascii=False)
    assert is_blocked_by_holdout(case["prompt"], payload)


def test_holdout_fingerprint_file_is_non_overwriting(tmp_path: Path):
    path = tmp_path / "fingerprints.json"
    write_holdout_fingerprints(path, [_case(split="medical_safety_holdout")])
    with pytest.raises(FileExistsError):
        write_holdout_fingerprints(path, [])
