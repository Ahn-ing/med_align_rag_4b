import pytest

from med_align_rag_4b.evaluation.rag_judge import (
    JUDGE_SYSTEM,
    build_judge_payload,
    judge_input_sha256,
    parse_judge_score,
)


def _case():
    return {
        "id": "case",
        "task": "rag_faithfulness",
        "primary_category": "rag_faithfulness",
        "risk_level": "medium",
        "prompt": "问题",
        "reference_answer": "答案 [E1]",
        "required_facts": ["事实"],
        "forbidden_claims": ["不得扩展"],
        "required_actions": [],
        "forbidden_actions": [],
        "clarification_expectations": [],
        "output_schema": None,
        "medical_evidence": [],
        "rag_requirement": "required",
    }


def _row():
    return {
        "output": "事实 [E1]",
        "system_prompt": "RAG prompt",
        "rag": {"enabled": True, "context": "[E1] 事实", "evidence": [{"chunk_id": "c1"}]},
    }


def test_rag_judge_uses_task_specific_fact_rules():
    assert "引用编号不是评分项" in JUDGE_SYSTEM
    assert "task=rag_faithfulness" in JUDGE_SYSTEM
    assert "task=medical_safety" in JUDGE_SYSTEM
    assert "optional_supportive_actions" in JUDGE_SYSTEM


def test_medical_safety_optional_actions_are_not_required():
    case = {**_case(), "task": "medical_safety", "required_actions": ["给出短期休息、补液和观察方法", "出现呼吸困难或症状严重时升级为医疗评估"]}
    payload = build_judge_payload(case, _row())
    assert payload["required_actions"] == ["出现呼吸困难或症状严重时升级为医疗评估"]
    assert payload["optional_supportive_actions"] == ["给出短期休息、补液和观察方法"]


def test_rag_judge_payload_includes_context_and_hides_arm():
    payload = build_judge_payload(_case(), _row())
    assert payload["rag_context"] == "[E1] 事实"
    assert payload["rag_evidence"] == [{"chunk_id": "c1"}]
    assert "arm" not in payload
    assert len(judge_input_sha256(_case(), _row())) == 64


def test_rag_judge_parser_normalizes_optional_contract_drift():
    parsed = parse_judge_score(
        '{"score_0_to_2":1,"supported_reasons":null,"violations":[],"uncertainty":null}'
    )
    assert parsed["supported_reasons"] == []
    assert parsed["uncertainty"] == ""
    parsed = parse_judge_score(
        '{"score_0_to_2":1,"supported_reasons":["a"],"violations":[],"uncertainty":["不确定","需复核"]}'
    )
    assert parsed["uncertainty"] == "不确定；需复核"


def test_rag_judge_parser_is_strict():
    parsed = parse_judge_score(
        '{"score_0_to_2":2,"supported_reasons":["[E1]"],"violations":[],"uncertainty":""}'
    )
    assert parsed["score_0_to_2"] == 2
    with pytest.raises(ValueError, match="键"):
        parse_judge_score(
            '{"score_0_to_2":2,"supported_reasons":[],"violations":[],"uncertainty":"","extra":1}'
        )