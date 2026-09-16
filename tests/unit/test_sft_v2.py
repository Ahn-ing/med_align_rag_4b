from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from med_align_rag_4b.sft.v2_quality import (
    API_MEDICAL_TOKEN_MAXIMUM,
    API_MEDICAL_TOKEN_MINIMUM,
    API_MEDICAL_TOKEN_TARGET,
    API_TOKEN_TARGETS,
    automatic_quality_flags,
    iter_json_array,
    raw_medical_record,
    template_skeleton,
)
from scripts.sft.build_phase6_v2_api_generation_tasks import (
    category_for_support,
    evidence_id_for_question,
    main,
)


def test_streams_top_level_json_array_across_tiny_chunks(tmp_path: Path) -> None:
    source = tmp_path / "medical.json"
    source.write_text(
        json.dumps(
            [
                {
                    "instruction": "问题一",
                    "input": "",
                    "output": "这是足够长的回答内容。",
                },
                {
                    "instruction": "问题二",
                    "input": "补充",
                    "output": "这是另一条足够长的回答。",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert [row["instruction"] for row in iter_json_array(source, chunk_size=7)] == [
        "问题一",
        "问题二",
    ]


def test_streams_medical_source_with_json_extension_as_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "train_zh_0.json"
    rows = [
        {"instruction": "问题一", "input": "", "output": "第一条回答"},
        {"instruction": "问题二", "input": "", "output": "第二条回答"},
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert list(iter_json_array(source)) == rows


def test_medical_candidate_is_natural_two_turn_conversation() -> None:
    record = raw_medical_record(
        {
            "instruction": "高血压是什么？",
            "input": "",
            "output": "高血压是需要持续监测和管理的慢性疾病。",
        },
        12,
    )
    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert record["review"]["status"] == "ready_for_review"
    assert record["release_allowed"] is False


@pytest.mark.parametrize(
    "answer",
    ["记录要点：这是第 1 个独立情境。", "背景编号为 42，请按模板作答。"],
)
def test_rejects_generator_meta_language(answer: str) -> None:
    assert "generator_meta_language" in automatic_quality_flags("问题", answer)


def test_template_skeleton_does_not_treat_numbers_as_diversity() -> None:
    assert template_skeleton("这是第1个案例") == template_skeleton("这是第928个案例")


@pytest.mark.parametrize(
    ("answer", "flag"),
    [
        (
            "根据你的描述，考虑是感冒，可以服用某药试试。",
            "online_consultation_template",
        ),
        ("这种情况考虑炎症引起，祝早日康复。", "generic_consultation_closing"),
        (
            "病因：甲。症状：乙。检查：丙。诊断：丁。治疗：戊。",
            "dense_fixed_medical_schema",
        ),
    ],
)
def test_rejects_old_project_failure_patterns(answer: str, flag: str) -> None:
    assert flag in automatic_quality_flags("问题", answer)


def test_api_task_treats_medical_emergency_claim_as_high_risk_category() -> None:
    assert category_for_support("卒中是医疗急症") == "emergency_escalation"


def test_rehabilitation_urgency_is_not_misclassified_as_emergency() -> None:
    assert (
        category_for_support("卒中康复应在患者病情稳定后立即开始")
        != "emergency_escalation"
    )


def test_registry_topic_terms_route_queries_to_new_sources() -> None:
    sources = [
        {
            "evidence_id": "who_opioid_overdose_2025",
            "topic_terms": ["阿片过量", "海洛因", "芬太尼"],
        }
    ]
    assert (
        evidence_id_for_question("海洛因过量该怎么办？", sources)
        == "who_opioid_overdose_2025"
    )


def test_api_medical_token_policy_uses_7_5_percent_target() -> None:
    assert API_MEDICAL_TOKEN_MINIMUM == 150_000
    assert API_MEDICAL_TOKEN_TARGET == 225_000
    assert API_MEDICAL_TOKEN_MAXIMUM == 300_000
    assert sum(API_TOKEN_TARGETS.values()) == API_MEDICAL_TOKEN_TARGET


@pytest.mark.parametrize(
    ("question", "evidence_id"),
    [
        ("慢阻肺有哪些常见症状？", "who_copd_2026"),
        ("败血症需要马上就医吗？", "who_sepsis_2024"),
        ("登革热有哪些警示症状？", "who_dengue_2025"),
        ("抗生素耐药为什么危险？", "who_amr_2026"),
        ("宫颈癌可以筛查吗？", "who_cervical_cancer_2026"),
        ("乳腺癌早期有哪些表现？", "who_breast_cancer_2026"),
        ("抑郁症有哪些治疗方式？", "who_depression_2026"),
        ("艾滋病毒如何传播？", "who_hiv_2026"),
        ("疟疾应该如何预防？", "who_malaria_2025"),
        ("季节性流感有哪些症状？", "who_influenza_2025"),
        ("食品安全风险应该如何预防？", "who_food_safety_2026"),
        ("孕产妇死亡如何预防？", "who_maternal_mortality_2025"),
    ],
)
def test_v3_topics_route_to_expanded_evidence(question: str, evidence_id: str) -> None:
    assert evidence_id_for_question(question) == evidence_id


def test_normal_otc_purchase_language_is_not_treated_as_advertising() -> None:
    flags = automatic_quality_flags(
        "购买非处方药前要注意什么？",
        "购买或服用前应阅读标签，并向医生或药师确认相互作用。",
    )
    assert "advertising_or_unverifiable_claim" not in flags

def test_api_generation_tasks_preserve_each_reference_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "evidence.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "evidence_id": "who_cancer_2026",
                        "publisher": "World Health Organization",
                        "url": "https://example.test/cancer",
                        "supports": ["癌症筛查只对部分癌种适用"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pool = tmp_path / "rule_pool.jsonl"
    questions = ["癌症筛查是不是都能查？", "癌症筛查结果异常怎么办？"]
    rows = [
        {
            "id": f"sftv2_test_{index}",
            "domain": "medical",
            "risk_level": "medium",
            "primary_category": "action_followup",
            "assistant_supervised_tokens": 60,
            "messages": [{"role": "user", "content": question}],
        }
        for index, question in enumerate(questions)
    ]
    pool.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_phase6_v2_api_generation_tasks.py",
            "--evidence-registry",
            str(registry),
            "--rule-pool",
            str(pool),
            "--max-tasks-per-focus",
            "2",
            "--output",
            str(output),
        ],
    )
    main()
    generated = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    anchor_questions = [row["reference_question"] for row in generated[1:]]
    assert anchor_questions == questions

    limited_output = tmp_path / "tasks_limited.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_phase6_v2_api_generation_tasks.py",
            "--evidence-registry",
            str(registry),
            "--rule-pool",
            str(pool),
            "--max-tasks-per-focus",
            "1",
            "--output",
            str(limited_output),
        ],
    )
    main()
    limited = [json.loads(line) for line in limited_output.read_text(encoding="utf-8").splitlines()]
    assert len(limited) == 2
    assert len(limited[1:]) == 1
