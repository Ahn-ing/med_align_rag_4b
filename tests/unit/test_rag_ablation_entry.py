from pathlib import Path

from med_align_rag_4b.evaluation.rag_ablation import (
    ARM_ORDER,
    generate_case_record,
    get_arm,
    load_rag_ablation_config,
    load_validation_cases,
)
from med_align_rag_4b.rag.prompting import SYSTEM_INSTRUCTION

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/phase8_rag.yaml"


def test_full_rag_matrix_is_frozen():
    config = load_rag_ablation_config(CONFIG, ROOT)
    assert tuple(config["arms"]) == ARM_ORDER
    assert config["arms"]["R1"]["rag"] is True
    assert config["arms"]["R0"]["rag"] is False
    cases, cases_sha256 = load_validation_cases(ROOT, config["dataset"])
    assert len(cases) == 60
    assert len(cases_sha256) == 64


def test_rag_off_does_not_inject_context_or_rag_prompt():
    config = load_rag_ablation_config(CONFIG, ROOT)
    cases, _ = load_validation_cases(ROOT, config["dataset"])
    arm = get_arm(config, "R0")
    record = generate_case_record(
        cases[0],
        arm=arm,
        base_system_prompt=config["base_system_prompt"],
        generation_config=None,
        generate=lambda messages, generation_config: {"answer": "ok", "generation_config": {}},
    )
    assert record["messages"][0]["content"] == config["base_system_prompt"]
    assert record["rag"]["enabled"] is False
    assert record["rag"]["context"] is None
    assert record["rag"]["uses_rag_system_prompt"] is False
    assert "medical_knowledge_context" not in record["messages"][1]["content"]


def test_rag_on_uses_rag_system_prompt_and_controlled_context():
    config = load_rag_ablation_config(CONFIG, ROOT)
    cases, _ = load_validation_cases(ROOT, config["dataset"])
    case = next(row for row in cases if row["rag_requirement"] == "required")
    arm = get_arm(config, "R1")

    def retrieve(query, top_k):
        raise AssertionError("controlled context 不应重新检索")

    record = generate_case_record(
        case,
        arm=arm,
        base_system_prompt=config["base_system_prompt"],
        generation_config=None,
        generate=lambda messages, generation_config: {"answer": "ok", "generation_config": {}},
        retrieve=retrieve,
        rag_top_k=config["rag"]["top_k"],
        max_context_chars=config["rag"]["max_context_chars"],
        controlled_context_for_required=True,
    )
    assert record["messages"][0]["content"] == SYSTEM_INSTRUCTION
    assert record["rag"]["uses_rag_system_prompt"] is True
    assert record["rag"]["context_source"] == "controlled_case_context"
    assert "<medical_knowledge_context>" in record["messages"][1]["content"]


def test_rag_on_open_retrieval_uses_rag_prompt():
    config = load_rag_ablation_config(CONFIG, ROOT)
    cases, _ = load_validation_cases(ROOT, config["dataset"])
    case = next(row for row in cases if row["rag_requirement"] == "disabled")
    arm = get_arm(config, "R1")

    def retrieve(query, top_k):
        assert query == case["prompt"]
        assert top_k == 5
        return {"results": [{"index": 3, "score": 0.8, "text": "证据文本", "metadata": {"chunk_id": "c3", "question": "来源"}}]}

    record = generate_case_record(
        case,
        arm=arm,
        base_system_prompt=config["base_system_prompt"],
        generation_config=None,
        generate=lambda messages, generation_config: {"answer": "ok", "generation_config": {}},
        retrieve=retrieve,
        rag_top_k=config["rag"]["top_k"],
        max_context_chars=config["rag"]["max_context_chars"],
        controlled_context_for_required=True,
    )
    assert record["messages"][0]["content"] == SYSTEM_INSTRUCTION
    assert record["rag"]["context_source"] == "frozen_retriever"
    assert record["rag"]["evidence"][0]["chunk_id"] == "c3"