import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from scripts.sft.evaluate_public_regression import (
    ACCEPTANCE_REQUIRED_BEHAVIORS,
    ACCEPTANCE_STANDARD,
    DEFAULT_SYSTEM_PROMPT,
    _model_inputs,
    behavior_review,
    load_cases,
    mechanical_signals,
    messages_for_case,
)

ROOT = Path(__file__).resolve().parents[2]


def test_public_regression_loads_frozen_cases() -> None:
    cases = load_cases(ROOT / "data/evaluation/regression_v2_public.json")
    assert len(cases) == 5
    assert {case["category"] for case in cases} == {
        "medical_action_followup",
        "medical_emergency_escalation",
        "general_instruction",
        "general_knowledge",
        "structured_output",
    }


def test_public_regression_rejects_hidden_or_disabled_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"status": "frozen", "generation_allowed": False, "cases": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="公开回归集"):
        load_cases(path)


def test_mechanical_signals_validate_json_and_point_count() -> None:
    json_case = {"prompt": "p", "category": "structured_output"}
    json_signals = mechanical_signals(json_case, '{"cities": ["北京", "上海"]}')
    assert json_signals["json_valid"] is True
    assert json_signals["json_keys"] == ["cities"]

    list_case = {"prompt": "p", "category": "general_instruction"}
    list_signals = mechanical_signals(list_case, "1. 固定起床 2. 减少屏幕 3. 适度运动")
    assert list_signals["point_count"] == 3


def test_behavior_review_is_pending_for_medical_case() -> None:
    case = {
        "category": "medical_emergency_escalation",
        "required_behaviors": ["立即急救"],
        "forbidden_behaviors": ["自行观察"],
    }
    review = behavior_review(case, {})
    assert review["status"] == "pending_manual_review"
    assert review["standard"] == ACCEPTANCE_STANDARD
    assert review["required"][0]["status"] == "pending_manual_review"
    assert review["forbidden"][0]["status"] == "pending_manual_review"
    assert review["graded_required"][0]["status"] == "pending_manual_review"


def test_acceptance_rubric_covers_frozen_public_cases() -> None:
    cases = load_cases(ROOT / "data/evaluation/regression_v2_public.json")
    assert {case["id"] for case in cases} == set(ACCEPTANCE_REQUIRED_BEHAVIORS)
    assert all(
        isinstance(behaviors, tuple) for behaviors in ACCEPTANCE_REQUIRED_BEHAVIORS.values()
    )


def test_default_system_prompt_matches_training_runtime_prompt() -> None:
    assert DEFAULT_SYSTEM_PROMPT == "你是一个谨慎、清晰且遵循证据边界的助手。"


def test_messages_use_same_optional_system_prompt() -> None:
    case = {"prompt": "问题"}
    assert messages_for_case(case, "") == [{"role": "user", "content": "问题"}]
    assert messages_for_case(case, "system") == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "问题"},
    ]


def test_model_inputs_accept_mapping_not_only_dict() -> None:
    class FakeTensor:
        shape = (1, 3)

        def to(self, device: str) -> "FakeTensor":
            assert device == "cuda"
            return self

    class FakeEncoding(Mapping):
        def __init__(self) -> None:
            self.data = {"input_ids": FakeTensor()}

        def __getitem__(self, key: str):
            return self.data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self.data)

        def __len__(self) -> int:
            return len(self.data)

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return FakeEncoding()

    model_inputs, prompt_length = _model_inputs(FakeTokenizer(), [], "cuda")
    assert prompt_length == 3
    assert isinstance(model_inputs["input_ids"], FakeTensor)
