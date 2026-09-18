import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from med_align_rag_4b.dpo.api_config import load_api_config
from med_align_rag_4b.dpo.api_generation import (
    generation_messages,
    generation_messages_v2,
    parse_generation_content,
    parse_judge_content,
)
from med_align_rag_4b.dpo.config import (
    load_dpo_config,
    require_frozen_training_parameters,
    validate_phase7_config,
)
from med_align_rag_4b.dpo.data import (
    load_fingerprint_file,
    question_fingerprint,
    to_trl_record,
    validate_preference_dataset,
    validate_preference_record,
)
from med_align_rag_4b.dpo.loss import dpo_log_sigmoid_loss
from med_align_rag_4b.dpo.manifest import initial_manifest
from med_align_rag_4b.dpo.pilot import select_pilot_prompts
from med_align_rag_4b.dpo.prompt_pool import (
    classify_safety_category,
    select_stratum,
)
from med_align_rag_4b.dpo.style import (
    style_gate_failures,
    style_metrics,
    style_tag_for,
)
from med_align_rag_4b.sft.checkpoints import (
    REQUIRED_CHECKPOINT_FILES,
    validate_checkpoint,
    write_checkpoint_contract,
)
from scripts.dpo.evaluate_phase7 import parse_args as parse_eval_args
from scripts.dpo.generate_phase7_pilot import parse_args as parse_generation_args
from scripts.dpo.generate_phase7_v2_pilot import parse_args as parse_v2_args
from scripts.dpo.run_dpo import parse_args as parse_dpo_args

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/dpo/qwen3_4b_lora_phase7.yaml"
API_CONFIG_PATH = ROOT / "configs/dpo/phase7_api.yaml"
API_CONFIG_V2_PATH = ROOT / "configs/dpo/phase7_api_v2.yaml"
SYSTEM_PROMPT = "你是一个谨慎、清晰且遵循证据边界的助手。"


def _record(
    record_id: str,
    *,
    split: str = "train",
    group: str | None = None,
    question: str = "示例问题",
    task: str = "medical_behavior",
    category: str = "medical_qa",
    review_status: str = "pending_agent_review",
) -> dict:
    reviewed = review_status == "agent_evidence_reviewed"
    return {
        "id": record_id,
        "prompt_group_id": group or f"group-{record_id}",
        "split": split,
        "source_id": "project_authored_preference_v1",
        "source_sha256": "a" * 64,
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "chosen": "优先给出证据支持的谨慎回答。",
        "rejected": "给出没有证据支持的确定结论。",
        "task": task,
        "primary_category": category,
        "risk_level": "medium",
        "evidence_ids": ["who_example_2026"] if task.startswith("medical") else [],
        "review_status": review_status,
        "reviewer_role": "codex_research_reviewer" if reviewed else None,
        "reviewed_at_utc": "2026-09-16T00:00:00Z" if reviewed else None,
        "evidence_checked": reviewed,
        "prompt_tokens": 12,
        "chosen_tokens": 10,
        "rejected_tokens": 9,
        "pair_reason": "事实支持和风险边界",
        "chosen_source": "evidence_constrained",
        "rejected_source": "controlled_negative",
        "contains_personal_data": False,
        "release_allowed": False,
    }


def test_phase7_config_is_frozen_and_training_values_are_stable():
    config = load_dpo_config(CONFIG_PATH)
    assert config["training_allowed"] is True
    assert config["status"] == "frozen"
    assert {
        key: config["training"][key]
        for key in (
            "beta",
            "learning_rate",
            "per_device_train_batch_size",
            "global_batch_size",
            "epochs",
        )
    } == {
        "beta": 0.1,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 1,
        "global_batch_size": 16,
        "epochs": 1,
    }
    assert config["data"]["counts"] == {"train": 900, "preference_validation": 100}
    require_frozen_training_parameters(config)

    changed = deepcopy(config)
    changed["reference"]["base_only_allowed"] = True
    with pytest.raises(ValueError, match="reference.base_only_allowed"):
        validate_phase7_config(changed)

    contract_only = deepcopy(config)
    contract_only["status"] = "contract_only"
    contract_only["training_allowed"] = False
    for field in (
        "learning_rate",
        "per_device_train_batch_size",
        "global_batch_size",
        "epochs",
    ):
        contract_only["training"][field] = None
    validate_phase7_config(contract_only)
    with pytest.raises(ValueError, match="尚未冻结"):
        require_frozen_training_parameters(contract_only)


def test_preference_record_accepts_pending_and_reviewed_records():
    config = load_dpo_config(CONFIG_PATH)
    pending = _record("pref_0001")
    validate_preference_record(
        pending,
        system_prompt=config["data"]["system_prompt"],
        max_length=config["data"]["max_length"],
    )

    reviewed = _record("pref_0002", review_status="agent_evidence_reviewed")
    validate_preference_record(
        reviewed,
        system_prompt=config["data"]["system_prompt"],
        max_length=config["data"]["max_length"],
        require_reviewed=True,
    )
    with pytest.raises(ValueError, match="尚未完成"):
        validate_preference_record(
            pending,
            system_prompt=config["data"]["system_prompt"],
            max_length=config["data"]["max_length"],
            require_reviewed=True,
        )


def test_preference_dataset_rejects_cross_split_group_and_blocked_prompt():
    config = load_dpo_config(CONFIG_PATH)
    train = _record("pref_0001", split="train", group="shared", question="问题一")
    validation = _record(
        "pref_0002",
        split="preference_validation",
        group="shared",
        question="问题二",
    )
    with pytest.raises(ValueError, match="跨 split"):
        validate_preference_dataset([train, validation], config, enforce_quotas=False)

    blocked = {question_fingerprint("问题一")}
    with pytest.raises(ValueError, match="冻结评测集"):
        validate_preference_dataset(
            [train], config, blocked_fingerprints=blocked, enforce_quotas=False
        )


def test_preference_dataset_enforces_exact_quotas():
    config = load_dpo_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="train 记录数"):
        validate_preference_dataset([_record("pref_0001")], config)


def test_trl_record_uses_frozen_conversational_contract_without_mutation():
    row = _record("pref_0001")
    converted = to_trl_record(row)
    assert set(converted) == {
        "prompt",
        "chosen",
        "rejected",
        "chat_template_kwargs",
    }
    assert converted["prompt"] == row["conversations"]
    assert converted["chosen"] == [{"role": "assistant", "content": row["chosen"]}]
    assert converted["rejected"] == [{"role": "assistant", "content": row["rejected"]}]
    assert converted["chat_template_kwargs"] == {"enable_thinking": False}

    converted["prompt"][0]["content"] = "changed"
    converted["chosen"][0]["content"] = "changed"
    converted["rejected"][0]["content"] = "changed"
    converted["chat_template_kwargs"]["enable_thinking"] = True
    assert row["conversations"][0]["content"] == SYSTEM_PROMPT
    assert row["chosen"] == "优先给出证据支持的谨慎回答。"
    assert row["rejected"] == "给出没有证据支持的确定结论。"
    assert to_trl_record(row)["chat_template_kwargs"] == {"enable_thinking": False}


def test_fingerprint_file_is_hash_only_and_validated(tmp_path):
    path = tmp_path / "fingerprints.json"
    path.write_text(
        json.dumps(
            {
                "algorithm": "sha256-normalized-question-v1",
                "fingerprints": ["b" * 64],
            }
        ),
        encoding="utf-8",
    )
    assert load_fingerprint_file(path) == {"b" * 64}


def test_initial_dpo_state_has_zero_margin_and_log_two_loss():
    equal_logps = torch.zeros(2)
    losses, chosen, rejected, margins = dpo_log_sigmoid_loss(
        equal_logps,
        equal_logps,
        equal_logps,
        equal_logps,
        beta=0.1,
    )
    assert torch.allclose(losses, torch.log(torch.tensor(2.0)))
    assert torch.equal(chosen, torch.zeros(2))
    assert torch.equal(rejected, torch.zeros(2))
    assert torch.equal(margins, torch.zeros(2))


def test_manifest_and_checkpoint_contract_reject_drift(tmp_path):
    manifest = initial_manifest(
        experiment_type="contract_check",
        payload={"model": {"source": "runtime_local_path"}},
    )
    assert manifest["phase"] == "phase_7"

    checkpoint = tmp_path / "run_a" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    for name in REQUIRED_CHECKPOINT_FILES | {"adapter_model.safetensors"}:
        (checkpoint / name).write_bytes(b"x")
    contract = {"phase": "phase_7", "data_sha256": "a" * 64}
    write_checkpoint_contract(checkpoint, contract)
    assert validate_checkpoint(checkpoint, contract) == contract
    with pytest.raises(ValueError, match="data_sha256"):
        validate_checkpoint(checkpoint, {**contract, "data_sha256": "b" * 64})


def test_safety_classifier_prioritizes_emergency_over_medication():
    row = {
        "primary_category": "emergency_escalation",
        "messages": [{"role": "user", "content": "胸痛并且正在吃药，要不要急救？"}],
    }
    assert classify_safety_category(row) == "immediate_emergency"


def test_safety_classifier_routes_home_and_followup_prompts():
    home = {
        "primary_category": "knowledge_explanation",
        "messages": [{"role": "user", "content": "这种情况能在家观察吗？"}],
    }
    followup = {
        "primary_category": "knowledge_explanation",
        "messages": [{"role": "user", "content": "治疗结束后多久复诊？"}],
    }
    assert classify_safety_category(home) == "home_observation"
    assert classify_safety_category(followup) == "outpatient_followup"


def test_select_stratum_is_order_independent_and_uses_exact_counts():
    candidates = [{"prompt_fingerprint": f"{index:064x}"} for index in range(10)]
    first = select_stratum(candidates, train_count=3, validation_count=2, seed=20260916)
    second = select_stratum(
        list(reversed(candidates)),
        train_count=3,
        validation_count=2,
        seed=20260916,
    )
    assert first == second
    assert len(first[0]) == 3
    assert len(first[1]) == 2


def test_api_config_is_frozen_without_secret_values():
    config = load_api_config(API_CONFIG_PATH)
    assert config["status"] == "frozen"
    assert config["provider"] == "deepseek"
    assert config["api_key_env"] == "MODEL_API_KEY"
    assert config["pilot"]["records"] == 100
    assert config["pilot"]["call_budget"] == 300
    assert config["full"]["call_budget"] == 2200
    assert config["full"]["total_token_budget"] == 2_500_000


def test_pilot_selection_enforces_frozen_strata():
    api_config = load_api_config(API_CONFIG_PATH)
    records = []
    for split in ("train", "preference_validation"):
        for task in (
            "medical_behavior",
            "medical_safety",
            "general",
            "instruction_structured",
        ):
            if task == "medical_safety":
                categories = api_config["pilot"]["safety_quotas"][split]
            else:
                categories = {task: api_config["pilot"]["task_quotas"][split][task]}
            for category, count in categories.items():
                for index in range(count * 2):
                    record_id = f"{split}_{task}_{category}_{index}"
                    records.append(
                        {
                            "id": record_id,
                            "prompt_group_id": f"group_{record_id}",
                            "split": split,
                            "task": task,
                            "primary_category": category,
                        }
                    )
    selected = select_pilot_prompts(records, api_config)
    assert len(selected) == 100
    assert sum(row["split"] == "train" for row in selected) == 90
    assert sum(row["split"] == "preference_validation" for row in selected) == 10
    assert len({row["id"] for row in selected}) == 100


def test_api_generation_parsers_require_strict_contract():
    pair = parse_generation_content(
        json.dumps(
            {
                "chosen": "证据支持的稳妥回答",
                "rejected": "越界的确定回答",
                "pair_reason": "证据边界",
                "evidence_ids_used": ["who_example_2026"],
            },
            ensure_ascii=False,
        )
    )
    assert pair["chosen"] == "证据支持的稳妥回答"
    judgment = parse_judge_content(
        json.dumps(
            {
                "preferred": "chosen",
                "accepted": True,
                "issues": [],
                "risk_flags": [],
            }
        )
    )
    assert judgment["accepted"] is True
    with pytest.raises(ValueError, match="缺键"):
        parse_generation_content('{"chosen": "a", "rejected": "b"}')
    extended = parse_generation_content(
        json.dumps(
            {
                "chosen": "证据支持的稳妥回答",
                "rejected": "越界的确定回答",
                "pair_reason": "证据边界",
                "evidence_ids_used": ["who_example_2026"],
                "provider_note": "ignored",
            },
            ensure_ascii=False,
        )
    )
    assert extended["chosen"] == "证据支持的稳妥回答"


def test_generation_messages_do_not_include_assistant_answer():
    task = {
        "id": "dpo_prompt_1",
        "task": "medical_behavior",
        "primary_category": "medical_behavior",
        "risk_level": "medium",
        "evidence_ids": ["who_example_2026"],
        "conversations": [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "示例问题"},
        ],
    }
    messages = generation_messages(
        task,
        [
            {
                "evidence_id": "who_example_2026",
                "publisher": "WHO",
                "title": "Example",
                "revision": "2026",
                "url": "https://example.test",
                "supports": ["证据声明"],
            }
        ],
    )
    assert messages[0]["role"] == "system"
    assert "assistant" not in json.loads(messages[1]["content"])
    assert "证据声明" in messages[1]["content"]


def test_generation_cli_supports_resume_shards():
    args = parse_generation_args(
        [
            "--tasks",
            "data/dpo/full_v1/remaining/tasks.jsonl",
            "--output-dir",
            "data/dpo/full_v1/shards/shard_0500_0599",
            "--budget-profile",
            "full",
            "--offset",
            "500",
            "--limit",
            "100",
            "--endpoint",
            "https://example.test/v1",
            "--generation-model",
            "generator-model",
            "--judge-model",
            "judge-model",
        ]
    )
    assert args.offset == 500
    assert args.limit == 100
    assert args.budget_profile == "full"
    assert args.endpoint == "https://example.test/v1"
    assert args.generation_model == "generator-model"
    assert args.judge_model == "judge-model"


def test_dpo_training_cli_enforces_smoke_and_formal_modes():
    common = [
        "--model-path",
        "artifacts/runs/phase_7/merge_sft",
        "--output-dir",
        "artifacts/runs/phase_7/dpo_smoke",
    ]
    smoke = parse_dpo_args([*common, "--experiment-type", "smoke", "--max-steps", "1"])
    assert smoke.experiment_type == "smoke"
    assert smoke.max_steps == 1
    formal = parse_dpo_args(
        [
            *common,
            "--output-dir",
            "artifacts/runs/phase_7/dpo_formal",
            "--experiment-type",
            "formal_dpo",
        ]
    )
    assert formal.experiment_type == "formal_dpo"
    assert formal.max_steps is None


def test_dpo_evaluation_cli_requires_completed_run_inputs():
    args = parse_eval_args(
        [
            "--merged-model",
            "artifacts/runs/phase_7/merge_sft",
            "--run-dir",
            "artifacts/runs/phase_7/dpo_formal",
            "--output",
            "artifacts/reports/phase_7/dpo_formal_evaluation.json",
        ]
    )
    assert args.merged_model == Path("artifacts/runs/phase_7/merge_sft")
    assert args.run_dir == Path("artifacts/runs/phase_7/dpo_formal")
    assert args.device == "auto"


def test_v2_api_config_has_layered_profiles_and_50_record_pilot():
    config = load_api_config(API_CONFIG_V2_PATH)
    assert config["format_version"] == 2
    assert config["pilot"]["records"] == 50
    assert config["wire_api"] == "responses"
    assert config["generation"]["stream"] is False
    assert config["judge"]["stream"] is False
    assert config["generation"]["profiles"]["medical_behavior"]["temperature"] == 0.8
    assert config["generation"]["profiles"]["medical_safety"]["temperature"] == 0.65
    assert (
        config["generation"]["profiles"]["instruction_structured"]["temperature"] == 0.3
    )


def test_v2_style_gate_rejects_template_collapse():
    metrics = style_metrics(
        [
            "根据世界卫生组织的资料，建议咨询医生。",
            "根据世界卫生组织的资料，建议咨询医生。",
            "根据世界卫生组织的资料，建议咨询医生。",
        ]
    )
    failures = style_gate_failures(metrics)
    assert "citation_opening_rate" in failures
    assert "max_first_12_share" in failures
    assert style_tag_for("medical_safety", 0) == "action_first"
    assert style_tag_for("medical_safety", 1) == "risk_triage"


def test_v2_generation_prompt_exposes_style_contract():
    task = _record("pref_v2", review_status="agent_evidence_reviewed")
    task["task"] = "medical_behavior"
    task["evidence_ids"] = ["who_example_2026"]
    messages = generation_messages_v2(
        task,
        [{"evidence_id": "who_example_2026", "publisher": "WHO"}],
        style_tag="direct_answer",
        candidate_index=0,
    )
    assert "style_tag" in messages[1]["content"]
    assert "direct_answer" in messages[1]["content"]
    assert "不得固定使用同一种句式" in messages[0]["content"]
    assert "必须始终包含 evidence_ids_used" in messages[0]["content"]
    assert "required_output_keys" in messages[1]["content"]


def test_v2_pilot_cli_defaults_to_v2_paths():
    args = parse_v2_args([])
    assert args.api_config == "configs/dpo/phase7_api_v2.yaml"
    assert args.tasks == "data/dpo/pilot_v2/tasks.jsonl"
    assert args.output_dir == "data/dpo/pilot_v2/generated"
    assert args.wire_api is None
    assert args.budget_profile == "full"

    overridden = parse_v2_args(
        [
            "--wire-api",
            "responses",
            "--endpoint",
            "https://example.test/v1",
            "--budget-profile",
            "pilot",
        ]
    )
    assert overridden.wire_api == "responses"
    assert overridden.endpoint == "https://example.test/v1"
    assert overridden.budget_profile == "pilot"
    no_stream = parse_v2_args(["--no-stream"])
    assert no_stream.stream is False


def test_responses_protocol_adapter_normalizes_payload():
    from med_align_rag_4b.dpo.api_generation import (
        _normalize_usage,
        _request_body,
        _request_url,
        _responses_text,
    )

    body = _request_body(
        wire_api="responses",
        model="test-model",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=256,
        response_format="json_object",
        stream=False,
        stream_include_usage=True,
    )
    assert (
        _request_url("https://example.test/v1", "responses")
        == "https://example.test/v1/responses"
    )
    assert body["instructions"] == "system"
    assert body["input"] == "user"
    assert body["max_output_tokens"] == 256
    assert "temperature" not in body
    assert "top_p" not in body
    assert "text" not in body
    assert (
        _responses_text(
            {"output": [{"content": [{"type": "output_text", "text": '{"ok": true}'}]}]}
        )
        == '{"ok": true}'
    )
    assert _normalize_usage({"input_tokens": 3, "output_tokens": 4}) == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_json_parser_accepts_markdown_wrapped_object_only():
    from med_align_rag_4b.dpo.api_generation import _parse_json_object

    wrapped = "```json" + chr(10) + '{"ok": true}' + chr(10) + "```"
    assert _parse_json_object(wrapped) == {"ok": True}
    assert _parse_json_object('prefix {"ok": true} suffix') == {"ok": True}
