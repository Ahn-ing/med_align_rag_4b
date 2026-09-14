import json
from types import SimpleNamespace

import pytest
import torch

from med_align_rag_4b.sft.adapter_validation import (
    validate_target_modules,
    validate_trainable_parameters,
)
from med_align_rag_4b.sft.checkpoints import (
    REQUIRED_CHECKPOINT_FILES,
    validate_checkpoint,
)
from med_align_rag_4b.sft.collator import AssistantOnlyCollator
from med_align_rag_4b.sft.config import (
    derive_gradient_accumulation,
    derive_warmup_steps,
    sanitized_config,
)
from med_align_rag_4b.sft.data import validate_phase4_release, validate_phase5_eval
from med_align_rag_4b.sft.evaluation import (
    build_loss_records,
    load_fixtures,
    validate_fixture_output,
)
from med_align_rag_4b.sft.loss import token_normalized_causal_loss
from med_align_rag_4b.sft.run_manifest import initial_manifest
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer
from med_align_rag_4b.sft.training import EncodedDataset, evaluate_active_token_loss


def base_config():
    return {
        "model": {"model_id": "Qwen/Qwen3-4B", "model_path": "C:/secret/model"},
        "training": {"per_device_train_batch_size": 4, "global_batch_size": 64},
    }


def test_batch_derivation_and_sanitization():
    config = base_config()
    assert derive_gradient_accumulation(config) == 16
    config["training"]["per_device_train_batch_size"] = 2
    assert derive_gradient_accumulation(config) == 32
    clean = sanitized_config(config)
    assert "model_path" not in clean["model"]
    assert clean["model"]["model_source"] == "runtime_local_path"
    with pytest.raises(ValueError):
        derive_gradient_accumulation(config, world_size=2)
    assert derive_warmup_steps(16, 0.03) == 1
    assert derive_warmup_steps(100, 0.03) == 3


def test_dynamic_right_padding_masks_labels():
    batch = AssistantOnlyCollator(9)([
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]


def test_loss_ignores_prompt_and_padding_but_uses_assistant():
    logits = torch.zeros((1, 4, 3), dtype=torch.float32)
    labels = torch.tensor([[-100, -100, 1, -100]])
    baseline = token_normalized_causal_loss(logits, labels)
    prompt_changed = logits.clone()
    prompt_changed[0, 0, 2] = 20
    assert torch.equal(baseline, token_normalized_causal_loss(prompt_changed, labels))
    assistant_changed = logits.clone()
    assistant_changed[0, 1, 1] = 20
    assert token_normalized_causal_loss(assistant_changed, labels) < baseline
    padded = torch.cat([logits, torch.randn((1, 1, 3))], dim=1)
    padded_labels = torch.cat([labels, torch.tensor([[-100]])], dim=1)
    assert torch.allclose(baseline, token_normalized_causal_loss(padded, padded_labels))


def test_accumulation_uses_window_active_token_denominator():
    logits_a = torch.zeros((1, 3, 2))
    labels_a = torch.tensor([[-100, 0, -100]])
    logits_b = torch.zeros((1, 4, 2))
    labels_b = torch.tensor([[-100, 0, 1, -100]])
    combined = (
        token_normalized_causal_loss(logits_a, labels_a, num_items_in_batch=3)
        + token_normalized_causal_loss(logits_b, labels_b, num_items_in_batch=3)
    )
    assert torch.allclose(combined, torch.tensor(0.6931472), atol=1e-6)


def test_checkpoint_contract_and_required_files(tmp_path):
    contract = {"data_sha256": "abc", "global_batch_size": 64}
    for name in REQUIRED_CHECKPOINT_FILES | {"adapter_model.safetensors"}:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "phase5_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    assert validate_checkpoint(tmp_path, contract) == contract
    (tmp_path / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer.pt"):
        validate_checkpoint(tmp_path, contract)


def test_only_lora_parameters_are_trainable():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.lora_A = torch.nn.Linear(2, 1, bias=False)
            self.base.requires_grad_(False)

    stats = validate_trainable_parameters(Tiny())
    assert stats["trainable"] == 2
    assert 0 < stats["ratio"] < 1
    validate_target_modules(Tiny(), ["base", "lora_A"])
    with pytest.raises(ValueError, match="missing"):
        validate_target_modules(Tiny(), ["missing"])


def test_manifest_rejects_absolute_paths():
    with pytest.raises(ValueError, match="绝对路径"):
        initial_manifest(experiment_type="smoke", payload={"model_path": "C:\\models\\qwen"})


def test_phase4_release_gate_uses_current_frozen_contract():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    manifest = validate_phase4_release(
        root, root / "data/sft/released/v1/smoke_train.jsonl"
    )
    assert manifest["status"] == "completed"
    assert manifest["training_allowed"] is True


def test_tokenizer_encode_reuses_prefix_difference(monkeypatch, tmp_path):
    class FakeTokenizer:
        chat_template = "template"
        pad_token_id = None
        eos_token_id = 3
        eos_token = "<eos>"
        pad_token = None
        padding_side = "left"

        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            if add_generation_prompt:
                return [10, 11]
            return [10, 11, 2, 3]

    fake = FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: fake
    )
    encoder = QwenAssistantTokenizer(tmp_path, max_length=4)
    item = encoder.encode([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ])
    assert fake.padding_side == "right"
    assert fake.pad_token == "<eos>"
    assert item == {
        "input_ids": [10, 11, 2, 3],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 2, 3],
        "active_label_tokens": 2,
    }


def test_phase5_loss_subset_is_frozen_and_has_approved_quotas():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    from med_align_rag_4b.common.jsonl import iter_jsonl

    source = list(iter_jsonl(root / "data/evaluation/validation_v1/cases.jsonl"))
    first = build_loss_records(source)
    second = build_loss_records(list(reversed(source)))
    assert first == second
    assert len(first) == 18
    assert [row["id"] for row in first] == [
        "validation_general_004", "validation_general_007",
        "validation_instruction_001", "validation_instruction_005",
        "validation_medical_qa_004", "validation_medical_qa_010",
        "validation_medical_qa_013", "validation_medical_qa_019",
        "validation_rag_002", "validation_rag_003",
        "validation_safety_clarification_001",
        "validation_safety_home_observation_002",
        "validation_safety_immediate_emergency_001",
        "validation_safety_medication_boundary_001",
        "validation_safety_outpatient_followup_001",
        "validation_safety_uncertainty_001",
        "validation_structured_001", "validation_structured_004",
    ]


def test_phase5_eval_manifest_and_source_hashes_are_valid():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    manifest = validate_phase5_eval(
        root, root / "data/evaluation/phase5_loss_v1.jsonl"
    )
    assert manifest["records"] == 18
    assert manifest["gradient_update_allowed"] is False


def test_engineering_fixtures_are_isolated_and_mechanically_checked():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    fixtures = load_fixtures(
        root / "data/evaluation/phase5_engineering_fixtures_v1.json"
    )
    assert all(row["source"] == "phase5_engineering_fixture" for row in fixtures)
    assert all(row["evaluation_split"] is None for row in fixtures)
    assert validate_fixture_output(fixtures[0], "版本控制用于追踪并协作管理文件变更。")
    assert validate_fixture_output(fixtures[1], '{"status":"ready","count":3}')["json_valid"]
    with pytest.raises(ValueError, match="think block"):
        validate_fixture_output(fixtures[2], "<think>隐藏推理</think>普通信息")


def test_eval_loss_is_weighted_by_all_active_assistant_tokens():
    class ZeroLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask, labels
            logits = torch.zeros((*input_ids.shape, 2), device=input_ids.device)
            return SimpleNamespace(logits=logits + self.anchor)

    rows = [
        {"input_ids": [0, 0], "attention_mask": [1, 1], "labels": [-100, 0]},
        {"input_ids": [0, 0, 0], "attention_mask": [1, 1, 1], "labels": [-100, 0, 1]},
    ]
    loss = evaluate_active_token_loss(
        ZeroLogitModel(), EncodedDataset(rows), AssistantOnlyCollator(0)
    )
    assert loss == pytest.approx(0.6931472, abs=1e-6)
