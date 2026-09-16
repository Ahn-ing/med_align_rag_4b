import json
from pathlib import Path

import pytest

from med_align_rag_4b.sft.checkpoints import (
    REQUIRED_CHECKPOINT_FILES,
    validate_checkpoint,
)
from med_align_rag_4b.sft.config import (
    derive_optimizer_steps,
    load_sft_config,
    validate_phase6_config,
)
from med_align_rag_4b.sft.data import (
    load_eval_and_encode,
    validate_phase6_release,
    validate_phase6_validation,
)
from med_align_rag_4b.sft.evaluation import SYSTEM_PROMPT
from med_align_rag_4b.sft.run_manifest import initial_manifest
from scripts.sft.run_sft import parse_args, project_output

ROOT = Path(__file__).resolve().parents[2]


def test_phase6_schedule_is_frozen_to_two_epochs_and_298_steps():
    config = load_sft_config(ROOT / "configs/sft/qwen3_4b_lora_phase6.yaml")
    validate_phase6_config(config)
    assert derive_optimizer_steps(9495, 64, 2) == 298
    config["training"]["epochs"] = 3
    with pytest.raises(ValueError, match="training.epochs"):
        validate_phase6_config(config)


def test_formal_train_cli_rejects_max_steps_or_non_two_epochs():
    base = [
        "--model-path", "model", "--output-dir", "artifacts/runs/phase_6/run",
        "--experiment-type", "formal_train", "--eval-data",
        "data/evaluation/validation_v2/cases.jsonl",
    ]
    assert parse_args([*base, "--epochs", "2"]).epochs == 2
    with pytest.raises(SystemExit):
        parse_args([*base, "--epochs", "2", "--max-steps", "264"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--epochs", "3"])


def test_phase5_cli_remains_max_steps_based():
    args = parse_args([
        "--model-path", "model", "--output-dir", "artifacts/runs/phase_5/run",
        "--experiment-type", "smoke", "--max-steps", "1",
    ])
    assert args.max_steps == 1 and args.epochs is None


def test_phase6_output_is_isolated_from_phase5():
    expected = (ROOT / "artifacts/runs/phase_6/run").resolve()
    assert project_output(ROOT, expected, "formal_train") == expected
    with pytest.raises(ValueError, match="phase_6"):
        project_output(ROOT, ROOT / "artifacts/runs/phase_5/run", "formal_train")


def test_phase6_train_and_full_validation_gates():
    release = validate_phase6_release(ROOT, ROOT / "data/sft/released/v2_plus_repair_r2/train.jsonl")
    assert release["training_allowed"] is True
    assert release["counts"]["train"] == 9495
    validation = validate_phase6_validation(
        ROOT, ROOT / "data/evaluation/validation_v2/cases.jsonl"
    )
    assert validation["records"] == 60
    assert validation["gradient_update_allowed"] is False
    assert validation["generation_allowed"] is False


def test_full_validation_cases_are_converted_to_sft_messages(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "v1", "prompt": "问题", "reference_answer": "答案"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    class RecordingEncoder:
        def __init__(self):
            self.messages = None

        def encode(self, messages):
            self.messages = messages
            return {"active_label_tokens": 1}

    encoder = RecordingEncoder()
    assert load_eval_and_encode(path, encoder) == [{"active_label_tokens": 1}]
    assert encoder.messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案"},
    ]


def test_phase6_manifest_and_checkpoint_contract(tmp_path):
    manifest = initial_manifest(
        phase="phase_6", experiment_type="formal_train",
        payload={"model": {"source": "runtime_local_path"}},
    )
    assert manifest["phase"] == "phase_6"
    checkpoint = tmp_path / "run_a" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    for name in REQUIRED_CHECKPOINT_FILES | {"adapter_model.safetensors"}:
        (checkpoint / name).write_bytes(b"x")
    contract = {"phase": "phase_6", "data_sha256": "abc"}
    (checkpoint / "phase6_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    assert validate_checkpoint(checkpoint, contract) == contract
    with pytest.raises(ValueError, match="data_sha256"):
        validate_checkpoint(checkpoint, {**contract, "data_sha256": "other"})
