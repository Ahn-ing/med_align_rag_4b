"""Phase 7 DPO 配置加载与稳定不变量。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "preference_validation")
TASKS = (
    "medical_behavior",
    "medical_safety",
    "general",
    "instruction_structured",
)
SAFETY_CATEGORIES = (
    "clarification",
    "home_observation",
    "immediate_emergency",
    "medication_boundary",
    "outpatient_followup",
    "uncertainty",
)
EXPECTED_MODEL = {
    "model_id": "Qwen/Qwen3-4B",
    "enable_thinking": False,
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "trust_remote_code": False,
    "local_files_only": True,
}
EXPECTED_REFERENCE = {
    "strategy": "merged_sft",
    "base_only_allowed": False,
}
EXPECTED_LORA = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}
EXPECTED_TRAINING = {
    "beta": 0.1,
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 1,
    "evaluation_steps": 50,
    "save_steps": 50,
    "save_total_limit": 2,
    "seed": 42,
    "data_seed": 42,
}
FROZEN_TRAINING_FIELDS = (
    "learning_rate",
    "per_device_train_batch_size",
    "global_batch_size",
    "epochs",
)


def load_dpo_config(path: str | Path) -> dict[str, Any]:
    """加载 Phase 7 YAML，并拒绝契约漂移。"""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DPO 配置必须是 YAML 对象")
    config = deepcopy(value)
    validate_phase7_config(config)
    return config


def validate_phase7_config(config: dict[str, Any]) -> None:
    """校验 Phase 7 的模型、数据、reference、LoRA 和训练门禁。"""
    status = config.get("status")
    if status not in {"contract_only", "frozen"}:
        raise ValueError("Phase 7 配置 status 必须是 contract_only 或 frozen")
    expected_training_allowed = status == "frozen"
    if config.get("training_allowed") is not expected_training_allowed:
        raise ValueError("training_allowed 必须与配置 status 一致")

    _validate_exact_section(config, "model", EXPECTED_MODEL)
    _validate_exact_section(config, "reference", EXPECTED_REFERENCE)
    _validate_lora(config.get("lora"))
    _validate_training(config.get("training"), status=status)
    _validate_data(config.get("data"))


def require_frozen_training_parameters(config: dict[str, Any]) -> None:
    """正式训练入口调用前，要求配置已离开 contract_only。"""
    if config.get("status") != "frozen":
        raise ValueError("DPO 配置尚未冻结，禁止启动训练")
    if config.get("training_allowed") is not True:
        raise ValueError("DPO 配置未明确允许训练")
    missing = [
        name for name in FROZEN_TRAINING_FIELDS if config["training"].get(name) is None
    ]
    if missing:
        raise ValueError(f"DPO 训练参数尚未冻结: {', '.join(missing)}")


def _validate_exact_section(
    config: dict[str, Any], section: str, expected: dict[str, Any]
) -> None:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"DPO 配置缺少对象: {section}")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"DPO 配置 {section}.{key} 必须为 {expected_value}")


def _validate_lora(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("DPO 配置缺少 lora")
    for key in ("rank", "alpha", "dropout", "bias"):
        if value.get(key) != EXPECTED_LORA[key]:
            raise ValueError(f"DPO 配置 lora.{key} 必须为 {EXPECTED_LORA[key]}")
    if set(value.get("target_modules", [])) != set(EXPECTED_LORA["target_modules"]):
        raise ValueError("DPO 配置 lora.target_modules 不匹配已批准集合")


def _validate_training(value: Any, *, status: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("DPO 配置缺少 training")
    for key, expected in EXPECTED_TRAINING.items():
        if value.get(key) != expected:
            raise ValueError(f"DPO 配置 training.{key} 必须为 {expected}")
    for key in FROZEN_TRAINING_FIELDS:
        if key not in value:
            raise ValueError(f"DPO 配置 training.{key} 必须显式存在")
    if status == "contract_only":
        if any(value[key] is not None for key in FROZEN_TRAINING_FIELDS):
            raise ValueError("contract_only 配置不得提前填写训练参数")
        return
    if any(
        not isinstance(value[key], (int, float)) or value[key] <= 0
        for key in FROZEN_TRAINING_FIELDS
    ):
        raise ValueError("frozen 配置的训练参数必须为正数")
    if value["global_batch_size"] % value["per_device_train_batch_size"]:
        raise ValueError("global_batch_size 必须整除 per_device_train_batch_size")


def _validate_data(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("DPO 配置缺少 data")
    if value.get("max_length") != 2048:
        raise ValueError("DPO 配置 data.max_length 必须为 2048")
    if value.get("split_seed") != 20260916:
        raise ValueError("DPO 配置 data.split_seed 必须为 20260916")
    system_prompt = value.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("DPO 配置 data.system_prompt 必须非空")

    counts = value.get("counts")
    if counts != {"train": 900, "preference_validation": 100}:
        raise ValueError("DPO 配置 data.counts 必须固定为 900/100")

    quotas = value.get("task_quotas")
    if not isinstance(quotas, dict) or set(quotas) != set(SPLITS):
        raise ValueError("DPO 配置 data.task_quotas 缺少 split")
    for split in SPLITS:
        split_quotas = quotas[split]
        if not isinstance(split_quotas, dict) or set(split_quotas) != set(TASKS):
            raise ValueError(f"DPO 配置 {split} task_quotas 不完整")
        if any(
            not isinstance(item, int) or item <= 0 for item in split_quotas.values()
        ):
            raise ValueError(f"DPO 配置 {split} task_quotas 必须为正整数")
        if sum(split_quotas.values()) != counts[split]:
            raise ValueError(f"DPO 配置 {split} task_quotas 总和必须为 {counts[split]}")

    safety = value.get("safety_quotas")
    if not isinstance(safety, dict) or set(safety) != set(SPLITS):
        raise ValueError("DPO 配置 data.safety_quotas 缺少 split")
    for split in SPLITS:
        split_safety = safety[split]
        if not isinstance(split_safety, dict) or set(split_safety) != set(
            SAFETY_CATEGORIES
        ):
            raise ValueError(f"DPO 配置 {split} safety_quotas 不完整")
        if any(
            not isinstance(item, int) or item <= 0 for item in split_safety.values()
        ):
            raise ValueError(f"DPO 配置 {split} safety_quotas 必须为正整数")
        if sum(split_safety.values()) != quotas[split]["medical_safety"]:
            raise ValueError(f"DPO 配置 {split} safety_quotas 总和不匹配")
