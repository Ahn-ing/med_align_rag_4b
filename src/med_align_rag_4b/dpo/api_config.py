"""Phase 7 API 配置的冻结契约，不保存 API 密钥。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXPECTED_ROOT = {
    "status": "frozen",
    "provider": "deepseek",
    "endpoint": "https://api.deepseek.com",
    "api_key_env": "MODEL_API_KEY",
}
EXPECTED_GENERATION = {
    "model": "deepseek-flash",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1600,
    "response_format": "json_object",
    "timeout_seconds": 90,
    "max_attempts": 3,
}
EXPECTED_JUDGE = {
    "model": "deepseek-flash",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 600,
    "response_format": "json_object",
    "timeout_seconds": 90,
    "max_attempts": 3,
}
EXPECTED_FULL = {
    "call_budget": 2200,
    "total_token_budget": 2_500_000,
}
EXPECTED_PILOT = {
    "seed": 20260916,
    "records": 100,
    "call_budget": 300,
    "total_token_budget": 1_000_000,
    "task_quotas": {
        "train": {
            "medical_behavior": 46,
            "medical_safety": 22,
            "general": 18,
            "instruction_structured": 4,
        },
        "preference_validation": {
            "medical_behavior": 4,
            "medical_safety": 3,
            "general": 2,
            "instruction_structured": 1,
        },
    },
    "safety_quotas": {
        "train": {
            "immediate_emergency": 5,
            "outpatient_followup": 3,
            "home_observation": 3,
            "clarification": 3,
            "medication_boundary": 4,
            "uncertainty": 4,
        },
        "preference_validation": {
            "immediate_emergency": 0,
            "outpatient_followup": 1,
            "home_observation": 1,
            "clarification": 1,
            "medication_boundary": 0,
            "uncertainty": 0,
        },
    },
}


def load_api_config(path: str | Path) -> dict[str, Any]:
    """加载 v1 或 v2 API 配置，并拒绝契约漂移。"""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("API 配置必须是 YAML 对象")
    config = deepcopy(value)
    if config.get("format_version") == 2:
        validate_api_config_v2(config)
    else:
        validate_api_config(config)
    return config


def validate_api_config(config: dict[str, Any]) -> None:
    """拒绝 v1 provider、model、参数和预算的静默漂移。"""
    for key, expected in EXPECTED_ROOT.items():
        if config.get(key) != expected:
            raise ValueError(f"API 配置 {key} 必须为 {expected}")
    for section, expected in (
        ("generation", EXPECTED_GENERATION),
        ("judge", EXPECTED_JUDGE),
    ):
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"API 配置缺少 {section}")
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(f"API 配置 {section}.{key} 必须为 {expected_value}")
    for section, expected_section in (
        ("pilot", EXPECTED_PILOT),
        ("full", EXPECTED_FULL),
    ):
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"API 配置缺少 {section}")
        for key, expected in expected_section.items():
            if value.get(key) != expected:
                raise ValueError(f"API 配置 {section}.{key} 不匹配冻结值")


V2_TASKS = (
    "medical_behavior",
    "medical_safety",
    "general",
    "instruction_structured",
)


def validate_api_config_v2(config: dict[str, Any]) -> None:
    """校验 v2 分层采样、候选数、预算和 50 条 pilot 配额。"""
    if config.get("format_version") != 2:
        raise ValueError("v2 API 配置 format_version 必须为 2")
    if config.get("status") != "frozen":
        raise ValueError("v2 API 配置 status 必须为 frozen")
    for key in ("provider", "endpoint", "api_key_env"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"v2 API 配置 {key} 必须是非空字符串")
    wire_api = config.get("wire_api", "chat_completions")
    if wire_api not in {"chat_completions", "responses"}:
        raise ValueError("v2 API 配置 wire_api 必须是 chat_completions 或 responses")

    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("v2 API 配置缺少 generation")
    for key in ("model", "response_format"):
        if not isinstance(generation.get(key), str) or not generation[key].strip():
            raise ValueError(f"v2 generation.{key} 必须是非空字符串")
    for key in ("max_tokens", "timeout_seconds", "max_attempts"):
        if not isinstance(generation.get(key), int) or generation[key] <= 0:
            raise ValueError(f"v2 generation.{key} 必须为正整数")
    profiles = generation.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(V2_TASKS):
        raise ValueError("v2 generation.profiles 必须覆盖全部任务")
    for task, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"v2 profile 不是对象: {task}")
        temperature = profile.get("temperature")
        top_p = profile.get("top_p")
        candidates = profile.get("candidates")
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError(f"v2 profile temperature 无效: {task}")
        if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError(f"v2 profile top_p 无效: {task}")
        if not isinstance(candidates, int) or not 1 <= candidates <= 4:
            raise ValueError(f"v2 profile candidates 无效: {task}")

    judge = config.get("judge")
    if not isinstance(judge, dict):
        raise ValueError("v2 API 配置缺少 judge")
    for key, expected in EXPECTED_JUDGE.items():
        if judge.get(key) != expected:
            raise ValueError(f"v2 API 配置 judge.{key} 必须为 {expected}")

    pilot = config.get("pilot")
    if not isinstance(pilot, dict):
        raise ValueError("v2 API 配置缺少 pilot")
    if pilot.get("records") != 50 or pilot.get("seed") != 20260917:
        raise ValueError("v2 pilot 必须固定为 50 条、seed=20260917")
    for key in ("call_budget", "total_token_budget"):
        if not isinstance(pilot.get(key), int) or pilot[key] <= 0:
            raise ValueError(f"v2 pilot.{key} 必须为正整数")
    _validate_task_quotas(pilot, expected_total=50)
    _validate_safety_quotas(pilot)
    full = config.get("full")
    if not isinstance(full, dict):
        raise ValueError("v2 API 配置缺少 full")
    for key in ("call_budget", "total_token_budget"):
        if not isinstance(full.get(key), int) or full[key] <= 0:
            raise ValueError(f"v2 full.{key} 必须为正整数")


def _validate_task_quotas(pilot: dict[str, Any], *, expected_total: int) -> None:
    quotas = pilot.get("task_quotas")
    if not isinstance(quotas, dict) or set(quotas) != {
        "train",
        "preference_validation",
    }:
        raise ValueError("v2 pilot.task_quotas 必须覆盖两个 split")
    total = 0
    for split, values in quotas.items():
        if not isinstance(values, dict) or set(values) != set(V2_TASKS):
            raise ValueError(f"v2 pilot.task_quotas 缺少任务: {split}")
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError(f"v2 pilot.task_quotas 必须为非负整数: {split}")
        total += sum(values.values())
    if total != expected_total:
        raise ValueError(f"v2 pilot.task_quotas 总和必须为 {expected_total}")


def _validate_safety_quotas(pilot: dict[str, Any]) -> None:
    quotas = pilot.get("safety_quotas")
    if not isinstance(quotas, dict) or set(quotas) != {
        "train",
        "preference_validation",
    }:
        raise ValueError("v2 pilot.safety_quotas 必须覆盖两个 split")
    categories = {
        "clarification",
        "home_observation",
        "immediate_emergency",
        "medication_boundary",
        "outpatient_followup",
        "uncertainty",
    }
    for split, values in quotas.items():
        if not isinstance(values, dict) or set(values) != categories:
            raise ValueError(f"v2 pilot.safety_quotas 缺少类别: {split}")
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError(f"v2 pilot.safety_quotas 必须为非负整数: {split}")
        expected = pilot["task_quotas"][split]["medical_safety"]
        if sum(values.values()) != expected:
            raise ValueError(f"v2 pilot.safety_quotas 总和不匹配: {split}")
