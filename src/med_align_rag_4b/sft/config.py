"""Phase 5 SFT 配置加载与批量推导。"""

from __future__ import annotations

from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

import yaml


def load_sft_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML 并校验 Phase 5 不变量。"""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("配置必须是 YAML 对象")
    config = deepcopy(value)
    if config.get("model", {}).get("model_id") != "Qwen/Qwen3-4B":
        raise ValueError("逻辑模型必须是 Qwen/Qwen3-4B")
    if config["model"].get("enable_thinking") is not False:
        raise ValueError("Phase 5 必须关闭 thinking")
    if config["model"].get("local_files_only") is not True:
        raise ValueError("必须启用 local_files_only")
    if config["model"].get("trust_remote_code") is not False:
        raise ValueError("必须禁用 trust_remote_code")
    derive_gradient_accumulation(config)
    return config


def derive_gradient_accumulation(config: dict[str, Any], *, world_size: int = 1) -> int:
    """从 global batch 严格推导 gradient accumulation。"""
    training = config["training"]
    micro_batch = int(training["per_device_train_batch_size"])
    global_batch = int(training["global_batch_size"])
    denominator = micro_batch * world_size
    if micro_batch not in {2, 4}:
        raise ValueError("Phase 5 micro-batch 只允许 4 或 OOM 回退值 2")
    if world_size != 1:
        raise ValueError("Phase 5 只允许单卡运行")
    if global_batch % denominator:
        raise ValueError("global batch 必须整除 micro-batch × world size")
    return global_batch // denominator


def derive_warmup_steps(max_steps: int, warmup_ratio: float) -> int:
    """将 Transformers 5 已移除的 warmup_ratio 显式换算为 steps。"""
    if max_steps <= 0:
        raise ValueError("max_steps 必须为正数")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio 必须位于 [0, 1)")
    return ceil(max_steps * warmup_ratio)


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    """返回可持久化且不含本地模型绝对路径的配置。"""
    result = deepcopy(config)
    result["model"].pop("model_path", None)
    result["model"]["model_source"] = "runtime_local_path"
    return result
