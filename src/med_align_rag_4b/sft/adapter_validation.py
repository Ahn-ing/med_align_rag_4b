"""LoRA adapter 结构和可训练参数检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_target_modules(model: Any, expected: list[str]) -> None:
    """在注入 LoRA 前确认每个目标后缀都存在。"""
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(expected) - module_names
    if missing:
        raise ValueError(f"模型缺少 LoRA target modules: {sorted(missing)}")


def validate_trainable_parameters(model: Any) -> dict[str, float | int]:
    """确认只有 LoRA 参数可训练。"""
    total = 0
    trainable = 0
    unexpected = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            if "lora_" not in name:
                unexpected.append(name)
    if not trainable or unexpected:
        raise ValueError(f"LoRA 可训练参数异常: {unexpected}")
    return {"trainable": trainable, "total": total, "ratio": trainable / total}


def validate_adapter_directory(path: str | Path, expected: dict[str, Any]) -> dict[str, Any]:
    """不加载权重地校验 adapter 配置和文件。"""
    directory = Path(path)
    config_path = directory / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError("adapter_config.json 不存在")
    names = {item.name for item in directory.iterdir() if item.is_file()}
    if not ({"adapter_model.safetensors", "adapter_model.bin"} & names):
        raise ValueError("adapter 权重不存在")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks = {
        "r": expected["rank"], "lora_alpha": expected["alpha"],
        "lora_dropout": expected["dropout"], "bias": expected["bias"],
    }
    for key, value in checks.items():
        if config.get(key) != value:
            raise ValueError(f"adapter 配置不匹配: {key}")
    if set(config.get("target_modules", [])) != set(expected["target_modules"]):
        raise ValueError("adapter target_modules 不匹配")
    return config
