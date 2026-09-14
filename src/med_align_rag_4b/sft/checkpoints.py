"""LoRA Trainer checkpoint 完整性与恢复兼容性检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_CHECKPOINT_FILES = {
    "adapter_config.json", "optimizer.pt", "scheduler.pt",
    "trainer_state.json", "rng_state.pth",
}


def validate_checkpoint(path: str | Path, expected: dict[str, Any]) -> dict[str, Any]:
    """拒绝文件不全或关键运行契约不一致的 checkpoint。"""
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint 不存在: {checkpoint}")
    names = {item.name for item in checkpoint.iterdir() if item.is_file()}
    missing = REQUIRED_CHECKPOINT_FILES - names
    if not ({"adapter_model.safetensors", "adapter_model.bin"} & names):
        missing.add("adapter_model.safetensors|adapter_model.bin")
    if missing:
        raise ValueError(f"checkpoint 缺少文件: {sorted(missing)}")
    contract_path = checkpoint / "phase5_contract.json"
    if not contract_path.is_file():
        raise ValueError("checkpoint 缺少 phase5_contract.json")
    actual = json.loads(contract_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"checkpoint 契约不匹配: {key}")
    return actual


def write_checkpoint_contract(path: str | Path, contract: dict[str, Any]) -> None:
    """在 Trainer 已创建的 checkpoint 中补充可恢复契约。"""
    destination = Path(path) / "phase5_contract.json"
    destination.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
