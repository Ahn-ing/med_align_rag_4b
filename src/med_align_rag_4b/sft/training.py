"""Qwen3-4B BF16 LoRA SFT 的最小 Transformers 训练链。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.sft.loss import token_normalized_causal_loss


class EncodedDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class CudaMemoryCallback(TrainerCallback):
    """每次 Trainer 日志时记录当前 CUDA 显存，用于短跑增长审计。"""

    def __init__(self) -> None:
        self.samples: list[dict[str, int]] = []

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if torch.cuda.is_available():
            self.samples.append({
                "step": int(state.global_step),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            })


class CheckpointContractCallback(TrainerCallback):
    """checkpoint 保存完成时立即附加恢复契约。"""

    def __init__(self, output_dir: Path, contract: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.contract = contract

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        from med_align_rag_4b.sft.checkpoints import write_checkpoint_contract

        write_checkpoint_contract(
            self.output_dir / f"checkpoint-{int(state.global_step)}", self.contract
        )


class TokenNormalizedTrainer(Trainer):
    """Trainer 最小修正：accumulation window 按 active token 归一化。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True

    def compute_loss(
        self, model: Any, inputs: dict[str, torch.Tensor], return_outputs: bool = False,
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> Any:
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = token_normalized_causal_loss(
            outputs.logits, labels, num_items_in_batch=num_items_in_batch
        )
        return (loss, outputs) if return_outputs else loss


def evaluate_active_token_loss(
    model: Any, dataset: Dataset, collator: Any, *, batch_size: int = 1
) -> float:
    """在整个 eval 子集上按有效 assistant token 精确聚合 loss。"""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    device = next(model.parameters()).device
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            labels = batch["labels"]
            active = int(labels[..., 1:].ne(-100).sum().item())
            outputs = model(**batch)
            mean_loss = token_normalized_causal_loss(outputs.logits, labels)
            total_loss += mean_loss.double() * active
            total_tokens += active
    if total_tokens <= 0:
        raise ValueError("eval 有效 assistant token 数必须大于 0")
    return float((total_loss / total_tokens).item())


def inspect_local_model(path: str | Path) -> dict[str, Any]:
    """只读检查本地 Qwen3-4B 必需文件，不允许网络 fallback。"""
    directory = Path(path).resolve()
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    missing = [name for name in required if not (directory / name).is_file()]
    index = directory / "model.safetensors.index.json"
    single = directory / "model.safetensors"
    if not index.is_file() and not single.is_file():
        missing.append("model.safetensors.index.json|model.safetensors")
    if missing:
        raise ValueError(f"本地模型不完整: {missing}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("本地模型不是 Qwen3")
    shards: list[str]
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weights.values()))
        missing_shards = [name for name in shards if not (directory / name).is_file()]
        if missing_shards:
            raise ValueError(f"本地模型缺少 shard: {missing_shards}")
    else:
        shards = [single.name]
    return {
        "model_type": config["model_type"],
        "architectures": config.get("architectures", []),
        "model_revision": config.get("_commit_hash") or "unknown",
        "file_sha256": {
            name: sha256_file(directory / name)
            for name in sorted(required | ({index.name} if index.is_file() else set()))
        },
        "weight_shards": [{"name": name, "bytes": (directory / name).stat().st_size} for name in shards],
    }
