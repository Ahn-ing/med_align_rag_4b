"""assistant-only SFT 的动态 right-padding collator。"""

from __future__ import annotations

from typing import Any

import torch


class AssistantOnlyCollator:
    """将变长样本右填充，padding label 固定为 -100。"""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id
        self.samples = 0
        self.input_tokens = 0
        self.assistant_tokens = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("batch 不能为空")
        width = max(len(row["input_ids"]) for row in features)
        output: dict[str, list[list[int]]] = {
            "input_ids": [], "attention_mask": [], "labels": []
        }
        for row in features:
            size = len(row["input_ids"])
            if size != len(row["attention_mask"]) or size != len(row["labels"]):
                raise ValueError("input_ids/attention_mask/labels 长度不一致")
            padding = width - size
            output["input_ids"].append(list(row["input_ids"]) + [self.pad_token_id] * padding)
            output["attention_mask"].append(list(row["attention_mask"]) + [0] * padding)
            output["labels"].append(list(row["labels"]) + [-100] * padding)
            self.samples += 1
            self.input_tokens += sum(row["attention_mask"])
            self.assistant_tokens += sum(label != -100 for label in row["labels"][1:])
        return {name: torch.tensor(rows, dtype=torch.long) for name, rows in output.items()}
