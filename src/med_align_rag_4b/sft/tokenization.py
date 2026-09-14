"""Qwen chat template 与 assistant-only 监督统计。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class QwenAssistantTokenizer:
    """使用模板前缀差分可靠定位单轮 assistant 正文和结束 token。"""

    def __init__(self, tokenizer_path: str | Path, *, max_length: int = 2048) -> None:
        from transformers import AutoTokenizer

        self.path = Path(tokenizer_path)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=False, local_files_only=True
        )
        if not self.tokenizer.chat_template:
            raise ValueError("tokenizer 缺少 chat template")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer 同时缺少 pad/eos token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """使用与 Phase 4 一致的前缀差分构建 assistant-only labels。"""
        analysis = self.analyze(messages)
        if analysis["overlength"]:
            raise ValueError(f"样本超过 max_length={self.max_length}，禁止静默截断")
        prefix_result = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix = prefix_result["input_ids"] if isinstance(prefix_result, Mapping) else prefix_result
        full = full_result["input_ids"] if isinstance(full_result, Mapping) else full_result
        labels = [-100] * len(prefix) + list(full[len(prefix) :])
        return {
            "input_ids": list(full),
            "attention_mask": [1] * len(full),
            "labels": labels,
            "active_label_tokens": analysis["assistant_supervised_tokens"],
        }

    def analyze(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """返回完整 token 数和经模板前缀验证的 assistant 监督数。"""
        assistant_positions = [i for i, row in enumerate(messages) if row["role"] == "assistant"]
        if assistant_positions != [len(messages) - 1]:
            raise ValueError("快速版仅接受末尾单轮 assistant")
        prefix_result = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        full_result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        prefix = prefix_result["input_ids"] if isinstance(prefix_result, Mapping) else prefix_result
        full = full_result["input_ids"] if isinstance(full_result, Mapping) else full_result
        if full[: len(prefix)] != prefix:
            raise ValueError("chat template prefix mismatch")
        supervised = len(full) - len(prefix)
        if supervised <= 0:
            raise ValueError("assistant supervision 为 0")
        return {
            "original_total_tokens": len(full),
            "total_tokens": len(full),
            "assistant_supervised_tokens": supervised,
            "supervision_ratio": supervised / len(full),
            "truncated": False,
            "assistant_truncated": False,
            "has_supervised_eos": self.tokenizer.eos_token_id in full[len(prefix) :],
            "assistant_span_count": 1,
            "chat_template_valid": True,
            "overlength": len(full) > self.max_length,
        }
