"""assistant token 加权的 causal language-model loss。"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def token_normalized_causal_loss(
    logits: torch.Tensor, labels: torch.Tensor, *, num_items_in_batch: int | torch.Tensor | None = None
) -> torch.Tensor:
    """按有效 assistant label token 求和后归一化。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    active = shift_labels.ne(-100).sum()
    denominator = active if num_items_in_batch is None else torch.as_tensor(
        num_items_in_batch, device=logits.device
    )
    if int(denominator.item()) <= 0:
        raise ValueError("有效 assistant label token 数必须大于 0")
    return functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
        ignore_index=-100, reduction="sum",
    ) / denominator
