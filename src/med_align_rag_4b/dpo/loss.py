"""DPO 初始损失和 reward margin 的最小确定性实现。"""

from __future__ import annotations

import torch


def dpo_log_sigmoid_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回逐样本 loss、chosen reward、rejected reward 和 margin。"""
    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("DPO log-probability 输入必须是 torch.Tensor")
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors[1:]):
        raise ValueError("DPO log-probability 输入 shape 必须一致")
    if beta <= 0:
        raise ValueError("beta 必须为正数")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    margins = chosen_rewards - rejected_rewards
    losses = -torch.nn.functional.logsigmoid(margins)
    return losses, chosen_rewards, rejected_rewards, margins
