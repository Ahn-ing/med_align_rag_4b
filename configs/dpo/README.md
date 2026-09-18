# DPO 配置门禁

Phase 7 的训练配置已冻结：`learning_rate=5.0e-5`、micro-batch `1`、global batch `16`、epochs `1`，LoRA rank `8`、alpha `16`、dropout `0.05`。`training_allowed=true` 仅表示工程门禁允许进入训练入口；DPO smoke 和正式训练仍需用户单独批准后，在远程 RTX 5090 上运行。不得将旧项目的 DPO 数据、配置或实现复制到此目录。