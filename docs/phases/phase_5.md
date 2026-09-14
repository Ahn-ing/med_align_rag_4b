# Phase 5：Qwen3-4B LoRA 冒烟与短跑工程

状态：**已完成（COMPLETED）**。RTX 5090 smoke、50-step short run、loss-only eval、checkpoint、adapter reload、工程 fixture 与本地哈希审计均已通过；Phase 6 仍为 `LOCKED`。

## 边界和已有证据

用户负责 RTX 5090、Ubuntu/CUDA/Python/uv 环境和本地 `Qwen/Qwen3-4B`。项目只从 `--model-path` 接收模型目录，强制 `local_files_only=True`、`trust_remote_code=False`，不下载模型、不修改旧项目，不读取 test 或 medical safety holdout。

已传回并完成本地契约校验的 smoke 证据为：

- preflight 和 1 optimizer step 成功；
- step 8 checkpoint 显式恢复到 step 16，optimizer、scheduler、RNG、trainer state 和 LoRA 契约文件完整；
- final adapter 在新进程 reload 成功；
- step 8→16 段 `train_loss=1.3981631`，峰值 allocated `15,763,330,048` bytes，reserved `21,032,337,408` bytes；
- 记录的吞吐为 input `1302.63 tokens/s`、assistant `573.92 tokens/s`，W&B run ID 为 `n5g5fo1p`。

上述只是工程可执行性证据，不是模型能力或医疗质量成绩。

## loss-only validation 子集

`data/evaluation/phase5_loss_v1.jsonl` 从冻结 validation 60 条中按 seed `20260914`、类别配额和稳定 ID 哈希排序选出 18 条：general 2、instruction 2、medical QA 4、RAG faithfulness 2、structured output 2，以及六个 medical safety 子类各 1 条。

子集只用相同 Qwen chat template 计算无梯度 assistant-only loss，`generation_allowed=false`、`gradient_update_allowed=false`。它不进入参数更新，不用于调参、修改数据或生成式能力评测。原 validation manifest 保持不变。

## 5090 short run

将本轮文件同步到租卡项目后，先在当前 shell 中将 `MODEL_PATH` 设为本地 Qwen3-4B 目录，再执行：

```bash
cd med_align_rag_4b
uv sync --extra sft
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

bash scripts/sft/run_phase5_short_5090.sh \
  "$MODEL_PATH" \
  4
```

训练固定使用正式 `train.jsonl`、50 optimizer steps、micro-batch 4、accumulation 16、global batch 64、BF16、SDPA、gradient checkpointing、LoRA `r=16/alpha=32/dropout=0.05`。每 8 steps 计算 eval loss 并保存 checkpoint；训练结束后在新进程 reload adapter 并运行普通中文、结构化 JSON 和低风险医疗信息三条非 holdout 工程 fixture。

如 micro-batch 4 明确 OOM，只允许使用第三个参数指定新 run name，并将第二个参数改为 `2`，例如 `bash scripts/sft/run_phase5_short_5090.sh "$MODEL_PATH" 2 short_run_qwen3_4b_lora_mb2`。accumulation 自动改为 32，global batch 保持 64；不修改 max length、LoRA、learning rate、dtype 或数据。

## 传回和最终验收

下载完整的 `artifacts/runs/phase_5/short_run_qwen3_4b_lora/`，放入本地项目同名目录后执行：

```powershell
.\.venv\Scripts\python.exe scripts\sft\validate_phase5_results.py
```

验收器要求 smoke step 16 与 step 8 恢复证据、short run step 50、18 条 eval、finite train/eval loss、checkpoint 完整、adapter SHA256、新进程 reload 和三条 fixture 全部通过。成功后产生 `artifacts/runs/phase_5/phase_5_validation_report.json`，并按 short-run 实测 step 速率估算 Phase 6 GPU 时间。

最终报告位于 `artifacts/runs/phase_5/phase_5_validation_report.json`。short run 到达 step 50，train loss 为 1.764870，最终普通 eval loss 为 2.992352，按全部有效 assistant token 聚合的 eval loss 为 3.108658；这些数值只验证训练与评估链，不是医疗能力成绩。峰值 allocated/reserved 分别为 25,875,823,104/32,562,479,104 bytes，首尾 allocated 变化为 -270,336 bytes。运行日志有两次非致命 CUDA allocator 分配回退，但没有抛出致命 OOM，训练、最终评估和 adapter 验证均继续完成。W&B run ID 为 `lzo9e2zu`；该次实际写入旧默认项目 `huggingface`，后续入口已固定为 `med-align-rag-4b`，不为修正展示位置而重跑训练。

按实测 0.177 optimizer step/s 线性外推，Phase 6 假设 8,428 条、global batch 64、2 epochs 时约需 264 steps，即约 0.414 GPU 小时；不包含完整评测时间。Phase 6 仍保持 `LOCKED`，不得自动执行正式 SFT、test、medical safety holdout 或 DPO。
