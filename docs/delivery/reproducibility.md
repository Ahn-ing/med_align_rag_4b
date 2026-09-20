# 复现指南

本项目区分 **CPU/仓库复现** 和 **GPU 实验复现**。前者验证配置、数据契约、评分逻辑和小型产物；后者需要本地 Qwen3-4B/BGE-M3、冻结数据和 GPU。所有数据仅允许内部工程使用，`release_allowed=false`；test 和 medical safety holdout 未解锁。

## 1. 环境

- Python 3.10+；
- GPU 主流程以单卡 RTX 5090、BF16、SDPA 记录；
- 模型、embedding 和输出目录只通过参数/环境变量传入。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

RAG 安装 `.[rag,dev]`，SFT/DPO 研究环境安装 `.[research,dev]`。

## 2. CPU 契约检查

下列测试验证 Phase 8 入口、judge JSON 契约和评分逻辑；失败时应停止，不运行 GPU 矩阵。

```bash
python -m pytest \
  tests/unit/test_rag_ablation_entry.py \
  tests/unit/test_rag_judge.py \
  tests/unit/test_rag_scoring.py
```

按需执行更广的检查：

```bash
python -m compileall src tests
python -m pytest tests/unit
```

## 3. RAG 不变量

- BGE-M3 dense embedding，float32，L2 normalization；
- FAISS `IndexFlatIP`；
- `embeddings.npy` 第 i 行、`metadata.jsonl` 第 i 条与 FAISS id i 对齐；
- query 使用同一 model revision 和 normalization；
- Phase 8 `top_k=5`，不静默改变 chunking 或 embedding 参数。

详细证据见 `docs/phases/phase_1.md` 和 `docs/phases/phase_2.md`。

## 4. SFT

最终配置为 `configs/sft/qwen3_4b_lora_phase6.yaml`：LoRA rank 16 / alpha 32 / dropout 0.05，lr 1e-4，2 epochs，global batch 64，max length 2048，`enable_thinking=false`。

```bash
bash scripts/sft/run_phase6_sft_5090.sh \
  /path/to/Qwen3-4B \
  4 \
  qwen3_4b_lora_v2_plus_repair_r2_reproduction
```

包装脚本固定使用已批准的 `data/sft/released/v2_plus_repair_r2` 和 validation_v2，输出位于 `artifacts/runs/phase_6/<run-name>`。完成条件是 run manifest 为 completed、298 steps/2 epochs 完整且 final adapter 可在新进程 reload，而不是“命令已返回”。

## 5. DPO v2

最终配置为 `configs/dpo/qwen3_4b_lora_phase7.yaml`：LoRA rank 8 / alpha 16 / dropout 0.05，beta 0.1，lr 5e-5，1 epoch，global batch 16。reference 必须是 merged SFT。

```bash
export MODEL_PATH=/path/to/merged-sft
export DATA_RELEASE=/path/to/approved/phase7-v2-release
export OUTPUT_DIR=/path/to/new/dpo-v2-output
export EXPERIMENT_TYPE=formal_dpo
bash scripts/dpo/run_phase7_dpo_5090.sh
```

preference validation 和公开 regression 必须分开报告；preference accuracy 不能替代生成式评测。

## 6. Phase 8 消融

```bash
python scripts/evaluation/run_rag_ablation.py plan \
  --config configs/evaluation/phase8_rag.yaml

export MODEL_PATH=/path/to/Qwen3-4B
export MERGED_SFT_PATH=/path/to/merged-sft
export BGE_M3_PATH=/path/to/bge-m3
export OUTPUT_ROOT=/path/to/new/phase8-output
bash scripts/evaluation/run_phase8_rag_5090.sh
```

生成后按顺序运行 `score_rag_ablation.py`、`judge_rag_ablation.py` 和 `build_rag_degradation_review.py`。judge 凭证只通过 `MODEL_API_KEY` 传入，不得写入配置。最终协议是 `validation_v4 + formal_v4 + phase8-full-rag-judge-v4`。

## 7. 产物边界

Git 追踪代码、配置、冻结小数据、小型 manifest 和聚合报告。权重、adapter、checkpoint、embedding、FAISS index、原始生成与逐条评分保留在实验存储中。最终版本以 `docs/evaluation/final_version_register.md` 为准。

## 8. 不能由复现证明的内容

- 临床有效性、诊断能力或真实患者安全性；
- 未读取 test/medical safety holdout 的表现；
- 模型 judge/代理复核等同于医生审核；
- 工程重跑成功等同于获得公开发布许可。
