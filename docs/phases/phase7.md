# Phase 7：DPO 与偏好对齐

状态：**COMPLETED**。Phase 7 已完成并通过用户内部验收：v1/v2 偏好数据、release、preflight 和正式 DPO 均已完成；v2 release 为 train 900 对、preference validation 100 对。`dpo_formal_v2` 完成 57 个 optimizer steps、1 epoch，`wandb_mode=online`；100 条 preference validation 的 preference accuracy 为 `0.96`，mean reward margin 为 `6.320698074996471`，mean DPO loss 为 `0.15533391111189862`。公开 regression 审计为 sampled `5 pass / 0 partial / 0 fail`、greedy `4 pass / 1 partial / 0 fail`，唯一 partial 为高血压日常管理未明确提醒规律监测和遵医嘱用药。用户已接受 v2 表现；该 partial、v1/v2 margin 差异和 `release_allowed=false` 必须保留。Phase 8 已进入规划，但联合评测尚未运行。

本文按四个部分组织 Phase 7：阶段目标、实现步骤、实际遇到的问题、如何解决与调整。当前结论必须区分：

- Phase 6 SFT adapter 已通过工程验收，可作为 DPO 的起点；
- MedicalGPT 已作为只读范式参考，但不得迁移其 DPO 数据、配置、模型输出或评测成绩；
- Phase 7 v1/v2 DPO 已训练并完成内部工程评测；结论仅限当前冻结数据、协议和代理审核，不代表医疗能力或临床效果。

文档不代表医疗能力、临床效果或公开可发布性。

## 阶段目标

### 1. 核心目标：在冻结的 SFT 模型上完成可复现 DPO

Phase 7 的目标不是重新训练 SFT，也不是直接复用旧 DPO 资产，而是以 Phase 6 final adapter 生成的 merged SFT 模型为固定起点，训练一个新的偏好 LoRA，并证明训练过程和产物可复现。

“可复现 DPO”在本项目中需要同时满足：

- 使用 `Qwen/Qwen3-4B`，主流程关闭 thinking，保持与 SFT 一致的 tokenizer 和 system prompt；
- 使用 Phase 6 `qwen3_4b_lora_v2_plus_repair_r2_formal` 的 final adapter，校验配置和权重 SHA256；
- 从运行参数传入基础模型和 adapter，不硬编码或持久化本机绝对路径；
- 将 Phase 6 adapter 合并为阶段性 merged SFT，merged 权重作为冻结 reference；
- 在 merged SFT 上新建 DPO LoRA，policy 的 reference 与训练起点一致；
- 初始 reward margin 接近 0，初始 DPO loss 接近 `log(2)`；
- BF16、SDPA、gradient checkpointing 和单卡 RTX 5090 运行稳定，无致命 OOM；
- checkpoint、恢复状态、final adapter、数据哈希和 run manifest 完整；
- 新进程可以重新加载 final adapter；
- 不读取 test 或 medical safety holdout，不运行隐藏评测；
- 结果仅声明为内部工程验证，不宣称医疗能力或临床效果。

计划参考 `shibing624/MedicalGPT` commit `ef9985ab9a8b26c3b3647d60dc8f51fd41a15229`。只借鉴其 `conversations + chosen + rejected` JSONL、TRL DPO + LoRA，以及在 SFT 模型上继续训练偏好适配器的工程范式。

### 2. 数据目标：构建全新的、可追溯的偏好数据

Phase 7 需要一套全新的偏好数据，不能复制旧 DPO 数据或其模型输出。7.1 已将首版设计冻结为 1,000 对，其中 train 900 对、preference validation 100 对；生成前不再调整配额。

每条记录至少需要包含：

| 字段 | 含义 |
|---|---|
| `id` | 跨版本保持稳定的样本 ID |
| `source_id`、`source_sha256` | 上游来源和版本 |
| `conversations` | 与 SFT 一致的对话输入，包含 system prompt |
| `chosen`、`rejected` | TRL DPO 使用的偏好回答 |
| `task`、`primary_category` | 任务和医疗/安全类别 |
| `risk_level` | 风险级别 |
| `evidence_ids` | 医疗事实或行为依据 |
| `review_status`、`reviewer_role` | 审核门禁，不能等同于医生审核 |
| `prompt_tokens`、`chosen_tokens`、`rejected_tokens` | 长度和预算记录 |
| `pair_reason`、`chosen_source`、`rejected_source` | 偏好形成原因和候选来源 |

按任务冻结的配额如下：

| split | 合计 | 医疗与行为 | 医疗安全 | 通用 | 指令与结构化 |
|---|---:|---:|---:|---:|---:|
| train | 900 | 450 | 225 | 180 | 45 |
| preference validation | 100 | 50 | 25 | 20 | 5 |
| **合计** | **1,000** | **500** | **250** | **200** | **50** |

医疗安全的六个子类也必须在生成前固定。已冻结配额如下：

| 安全子类 | train | preference validation | 合计 |
|---|---:|---:|---:|
| immediate_emergency | 38 | 5 | 43 |
| outpatient_followup | 38 | 4 | 42 |
| home_observation | 38 | 4 | 42 |
| clarification | 37 | 4 | 41 |
| medication_boundary | 37 | 4 | 41 |
| uncertainty | 37 | 4 | 41 |
| **合计** | **225** | **25** | **250** |

split 使用 `seed=20260916` 和 `prompt_group_id` 在生成前切分；同一组 prompt 不得跨 split。

偏好数据生成遵守以下边界：

- 医疗和安全部分优先使用权威证据约束的 chosen；rejected 来自当前 SFT 的错误、遗漏、越界或经控制构造的弱回答；
- 通用和指令跟随部分可以比较当前 SFT 的两种候选输出，但必须使用固定规则或固定版本 judge 形成偏好；
- 高风险样本保留代理审核记录；自动 judge 结果不得称为临床准确率；
- 训练集和偏好验证集在生成前切分，prompt 不重叠；
- 所有样本与 validation、test、medical safety holdout 和 RAG 指纹做泄漏检查；
- 旧项目 SFT/DPO 数据、adapter、checkpoint、模型输出和评测成绩保持隔离。

### 3. 质量目标：先保证 reference 正确和评测隔离

DPO 的数据可以少，但 reference、偏好方向和数据边界不能含糊：

- 不使用基础模型直接作为 DPO reference，避免把 SFT 已经学到的行为重新抵消；
- merged SFT 必须只读，训练过程中不得被修改；
- policy 使用 fresh LoRA，初始状态必须与 reference 等价；
- chosen/rejected 不能用未经审核的模型自由生成结果直接充当事实标准；
- 失败的合并、数据构建或训练版本进入 quarantine，不覆盖既有产物；
- test 和 medical safety holdout 不参与数据生成、超参数选择或训练；
- 公开 regression 只用于已批准的工程行为检查，不作为临床准确率。

### 4. 交付目标：形成可复现的 DPO 证据

Phase 7 最终需要留下：

- 全新的偏好 train、preference validation、rejection/quarantine 和 release manifest；
- merged SFT 的来源、配置和 SHA256；
- DPO 数据校验、模型合并、训练和 adapter reload 入口；
- run manifest、resolved config、checkpoint、loss/reward margin、显存和吞吐记录；
- final DPO adapter 及其 SHA256；
- SFT 与 DPO 在偏好验证集和公开 regression 上的内部对比；
- 明确的数据、模型和评测限制；
- 可复现实验命令和阶段总结。

## 实现步骤

### 1. 冻结 Phase 6 输入和运行契约

状态：**已完成**。冻结结果写入 `manifests/phase_7/input_freeze.json`，manifest SHA256 为 `ad08ba75e90ea2c4b8c9f32304dd63ee29d69cf63c81df7e984a97693477f5c8`。

- Phase 6 run：`artifacts/runs/phase_6/qwen3_4b_lora_v2_plus_repair_r2_formal`；
- Phase 6 run manifest SHA256：`f4df855e6cfebfa793d59900c334e87d26b8f34f6f86b4a2cac84d76291b931e`；
- final adapter config SHA256：`91f41100779d287ca574377a278b5ce0d93716ff985b367d363a005002b033e4`；
- final adapter weight SHA256：`d80812b88eb9037630f626c2ce38c0b411bfd33dbe9fbf1ac86ec7a430f98f96`；
- 基础模型 upstream revision：`1cfa9a7208912126459214e8b04321603b3df60c`，来源为本地 Hugging Face 缓存 metadata；
- 基础模型本地文件：13 个、8,060,926,626 bytes，聚合指纹 `2ea852284b4461e0f10b81a54f720da7d853b6ef1a7072ba46925255bade06d1`；
- system prompt：`你是一个谨慎、清晰且遵循证据边界的助手。`，SHA256 为 `92f31efce9d998f7f2e229defdce4fd2d9255af3ae0d38b4970f9805c4d094ad`；
- `enable_thinking=false`，模型路径继续由运行参数传入；
- Phase 6 run manifest 的 `model_revision=unknown` 是因为训练时使用本地目录；本地 HF cache metadata 中的 13 个文件均指向 revision `1cfa9a7208912126459214e8b04321603b3df60c`，Phase 7 以此为实际 revision 证据。历史 run manifest 保持只读，不回改。

### 2. 通过偏好数据设计门禁

状态：**设计、生成、统一审核、批量修订和 release 冻结均已完成**。当前冻结：

- 总规模 1,000 对，train 900、preference validation 100；任务和六个安全子类配额见“数据目标”；
- 医疗与行为、医疗安全数据只能引用 `source_registry.json` 和 `evidence_registry_v5_scale.json` 中已登记来源，并记录来源 SHA256 和 evidence IDs；
- 通用与指令数据的 prompt 只能来自已批准的 SFT 训练来源或项目自编任务，禁止读取 validation、test、medical safety holdout 和 RAG 评测 query；
- 医疗与安全的 chosen 必须由权威证据约束；rejected 来自当前 SFT 的错误、遗漏、越界或控制构造的弱回答；
- 用户已批准商业 API 参与偏好数据构建；API 可用于生成候选回答和固定 judge，但首次调用前必须冻结 provider、精确 model 标识、temperature、top_p、max_tokens、重试/并发和总调用预算，密钥只允许通过环境变量读取；
- 医疗与安全的 chosen 仍由权威证据约束，API 只能辅助生成或重写，不能改变证据边界；通用与指令数据可使用 API 生成候选，并由固定版本 judge 或确定性规则形成偏好；
- `review_status` 初始为 `pending_agent_review`；若后续单独批准代理审核，才可使用 `agent_evidence_reviewed`、`reviewer_role=codex_research_reviewer` 和 `release_allowed=false`，且它不是医生审核或临床验证；
- 所有 prompt 做 NFKC、空白归一化和 casefold 指纹检查，重复、冲突、关键截断、零 chosen/rejected 或跨 split prompt group 直接拒绝；
- 失败样本进入 `quarantine`，不覆盖 accepted 数据集。

用户已批准数据构建调用 API。7.3A.1 已完成：prompt pool 共 1,000 条，train 900、preference validation 100，六个安全子类配额全部满足，212 条冻结评测/RAG 指纹零冲突。

API pilot 准备已完成：

- `configs/dpo/phase7_api.yaml` 固定 DeepSeek provider、`deepseek-flash`、生成/judge 参数；pilot profile 为 300 次调用和 1,000,000 token，full profile 为 2,200 次调用和 2,500,000 token；API key 只从 `MODEL_API_KEY` 环境变量读取；
- `scripts/dpo/select_phase7_pilot.py` 已生成 `data/dpo/pilot_v1/tasks.jsonl`，共 100 条，train 90、preference validation 10；任务 manifest SHA256 为 `9e6e87a55ad6ce08e1ae6440d98e7f3464e7112531aea07180877d00ce8c799a`；
- `scripts/dpo/generate_phase7_pilot.py` 实现严格 JSON 生成、独立 judge、重试、调用/token 预算、证据子集检查和 quarantine；
- `scripts/dpo/run_phase7_pilot.ps1` 和 `scripts/dpo/run_phase7_remaining.ps1` 是手动运行入口；
- `data/dpo/full_v1/remaining/tasks.jsonl` 已包含剩余 900 条任务，train 810、preference validation 90。

API pilot 已在本机环境完成：100 个任务均生成 pair，quarantine 为 0；生成/judge 共记录 202 次调用和 192,776 token。pilot 自动门禁报告 SHA256 为 `fff4549dda2872e598792a971d5563ba0e5a4f28b9d73123df5c357c8835a054`。剩余 900 对合并到 `data/dpo/full_v1/generated`，自动门禁报告为 `passed:true`，调用 1,802 次、总 token 1,774,163。统一代理审核对 124 个 high-risk/高相似候选做了人工式审阅，其余 776 个通过全量规则筛查和生成器 judge；remaining 897 接受、3 修订。随后对 pilot + remaining 的 5 条 `needs_revision` 做了 Codex agent 批量修订，并用 Qwen tokenizer 重新计算长度；最终 release 位于 `data/dpo/released/phase7_v1`，train 900、preference validation 100。复现运行入口为：

```powershell
$env:MODEL_API_KEY = "<本机密钥>"
& .\scripts\dpo\run_phase7_remaining.ps1
```

当前剩余任务曾运行到约第 501/900 条后因 API 402 中断；旧脚本只在全部完成后落盘，因此没有可恢复的进度文件。必须从剩余任务索引 0 开始，但改用 100 对分片发布：

```powershell
& .\scripts\dpo\run_phase7_remaining_shards.ps1 -StartOffset 0 -EndOffset 900 -ShardSize 100
```

若中途失败，重复执行同一条命令；已有完整 manifest 的分片会自动跳过，只重跑当前失败分片。不得把 offset 500 描述为恢复此前未落盘的进度。

如果更换 provider 或 OpenAI-compatible 端点，可在运行时覆盖 endpoint 和模型，无需修改冻结配置：

```powershell
& .\scripts\dpo\run_phase7_remaining_shards.ps1 `
    -StartOffset 0 -EndOffset 900 -ShardSize 100 `
    -Endpoint "https://new-endpoint.example/v1" `
    -GenerationModel "your-generation-model" `
    -JudgeModel "your-judge-model"
```

endpoint 可以传 base URL（自动追加 `/chat/completions`），也可以传完整 `/chat/completions` URL。manifest 会记录本次实际使用的端点和模型。

不要在聊天或 manifest 中提交 API key。

### 3. 实现最小 DPO 子系统

状态：**已完成**。新增：

- `src/med_align_rag_4b/dpo/config.py`：`contract_only`/`frozen` 配置、配额、reference、LoRA 和训练参数门禁；
- `src/med_align_rag_4b/dpo/data.py`：偏好 schema、review、配额、问题指纹、泄漏检查和 TRL `prompt/chosen/rejected` 转换；
- `src/med_align_rag_4b/dpo/loss.py`：初始 DPO loss、reward 和 margin 的确定性 CPU 实现；
- `src/med_align_rag_4b/dpo/manifest.py`：Phase 7 run manifest 装配入口；
- `scripts/dpo/validate_preference_data.py`：只校验 JSONL，不加载模型；
- `configs/dpo/qwen3_4b_lora_phase7.yaml`：当前固定为 `frozen`、`training_allowed=true`，训练参数为 `learning_rate=5.0e-5`、micro-batch `1`、global batch `16`、epochs `1`；
- `tests/unit/test_dpo_phase7.py`：覆盖配置漂移、review、跨 split、泄漏、配额、TRL 字段、初始 `log(2)` loss、manifest 和 checkpoint 契约。

验证：`ruff format --check` 和 `ruff check` 通过；加入评测入口后 DPO 目标测试 `18 passed`，完整 `tests/unit` 为 `116 passed`；语法编译通过。此步骤没有加载 Qwen3-4B、没有使用 GPU、没有生成偏好数据，也没有启动 DPO。

### 4. 合并 Phase 6 adapter 并准备 reference

状态：**已完成远程执行**。merged SFT 位于 `artifacts/runs/phase_7/merge_sft`，并已通过 DPO reference 初始前向预检。远程执行命令：

```bash
export BASE_MODEL=/path/to/Qwen3-4B
export ADAPTER=/path/to/phase6/final_adapter
bash scripts/dpo/run_phase7_merge.sh
```

默认输出：`artifacts/runs/phase_7/merge_sft`。merge 成功后执行初始 log-ratio 预检：

```bash
export MERGED_MODEL=artifacts/runs/phase_7/merge_sft
bash scripts/dpo/run_phase7_preflight.sh
```

默认报告：`artifacts/reports/phase_7/preflight.json`。脚本要求 CUDA 可见、仅一张 GPU、BF16 可用，且只执行 forward，不更新参数、不启动 DPO。

从运行参数接收基础模型目录和 Phase 6 adapter：

1. 校验 adapter 配置、权重哈希和 Phase 6 manifest；
2. 将 adapter 合并到基础模型，生成阶段性 merged SFT；
3. 校验 merged 模型的配置、文件数量和 SHA256；
4. 将 merged SFT 固定为 DPO reference；
5. 在 merged SFT 上新建 DPO LoRA。

该策略与 MedicalGPT 的“SFT 模型作为 DPO 输入，再新建 LoRA”范式一致。备选方案是同时加载 policy 和 SFT adapter 形成两个可训练模型，但内存和恢复复杂度更高，不作为首版方案。

### 5. 适配偏好数据并运行预检

状态：**已完成**。远程 preflight 报告 `artifacts/reports/phase_7/preflight.json` 为 `passed`，使用 4 条样本，reward margin 和 max logit diff 均为 0，DPO loss 为 `log(2)`。本步骤只执行 forward，未更新参数、未启动 DPO。

- 将偏好 JSONL 转换为 TRL 需要的 `prompt`、`chosen`、`rejected`；
- 保持与 SFT 一致的 chat template 和 system prompt；
- 检查 prompt mask、padding、`max_length`、截断率和样本行对齐；
- 在 1–8 条样本上验证初始 reward margin 接近 0；
- 验证初始 DPO loss 接近 `log(2)`；
- 检查 policy 和 reference 的 tokenizer、dtype、device 和 LoRA 状态；
- 记录依赖版本、数据类型、token 统计和运行参数；
- preflight 报告 SHA256 为 `476d776cd7e396319e05053ce7f319d154eda35b8ec97f75960906001537ab3b`。

### 6. 执行 GPU 冒烟

状态：**1-step contract-correct smoke 和新进程 adapter reload 已完成**。`dpo_smoke_v3` 在 RTX 5090 上完成训练、eval、final adapter 保存和 manifest 收尾；初始 train/eval loss 均为 `0.6931`（`log(2)`），reward margin 为 0。reload 报告为 `passed`，显式使用 `enable_thinking=false`，forward loss 为 `3.7698967456817627`，logits 全部有限。该结果只证明工程链可运行，不代表模型质量提升。实际产出了 `checkpoint-1`，包含 adapter、optimizer、scheduler、RNG state 和 trainer state；随后从该 checkpoint 启动 2-step resume run，status 为 `completed`，`last_completed_step=2`，并生成契约完整的 `checkpoint-2`。`trainer_state.json` 保留了 step 1 原始日志并在 step 2 继续，final adapter 权重/配置哈希与 resume run manifest 一致。注意 resume run manifest 的聚合 `train_loss=0.3465736508369446` 只用于日志核对，不代表偏好质量改善；step 2 的 trainer log loss 仍为 `0.6931473016738892`。

冒烟目标是验证工程链而不是模型质量：

- BF16、SDPA、gradient checkpointing；
- 单卡 RTX 5090；
- 无致命 OOM，记录 allocated/reserved 显存；
- checkpoint 完整性和中断恢复；
- final adapter 保存；
- 新进程 reload；
- 不读取 test 或 medical safety holdout。

首版参数已冻结：LoRA rank 8、alpha 16、dropout 0.05；learning rate `5.0e-5`、micro-batch `1`、global batch `16`、epochs `1`；由此推导 gradient accumulation 为 16，900 条 train 样本对应 57 个 optimizer steps，warmup steps 为 2。预检通过后不再调整这些参数。

### 7. 执行正式 DPO

状态：**v1/v2 正式 DPO 远程执行和 7.6 验收均已完成，v2 为最终接受版本**。`scripts/dpo/run_dpo.py` 支持 smoke 与 formal_dpo，检查 release/preflight/config 门禁并保存 checkpoint、final adapter 和 run manifest；1-step contract-correct smoke 已通过。正式 run `dpo_formal` status 为 `completed`，完成 57 个 optimizer steps、1.0 epoch；final train loss 为 `0.13152768742120793`，step 50 eval loss 为 `0.057531099766492844`。final adapter 权重 SHA256 为 `a6fda2daa82eb35207747459da9d96348d6f771c351bef90633f4f69cb10979b`，配置 SHA256 为 `de05aa5b1d7617b72a9f95a527cdd56f0dc87dbd19b8c3107836fa2c60844531`，run manifest SHA256 为 `cf97ebcf1f4453763a5c2621590cd6eac2defcfcdd7e4c4ec1b41126e81f3728`。manifest 记录的 `wandb_mode=online` 与执行说明建议的 `offline` 不一致，需要保留实际 W&B run 证据。冻结 manifest 仍位于 `manifests/phase_7/training_config_freeze.json`，SHA256 为 `cc4c766148de6e1e7c83b95218233b5ba38abb860566b4233e0fa562a67e088d`。

远程 smoke 示例：

```bash
export MODEL_PATH=artifacts/runs/phase_7/merge_sft
export OUTPUT_DIR=artifacts/runs/phase_7/dpo_smoke
export EXPERIMENT_TYPE=smoke
export MAX_STEPS=1
export WANDB_MODE=offline
bash scripts/dpo/run_phase7_dpo_5090.sh
```

正式 DPO 示例（只有冒烟通过并单独获批后运行）：

```bash
export MODEL_PATH=artifacts/runs/phase_7/merge_sft
export OUTPUT_DIR=artifacts/runs/phase_7/dpo_formal
export EXPERIMENT_TYPE=formal_dpo
bash scripts/dpo/run_phase7_dpo_5090.sh
```

冒烟通过并单独批准后执行正式训练：

- 使用一个冻结配置，不同时比较多个关键变量；
- 使用 TRL `DPOConfig` 的 `beta=0.1` 作为首版基线；
- 使用 fresh LoRA，merged SFT reference 保持冻结；
- 周期性保存 checkpoint 和 run manifest；
- 记录 loss、reward margin、显存、步数、吞吐和恢复边界；
- 只使用训练集和 preference validation，不读取冻结测试集或安全 holdout；
- 出现 NaN、持续显存增长、恢复失败或数据越界时立即停止并保留失败结果。

### 8. 工程与内部验收

状态：**v1/v2 preference validation、v2 公开 regression 和内部验收均已完成**。评测入口为 `scripts/dpo/evaluate_phase7.py`，包装脚本为 `scripts/dpo/run_phase7_eval_5090.sh`。v1 报告为 `artifacts/reports/phase_7/dpo_formal_evaluation.json`，SHA256 为 `2ebc15da23fefa4e1e713e33bfb49e53fff98915e3e68c790321aab03f227394`；v2 报告为 `artifacts/reports/phase_7/dpo_formal_v2_evaluation.json`，preference accuracy 为 `0.96`，mean reward margin 为 `6.320698074996471`，mean DPO loss 为 `0.15533391111189862`，SHA256 为 `63689cde30cdea92a6501aa96943af2eafe9b8807fd57721a46685afe585c2de`。公开 regression 审计报告为 `artifacts/evaluations/phase_7_v2_public_regression/public_regression_v2_audit.json`，sampled 为 `5 pass / 0 partial / 0 fail`，greedy 为 `4 pass / 1 partial / 0 fail`；该 partial 已保留在结论中。

远程评测命令：

```bash
cd /root/autodl-tmp/med_4b
export MERGED_MODEL=artifacts/runs/phase_7/merge_sft
export RUN_DIR=artifacts/runs/phase_7/dpo_formal
export OUTPUT=artifacts/reports/phase_7/dpo_formal_evaluation.json
export DEVICE=cuda
bash scripts/dpo/run_phase7_eval_5090.sh
```

- 校验数据、模型、checkpoint 和 final adapter SHA256；
- 新进程重新加载 adapter 并执行机械 fixture；
- 比较 SFT 与 DPO 的 DPO preference validation 指标；
- 在已批准的公开 regression 上比较行为边界和通用能力；
- 自动规则、模型辅助评分和代理审核分开报告；
- 若出现退化，先保留失败结果并分析原因，不通过调门禁掩盖结果；
- 不运行未获批的隐藏评测；Phase 8 的执行需在评测输入和协议冻结后单独获批。

## 实际遇到问题

### 1. reference 模型选择容易混淆

DPO 的 reference 应当是 SFT 初始化，而不是原始 base。若直接使用 base 作为 reference，DPO 可能部分抵消 SFT 已经学到的行为。另一个方案是同时加载 policy 和 reference 两份完整模型，但会增加 RTX 5090 的显存和恢复复杂度。

### 2. Phase 6 run manifest 未自动记录 revision

Phase 6 两个 run 的 `model.model_revision` 都记录为 `unknown`，原因是训练时基座通过本地目录传入。后续只读复核从本地 HF cache metadata 恢复了实际 revision `1cfa9a7208912126459214e8b04321603b3df60c`，并与 Phase 4 tokenizer revision 一致。限制仍然存在：Phase 6 run manifest 只按名称和字节数记录三个权重分片，没有记录分片 SHA256，因此不能声称完成了所有权重的哈希级证明。Phase 7 使用该 revision 加完整本地文件指纹继续记录证据，历史 manifest 不回改。

### 3. 偏好数据容易出现来源和审核强度不一致

医疗数据需要权威证据和风险边界，通用数据需要语言质量与指令遵循，两类样本不能使用同一套审核标准。如果自动 judge 直接充当事实标准，容易把模型偏好误当成医学证据。

### 4. DPO 训练比 SFT 多一个 reference 计算路径

即使只训练 LoRA，DPO 仍需要同时计算 policy 和 reference log-probability。额外前向、序列拼接和 padding 都可能增加显存压力。正式训练前必须通过初始 log-ratio、显存和恢复检查。

### 5. 数据设计、生成和 release 都需要分别留痕

目标规模、任务配额、split、安全子类、来源边界和审核规则已经冻结；实际生成的 1,000 对样本也已完成代理审核和 release 冻结。冻结设计与数据 release 是两个独立门禁，不能用其中一项替代另一项。

### 6. TRL 0.29.1 要求 DPO 样本字段格式一致

首次 GPU smoke 在 `DPOTrainer` tokenize 阶段失败，报错为 `text input must be of type str`。原因是 `to_trl_record` 返回了 conversational 格式的 `prompt`，但 `chosen` 和 `rejected` 仍是字符串；TRL 会把这种混合结构判断为非 conversational，随后把消息列表直接传给 tokenizer。修复同时补上 `chat_template_kwargs={"enable_thinking": False}`，避免训练模板与 preflight/SFT 的 thinking 设置漂移。

### 7. 首版 DPO 数据出现模板化，最终改用 v2

v1 formal DPO 的 preference validation 指标较高，但公开 regression 暴露出固定来源和开头风格可能造成模板化及行为退化。Phase 7 因此保留 v1 作为对照，重新构建 v2 数据并完成 `dpo_formal_v2`。v2 sampled regression 为 `5 pass / 0 partial / 0 fail`，greedy 为 `4 pass / 1 partial / 0 fail`；本文按实际 v2 报告记录结果，不把 v1 的模板化问题或 v2 的 partial 隐藏。

## 如何解决与调整

### 1. 使用 merged SFT 作为统一起点

把 Phase 6 adapter 合并为阶段性 SFT 模型，再在其上新建 DPO LoRA；reference 使用同一 merged SFT 并保持冻结。这样 policy 初始状态与 reference 一致，也能沿用 MedicalGPT 的 SFT→DPO 训练范式。

### 2. 按已冻结设计生成数据

7.1 的设计已经完成，但实际生成仍需单独批准。生成时严格按已冻结配额、split、来源和审核规则执行；实际构建后先通过 schema、来源、重复、冲突、泄漏、截断和审核门禁，再冻结 train 和 preference validation，不再修改配额。

### 3. 用初始 log-ratio 验证 reference 正确性

在正式训练前检查初始 reward margin 和 DPO loss。如果 margin 不接近 0、loss 不接近 `log(2)`，优先排查 merged SFT、LoRA 初始化、chat template、tokenizer 或数据拼接问题，而不是直接开始训练。

### 4. 用顺序门禁控制成本和风险

严格执行：输入冻结 → 数据设计 → CPU 契约测试 → 模型/数据预检 → GPU 冒烟 → 正式 DPO → 内部验收。每一步完成后停止，等待下一步批准；不同时运行多个关键变量。

### 5. 把工程稳定性与模型质量分开报告

Phase 7 可以先验收训练稳定性、checkpoint、恢复、reload 和哈希，但不能据此宣称医疗能力提升。偏好验证、公开 regression 和后续 Phase 8 必须分开报告，test 与 medical safety holdout 保持锁定。

### 6. 统一 DPO conversational 格式

`to_trl_record` 将 `prompt`、`chosen`、`rejected` 统一转换为消息列表：chosen 和 rejected 分别包成单条 `assistant` message，并显式传入 `enable_thinking=false`。这样 TRL 会走 chat-template 分支，与 Qwen3-4B 的系统提示、Phase 6 SFT 和 preflight 保持一致；该契约由单元测试和 token 前缀对齐检查覆盖。

### 7. 当前交付边界与下一步

已完成：

- Phase 7 状态由 `IN PROGRESS` 转为 `COMPLETED`；
- 确认 MedicalGPT 的偏好数据与 DPO 训练范式；
- 完成 7.0 输入冻结与 `manifests/phase_7/input_freeze.json`；
- 完成 7.1 数据设计门禁：1,000 对、900/100 split、任务配额、六个安全子类、来源和审核规则；
- 完成 7.2 CPU 实现与契约测试，目标测试 `8 passed`；
- 完成 7.3A.1 prompt pool 冻结：`data/dpo/phase7_v1/prompt_pool.jsonl` SHA256 为 `d4cbd6e29f8376bc474b2e48467cd46822de3e1b781c408d341ea8c7c1a4c15a`，`blocked_fingerprints.json` SHA256 为 `a5866dfb99493ef27b01336a6cbd9cf301fb4a6b035ca05b6f538d411b51f5e1`；
- 完成 API 配置冻结和 100 条 pilot 任务选择；
- 完成 100 对 API pilot 生成和自动门禁，DPO 目标测试 `16 passed`，完整 `tests/unit` 为 `114 passed`；
- 完成 pilot 代理语义审核：98 对 `agent_evidence_reviewed`，2 对 deferred revision；审核结果 `reviewed_candidates.jsonl` SHA256 为 `9ffcfa8e6ae7cbc5ba95a617014a86d96df6f564eb4e84b22727bda23c326989`，拒绝记录 SHA256 为 `0e118003c4a7239e254a09878109075966068145ecdc7048b3b79b1907721c58`；
- 完成剩余 900 条任务选择：train 810、preference validation 90，API 预算 profile 为 `full`；
- 完成剩余 900 对 API 生成和合并：`data/dpo/full_v1/generated/candidates.jsonl` SHA256 为 `d8444b56ec6fa5fb20d59f7d49fceec952479c7009c023dd9a0e120e9b357077`；
- 完成 remaining 统一代理审核：897 接受、3 `needs_revision`；
- 完成 pilot + remaining 合并审核：995 接受、5 `needs_revision`；
- 完成 5 条 Codex agent 批量修订，最终 1,000 对全部为 `agent_evidence_reviewed`；修复后 candidates 位于 `data/dpo/full_v1/repaired/candidates.jsonl`；
- 完成 final release：`data/dpo/released/phase7_v1/train.jsonl` 900 条、`preference_validation.jsonl` 100 条，release manifest SHA256 为 `3d371336765a18b43db9ef77bb58440fba46f1cdec8c3096b87fa2e91989ac6d`；
- 完成 7.3B merged SFT 和远程 preflight：报告 SHA256 为 `476d776cd7e396319e05053ce7f319d154eda35b8ec97f75960906001537ab3b`，status 为 `passed`；
- 完成训练参数冻结：配置 SHA256 为 `799cbe9b70d7e147027852bf4613a6088584b526a92179c6bf2e92378b4bfa3e`，冻结 manifest SHA256 为 `cc4c766148de6e1e7c83b95218233b5ba38abb860566b4233e0fa562a67e088d`；
- 完成 7.4 contract-correct 1-step smoke：`dpo_smoke_v3` status 为 `completed`，train/eval loss 为 `0.6931`，reward margin 为 0；final adapter 已保存；
- 完成 7.4 新进程 reload：报告 `artifacts/reports/phase_7/dpo_smoke_v3_adapter_reload.json` 为 `passed`，SHA256 为 `0e4af49e74bd0bf4b072072284a7df7cb116aafcbba676e0acf13f9b015e1b5c`，forward logits 全部有限；
- 完成 7.4 checkpoint recovery：从 `dpo_smoke_v3/checkpoint-1` 恢复并完成到 step 2，`checkpoint-2` 契约通过，resume manifest 为 `completed`；resume manifest SHA256 为 `d444d9e885dbee2be536f8db96e51f15fef2d81bc300841d15dd504f5ebb56d4`，final adapter 权重哈希为 `ebbf4be2e8962637558c6ac4d621bf8eece7a2f88a57b5739f98209bcfe483ad`；
- 完成 7.5 正式 DPO：`dpo_formal` status 为 `completed`，完成 57 steps、1 epoch，生成 `checkpoint-50`、`checkpoint-57` 和 final adapter；run manifest SHA256 为 `cf97ebcf1f4453763a5c2621590cd6eac2defcfcdd7e4c4ec1b41126e81f3728`；
- 完成正式产物本地完整性检查：checkpoint-50/57 契约、manifest 与 final adapter 权重/配置哈希全部匹配；
- 完成 7.6 preference validation：100 条中 98 条 reward margin 为正，preference accuracy 为 0.98，mean reward margin 为 8.407423021793365，mean DPO loss 为 0.057569396920201896；policy 原始 chosen/rejected log-ratio accuracy 为 0.74，reference 为 0.21，汇总报告 SHA256 为 d89616f59a272892777645143f60eeb97103fa2491e3f9e9ebef1e1f810b297c；
- 新增 7.6 评测入口：`scripts/dpo/evaluate_phase7.py` 和 `scripts/dpo/run_phase7_eval_5090.sh`，远程 preference validation 已完成。
- 完成 Phase 7 v2 模型/data preflight：`artifacts/reports/phase_7/preflight_v2.json` status 为 `passed`，SHA256 为 `83e07c398c1b040fee3c6ad6969e8b42784c789baf39deb8c6534f7c084a8658`；
- 完成 Phase 7 v2 public regression：v2 相比 v1 移除了固定来源模板，结构化输出恢复严格 JSON。sampled protocol 为 `5 pass / 0 partial / 0 fail`，forbidden behavior 零命中；greedy protocol 为 `4 pass / 1 partial / 0 fail`，唯一 partial 为高血压日常管理未明确提醒规律监测和遵医嘱用药。审计报告 SHA256 为 `93b71dd010f6b5ec96bc7ff4aa8c6b9153612f86a5b409f5b11265cfd4d3fdcb`；用户已明确表示接受 v2 的公开 regression 表现，但该 partial 不能被写成无退化。
- 完成 Phase 7 v2 preference validation：100 条中 96 条 reward margin 为正，preference accuracy 为 `0.96`，mean reward margin 为 `6.320698074996471`，mean DPO loss 为 `0.15533391111189862`；policy 原始 chosen/rejected log-ratio accuracy 为 `0.84`，reference 为 `0.43`。v1/v2 对比报告为 `artifacts/reports/phase_7/dpo_formal_v1_v2_comparison.json`，SHA256 为 `603471e3b5385fb77f2e782cd2cb26166bb79b7a3c880d025f2816e90233e9ae`；
- 完成 Phase 7 v2 formal DPO：`dpo_formal_v2` 完成 57 steps、1 epoch，`wandb_mode=online`，final train loss 为 `0.22498572320399576`，step 50 eval loss 为 `0.15539364516735077`；run manifest SHA256 为 `7d2d99c30051318931314539082ce25db175cfde3ff9bba8bb501cfebab9f7f2`，final adapter 权重 SHA256 为 `1a8dff93b8ea1e21897b710f88f3ccf0448d2af6f4c33d2d8d87d96e336a7126`；checkpoint-50/57 契约和 final adapter 哈希本地校验通过。
- 完成 Phase 7 v2 reviewed candidates 与 release 构建：`data/dpo/released/phase7_v2/train.jsonl` 900 条、`preference_validation.jsonl` 100 条；release manifest SHA256 为 `d478e78e6411d233326db7cad0d633c6eabb1fe95eb2ba87ff0cb121f32e89c3`，train SHA256 为 `99a35fc277fb1ca4decd461f33742d527143106be756b09c5be3ed04fd05abd3`，validation SHA256 为 `b4432e6dc8ec55e4297c06ca8f7cc038e6638b0cb49f7be8eb0235214fe4d56a`。

Phase 7 已完成：

- v1/v2 preference validation、v1/v2 对比和 v2 公开 regression；
- v2 final adapter、release、preflight、run manifest 和内部验收记录；
- Phase 8 的联合评测范围、冻结输入和门禁已移入 `docs/phases/phase8.md`，但尚未运行联合评测。

下一步是进入 Phase 8，但正式执行必须等待输入冻结、评测入口 smoke 和用户明确批准。test 与 medical safety holdout 继续锁定；`release_allowed=false`，所有结论仍不是医生审核或临床验证。
