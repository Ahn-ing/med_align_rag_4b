# Phase 7：DPO 与偏好对齐

状态：**IN PROGRESS**。截至 2026-09-16，仅完成 Phase 7 的计划、参考范式核对和文档拆分；尚未运行模型、商业 API、偏好数据生成、模型合并或 DPO 训练。以下内容是实现计划与执行记录模板，不代表已完成结果。

本文按四个部分组织 Phase 7：阶段目标、实现步骤、实际遇到的问题、如何解决与调整。当前结论必须区分：

- Phase 6 SFT adapter 已通过工程验收，可作为 DPO 的起点；
- MedicalGPT 已作为只读范式参考，但不得迁移其 DPO 数据、配置、模型输出或评测成绩；
- Phase 7 尚未训练，所有 DPO 效果、医疗能力和安全结论均未产生。

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

Phase 7 需要一套全新的偏好数据，不能复制旧 DPO 数据或其模型输出。首版建议规模为 800–1,200 对，最终数量和配比只能冻结一次，不能边训练边调整。

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

数据计划按任务分层，初始配比建议为医疗与行为约 50%、医疗安全约 25%、通用约 20%、指令与结构化约 5%。具体配额需要在 7.1 数据设计门禁中单独批准。

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

在实现前固定：

- Phase 6 run：`artifacts/runs/phase_6/qwen3_4b_lora_v2_plus_repair_r2_formal`；
- final adapter 的 `adapter_config.json` 和 `adapter_model.safetensors` SHA256；
- 基础模型文件、tokenizer revision、system prompt 和 `enable_thinking=false`；
- 当前 Phase 6 manifest 中的模型 revision 如果仍是 `unknown`，Phase 7 manifest 必须原样记录限制，不能伪造 revision；
- DPO 配置、数据 schema、实验命名和输出目录。

### 2. 通过偏好数据设计门禁

在生成任何样本前提交并批准：

- 来源清单和许可证边界；
- 任务、风险、安全子类配额；
- chosen/rejected 的生成和审核方法；
- train 与 preference validation 的切分方式；
- 泄漏、重复、冲突、截断和零监督门禁；
- 高风险样本的代理审核规则；
- 失败样本的 quarantine 规则。

此步骤只完成数据设计，实际生成仍需单独批准。

### 3. 实现最小 DPO 子系统

只新增与 Phase 7 直接相关的文件：

- `src/med_align_rag_4b/dpo/`：配置、数据校验、manifest 和 checkpoint 契约；
- `scripts/dpo/`：数据校验、model merge、DPO 训练和 adapter reload；
- `configs/dpo/`：冻结配置；
- `tests/unit/`：schema、泄漏、初始 log-ratio、配置和 manifest 测试。

复用现有哈希、JSONL、路径和 tokenizer 约定。没有明确收益时不新增依赖，不重写正常工作的 SFT 代码。

### 4. 合并 Phase 6 adapter 并准备 reference

从运行参数接收基础模型目录和 Phase 6 adapter：

1. 校验 adapter 配置、权重哈希和 Phase 6 manifest；
2. 将 adapter 合并到基础模型，生成阶段性 merged SFT；
3. 校验 merged 模型的配置、文件数量和 SHA256；
4. 将 merged SFT 固定为 DPO reference；
5. 在 merged SFT 上新建 DPO LoRA。

该策略与 MedicalGPT 的“SFT 模型作为 DPO 输入，再新建 LoRA”范式一致。备选方案是同时加载 policy 和 SFT adapter 形成两个可训练模型，但内存和恢复复杂度更高，不作为首版方案。

### 5. 适配偏好数据并运行预检

- 将偏好 JSONL 转换为 TRL 需要的 `prompt`、`chosen`、`rejected`；
- 保持与 SFT 一致的 chat template 和 system prompt；
- 检查 prompt mask、padding、`max_length`、截断率和样本行对齐；
- 在 1–8 条样本上验证初始 reward margin 接近 0；
- 验证初始 DPO loss 接近 `log(2)`；
- 检查 policy 和 reference 的 tokenizer、dtype、device 和 LoRA 状态；
- 记录依赖版本、数据类型、token 统计和运行参数。

### 6. 执行 GPU 冒烟

单独获批后执行极短 DPO，目标是验证工程链而不是模型质量：

- BF16、SDPA、gradient checkpointing；
- 单卡 RTX 5090；
- 无致命 OOM，记录 allocated/reserved 显存；
- checkpoint 完整性和中断恢复；
- final adapter 保存；
- 新进程 reload；
- 不读取 test 或 medical safety holdout。

首版参考 MedicalGPT 范式，LoRA 建议从 rank 8、alpha 16、dropout 0.05 起步。learning rate、micro-batch、global batch 和 epochs 由预检结果一次性冻结。

### 7. 执行正式 DPO

冒烟通过并单独批准后执行正式训练：

- 使用一个冻结配置，不同时比较多个关键变量；
- 使用 TRL `DPOConfig` 的 `beta=0.1` 作为首版基线；
- 使用 fresh LoRA，merged SFT reference 保持冻结；
- 周期性保存 checkpoint 和 run manifest；
- 记录 loss、reward margin、显存、步数、吞吐和恢复边界；
- 只使用训练集和 preference validation，不读取冻结测试集或安全 holdout；
- 出现 NaN、持续显存增长、恢复失败或数据越界时立即停止并保留失败结果。

### 8. 工程与内部验收

- 校验数据、模型、checkpoint 和 final adapter SHA256；
- 新进程重新加载 adapter 并执行机械 fixture；
- 比较 SFT 与 DPO 的 DPO preference validation 指标；
- 在已批准的公开 regression 上比较行为边界和通用能力；
- 自动规则、模型辅助评分和代理审核分开报告；
- 若出现退化，先保留失败结果并分析原因，不通过调门禁掩盖结果；
- 不运行隐藏评测，不进入 Phase 8。

## 实际遇到问题

### 1. reference 模型选择容易混淆

DPO 的 reference 应当是 SFT 初始化，而不是原始 base。若直接使用 base 作为 reference，DPO 可能部分抵消 SFT 已经学到的行为。另一个方案是同时加载 policy 和 reference 两份完整模型，但会增加 RTX 5090 的显存和恢复复杂度。

### 2. Phase 6 模型 revision 信息不完整

Phase 6 run manifest 已记录基础模型文件 SHA256，但模型 revision 当前记录为 `unknown`。Phase 7 不能把未知信息伪装成已知 revision；必须保留该限制，并优先通过本地文件指纹和 tokenizer revision 建立可复现合同。

### 3. 偏好数据容易出现来源和审核强度不一致

医疗数据需要权威证据和风险边界，通用数据需要语言质量与指令遵循，两类样本不能使用同一套审核标准。如果自动 judge 直接充当事实标准，容易把模型偏好误当成医学证据。

### 4. DPO 训练比 SFT 多一个 reference 计算路径

即使只训练 LoRA，DPO 仍需要同时计算 policy 和 reference log-probability。额外前向、序列拼接和 padding 都可能增加显存压力。正式训练前必须通过初始 log-ratio、显存和恢复检查。

### 5. 数据设计尚未批准

目标规模、任务配额、生成方式、审核角色和偏好验证集尚未冻结。当前文档中的 800–1,200 对和初始配比只是计划建议，不是已经批准的实验参数。

### 6. 当前没有 DPO 训练结果

现在不能报告 DPO loss、reward margin、偏好胜率、医疗能力或通用能力变化。任何后续结果都必须来自实际运行的 checkpoint，并区分自动规则、模型 judge 和代理审核。

## 如何解决与调整

### 1. 使用 merged SFT 作为统一起点

把 Phase 6 adapter 合并为阶段性 SFT 模型，再在其上新建 DPO LoRA；reference 使用同一 merged SFT 并保持冻结。这样 policy 初始状态与 reference 一致，也能沿用 MedicalGPT 的 SFT→DPO 训练范式。

### 2. 先通过数据设计门禁，再生成数据

在 7.1 批准前不生成样本。数据建立后先通过 schema、来源、重复、冲突、泄漏、截断和审核门禁，再冻结 train 和 preference validation，避免训练数据反向影响评测或反复修改配额。

### 3. 用初始 log-ratio 验证 reference 正确性

在正式训练前检查初始 reward margin 和 DPO loss。如果 margin 不接近 0、loss 不接近 `log(2)`，优先排查 merged SFT、LoRA 初始化、chat template、tokenizer 或数据拼接问题，而不是直接开始训练。

### 4. 用顺序门禁控制成本和风险

严格执行：输入冻结 → 数据设计 → CPU 契约测试 → 模型/数据预检 → GPU 冒烟 → 正式 DPO → 内部验收。每一步完成后停止，等待下一步批准；不同时运行多个关键变量。

### 5. 把工程稳定性与模型质量分开报告

Phase 7 可以先验收训练稳定性、checkpoint、恢复、reload 和哈希，但不能据此宣称医疗能力提升。偏好验证、公开 regression 和后续 Phase 8 必须分开报告，test 与 medical safety holdout 保持锁定。

### 6. 当前交付边界与下一步

已完成：

- Phase 7 状态进入 `IN PROGRESS`；
- 确认 MedicalGPT 的偏好数据与 DPO 训练范式；
- 完成 Phase 7 的计划和文档结构。

未完成：

- 7.0 输入冻结与契约；
- 7.1 偏好数据设计、生成、审核和冻结；
- 7.2–7.3 代码实现、模型合并与预检；
- 7.4–7.5 GPU 冒烟与正式 DPO；
- 7.6 内部验收；
- Phase 8 联合评测。

下一步需要单独批准 `7.0 输入冻结与契约` 和 `7.1 偏好数据设计门禁`。在批准前不创建偏好样本、不调用模型或 API、不合并权重，也不运行 DPO。