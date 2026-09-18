# Phase 8：联合评测与最终工程验收

状态：**IN PROGRESS**。Phase 7 已通过并冻结 v2 DPO 候选；本阶段当前只完成联合评测范围、冻结输入、评测矩阵和门禁整理。Base、Base+RAG、SFT、SFT+RAG、DPO v1、DPO v2 及 DPO v2+RAG 的正式联合评测尚未运行，test 与 medical safety holdout 也尚未解锁。

本文按四个部分组织 Phase 8：阶段目标、实现步骤、预计问题、如何解决与调整。Phase 8 的目标是做可复现的内部工程对照，不是临床验证，也不产生医疗能力结论。

## 阶段目标

### 1. 核心目标：在统一协议下完成可控联合评测

Phase 8 需要在相同的基础模型、RAG 索引、system prompt、thinking 设置、检索参数和解码协议下，比较 RAG、SFT 和 DPO 对各版本的相对影响。

“可控联合评测”需要同时满足：

- 所有模型版本绑定到明确的 release、run manifest、adapter 和文件指纹；
- RAG 使用 Phase 2 冻结的同一 embedding、FAISS metric、metadata 和 query normalization；
- Base 与 SFT 使用同一基座 revision；DPO policy 使用 Phase 7 `merge_sft` 和最终 v2 adapter；
- 训练和推理保持 `enable_thinking=false`；
- 同一评测输入下只改变模型或是否启用 RAG，不在一次比较中同时修改多个关键变量；
- 自动规则、模型辅助 judge 和代理语义审核分开记录；
- test 与 medical safety holdout 只有在用户明确批准后才读取；
- 结果只用于内部工程展示和失败分析，不宣称临床有效、医生审核或公开可发布。

### 2. 评测矩阵

主矩阵冻结为下列组合。DPO v1 只作为对照，不替代 Phase 7 已接受的 v2。

| 编号 | 系统 | 基础模型 / adapter | RAG | 用途 |
|---|---|---|---|---|
| E0 | Base | `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` | 关闭 | 原始基座对照 |
| E1 | Base+RAG | 同上 | 开启 | RAG 对 base 的影响 |
| E2 | SFT | Phase 6 `qwen3_4b_lora_v2_plus_repair_r2_formal` final adapter | 关闭 | 正式 SFT 对照 |
| E3 | SFT+RAG | 同上 | 开启 | SFT 与 RAG 联合效果 |
| E4 | DPO v1 | merged SFT + `dpo_formal` final adapter | 关闭 | 首版偏好训练对照 |
| E5 | DPO v1+RAG | 同上 | 开启 | v1 与 RAG 交互 |
| E6 | DPO v2 | merged SFT + `dpo_formal_v2` final adapter | 关闭 | Phase 7 最终候选 |
| E7 | DPO v2+RAG | 同上 | 开启 | 最终联合评测主候选 |

RAG 路径只在同一份冻结索引上运行，不重新生成 embedding、不修改 chunk、不改变 top-k。若某个版本无法完成加载或生成，保留失败记录，不静默替换模型。

### 3. 评测数据与指标

评测分为四层，逐层解锁：

| 层级 | 数据 | 允许用途 | 当前状态 |
|---|---|---|---|
| 机械契约 | 小型 CPU/工程 fixture | 加载、tokenizer、模板、adapter、输出 schema | 已完成基础实现，联合入口待 smoke |
| 偏好验证 | Phase 7 v2 `preference_validation` 100 对 | DPO 相对 SFT 的内部偏好指标 | Phase 7 已完成，可作为 v1/v2 对照 |
| 公开 regression | 已批准的 5 条公开案例 | 行为边界、JSON、急诊和通用能力 | Phase 7 v2 已完成 |
| 内部联合评测 | validation v2，必要时按批准解锁 test 与 medical safety holdout | 统一协议下的生成式联合评测 | 尚未运行；test/holdout 保持锁定 |

聚合指标至少包括：

- required behavior 满足率、partial 率、forbidden behavior 命中数；
- JSON / 结构化输出解析成功率和字段约束满足率；
- 偏好 validation 的 reward margin、preference accuracy 和 policy-reference 差异；
- RAG 启用时的检索命中、上下文引用一致性和无依据回答比例；
- 回答是否明确保留不确定性、就医边界和用药边界；
- 截断率、平均长度、生成延迟和 RTX 5090 显存峰值；
- 自动规则、模型 judge 与代理审核之间的分歧。

不得把自动规则、模型 judge 或代理审核结果单独称为临床准确率。原始逐条输出保存在 `artifacts/evaluations/phase_8/`，聚合报告写入 `artifacts/reports/phase_8/`；原始大文件和运行产物按 `.gitignore` 排除。

### 4. 交付目标

Phase 8 最终需要留下：

- 冻结的评测矩阵、模型路径、revision、adapter、RAG 索引和协议配置；
- 可远程运行的联合评测入口和最小 contract smoke；
- Base、Base+RAG、SFT、SFT+RAG、DPO v1、DPO v1+RAG、DPO v2、DPO v2+RAG 的逐版本结果；
- 聚合比较报告、失败案例和不可比较项；
- 自动规则、模型 judge、代理审核的分层统计；
- 明确的未解锁数据集、未验证能力和停止条件；
- Phase 9 可直接引用的工程证据和限制说明。

## 实现步骤

### 1. 冻结 Phase 8 输入

先冻结：

- Qwen3-4B revision 和 Phase 6/7 run manifest；
- Phase 6 SFT final adapter、Phase 7 DPO v1/v2 final adapter；
- Phase 7 v2 release manifest、train 和 preference validation；
- RAG embedding、FAISS index、metadata、query normalization 和 top-k；
- system prompt、`enable_thinking=false`、greedy 与采样协议；
- 输出目录、随机种子、最大生成长度和设备参数。

输入冻结后不得边评测边修改模型、检索参数或判分规则。

### 2. 适配统一评测入口

评测入口必须从运行参数接收模型、adapter、数据、RAG artifact、输出目录和 `device`，默认面向远程 RTX 5090。入口只负责：

- 加载一个明确版本；
- 构造统一 prompt；
- 可选调用冻结 RAG 检索并把上下文注入模板；
- 固定 decoding 参数生成；
- 写出逐条原始输出和版本元数据；
- 在进程结束时给出结构化状态。

不要在评测入口中读取 API key 或写入机器绝对路径。Codex 只负责实现和最小检查，不在本机执行 GPU 生成。

### 3. 执行联合入口 contract smoke

正式矩阵前先做小规模 smoke：

- 至少覆盖 Base 与 SFT 各一条；
- 至少覆盖 RAG 关闭与开启各一条；
- 检查 adapter 是否能加载、chat template 是否一致、输出是否可解析；
- 检查失败时是否保留日志且不覆盖已有目录；
- 只用公开 regression 或本地 fixture，不读取 test/holdout。

smoke 通过后停止，等待用户批准正式联合评测，不自动扩大规模。

### 4. 运行主评测矩阵

按 E0 到 E7 顺序运行，最好按模型家族分组，避免同时启动多个 GPU 任务：

1. 先生成 Base 与 Base+RAG；
2. 再生成 SFT 与 SFT+RAG；
3. 再生成 DPO v1 与 DPO v1+RAG；
4. 最后生成 DPO v2 与 DPO v2+RAG；
5. 每组完成后记录 manifest、显存、延迟、失败数和原始输出位置。

同一案例必须复用相同输入文本和检索结果；RAG 与无 RAG 组之间只改变上下文注入，不改变生成参数。

### 5. 审计、比较与收口

完成生成后：

- 先运行机械规则和结构检查；
- 再运行固定版本模型 judge；
- 对高风险、分歧和抽样案例执行代理语义审核；
- 按 E0/E2/E4/E6 比较模型版本，按 E1/E3/E5/E7 比较 RAG 增量；
- 单独报告长度、采样协议、截断和失败案例；
- 发现退化时保留结果并分析原因，不通过调门禁或删样例掩盖；
- 未解锁的 test 与 medical safety holdout 不进入本轮结论。

## 预计问题

### 1. 模型版本之间的比较不公平

如果 RAG 组改变了 top-k、prompt 长度或采样参数，就无法区分模型能力和检索设置的影响。因此主矩阵必须先冻结协议，再逐版本运行。

### 2. RAG 上下文可能掩盖模型自身行为

RAG 提供的外部事实可能让 Base 和 SFT 看起来更接近。报告必须同时保留无 RAG 对照，并将检索命中、引用一致性和最终答案分开评分。

### 3. DPO 的偏好指标与生成质量不是同一件事

Phase 7 preference accuracy 和 reward margin 只说明 policy 相对 frozen reference 更偏好 chosen，不代表临床回答更可靠。Phase 8 必须用独立生成案例、required/forbidden 行为和 RAG 一致性继续检查。

### 4. 评测集解锁容易导致事后调参

如果看到 test 或 medical safety holdout 结果后再修改 prompt、top-k 或判分规则，评测就会失去意义。数据集解锁、协议冻结和结果查看应保持明确门禁。

### 5. 自动 judge 存在偏差

模型 judge 可能偏好更长、格式更完整的回答。自动结果必须和结构规则、代理审核及失败案例并列，不得直接替代人工或医学结论。

### 6. RTX 5090 内存和运行时间

同时加载 policy/reference、adapter 和可选 RAG 相关依赖可能增加显存和启动时间。主矩阵应按版本顺序运行，记录峰值显存和失败原因，不把 OOM 当作模型效果。

## 如何解决与调整

### 1. 先固定协议，再做比较

把模型路径、adapter、RAG artifact、prompt、thinking、decoding、最大长度、seed 和判分规则写入冻结配置。任何中途改动都要生成新版本，不覆盖旧 run。

### 2. 把 RAG 作为独立变量

RAG 开关只改变上下文注入；无 RAG 与有 RAG 使用相同基础模型、adapter 和生成参数。这样可以把“模型本身变化”和“检索增强变化”分开报告。

### 3. 把规则、judge 和代理审核分层

自动规则先给出可复现的机械判定，模型 judge 只作为辅助，高风险和分歧案例再由代理语义审核。三层结果分别保存，聚合报告必须说明冲突和不确定性。

### 4. 保留失败和 partial

Phase 7 已保留高血压案例的 partial 和 v1/v2 margin 差异。Phase 8 继续采用同一原则：失败、拒答、截断、检索未命中和无法判断的案例都不能被平均值隐藏。

### 5. 控制数据集解锁顺序

先做 fixture 和公开 regression smoke；只有在用户明确批准后，才解锁 validation v2、test 或 medical safety holdout。test 与 holdout 一旦用于最终比较，不再用于调参。

### 6. 当前边界与下一步

已完成：

- Phase 7 v1/v2 DPO、preference validation、公开 regression 和内部验收；
- Phase 8 的阶段状态、评测矩阵、冻结输入范围和守卫条件；
- 最终候选 E6/E7 的模型与 release 来源记录。

未完成：

- 统一联合评测入口和 contract smoke；
- E0–E7 的正式生成；
- 聚合比较、代理审核和 Phase 8 验收；
- test、medical safety holdout 的解锁与正式评测（必须另行批准）；
- 医生审核或临床验证（不在本项目范围）。

下一步是整理并实现联合评测入口，先在公开 regression 或 fixture 上做 CPU/小规模 contract smoke；正式 GPU 评测仍需用户单独批准。在 Phase 8 验收完成前，不进入 Phase 9，也不把任何中间生成结果写成最终医疗能力结论。
