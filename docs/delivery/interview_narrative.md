# 求职讲述稿

这份文档用于把 `med_align_rag_4b` 讲清楚，不是现场演示脚本。所有结果仅限学习、研究和内部工程评测，`release_allowed=false`，不代表医生审核或临床验证。

## 1. 30 秒版本

> 我做了一个基于 Qwen3-4B 的可复现医疗 LLM 工程项目，完整走过 RAG 迁移、SFT 数据治理和训练、DPO 偏好对齐以及 RAG/SFT/DPO 消融评测。我最重视的不是只报一个更高分数，而是冻结数据与版本、防止泄漏、分层报告规则/judge/复核，并保留失败结果。最后我发现 SFT 能改善医疗内容和简洁表达，但 DPO 出现通用/格式退化，而 RAG 在 Base 上为正、在 SFT 上明显为负，说明训练分布和 RAG 接口存在不匹配。

## 2. 3 分钟版本

### 背景和目标

> 这个项目的目标是用一个 4B 模型做完整但成本可控的对齐工程，而不是声称做出了可临床使用的医疗模型。我把项目拆成 RAG、SFT、DPO 和消融评测，每阶段有明确输入、冻结版本、验收门禁和停止条件。

### RAG

> 我先把旧项目的 RAG 资产迁移到新项目，保持 BGE-M3 float32 + L2 normalization、FAISS Inner Product，並确保 embedding 第 i 行、metadata 第 i 条和 FAISS id i 对齐。然后用 32 条冻结 query 在 top-k 1/3/5/10 上验证新旧检索结果完全一致。这一步证明的是工程等价，不是生成质量。

### SFT

> SFT 前我先冻结 validation、test 和 medical-safety holdout，再做来源、许可证、去重、证据和审核治理。最终 R2 训练集有 9,495 条、1,117,446 个 assistant 监督 token。单卡 RTX 5090 上用 LoRA rank 16、lr 1e-4 训练 2 epochs/298 steps，active-assistant validation loss 从 1.9274 降到 1.8060。生成式消融显示医疗问答和简洁表达有改善，但安全和指令目标只部分达成。

### DPO

> DPO v2 使用 900 个训练偏好对和 100 个 held-out preference validation，以 merged SFT 作为 reference。它的 preference accuracy 是 0.96，mean reward margin 是 6.3207，但这只证明对冻结偏好对的排序，不是生成式 blind win rate。在独立消融中，DPO 医疗/安全方向有改善迹象，但通用、指令和结构化任务明显退化。训练数据 75% 是医疗行为+安全，instruction structured 只有 5%，这是一个重要风险因素，但还不是经过单变量验证的唯一因果。

### 消融和最重要的发现

> 最终我用 validation_v4 的 60 条案例做 R0–R5 六组消融。RAG 对 Base 是 +0.1500，对 SFT 是 -0.2500，对 DPO v2 是 +0.0500。SFT+RAG 的主要问题在医疗安全：检索结果相关，但往往不包含案例需要的分诊和升级行动；而 SFT 从没见过 RAG system prompt 和 Context wrapper。所以当前证据支持“训练分布与 RAG 接口不匹配”，不支持简单说 RAG 无效。

### 我如何处理失败

> 我没有只报 judge 均分。最终退化队列有 58 条，代理复核区分了 50 个真实退化、5 个规则误判和 4 个待补证据，其中 1 个类别重叠。例如 RAG OFF 样本曾被 Context 忠实性规则误扣分；另一些急救案例则是真实的安全行动消失或错误用药建议。这些必须分开讲。

## 3. 技术深挖顺序

如果面试官愿意继续深挖，建议按以下顺序：

1. **数据泄漏与冻结顺序**：为什么训练前冻结 validation/test/holdout；
2. **LoRA 训练契约**：device/dtype、assistant-only loss、checkpoint/reload 和 manifest；
3. **DPO reference 与指标边界**：为什么使用 merged SFT，为什么 preference accuracy 不是生成质量；
4. **RAG 组件定义**：RAG ON 同时包含检索、专用 system prompt 和 Context wrapper；
5. **归因边界**：为什么当前只能声明完整 RAG 组件的效果，不能单独归因于检索器；
6. **失败案例**：急救行动消失、Context 外事实扩展、judge 误判和检索充分性。

## 4. 常见追问

### 为什么选 4B 模型？

> 为了在单卡消费级/租赁 GPU 成本下完整跑通数据治理、SFT、DPO、恢复和消融，将重点放在工程契约和评测设计上，而不是用更大模型掩盖流程问题。

### 为什么不直接报 loss？

> loss 用来判断优化和工程过程，不能代表生成安全性。本项目另外保留公开 regression、偏好对排序、六组生成式消融、机械规则、judge 和复核。

### DPO 0.96 是不是证明 DPO 很成功？

> 它证明对当前 100 个 held-out 偏好对的排序很强。但生成式消融中总分下降 0.1500，通用、指令和结构化任务退化，所以更合理的结论是“偏好优化目标较窄”，而不是“整体能力提升”。

### 为什么 SFT+RAG 更差？

> 有两类证据：第一，SFT 训练没有 RAG-conditioned 样本，模型未见过 RAG prompt/wrapper；第二，检索文本虽然相关，但常缺少安全任务需要的分诊行动。严格依据 Context 反而会抑制已学行为。但因为 prompt 和 Context 同时改变，我不把它写成单一因果。

### 你最想改的是什么？

> 第一优先级是做 fixed-prompt/context-only 对照，先把 RAG 组件归因拆开；第二是在不改其他训练参数的前提下增加 RAG-aware SFT；第三是提高 DPO 中通用、指令和结构化偏好对比例。这些都应该是单变量实验，不一次改多个因素。

### 这个项目能用于医疗吗？

> 不能。它是内部工程和求职展示项目，数据与产物是 `release_allowed=false`，没有医生审核或临床验证，test 和 medical safety holdout 也仍未读取。

## 5. 讲述时不要这样说

| 不准确表述 | 建议改为 |
|---|---|
| “DPO 胜率 96%” | “在自建 100 条 held-out 偏好对上 preference accuracy 为 0.96” |
| “RAG 使 SFT 失效” | “当前完整 RAG 组件与 SFT 呈负交互，归因尚未拆分” |
| “医疗安全率提升” | “在当前内部协议的特定医疗安全案例上观察到该差值” |
| “人工审核通过” | “代理工程复核完成，不是医生审核” |
| “项目可以用于诊断” | “项目仅用于学习、研究和工程展示” |

## 6. 证据入口

- 项目总览：`README.md`
- 复现指南：`docs/delivery/reproducibility.md`
- 数据卡：`docs/delivery/data_card.md`
- 模型卡：`docs/delivery/model_card.md`
- 实验/失败分析：`docs/delivery/experiment_failure_analysis.md`
- 最终版本登记：`docs/evaluation/final_version_register.md`
- Phase 8 综合复核：`artifacts/reports/phase_8_rag/formal_v4_comprehensive_review.md`

讲述的主线始终是：**先建立可复现契约，再训练；先分层报告证据，再解释分数；发现负面结果时保留并分析，而不是调整门禁把它隐藏。**
