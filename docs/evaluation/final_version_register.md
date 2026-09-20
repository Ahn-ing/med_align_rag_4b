# 最终版本登记与废弃说明

本文件是项目最终版本和废弃结果的统一入口。历史产物如仍存在，仅用于审计，不得作为最终交付、演示、验收或后续比较依据。

## 1. DPO 最终版本

最终版本：`dpo_formal_v2`

- 训练数据：900 对 train，100 对 preference validation
- 训练设置：LoRA rank 8 / alpha 16，beta 0.1，1 epoch，57 steps，global batch 16，lr 5e-5
- train loss：0.2250
- eval loss：0.1554
- held-out preference accuracy：0.96（96/100）
- mean reward margin：6.3207
- policy raw chosen-vs-rejected accuracy：0.84
- reference raw accuracy：0.43

废弃版本：`dpo_formal_v1`

v1 不再作为最终 DPO 产物、验收对象、对比基线或 Phase 8 消融输入。`dpo_formal_v1_v2_comparison.json` 仅保留为历史诊断；v1/v2 使用不同版本的 preference validation，数值不能直接横向比较。

### DPO 已知限制

- held-out preference accuracy 是偏好对排序指标，不是生成式 A/B blind win rate。
- preference validation 由项目自建，不是医生审核或临床验证。
- DPO 在医疗偏好对方向有改善，但消融评测显示通用、指令和结构化任务存在退化风险。
- 训练集医疗行为和安全占 75%，instruction_structured 仅占 5%，存在过度医疗化和格式能力回落的限制。

## 2. Phase 8 消融最终版本

最终组合：

```text
validation_v4
formal_v4
phase8-full-rag-judge-v4
```

- `validation_v4`：60 条冻结 validation
- `formal_v4`：R0–R5 最终原始输出
- judge v4：事实忠实性评分；引用编号不作为评分项；medical_safety 核心行为与可选支持性建议显式分离
- 偏好评估：使用 DPO v2 的 100 条 held-out preference validation

最终均分：

```text
R0 = 1.7167
R1 = 1.8667
R2 = 1.8333
R3 = 1.5833
R4 = 1.6833
R5 = 1.7333
```

最终配对差值：

```text
R0->R1 = +0.1500
R2->R3 = -0.2500
R4->R5 = +0.0500
R0->R2 = +0.1167
R2->R4 = -0.1500
R1->R3 = -0.2833
R3->R5 = +0.1500
```

### 废弃版本

以下版本不得作为 Phase 8 最终结论：

```text
validation_v1 / validation_v2 / validation_v3
formal_v1 / formal_v2 / formal_v3
judge v1 / judge v2 / judge v3 / judge v5
```

废弃原因：

- validation_v1/v2/v3 已被 v4 取代；v4 修正了结构化案例歧义并冻结事实忠实性 rubric。
- formal_v1/v2/v3 使用旧数据版本、旧 RAG 标准或旧 judge 规则。
- judge v1/v2 使用引用编号门禁。
- judge v3 仍存在 medical_safety 与一般支持性建议的冲突。
- judge v5 未被项目采纳，v5 相关 judge、退化队列和综合报告已废除。

### Phase 8 已知限制

- judge v4 仍存在少量 RAG OFF 被错误按 Context 支持规则判断的情况；人工审核中的规则误判必须单列，不并入真实退化。
- RAG ON 同时改变 system prompt、Context 和任务分布，不能把结果单独归因为检索器或模型能力。
- RAG 在 SFT 上为明显负交互，尤其体现在 medical_safety；相关结论受检索覆盖和训练分布影响。
- 当前没有 SFT 与 DPO 的生成式 A/B blind win rate；已有的是 held-out 偏好对 accuracy 和 reward margin。
- validation_v4 只有 60 条；test 与 medical safety holdout 未读取。
- 所有结论仅限内部工程评测，不是医生审核、临床验证或公开发布许可。
