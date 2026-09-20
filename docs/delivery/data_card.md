# 数据卡

## 1. 范围

本数据卡覆盖 RAG 知识资产、SFT 数据、DPO 偏好对和 Phase 8 validation_v4。数据仅用于学习、研究和内部工程评测，不是临床数据产品。详细来源字段见 `data/sft/sources/source_registry.json`。

## 2. 组成

| 子系统 | 最终规模 | 用途 | 状态 |
|---|---:|---|---|
| RAG 资产 | 冻结 chunk/embedding/metadata/FAISS | 检索与 RAG ON | 大文件不入 Git |
| SFT R2 | 9,495 条，1,117,446 监督 token | LoRA SFT | `release_allowed=false` |
| DPO v2 | train 900 对 + preference validation 100 对 | DPO/偏好排序评估 | `release_allowed=false` |
| validation_v4 | 60 条 | Phase 8 R0–R5 | `release_allowed=false` |
| test | 90 条 | 保留评测 | 未读取 |
| medical safety holdout | 30 条 | 安全保留评测 | 未读取 |

## 3. 来源和许可证边界

| 来源 | 用途 | 记录状态 | 限制 |
|---|---|---|---|
| `shibing624/medical` 固定 revision | 医疗候选 | Apache-2.0（上游数据卡） | 逐条来源未充分核验，禁止公开发布 |
| `databricks/databricks-dolly-15k` | 通用指令 | CC-BY-SA-3.0 | 排除医疗关键词、联系方式和评测重叠 |
| `m-a-p/COIG-CQIA` | 分析/候选 | 总许可证待定 | 不作为已批准内部训练来源 |
| 项目自编数据 | 安全、行为补丁、validation | internal-only | 代理审核不是医生审核 |
| evidence-constrained API 候选 | SFT 医疗数据 | internal-training-only | 模型生成，必须绑定证据并复核 |

许可证字段是项目内使用边界，不构成法律意见。

## 4. SFT 构成

| 来源类型 | 记录数 | token | 占比 |
|---|---:|---:|---:|
| rule pool salvage | 5,087 | 651,387 | 58.29% |
| API evidence | 1,507 | 217,152 | 19.43% |
| general | 2,620 | 215,106 | 19.25% |
| rule pool repair | 271 | 33,378 | 2.99% |
| medical behavior patch | 10 | 423 | 0.04% |

最终 R2 统一携带训练 system prompt。该分布能解释医疗内容/简洁表达改善，也提示安全与指令目标只部分达成。

## 5. DPO 构成

900 个训练对包含 medical behavior 450、medical safety 225、general 180、instruction structured 45。instruction structured 仅占 5%，医疗行为+安全占 75%，是过度医疗化和格式能力回落的分布风险。

## 6. 分割、泄漏和审核

- validation/test/medical safety holdout 在训练抽样前冻结；
- 稳定 ID/指纹排除评测和 RAG overlap；
- 重复、冲突、关键截断或零 assistant 监督记录不进入 release；
- Phase 8 仅使用 validation_v4，不根据保留评测回调训练数据；
- `agent_evidence_reviewed`/`codex_research_reviewer` 是内部工程复核，不是医生审核或临床验证。

安全数据覆盖急救、门诊复诊、家庭观察、信息澄清、药物边界和不确定性，目标是学习分级行为，而不是对所有问题统一建议就医。

## 7. 限制

- 部分医疗聚合数据的逐条来源/许可证无法充分核验，限于内部使用。
- 模型生成候选可能带有事实、风格或过度诊疗偏差。
- validation_v4 仅 60 条，不代表真实世界医疗分布。
- 未读取 holdout 不能被用来声称任何结果。
