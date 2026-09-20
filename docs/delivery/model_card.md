# 模型卡

## 1. 最终版本

| 角色 | 标识 | 基础 |
|---|---|---|
| Base | `Qwen/Qwen3-4B` | revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| SFT | `qwen3_4b_lora_v2_plus_repair_r2_formal` | Base + LoRA |
| DPO | `dpo_formal_v2` | merged SFT + LoRA |

`dpo_formal_v1`、Phase 8 `formal_v1/v2/v3` 和 judge v1/v2/v3/v5 仅作历史审计。

## 2. 用途

适用于研究/展示 Qwen3-4B 上的 RAG、SFT、DPO、adapter 验收和消融工程。不得用于临床诊断、治疗决策、急救指导、自主用药或面向患者的无监督服务。

## 3. 训练

### SFT

- 9,495 条，1,117,446 assistant 监督 token；
- LoRA rank 16 / alpha 32 / dropout 0.05；
- lr 1e-4，2 epochs，298 steps，global batch 64，max length 2048；
- BF16、SDPA、gradient checkpointing，`enable_thinking=false`；
- active-assistant validation loss 1.9274 → 1.8060。

### DPO v2

- train 900 对，preference validation 100 对；
- merged SFT reference；
- LoRA rank 8 / alpha 16 / dropout 0.05，beta 0.1；
- lr 5e-5，1 epoch，57 steps，global batch 16。

DPO 训练中医疗行为+安全占 75%，instruction structured 只占 5%，是过度医疗化和格式能力回落的风险。

## 4. 评测

### SFT regression

- greedy：LoRA 5 pass / 0 partial / 0 fail，Base 3 / 1 / 1；
- sampled：LoRA 和 Base 均 5 / 0 / 0；
- 两套协议 forbidden behavior 零命中。

这是 5 条公开工程案例，不是医疗准确率。

### DPO v2

- held-out preference accuracy 0.96（96/100）；
- mean reward margin 6.3207；
- policy/reference raw accuracy 0.84/0.43；
- public sampled 5 / 0 / 0，greedy 4 / 1 / 0。

preference accuracy 是自建偏好对排序指标，不是生成式 blind win rate。

### Phase 8

最终协议为 `validation_v4 + formal_v4 + phase8-full-rag-judge-v4`，R0–R5 各 60 条：

| Arm | 系统 | 均分 |
|---|---|---:|
| R0 | Base | 1.7167 |
| R1 | Base + RAG | 1.8667 |
| R2 | SFT | 1.8333 |
| R3 | SFT + RAG | 1.5833 |
| R4 | DPO v2 | 1.6833 |
| R5 | DPO v2 + RAG | 1.7333 |

主要差值：R0→R1 +0.1500，R2→R3 -0.2500，R4→R5 +0.0500，R0→R2 +0.1167，R2→R4 -0.1500。

## 5. 解读

- SFT 改善医疗内容、简洁表达和部分证据边界；安全/指令目标只部分达成。
- DPO 在医疗/安全偏好方向有改善迹象，但通用、指令和结构化任务存在退化。
- RAG 在 Base 上为正、在 SFT 上为负；证据更支持训练分布与 RAG prompt/Context wrapper 不匹配，不支持简单声称 RAG 无效。
- RAG ON 同时开启检索、RAG system prompt 和 Context，差值不能单独归因于检索器。

## 6. 安全与限制

- `release_allowed=false`，未获得公开发布或临床使用许可；
- 模型 judge/代理复核不是医生审核；
- validation_v4 仅 60 条，test 和 medical safety holdout 未读取；
- 检索 Context 可能缺少分诊/升级行动，从而抑制模型已学安全行为；
- judge v4 有少量 RAG OFF 规则误判，已在代理复核中单列。

复现命令见 `docs/delivery/reproducibility.md`，数据边界见 `docs/delivery/data_card.md`，最终版本见 `docs/evaluation/final_version_register.md`。
