# med_align_rag_4b

一个以 `Qwen/Qwen3-4B` 为默认逻辑模型标识的低成本医疗大模型工程项目，规划流程为：

```text
RAG → SFT → Evaluation → DPO → Ablation
```

本项目只用于学习、研究和工程能力展示，不用于真实临床部署，不提供医疗诊断。

## 当前状态

Phase 0–9 已完成。Phase 8 最终组合为 `validation_v4 + formal_v4 + phase8-full-rag-judge-v4`：R0–R5 各 60 条，最终 judge 覆盖 360 条，并已完成确定性规则、模型 judge 和代理复核。RAG 对 Base 的增量为 +0.1500，对 SFT 的增量为 -0.2500，表明当前 SFT 与 RAG 接口存在明显负交互。

## 关键工程结果

| 阶段 | 最终版本 | 结果 | 边界 |
|---|---|---|---|
| SFT | `qwen3_4b_lora_v2_plus_repair_r2_formal` | 9,495 条、298 steps、2 epochs；validation loss 1.9274 → 1.8060 | loss 不等于医疗能力 |
| DPO | `dpo_formal_v2` | preference accuracy 0.96，mean reward margin 6.3207 | 偏好对排序指标，不是 blind win rate |
| 消融 | `formal_v4 + judge-v4` | RAG 对 Base +0.1500，对 SFT -0.2500，对 DPO v2 +0.0500 | validation_v4 仅 60 条 |

最重要的发现是：当前证据支持“训练分布与 RAG 接口不匹配”，不支持简单声称 RAG 本身无效。

## 快速验证

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/unit/test_rag_ablation_entry.py tests/unit/test_rag_judge.py tests/unit/test_rag_scoring.py
```

RAG 使用 `.[rag,dev]`，SFT/DPO 研究环境使用 `.[research,dev]`。完整 CPU/GPU 入口、数据契约和产物路径见 [复现指南](docs/delivery/reproducibility.md)。

## 关键约束

- 旧项目 `med_align_rag` 只读。
- 本地模型目录必须通过运行参数传入，代码和配置不保存机器绝对路径。
- 数据、配置、模型运行和评测产物保留必要的版本/路径记录；哈希只在具体验收门禁需要时计算。
- 一次只执行一个经批准的阶段。
- 医疗安全评测区分自动规则、模型辅助评分与人工语义审核。

## Phase 0 验证

```bash
python -m compileall src tests
python -m pytest tests/unit/test_dpo_phase7.py
```

Phase 1 的迁移范围、数据契约、风险和验证结果见 `docs/phases/phase_1.md`。

Phase 2 的冻结 query、模型文件指纹、新旧检索比较和限制见 `docs/phases/phase_2.md`。

Phase 7 的偏好数据、DPO 训练、preference validation 和公开 regression 见 `docs/phases/phase7.md`；Phase 8 的最终消融矩阵、冻结输入、评测分层和门禁见 `docs/phases/phase8.md`；最终/废弃版本的唯一登记入口见 `docs/evaluation/final_version_register.md`。

## 交付文档

- [复现指南](docs/delivery/reproducibility.md)
- [数据卡](docs/delivery/data_card.md)
- [模型卡](docs/delivery/model_card.md)
- [实验结果与失败分析](docs/delivery/experiment_failure_analysis.md)
- [求职讲述稿](docs/delivery/interview_narrative.md)
- [最终证据目录](docs/delivery/phase9_evidence_catalog.md)
- [最终版本与废弃说明](docs/evaluation/final_version_register.md)

## 重要限制

- 数据、adapter 和评测结果均为 `release_allowed=false`。
- 模型 judge 和代理复核不是医生审核或临床验证。
- test 与 medical safety holdout 未读取。
- `formal_v1/v2/v3`、旧 judge 和 v5 只作历史审计。
