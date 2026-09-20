# Phase 9 最终证据目录

本目录是求职交付材料的引用起点。最终版本与废弃说明以 `docs/evaluation/final_version_register.md` 为准。

## 1. 可引用证据

| 主题 | 可声明结论 | 主证据 | 必须保留的限制 |
|---|---|---|---|
| RAG 迁移 | 索引、embedding、metadata 与 FAISS id 完成对齐迁移 | `docs/phases/phase_1.md` | 不代表生成质量 |
| RAG 兼容性 | 32 条冻结 query 在 top-k 1/3/5/10 上新旧检索完全一致 | `docs/phases/phase_2.md` | 只证明冻结集合与协议下的工程等价 |
| SFT 数据 | 最终内部训练集 9,495 条、1,117,446 监督 token | `docs/phases/phase_6.md`、`data/sft/sources/source_registry.json` | 代理证据审核不是医生审核；`release_allowed=false` |
| SFT 训练 | 2 epochs、298 optimizer steps，active-assistant validation loss 1.9274 → 1.8060 | `docs/phases/phase_6.md`、Phase 6 run manifest | loss 不等于医疗能力 |
| DPO v2 | held-out preference accuracy 0.96，mean reward margin 6.3207 | `docs/phases/phase7.md`、`docs/evaluation/final_version_register.md` | 自建偏好对排序指标，不是生成式 blind win rate |
| Phase 8 消融 | R0–R5 各 60 条；RAG 在 Base 上 +0.1500，在 SFT 上 -0.2500 | `docs/phases/phase8.md`、`artifacts/reports/phase_8_rag/formal_v4_comprehensive_review.md` | validation_v4 仅 60 条；RAG ON 同时改变 prompt 和 Context |
| 失败分析 | SFT/RAG 存在训练分布与接口不匹配；DPO 存在过度医疗化和格式能力回落风险 | `artifacts/reports/phase_8_rag/formal_v4_comprehensive_review.md` | 规则误判、检索覆盖不足和真实退化必须分开 |

## 2. Phase 8 Git 交付边界

应追踪：

- `configs/evaluation/phase8_rag.yaml` 和 `phase8_rag_judge.yaml`；
- `src/med_align_rag_4b/evaluation/` 中的消融、judge 和评分实现；
- `scripts/evaluation/` 中最终 v4 构建、生成、评分、judge 和复核入口；
- `tests/unit/test_rag_ablation_entry.py`、`test_rag_judge.py`、`test_rag_scoring.py`；
- `data/evaluation/validation_v4/`、`review_registry_v4.json`、v4 rubric 和最终版本登记；
- `formal_v4` 的 R0–R5 manifest、judge v4 manifest、聚合规则报告、v4 审核报告和综合结论。

应忽略：

- R0–R5 原始生成、逐条 judge、规则分数和审核队列 JSONL；
- smoke、formal_v1/v2/v3、旧 judge 和未采纳的 v5 产物；
- validation_v3、v3 rubric、v3 构建脚本与 v4 中间 candidates；
- 模型权重、adapter、checkpoint、optimizer 状态、索引和 embedding 大文件。

## 3. 不得声明

- 不得宣称具有医疗诊断能力或临床有效性。
- 不得将自动规则、模型 judge 或代理复核称为医生审核。
- 不得声称 test 或 medical safety holdout 已评测。
- 不得将 DPO preference accuracy 称为生成式总体胜率。
- 不得将 RAG/SFT 负交互单独归因于检索器或模型本身。
