# med_align_rag_4b

一个以 `Qwen/Qwen3-4B` 为默认逻辑模型标识的低成本医疗大模型工程项目，规划流程为：

```text
RAG → SFT → Evaluation → DPO → Ablation
```

本项目只用于学习、研究和工程能力展示，不用于真实临床部署，不提供医疗诊断。

## 当前状态

Phase 0–8 已完成。Phase 8 最终组合为 `validation_v4 + formal_v4 + phase8-full-rag-judge-v4`：R0–R5 各 60 条，最终 judge 覆盖 360 条，并已完成确定性规则、模型 judge 和代理复核。RAG 对 Base 的增量为 +0.1500，对 SFT 的增量为 -0.2500，表明当前 SFT 与 RAG 接口存在明显负交互。`formal_v1/v2/v3`、旧 judge 和 v5 相关结果仅保留为历史审计；Phase 9 仍未解锁。

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
