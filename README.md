# med_align_rag_4b

一个以 `Qwen/Qwen3-4B` 为默认逻辑模型标识的低成本医疗大模型工程项目，规划流程为：

```text
RAG → SFT → Evaluation → DPO → Combined Evaluation
```

本项目只用于学习、研究和工程能力展示，不用于真实临床部署，不提供医疗诊断。

## 当前状态

Phase 0–7 已完成；Phase 7 v2 DPO 已通过内部工程验收和公开 regression。**Phase 8：联合评测** 已进入进行中状态，当前只整理评测范围、冻结输入和入口，尚未运行联合评测；test、medical safety holdout 和未批准的生成式评测保持锁定。

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

Phase 7 的偏好数据、DPO 训练、preference validation 和公开 regression 见 `docs/phases/phase7.md`；Phase 8 的联合评测矩阵、冻结输入和门禁见 `docs/phases/phase8.md`。
