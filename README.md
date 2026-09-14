# med_align_rag_4b

一个以 `Qwen/Qwen3-4B` 为默认逻辑模型标识的低成本医疗大模型工程项目，规划流程为：

```text
RAG → SFT → Evaluation → DPO → Combined Evaluation
```

本项目只用于学习、研究和工程能力展示，不用于真实临床部署，不提供医疗诊断。

## 当前状态

Phase 0、Phase 1 与 **Phase 2：RAG 兼容性验收** 已完成。Phase 3 及后续阶段保持锁定；Qwen 模型下载、训练、holdout 和 DPO 均未执行。

## 关键约束

- 旧项目 `med_align_rag` 只读。
- 本地模型目录必须通过运行参数传入，代码和配置不保存机器绝对路径。
- 数据、配置、模型运行和评测产物使用 SHA256 manifest 追踪。
- 一次只执行一个经批准的阶段。
- 医疗安全评测区分自动规则、模型辅助评分与人工语义审核。

## Phase 0 验证

```bash
python -m pytest
python -m compileall src tests
```

Phase 1 的迁移范围、数据契约、风险和验证结果见 `docs/phases/phase_1.md`。

Phase 2 的冻结 query、模型文件指纹、新旧检索比较和限制见 `docs/phases/phase_2.md`。
