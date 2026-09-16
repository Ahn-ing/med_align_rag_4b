# 项目代理规则

## 项目定位与阶段控制

`med_align_rag_4b` 是按阶段构建的可复现求职展示项目：`RAG → SFT → Evaluation → DPO → 联合评测`。默认逻辑模型标识为 `Qwen/Qwen3-4B`。项目不面向临床部署，不得声称具有医疗诊断能力。

1. 遵循 `PLANS.md`，一次只执行一个明确批准的阶段。
2. 未经用户确认不得进入下一阶段；当前阶段获批不代表训练或后续评测获批。
3. 当前阶段未明确允许时，不得运行冻结安全 holdout、下载模型或开展 DPO。

## 旧项目边界

同级旧项目 `med_align_rag` 只作为经验和获批 RAG 资产的只读来源，不得修改、删除、覆盖或重组其中任何文件。每项迁移必须记录源/目标相对路径和 SHA256。不得迁移旧 SFT/DPO 数据、adapter、checkpoint、模型输出、不安全回答、抽样结果或评测成绩。

## 模型、路径和公共函数

- 本地模型目录只能通过运行参数传入；不得硬编码或持久化机器绝对路径。
- 运行前在 manifest 中固定获批模型 revision；训练和推理保持一致的 thinking 设置。
- 只有确有复用的辅助函数才放入合适的 `utils.py` 或具名公共模块。
- 仅在子系统内复用时优先放在子系统；跨子系统复用才提升至 `common/`。
- 哈希、路径、JSONL、manifest 等明确职责使用独立模块。
- 不创建空的或为假设需求准备的工具模块，不重复实现等价函数，不使用通配符导入。
- 公共函数不得隐式依赖本机路径、可变全局配置、GPU 或网络。

## RAG 不变量

- BGE-M3 dense embedding 使用 float32 和 L2 normalization；query 使用相同模型 revision 和 normalization。
- FAISS 使用 Inner Product；`embeddings.npy` 第 i 行对应 `metadata.jsonl` 第 i 条和 FAISS id i。
- 使用前校验哈希、数量、shape、dtype、norm、维度和 metric。
- 不得静默修改 chunking 或 embedding 参数。

## SFT 数据与医疗安全

训练抽样前冻结 validation、test 和医疗安全 holdout。正式记录必须具有稳定 ID、来源、许可证、医疗权威依据和审核通过状态，且无评测泄漏、未解决重复/冲突、关键截断或零 assistant 监督。Phase 3/4 快速版经用户明确批准，允许 `agent_evidence_reviewed` 分别作为内部评测冻结和内部工程训练的审核状态，审核角色固定为 `codex_research_reviewer` 且 `release_allowed=false`；它不是医生审核、临床验证或公开发布许可。配比按有效 assistant 监督 token 计算。

安全数据需覆盖急救、门诊复诊、家庭观察、信息澄清、药物边界和不确定性，不得训练成所有问题都建议去医院。自动规则、模型辅助评分和人工审核分别报告；自动结果不得称为临床准确率。

## 工程与验证

- 使用 Python 3.10+、`pathlib`、类型标注和简洁 docstring。
- 优先简单、直接、可维护的实现，不引入无明确收益的依赖或抽象。
- 采用原子产物发布，拒绝意外覆盖已有实验。
- 不修改无关文件；未经要求不 commit 或 push。
- 修改后执行语法、最小有效测试、相关契约/集成测试以及哈希和对齐检查，并如实报告未验证内容。
- 每次修改文件后，只调用 `open_in_codex` 打开两份 Review：当前对话轮次（`view: "last-turn"`）和累计未提交改动（`view: "unstaged"`）。不要额外打开文件面板。被 `.gitignore` 忽略、未出现在 Review 中或需要补充查看的文件，必须在最终总结中列出相对路径、类型、数量和用途；只有用户明确要求时才单独打开具体文件。
