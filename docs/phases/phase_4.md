# Phase 4：快速版 SFT 清洗、审核与构建

状态：**已完成（COMPLETED）**。本阶段只构建数据，Phase 5 保持锁定。

## 输入与边界

入口处重新确认 Phase 3 的 validation 60、test 90 和 medical safety holdout 30 已冻结且 manifest 有效。Phase 4 对 holdout 只读取不可逆问题指纹；没有运行或输出 holdout 内容。

通用候选来自固定 revision 的 Databricks Dolly 15k 原始公开 JSONL。医疗、安全和指令数据按权威事实表独立编写。旧项目保持只读且本阶段未读取其 SFT 派生数据、抽样、模型输出、adapter、checkpoint 或 DPO 资产。

## 数据流与门禁

处理顺序为：来源登记 → 统一候选 → 稳定 ID → NFKC 最小清洗 → 许可证和个人数据门禁 → 重复与冲突 → 评测/RAG 指纹泄漏 → 安全结构规则 → 代理证据审核 → Qwen chat template → 长度与 assistant supervision → 分层 token 选样 → 原子发布。

每条医疗记录必须关联 evidence registry，审核状态固定为 `agent_evidence_reviewed`、审核角色固定为 `codex_research_reviewer`。该状态仅用于内部工程训练，不是医生审核或临床验证，公开发布仍被禁止。

候选守恒计数：26,439 = 672 来源/安全拒绝 + 16 重复 + 297 冲突记录 + 0 泄漏 + 26 超长 + 17,000 合格未选中 + 8,428 正式记录。所有同问异答冲突组整体隔离，没有随机保留其中答案。

## Tokenizer、选样和输出

仅下载并使用 `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` 的 tokenizer/config 文件，没有下载模型权重。使用官方 chat template、关闭 thinking，通过模板前缀差分定位 assistant 正文和结束标记；零监督或超过 2,048 token 的记录禁止发布。

固定 seed `20260914`，按有效 assistant token 选出 500,041 token：医疗 275,011、通用 100,000、安全 100,026、指令 25,004。六个安全子类均在批准目标 ±1% 内。正式集 8,428 条，smoke 子集 180 条。

输出包括 quarantine、reviewed registry、正式 train/smoke、九类审计报告、数据卡和 SHA256 manifest。`training_allowed=true` 只允许进入后续获批的内部 Phase 5 工程训练；`release_allowed=false`。

## 限制与停止点

项目自编来源占约 80% token，医疗事实主题和表达模板仍较集中，因此本版本适合低成本跑通流程，不证明医疗效果或来源多样性。没有加载模型权重、运行推理、训练、validation/test/holdout、DPO、commit 或 push。Phase 5 必须等待用户单独批准。
