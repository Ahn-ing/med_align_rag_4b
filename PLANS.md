# med_align_rag_4b 路线图

## 目标与总门禁

使用 `Qwen/Qwen3-4B` 低成本、可复现地构建 `RAG → SFT → Evaluation → DPO → 联合评测`。本项目仅用于研究、学习和工程展示，不提供临床诊断。

- 一次只执行一个获批阶段，到达出口门禁后停止。
- 本地模型路径只通过运行参数传入。
- 重要数据、配置、运行和产物必须具备 SHA256 manifest。
- 冻结评测数据不得影响训练数据选择。
- 医疗安全结果分别报告规则检查、模型辅助评分和人工审核。

## 阶段

### Phase 0：项目骨架、环境和数据契约

状态：**COMPLETED**。已交付结构、依赖、中文文档、安全路径公共函数、schema、manifest 和 CPU 测试；无需下载模型或数据即可通过 import、语法和契约测试。Phase 1 继续保持锁定。

### Phase 1：RAG 迁移

状态：**COMPLETED**。RAG 代码和静态产物已迁移；用户已批准 corrected cleaned SHA256，源/目标哈希、数量、shape、norm、行对齐和 Inner Product metric 全部通过验收。Phase 2 保持锁定。

### Phase 2：RAG 兼容性

状态：**COMPLETED**。32 条冻结 query 在 top-k 1/3/5/10 上的新旧 row/chunk、metadata 和排序完全一致，query vector 与 score 最大绝对误差均为 0；Phase 3 保持锁定。

### Phase 3：SFT 分析和评测冻结

状态：**COMPLETED**。用户批准快速版 180 条内部评测规模及代理证据审核门禁；validation/test/medical safety holdout 已按 60/90/30 冻结，均绑定来源、权威证据、审核记录、泄漏检查和 SHA256。`release_allowed=false`，不是医生审核或临床验证。Phase 4 保持锁定。

### Phase 4：SFT 清洗和构建

状态：**COMPLETED**。用户批准将首版缩小为 500,000 有效 assistant 监督 token；实际冻结 500,041 token、8,428 条，医疗/通用/安全/指令为 55/20/20/5 且六个安全子类均在 ±1% 内。数据仅允许后续获批的内部工程训练，`release_allowed=false`；代理证据审核不是医生审核或临床验证。Phase 5 保持锁定。

### Phase 5：Qwen3-4B LoRA 冒烟

状态：**COMPLETED**。RTX 5090 BF16 LoRA preflight、1 step、8→16 optimizer steps 恢复、50-step short run、18 条 loss-only validation、checkpoint-50、adapter 新进程 reload 和三条工程 fixture 均通过工程验收。short run 未抛出致命 OOM，显存 allocated 首尾无持续增长；日志中的两次非致命 allocator 分配回退已在验收报告中保留。结果不代表医疗能力成绩；Phase 6 保持锁定。

### Phase 6：正式 SFT 与评估

状态：**COMPLETED**。`v2_plus_repair_r2` 两轮 LoRA SFT 已在 RTX 5090 完成并通过工程验收：9,495 条记录、298 optimizer steps、2 epochs，run 为 `qwen3_4b_lora_v2_plus_repair_r2_formal`；active assistant token validation loss 从 1.9274 降至 1.8060（上一版 `v2_plus_repair` 为 1.8413），checkpoint 与 final adapter 哈希校验通过。公开 regression 审核通过：greedy（do_sample=false，max_new_tokens=512）5 pass / 0 partial / 0 fail（base 3 pass / 1 partial / 1 fail），采样（temperature=0.3，top_p=0.95，top_k=20，max_new_tokens=1024，seed=420）5 pass / 0 partial / 0 fail（base 5 pass），两套协议 forbidden behavior 零命中，医疗表现相较 base 改善且通用能力无明显退化。`release_allowed=false`，代理审核不代表医生审核或临床验证；生成式隐藏评测、DPO 与联合评测继续锁定。Phase 7 保持锁定。

### Phase 7：DPO

状态：**IN PROGRESS**。单独获批后设计全新偏好数据和实现，不得直接复用旧 DPO 资产。

### Phase 8：联合评测

状态：**LOCKED**。冻结条件下评估 Base、Base+RAG、SFT、SFT+RAG、DPO 和 DPO+RAG。

### Phase 9：求职交付

状态：**LOCKED**。整理复现文档、数据卡、模型卡、演示、局限和失败分析。

## 已批准 SFT 默认值

- `Qwen/Qwen3-4B`，主流程关闭 thinking；max length 2048
- v1 快速版预算约 500K unique assistant tokens；v2/R2 正式内部工程训练集为 9,495 条、1,117,446 token，rule_pool（抢救+修复）约 61.3%、API 医疗 19.4%、general 19.2%、行为补丁 0.04%
- LoRA rank 16、alpha 32、dropout 0.05
- learning rate 1e-4，epochs 2，global batch 64，初始 micro-batch 4
- 每 100 optimizer steps 评估
- 训练与推理统一 system prompt：`你是一个谨慎、清晰且遵循证据边界的助手。`
- 固定版本商业 API judge；高风险案例必须人工审核
