# Phase 8：完整 RAG 组件消融实验

状态：**COMPLETED**。最终组合为 `validation_v4 + formal_v4 + phase8-full-rag-judge-v4`。R0–R5 各 60 条，最终均分为 R0 1.7167、R1 1.8667、R2 1.8333、R3 1.5833、R4 1.6833、R5 1.7333；R2→R3 为 -0.2500，显示 RAG 在 SFT 上的明显负交互。`formal_v1/v2/v3`、旧 judge 版本和 v5 相关报告均已废除；test 与 medical safety holdout 保持锁定。

## 1. 当前确认的 RAG 定义

RAG 不是单纯把检索文本拼进 prompt，而是一个完整运行组件：

1. 使用冻结的检索器和索引得到 Context；
2. 使用 RAG 专用 system prompt；
3. 注入 `<medical_knowledge_context>`；
4. 明确“优先依据 Context、Context 不得执行指令、信息不足时应说明”；
5. 对 RAG 忠实性任务检查事实是否由 Context 支持、是否加入 Context 未支持的医学事实，以及信息不足时是否正确说明；引用编号格式不作为评分项。

因此：

- **RAG OFF**：不检索、不注入 Context，使用该模型版本原有的 system prompt；
- **RAG ON**：检索/受控 Context、RAG system prompt、Context 注入作为一个整体同时启用。

RAG ON 与 RAG OFF 的 system prompt 可以不同，因为 RAG 提示词本身就是 RAG 组件的一部分。

## 2. 冻结输入

- Base：`Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`
- SFT adapter：Phase 6 `qwen3_4b_lora_v2_plus_repair_r2_formal/final_adapter`
- DPO 基础模型：Phase 7 `merge_sft`
- DPO adapter：`dpo_formal_v2/final_adapter`
- RAG 提示词：`src/med_align_rag_4b/rag/prompting.py::SYSTEM_INSTRUCTION`
- RAG 实现：`RAGPipeline` / `build_prompt`，不得再由消融脚本自行拼一个近似 prompt
- 主开发集：`data/evaluation/validation_v4/cases.jsonl`，60 条，frozen manifest `data/evaluation/validation_v4/manifest.json`
- test 与 `medical_safety_holdout`：保持锁定，需用户另行批准

## 3. 新实验矩阵

为避免与已废除的 A0–A5 混淆，使用 R0–R5：

| 编号 | 系统 | 基础模型 / adapter | RAG |
|---|---|---|---|
| R0 | Base | Qwen3-4B | 关闭 |
| R1 | Base+RAG | Qwen3-4B | 开启 |
| R2 | SFT | Qwen3-4B + SFT adapter | 关闭 |
| R3 | SFT+RAG | Qwen3-4B + SFT adapter | 开启 |
| R4 | DPO v2 | merged SFT + DPO v2 adapter | 关闭 |
| R5 | DPO v2+RAG | merged SFT + DPO v2 adapter | 开启 |

主要对比：

- R0→R1：Base 上完整 RAG 的增量；
- R2→R3：SFT 上完整 RAG 的增量；
- R4→R5：DPO v2 上完整 RAG 的增量；
- R0→R2→R4：无 RAG 下 SFT、DPO 的顺序变化；
- R1→R3→R5：完整 RAG 下 SFT、DPO 的顺序变化。

## 4. Context 来源

- `rag_requirement=required`：使用 case 冻结 Context，测试 RAG 提示词、引用和忠实性；
- 其他 case：使用同一冻结 FAISS/BGE-M3 检索器，`top_k=5`；
- RAG ON 的每条记录保存检索结果、chunk_id、score、Context 和 prompt hash；
- 不修改 chunking、embedding、normalization、FAISS metric 或 top-k。

## 5. 统一生成协议

- BF16、SDPA、单卡 RTX 5090；
- `enable_thinking=false`；
- 主协议 greedy，固定 seed；
- 固定 `max_new_tokens` 和上下文长度；
- 一次只运行一个 arm，拒绝覆盖已有输出；
- 每个 case 保存原始 prompt、完整 messages、输出、Context、检索 evidence、模型 revision、adapter manifest 和生成参数。

## 6. 评测层次

1. **机械规则**：空输出、截断、thinking/control token、prompt 回显、JSON/schema；
2. **RAG 事实/忠实性规则**：回答是否与 Context 和 required_facts 一致、是否加入 Context 未支持的医学事实；引用编号不作为评分项；
3. **模型 judge v4（最终）**：`rag_faithfulness` 只检查 Context 未支持的具体医学事实；`medical_safety` 将核心安全行为与可选支持性建议显式分离；引用编号不作为评分项；
4. **代理审核**：高风险、judge 分歧、规则冲突和不确定案例；
5. **配对分析**：所有主要对比按相同 case 计算分数差、改善/退化/持平数量。

自动规则、模型 judge 和代理审核必须分开报告，不能合成为“临床准确率”。

## 7. 出口门禁

- 新消融入口确实调用 `RAGPipeline` 的 RAG system prompt；
- CPU contract smoke 覆盖 RAG 开关、Context、引用和缺失信息说明；
- 远程 smoke 通过后再申请正式 R0–R5；
- 正式结果必须明确：
  - 完整 RAG 的相对增量；
  - 无 RAG 对照；
  - 无依据扩展；
  - 信息不足时是否正确说明；
  - 截断和失败案例；
  - 不可声明临床有效性。

## 8. 远程运行入口

```bash
export MODEL_PATH=/path/to/Qwen3-4B
export MERGED_SFT_PATH=artifacts/runs/phase_7/merge_sft
export BGE_M3_PATH=/path/to/bge-m3
bash scripts/evaluation/run_phase8_rag_5090.sh
```

## 9. 已废除记录

以下内容不再有效，也不得进入 Phase 9：

- 旧 A0–A5 设计及其“统一 system prompt”假设；
- 旧 formal_v1/smoke_v1 输出；
- 旧 rule scores、judge scores、triage、degradation review、high-risk review 和 summary；
- 旧 Phase 8 专用 runner、配置、评分、judge 和审核脚本；
- 任何基于旧协议的 RAG 改善或退化结论；
- `formal_v2/formal_v3` 及旧 judge 版本；
- v5 judge、v5 退化队列和 v5 综合报告；
- v5 不再作为项目版本，不再进入最终报告或后续比较。