# Phase 3：SFT 数据分析和快速版评测冻结

状态：**已完成（COMPLETED）**。用户明确批准 180 条快速版规模及代理证据审核门禁；validation、test 和 medical safety holdout 已在任何正式训练集构建前冻结。Phase 4 保持锁定。

## 输入、来源与边界

Phase 1/2 manifest、RAG 静态产物和 32 条 compatibility query 均重新通过哈希门禁。只读分析使用旧项目的两个上游原始缓存：`shibing624/medical` revision `6e219f1a14856833ee436063d3b73c5f1ab9cfb9` 和 `m-a-p/COIG-CQIA` revision `8b55868c6168adf86c30e7ca0f782cca1c514297`。

medical 上游声明 Apache-2.0；仅 6 个经过 PII、重复和 overlap 门禁的原始问法进入 validation，原答案未使用。COIG 总许可证未知，仅用于任务分类与难度参考，未复制文本。其余 174 条为项目独立编写。没有读取或复制旧项目的派生 analysis、eval、experiments、corrective safety、模型输出或 DPO 数据，也未修改旧项目。

## 原始数据分析

共只读分析 1,995,666 条原始记录。规范化采用 Unicode NFKC、常见标点、连续空白和英文 casefold，不删除医学数字、剂量、单位、药名或否定词。

| 指标 | 数量 |
|---|---:|
| 原文 QA 额外重复记录 | 41,717 |
| 规范化 QA 额外重复记录 | 41,801 |
| 规范化问题额外重复记录 | 51,299 |
| 同问异答问题组 | 9,159 |
| 跨 source 问题组 | 81 |
| 跨原始 split 问题组 | 135 |
| medical train / valid / test 与 RAG question overlap | 367,007 / 80 / 74 |
| COIG 与 RAG question overlap | 82 |
| medical train 与 compatibility query overlap | 32 |

## 快速版评测集

用户批准将原计划 920 条缩小为 180 条。Validation 60 条：医疗问答 20、RAG 忠实性 8、通用 10、指令跟随 6、结构化输出 6、医疗安全 10。Test 90 条：29/13/14/9/9/16。Medical safety holdout 30 条：立即急救 6、门诊复诊 4、家庭观察 5、信息澄清 4、药物边界 6、不确定性 5。

所有医疗案例引用 `data/evaluation/evidence_registry_v1.json` 中的 WHO、CDC、FDA 或 NHS 一手来源。审核状态为 `agent_evidence_reviewed`，角色固定为 `codex_research_reviewer`，全部 `release_allowed=false`。这表示基于权威来源的内部工程审核，不是医生审核、临床验证或公开发布许可。

RAG 忠实性案例携带独立权威 context，不复制许可证未知的旧 RAG 文本。冻结前已验证稳定 ID、原文/规范化问题和 QA、同问异答、group、原始 split、RAG corpus 和 compatibility query；禁止级 overlap 为 0。

## 冻结产物与隔离

- Validation：60 条，SHA256 `5e0c19127e2e135f1035c3b5c67a1c10d016b3454b62cd30cac0658d24301807`。
- Test：90 条，SHA256 `bed02eaae12ff229ed8fd077ebade3607509944dec05120e95c8c366c6679380`。
- Medical safety holdout：30 条，SHA256 `35ff383c07928d99113169d99c61c6a555399b90ebdf11ee2a77b9e6afb70dcf`。
- Holdout fingerprints：30 个，SHA256 `f5e5034f2bcef7afb60004a35226a18d268ee2c11ef1d714715a40d3511e2006`。

三个 split 均按固定 seed `20260914`、稳定排序、原子 JSONL 和拒绝覆盖策略冻结，manifest 绑定 source、review、rubric 和 evidence registry SHA256。Phase 4 默认只能消费 holdout fingerprints；完整 prompt 不写入分析或 rejection log。

## 验证与限制

实际执行 compileall、ruff、全部单元/集成及 Phase 0–2 回归测试；39 项通过。另行重新加载三个冻结集并核对 count/hash、180 条 QA 唯一性、30 个 holdout 指纹及 prompt 不泄漏。

本阶段未加载或运行 Qwen3-4B，未调用商业 API judge，未生成目标模型回答，未构建正式训练集，未训练，未运行 holdout，未执行 DPO，未 commit/push，未进入 Phase 4。正式训练前仍需使用获批 Qwen3-4B tokenizer 完成 token 长度门禁；任何公开发布或医学效果主张仍需许可证复核和医学专家审核。
